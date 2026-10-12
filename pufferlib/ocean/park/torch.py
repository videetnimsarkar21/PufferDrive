"""CoPark actor-critic: a bounded residual over a phase-gated, threat-released Stanley log-prior (arXiv:2606.04149)."""

import torch
from torch import nn

import pufferlib.pytorch

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


def make_encoder(in_features, out_features, activation, layer_norm):
    layers = [pufferlib.pytorch.layer_init(nn.Linear(in_features, out_features))]
    if layer_norm:
        layers.append(nn.LayerNorm(out_features))
    layers += [ACTIVATIONS[activation](), pufferlib.pytorch.layer_init(nn.Linear(out_features, out_features))]
    return nn.Sequential(*layers)


def make_head(in_features, hidden_size, out_features, activation, output_std):
    return nn.Sequential(
        pufferlib.pytorch.layer_init(nn.Linear(in_features, hidden_size)),
        ACTIVATIONS[activation](),
        pufferlib.pytorch.layer_init(nn.Linear(hidden_size, out_features), std=output_std),
    )


def masked_max_pool(encoded, valid_counts):
    """Max over the first valid_counts slots; rows with no valid slot pool to zeros."""
    slot_idx = torch.arange(encoded.shape[1], device=encoded.device)
    valid = slot_idx[None, :] < valid_counts[:, None]
    pooled = encoded.masked_fill(~valid[:, :, None], torch.finfo(encoded.dtype).min).amax(dim=1)
    return torch.where(valid_counts[:, None] > 0, pooled, torch.zeros_like(pooled))


class CoPark(nn.Module):
    def __init__(
        self,
        env,
        context_hidden_size=128,
        entity_hidden_size=64,
        backbone_hidden_size=256,
        backbone_num_layers=2,
        actor_hidden_size=256,
        critic_hidden_size=256,
        activation="gelu",
        layer_norm=True,
        use_prior=True,
        residual_logit_bound=4.0,
        reliance_min=1.0,
        reliance_max=4.0,
        release_rate_accel=0.95,
        release_rate_steer=0.40,
        prior_sigma_accel_mps2=0.5,
        prior_sigma_steer_rad=0.1,
    ):
        super().__init__()
        self.is_continuous = False
        self.use_prior = use_prior
        self.residual_logit_bound = residual_logit_bound
        self.reliance_min = reliance_min
        self.reliance_max = reliance_max
        self.release_rate_accel = release_rate_accel
        self.release_rate_steer = release_rate_steer
        self.prior_sigma_accel_mps2 = prior_sigma_accel_mps2
        self.prior_sigma_steer_rad = prior_sigma_steer_rad

        self.context_size = env.partner_offset
        self.tail_offset = env.ego_features + env.conditioning_features
        self.phase_offset = env.ego_phase_offset
        self.phase_maneuver = env.phase_maneuver
        self.phase_refine = env.phase_refine
        self.tail_threat_idx = env.tail_threat_idx
        self.tail_teacher_accel_idx = env.tail_teacher_accel_idx
        self.tail_teacher_steer_idx = env.tail_teacher_steer_idx
        self.max_accel_mps2 = env.max_accel_mps2
        self.max_steer_rad = env.max_steer_rad
        self.entity_layout = (
            (env.partner_offset, env.partner_slots, env.partner_features),
            (env.lane_offset, env.lane_slots, env.lane_features),
            (env.boundary_offset, env.boundary_slots, env.boundary_features),
        )
        self.count_offset = env.count_offset
        self.register_buffer("accel_values", torch.tensor(env.accel_values_mps2, dtype=torch.float32), persistent=False)
        self.register_buffer("steer_values", torch.tensor(env.steer_values_rad, dtype=torch.float32), persistent=False)
        self.action_count = len(env.accel_values_mps2) * len(env.steer_values_rad)

        self.context_encoder = make_encoder(self.context_size, context_hidden_size, activation, layer_norm)
        self.entity_encoders = nn.ModuleList(
            make_encoder(features, entity_hidden_size, activation, layer_norm) for _, _, features in self.entity_layout
        )
        backbone_layers = []
        backbone_in = context_hidden_size + len(self.entity_layout) * entity_hidden_size
        for _ in range(backbone_num_layers):
            backbone_layers += [
                ACTIVATIONS[activation](),
                pufferlib.pytorch.layer_init(nn.Linear(backbone_in, backbone_hidden_size)),
            ]
            backbone_in = backbone_hidden_size
        backbone_layers.append(ACTIVATIONS[activation]())
        self.backbone = nn.Sequential(*backbone_layers)
        self.actor_head = make_head(backbone_in, actor_hidden_size, self.action_count, activation, output_std=0.01)
        self.reliance_head = pufferlib.pytorch.layer_init(nn.Linear(backbone_in, 1), std=0.01)
        self.critic_head = make_head(backbone_in, critic_hidden_size, 1, activation, output_std=1.0)

    def encode_observations(self, observations):
        features = [self.context_encoder(observations[:, : self.context_size])]
        for encoder_idx, (offset, slots, slot_features) in enumerate(self.entity_layout):
            entities = observations[:, offset : offset + slots * slot_features].view(-1, slots, slot_features)
            valid_counts = observations[:, self.count_offset + encoder_idx].long().clamp(0, slots)
            features.append(masked_max_pool(self.entity_encoders[encoder_idx](entities), valid_counts))
        return self.backbone(torch.cat(features, dim=1))

    def stanley_log_prior(self, observations, hidden):
        """Channel-wise Gaussian log-priors around the Stanley command, scaled by threat-released reliance."""
        tail = observations[:, self.tail_offset :].float()
        teacher_accel = tail[:, self.tail_teacher_accel_idx] * self.max_accel_mps2
        teacher_steer = tail[:, self.tail_teacher_steer_idx] * self.max_steer_rad
        accel_gap = (self.accel_values[None, :] - teacher_accel[:, None]) / self.prior_sigma_accel_mps2
        steer_gap = (self.steer_values[None, :] - teacher_steer[:, None]) / self.prior_sigma_steer_rad
        log_prior_accel = torch.log_softmax(-0.5 * accel_gap.square(), dim=1)
        log_prior_steer = torch.log_softmax(-0.5 * steer_gap.square(), dim=1)
        threat_intensity = tail[:, self.tail_threat_idx]
        reliance_span = self.reliance_max - self.reliance_min
        reliance = self.reliance_min + reliance_span * torch.sigmoid(self.reliance_head(hidden).float().squeeze(1))
        reliance_accel = reliance * (1.0 - self.release_rate_accel * threat_intensity)
        reliance_steer = reliance * (1.0 - self.release_rate_steer * threat_intensity)
        joint_prior = (reliance_accel[:, None, None] * log_prior_accel[:, :, None]) + (
            reliance_steer[:, None, None] * log_prior_steer[:, None, :]
        )
        phases = observations[:, self.phase_offset :].float()
        maneuver_gate = phases[:, self.phase_maneuver] + phases[:, self.phase_refine]
        return maneuver_gate[:, None] * joint_prior.reshape(-1, self.action_count)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        raw_logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        if not self.use_prior:
            return raw_logits, value
        residual = self.residual_logit_bound * torch.tanh(raw_logits.float())
        return residual + self.stanley_log_prior(observations, hidden), value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)
