"""CoPark reactive multi-agent parking environment (arXiv:2606.04149) as a native PufferEnv."""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.park import binding

LOG_MIN_AGENT_EPISODES = 64


class Park(pufferlib.PufferEnv):
    def __init__(
        self,
        num_agents=1024,
        max_agents_per_env=32,
        scene_bank_path="pufferlib/resources/park/park_scenes.bin",
        dt=0.1,
        episode_steps=400,
        max_controlled_per_scene=32,
        obs_partner_slots=8,
        obs_lane_slots=48,
        obs_boundary_slots=64,
        reward_conditioning=True,
        goal_weight_min=0.5,
        goal_weight_max=2.0,
        collision_weight_min=0.5,
        collision_weight_max=3.0,
        goal_weight_default=1.25,
        collision_weight_default=1.75,
        reward_success=1.0,
        reward_collision=-1.0,
        reward_offroad=-1.0,
        reward_handoff=0.2,
        threat_penalty_weight=0.10,
        reward_static_clearance=-0.02,
        reward_route_progress_per_m=0.02,
        reward_subgoal=0.02,
        reward_speed_target=0.005,
        speed_target_mps=2.5,
        speed_target_band_mps=2.5,
        reward_plan_progress_per_m=0.03,
        imitation_weight=0.20,
        imitation_scale=0.50,
        refine_gain=0.25,
        refine_hold_bonus=0.08,
        refine_tanh_scale_m=0.10,
        threat_decay_m=4.5,
        threat_squash_mps=1.0,
        trust_cross_track_scale_m=1.5,
        trust_clearance_slope_per_m=20.0,
        trust_clearance_threshold_m=0.35,
        success_position_tolerance_m=0.8,
        success_heading_tolerance_rad=0.1745,
        success_speed_tolerance_mps=0.35,
        success_hold_steps=5,
        refine_budget_steps=80,
        curriculum_ramp_steps=0,
        tier_weight_sparse=0.25,
        tier_weight_medium=0.25,
        tier_weight_dense=0.5,
        stagger_first_episode=True,
        report_interval=32,
        buf=None,
        seed=0,
    ):
        if num_agents % max_agents_per_env != 0:
            raise pufferlib.APIUsageError(
                f"num_agents ({num_agents}) must be a multiple of max_agents_per_env ({max_agents_per_env})"
            )
        self.bank_info = binding.scene_bank_info(scene_bank_path)
        controlled_capacity = min(self.bank_info["max_scene_agent_count"], max_controlled_per_scene)
        if controlled_capacity > max_agents_per_env:
            raise pufferlib.APIUsageError(
                f"{scene_bank_path} controls up to {controlled_capacity} agents per scene, "
                f"more than max_agents_per_env={max_agents_per_env}"
            )
        self.num_agents = num_agents
        self.max_agents_per_env = max_agents_per_env
        self.num_envs = num_agents // max_agents_per_env
        self.report_interval = report_interval
        self.tick = 0
        self.rng = np.random.default_rng(seed)

        self.ego_features = binding.EGO_FEATURES
        self.conditioning_features = binding.CONDITIONING_FEATURES
        self.tail_features = binding.TAIL_FEATURES
        self.partner_slots = obs_partner_slots
        self.partner_features = binding.PARTNER_FEATURES
        self.lane_slots = obs_lane_slots
        self.lane_features = binding.LANE_FEATURES
        self.boundary_slots = obs_boundary_slots
        self.boundary_features = binding.BOUNDARY_FEATURES
        self.count_features = binding.COUNT_FEATURES
        self.ego_phase_offset = binding.EGO_PHASE_OFFSET
        self.num_phases = binding.NUM_PHASES
        self.phase_maneuver = binding.PHASE_MANEUVER
        self.phase_refine = binding.PHASE_REFINE
        self.tail_threat_idx = binding.TAIL_THREAT_IDX
        self.tail_teacher_accel_idx = binding.TAIL_TEACHER_ACCEL_IDX
        self.tail_teacher_steer_idx = binding.TAIL_TEACHER_STEER_IDX
        self.accel_values_mps2 = binding.ACCEL_VALUES_MPS2
        self.steer_values_rad = binding.STEER_VALUES_RAD
        self.max_accel_mps2 = binding.MAX_ACCEL_MPS2
        self.max_steer_rad = binding.MAX_STEER_RAD

        context_size = self.ego_features + self.conditioning_features + self.tail_features
        self.partner_offset = context_size
        self.lane_offset = self.partner_offset + obs_partner_slots * self.partner_features
        self.boundary_offset = self.lane_offset + obs_lane_slots * self.lane_features
        self.count_offset = self.boundary_offset + obs_boundary_slots * self.boundary_features
        self.num_obs = self.count_offset + self.count_features
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(binding.NUM_ACTIONS)
        # Slot counts are raw integers, not normalized features.
        self.obs_stats_feature_mask = np.ones(self.num_obs, dtype=bool)
        self.obs_stats_feature_mask[self.count_offset :] = False

        super().__init__(buf=buf)
        env_kwargs = dict(
            scene_bank_path=scene_bank_path,
            dt=float(dt),
            episode_steps=int(episode_steps),
            max_agents=int(max_agents_per_env),
            max_controlled_per_scene=int(max_controlled_per_scene),
            obs_partner_slots=int(obs_partner_slots),
            obs_lane_slots=int(obs_lane_slots),
            obs_boundary_slots=int(obs_boundary_slots),
            reward_conditioning=int(reward_conditioning),
            goal_weight_min=float(goal_weight_min),
            goal_weight_max=float(goal_weight_max),
            collision_weight_min=float(collision_weight_min),
            collision_weight_max=float(collision_weight_max),
            goal_weight_default=float(goal_weight_default),
            collision_weight_default=float(collision_weight_default),
            reward_success=float(reward_success),
            reward_collision=float(reward_collision),
            reward_offroad=float(reward_offroad),
            reward_handoff=float(reward_handoff),
            threat_penalty_weight=float(threat_penalty_weight),
            reward_static_clearance=float(reward_static_clearance),
            reward_route_progress_per_m=float(reward_route_progress_per_m),
            reward_subgoal=float(reward_subgoal),
            reward_speed_target=float(reward_speed_target),
            speed_target_mps=float(speed_target_mps),
            speed_target_band_mps=float(speed_target_band_mps),
            reward_plan_progress_per_m=float(reward_plan_progress_per_m),
            imitation_weight=float(imitation_weight),
            imitation_scale=float(imitation_scale),
            refine_gain=float(refine_gain),
            refine_hold_bonus=float(refine_hold_bonus),
            refine_tanh_scale_m=float(refine_tanh_scale_m),
            threat_decay_m=float(threat_decay_m),
            threat_squash_mps=float(threat_squash_mps),
            trust_cross_track_scale_m=float(trust_cross_track_scale_m),
            trust_clearance_slope_per_m=float(trust_clearance_slope_per_m),
            trust_clearance_threshold_m=float(trust_clearance_threshold_m),
            success_position_tolerance_m=float(success_position_tolerance_m),
            success_heading_tolerance_rad=float(success_heading_tolerance_rad),
            success_speed_tolerance_mps=float(success_speed_tolerance_mps),
            success_hold_steps=int(success_hold_steps),
            refine_budget_steps=int(refine_budget_steps),
            curriculum_ramp_steps=int(curriculum_ramp_steps),
            tier_weight_sparse=float(tier_weight_sparse),
            tier_weight_medium=float(tier_weight_medium),
            tier_weight_dense=float(tier_weight_dense),
            stagger_first_episode=int(stagger_first_episode),
        )
        env_ids = []
        for env_idx in range(self.num_envs):
            rows = slice(env_idx * max_agents_per_env, (env_idx + 1) * max_agents_per_env)
            env_ids.append(
                binding.env_init(
                    self.observations[rows],
                    self.actions[rows],
                    self.rewards[rows],
                    self.terminals[rows],
                    self.truncations[rows],
                    self.masks[rows],
                    self.random_seed,
                    **env_kwargs,
                )
            )
        self.c_envs = binding.vectorize(*env_ids)

    @property
    def random_seed(self):
        return int(self.rng.integers(0, 2**63, dtype=np.int64))

    def reset(self, seed=None):
        if seed is None:
            binding.vec_reset(self.c_envs)
        else:
            self.rng = np.random.default_rng(seed)
            binding.vec_reset(self.c_envs, [self.random_seed for _ in range(self.num_envs)])
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, LOG_MIN_AGENT_EPISODES)
            if log:
                info.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def teacher_actions(self):
        """Hybrid A* + Stanley baseline actions (pure pursuit on the route before the maneuver)."""
        actions = np.zeros(self.num_agents, dtype=np.int32)
        binding.vec_teacher_actions(self.c_envs, actions)
        return actions

    def agent_states(self):
        """Per agent: x, y, heading, speed, steer, status, phase, outcome, goal x/y/heading, threat intensity."""
        states = np.zeros((self.num_agents, binding.AGENT_STATE_FIELDS), dtype=np.float32)
        binding.vec_agent_states(self.c_envs, states)
        return states

    def scene_indices(self):
        return binding.vec_scene_indices(self.c_envs)

    def close(self):
        binding.vec_close(self.c_envs)
