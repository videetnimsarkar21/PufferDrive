"""Structured schema and validation for the puffer_park configuration (pufferlib/config/puffer_park.yaml)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from pufferlib.config_schema import (
    FINITE_NUMBER_CONSTRAINT,
    MAX_C_SEED,
    NONEMPTY_STRING_CONSTRAINT,
    NONNEGATIVE_INT_CONSTRAINT,
    NONNEGATIVE_NUMBER_CONSTRAINT,
    POSITIVE_INT_CONSTRAINT,
    POSITIVE_NUMBER_CONSTRAINT,
    PROBABILITY_CONSTRAINT,
    Activation,
    PackageName,
    TrainingConfig,
    VectorConfig,
    _constrained_field,
    _raise_config_error,
    _validate_field_constraints,
    _validate_value_constraint,
)


class ParkEnvironmentName(Enum):
    puffer_park = 0


class ParkPolicyName(Enum):
    CoPark = 0


@dataclass
class ParkEnvConfig:
    num_agents: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    max_agents_per_env: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    scene_bank_path: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    dt: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    episode_steps: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    max_controlled_per_scene: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    obs_partner_slots: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    obs_lane_slots: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    obs_boundary_slots: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    reward_conditioning: bool = MISSING
    goal_weight_min: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    goal_weight_max: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    collision_weight_min: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    collision_weight_max: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    goal_weight_default: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    collision_weight_default: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    reward_success: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_collision: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_offroad: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_handoff: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    threat_penalty_weight: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    reward_static_clearance: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_route_progress_per_m: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_subgoal: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_speed_target: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    speed_target_mps: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    speed_target_band_mps: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    reward_plan_progress_per_m: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    imitation_weight: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    imitation_scale: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    refine_gain: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    refine_hold_bonus: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    refine_tanh_scale_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    threat_decay_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    threat_squash_mps: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    trust_cross_track_scale_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    trust_clearance_slope_per_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    trust_clearance_threshold_m: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    success_position_tolerance_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    success_heading_tolerance_rad: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    success_speed_tolerance_mps: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    success_hold_steps: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    refine_budget_steps: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    curriculum_ramp_steps: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    tier_weight_sparse: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    tier_weight_medium: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    tier_weight_dense: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    stagger_first_episode: bool = MISSING
    report_interval: int = _constrained_field(POSITIVE_INT_CONSTRAINT)


@dataclass
class ParkPolicyConfig:
    context_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    entity_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    backbone_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    backbone_num_layers: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    actor_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    critic_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    activation: Activation = MISSING
    layer_norm: bool = MISSING
    use_prior: bool = MISSING
    residual_logit_bound: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    reliance_min: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    reliance_max: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    release_rate_accel: float = _constrained_field(PROBABILITY_CONSTRAINT)
    release_rate_steer: float = _constrained_field(PROBABILITY_CONSTRAINT)
    prior_sigma_accel_mps2: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    prior_sigma_steer_rad: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)


@dataclass
class PufferParkConfig:
    load_model_path: str | None = MISSING
    load_id: str | None = MISSING
    wandb: bool = MISSING
    wandb_project: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    wandb_group: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    run_name: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    neptune: bool = MISSING
    neptune_name: str = MISSING
    neptune_project: str = MISSING
    tb: bool = MISSING
    tag: str | None = MISSING
    package: PackageName = MISSING
    env_name: ParkEnvironmentName = MISSING
    policy_name: ParkPolicyName = MISSING
    rnn_name: str | None = MISSING
    vec: VectorConfig = MISSING
    env: ParkEnvConfig = MISSING
    policy: ParkPolicyConfig = MISSING
    train: TrainingConfig = MISSING
    sweep: dict = MISSING
    eval: Any = None
    no_model_upload: bool = False
    git: dict | None = None


def normalize_puffer_park_config(config, context="load"):
    """Validate config structure and types, then return a normalized plain dictionary."""
    if not isinstance(config, Mapping):
        _raise_config_error(context, "root", "must be a mapping")
    try:
        validated = OmegaConf.merge(OmegaConf.structured(PufferParkConfig), dict(config))
        missing_keys = sorted(OmegaConf.missing_keys(validated))
        if missing_keys:
            _raise_config_error(context, missing_keys[0], "is required")
        container = OmegaConf.to_container(validated, resolve=True, enum_to_str=True, throw_on_missing=True)
    except (OmegaConfBaseException, TypeError, ValueError) as exc:
        error_path = getattr(exc, "full_key", None) or "root"
        _raise_config_error(context, error_path, str(exc).splitlines()[0])
    return container


def _validate_park_env(env, context):
    from pufferlib.ocean.park import binding

    if env["max_agents_per_env"] > binding.MAX_AGENTS_PER_SCENE:
        _raise_config_error(context, "env.max_agents_per_env", f"must not exceed {binding.MAX_AGENTS_PER_SCENE}")
    if env["num_agents"] % env["max_agents_per_env"] != 0:
        _raise_config_error(context, "env.num_agents", "must be a multiple of env.max_agents_per_env")
    slot_limits = (
        ("obs_partner_slots", binding.MAX_PARTNER_SLOTS),
        ("obs_lane_slots", binding.MAX_LANE_SLOTS),
        ("obs_boundary_slots", binding.MAX_BOUNDARY_SLOTS),
    )
    for field_name, limit in slot_limits:
        if env[field_name] > limit:
            _raise_config_error(context, f"env.{field_name}", f"must not exceed {limit}")
    for weight_name in ("goal_weight", "collision_weight"):
        if env[f"{weight_name}_min"] > env[f"{weight_name}_max"]:
            _raise_config_error(context, f"env.{weight_name}_min", f"must not exceed env.{weight_name}_max")
    if env["tier_weight_sparse"] + env["tier_weight_medium"] + env["tier_weight_dense"] <= 0:
        _raise_config_error(context, "env.tier_weight_dense", "tier weights must not all be zero")


def _validate_park_training(config, context):
    train = config["train"]
    if train["evaluation_interval_epochs"] is not None or train["cosim_debug_evals"] or train["render"]:
        _raise_config_error(context, "train", "puffer_park has no benchmark, co-sim or replay-render hooks yet")
    for field_name in ("batch_size", "bptt_horizon"):
        if train[field_name] != "auto":
            _validate_value_constraint(train[field_name], POSITIVE_INT_CONSTRAINT, context, f"train.{field_name}")
    if train["bptt_horizon"] == "auto":
        _raise_config_error(context, "train.bptt_horizon", "must be set explicitly for puffer_park")
    effective_minibatch_size = min(train["minibatch_size"], train["max_minibatch_size"])
    if effective_minibatch_size % train["bptt_horizon"] != 0:
        _raise_config_error(context, "train.minibatch_size", "the effective minibatch size must divide by bptt_horizon")
    total_agents = config["vec"]["num_envs"] * config["env"]["num_agents"]
    batch_size = total_agents * train["bptt_horizon"] if train["batch_size"] == "auto" else train["batch_size"]
    if batch_size < train["minibatch_size"]:
        _raise_config_error(context, "train.minibatch_size", f"exceeds the batch size ({batch_size})")
    if batch_size // train["bptt_horizon"] < total_agents:
        _raise_config_error(context, "train.batch_size", f"must hold one bptt segment per agent ({total_agents})")


def validate_puffer_park_config(config, context) -> None:
    """Validate field and cross-field semantics on a normalized puffer_park config."""
    _validate_field_constraints(config, PufferParkConfig, context)
    if config["load_id"] is not None and not (config["wandb"] or config["neptune"]):
        _raise_config_error(context, "load_id", "requires wandb or neptune")
    if config["rnn_name"] is not None:
        _raise_config_error(context, "rnn_name", "CoPark is feed-forward; set rnn_name to null")
    _validate_park_env(config["env"], context)
    policy = config["policy"]
    if policy["reliance_min"] > policy["reliance_max"]:
        _raise_config_error(context, "policy.reliance_min", "must not exceed policy.reliance_max")
    _validate_park_training(config, context)
    vector_config = config["vec"]
    if vector_config["num_workers"] != "auto":
        _validate_value_constraint(vector_config["num_workers"], POSITIVE_INT_CONSTRAINT, context, "vec.num_workers")
    if vector_config["seed"] is not None and vector_config["seed"] > MAX_C_SEED:
        _raise_config_error(context, "vec.seed", f"must not exceed the C RNG maximum {MAX_C_SEED}")


def validate_puffer_park_resources(config, context):
    """Check filesystem inputs separately from config semantics."""
    scene_bank_path = config["env"]["scene_bank_path"]
    if not os.path.isfile(scene_bank_path):
        _raise_config_error(
            context,
            "env.scene_bank_path",
            f"file does not exist: {scene_bank_path}; generate it with python -m pufferlib.ocean.park.scenes",
        )
    load_model_path = config.get("load_model_path")
    if load_model_path not in (None, "latest") and not os.path.isfile(load_model_path):
        _raise_config_error(context, "load_model_path", f"checkpoint does not exist: {load_model_path}")
