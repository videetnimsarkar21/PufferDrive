#ifndef PUFFERLIB_OCEAN_PARK_PARK_H
#define PUFFERLIB_OCEAN_PARK_PARK_H

#include "geometry.h"
#include "scene_bank.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARK_NUM_ACCEL_ACTIONS 7
#define PARK_NUM_STEER_ACTIONS 13
#define PARK_NUM_ACTIONS (PARK_NUM_ACCEL_ACTIONS * PARK_NUM_STEER_ACTIONS)
static const float PARK_ACCEL_VALUES_MPS2[PARK_NUM_ACCEL_ACTIONS] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
static const float PARK_STEER_VALUES_RAD[PARK_NUM_STEER_ACTIONS]
    = {-0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
#define PARK_MAX_ACCEL_MPS2 2.0f
#define PARK_MAX_STEER_RAD 0.6f
#define PARK_MAX_FORWARD_SPEED_MPS 4.0f
#define PARK_MAX_REVERSE_SPEED_MPS 2.0f
#define PARK_MOTION_SPEED_EPSILON_MPS 0.05f

#define PARK_PHASE_NAVIGATE 0
#define PARK_PHASE_MANEUVER 1
#define PARK_PHASE_REFINE 2
#define PARK_NUM_PHASES 3

#define PARK_STATUS_INACTIVE 0
#define PARK_STATUS_ACTIVE 1
#define PARK_STATUS_PARKED 2
#define PARK_STATUS_REMOVED 3

#define PARK_OUTCOME_NONE 0
#define PARK_OUTCOME_SUCCESS 1
#define PARK_OUTCOME_COLLISION_VEHICLE 2
#define PARK_OUTCOME_COLLISION_STATIC 3
#define PARK_OUTCOME_OFFROAD 4
#define PARK_OUTCOME_REFINE_TIMEOUT 5
#define PARK_OUTCOME_HORIZON 6

#define PARK_EDGE_SOURCE_CURB 0
#define PARK_EDGE_SOURCE_OBSTACLE 1
#define PARK_EDGE_SOURCE_STATIC_VEHICLE 2

#define PARK_EGO_FEATURES 22
#define PARK_EGO_PHASE_OFFSET 5
#define PARK_CONDITIONING_FEATURES 2
#define PARK_TAIL_FEATURES 10
#define PARK_TAIL_THREAT_IDX 0
#define PARK_TAIL_CLEARANCE_IDX 1
#define PARK_TAIL_SLOT_LONG_IDX 2
#define PARK_TAIL_SLOT_LAT_IDX 3
#define PARK_TAIL_SLOT_HEADING_IDX 4
#define PARK_TAIL_TEACHER_ACCEL_IDX 5
#define PARK_TAIL_TEACHER_STEER_IDX 6
#define PARK_TAIL_TEACHER_CROSS_TRACK_IDX 7
#define PARK_TAIL_TEACHER_HEADING_IDX 8
#define PARK_TAIL_SPEED_IDX 9
#define PARK_PARTNER_FEATURES 11
#define PARK_LANE_FEATURES 5
#define PARK_BOUNDARY_FEATURES 6
#define PARK_COUNT_FEATURES 3
#define PARK_MAX_PARTNER_SLOTS 16
#define PARK_MAX_LANE_SLOTS 128
#define PARK_MAX_BOUNDARY_SLOTS 128

#define PARK_OBS_SPEED_NORM_MPS 4.0f
#define PARK_OBS_GOAL_NORM_M 30.0f
#define PARK_OBS_ROUTE_NORM_M 60.0f
#define PARK_OBS_LENGTH_NORM_M 5.0f
#define PARK_OBS_WIDTH_NORM_M 2.5f
#define PARK_OBS_PARTNER_RANGE_M 25.0f
#define PARK_OBS_ROAD_RANGE_M 20.0f
#define PARK_OBS_SEGMENT_NORM_M 5.0f
#define PARK_OBS_SLOT_ERROR_NORM_M 5.0f
#define PARK_OBS_CLEARANCE_NORM_M 2.0f
#define PARK_OBS_CROSS_TRACK_NORM_M 2.0f

#define PARK_STANLEY_GAIN 1.5f
#define PARK_STANLEY_SOFTENING_MPS 0.5f
#define PARK_TEACHER_SPEED_MPS 1.0f
#define PARK_TEACHER_DECEL_MPS2 0.8f
#define PARK_TEACHER_SPEED_GAIN_PER_S 4.0f
#define PARK_REFINE_POSITION_GAIN_PER_S 0.8f
#define PARK_REFINE_TEACHER_SPEED_MPS 0.5f
#define PARK_REFINE_SWEEP_M 0.6f
#define PARK_REFINE_ALIGNED_FRACTION 0.5f
#define PARK_NAV_TEACHER_SPEED_MPS 2.5f
#define PARK_NAV_TEACHER_DECEL_MPS2 0.6f
#define PARK_PLAN_SEARCH_WINDOW_POINTS 40
#define PARK_CUSP_REACH_M 0.35f
#define PARK_CUSP_SPEED_MPS 0.3f
#define PARK_PLAN_END_REACH_M 0.35f
#define PARK_PLAN_STALL_PROGRESS_M 0.01f
#define PARK_PLAN_STALL_STEPS 30
#define PARK_PLAN_STALL_MIN_FRACTION 0.7f
#define PARK_PREP_RADIUS_M 1.5f
#define PARK_PREP_HEADING_TOLERANCE_RAD 0.5f
#define PARK_ROUTE_SEARCH_WINDOW_SEGMENTS 4
#define PARK_SUBGOAL_REACH_TOLERANCE_M 1.0f
#define PARK_CLEARANCE_QUERY_RANGE_M 3.0f
#define PARK_RADIANS_TO_DEGREES (180.0f / PARK_PI)

typedef struct {
    uint64_t s[4];
} ParkRng;

static inline uint64_t park_splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline void park_rng_seed(ParkRng *rng, uint64_t seed) {
    uint64_t splitmix_state = seed;
    for (int word_idx = 0; word_idx < 4; word_idx++) {
        rng->s[word_idx] = park_splitmix64_next(&splitmix_state);
    }
}

static inline uint64_t park_rotl64(uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

// xoshiro256++ (Blackman & Vigna, public domain).
static inline uint64_t park_rng_next(ParkRng *rng) {
    uint64_t *s = rng->s;
    uint64_t result = park_rotl64(s[0] + s[3], 23) + s[0];
    uint64_t shifted = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= shifted;
    s[3] = park_rotl64(s[3], 45);
    return result;
}

static inline float park_rng_uniform(ParkRng *rng) {
    return (float) (park_rng_next(rng) >> 40) * (1.0f / 16777216.0f);
}

static inline int park_rng_below(ParkRng *rng, int upper_exclusive) {
    return (int) (((park_rng_next(rng) >> 32) * (uint64_t) upper_exclusive) >> 32);
}

// All fields are floats: the binding sums and averages logs element-wise.
typedef struct {
    float n;
    float score;
    float success_rate;
    float collision_rate;
    float collision_vehicle_rate;
    float collision_static_rate;
    float offroad_rate;
    float refine_timeout_rate;
    float horizon_timeout_rate;
    float handoff_rate;
    float refine_entered_rate;
    float success_position_error_m;
    float success_heading_error_deg;
    float path_length_m;
    float maneuver_count;
    float episode_return;
    float episode_length;
    float mean_threat;
    float imitation_reward;
    float scene_agent_count;
    float scene_tier;
} ParkLog;

typedef struct {
    float dt_s;
    int episode_steps;
    int max_agents;
    int max_controlled_per_scene;
    int partner_slots;
    int lane_slots;
    int boundary_slots;
    int reward_conditioning;
    float goal_weight_min;
    float goal_weight_max;
    float collision_weight_min;
    float collision_weight_max;
    float goal_weight_default;
    float collision_weight_default;
    float reward_success;
    float reward_collision;
    float reward_offroad;
    float reward_handoff;
    float threat_penalty_weight;
    float reward_static_clearance;
    float reward_route_progress_per_m;
    float reward_subgoal;
    float reward_speed_target;
    float speed_target_mps;
    float speed_target_band_mps;
    float reward_plan_progress_per_m;
    float imitation_weight;
    float imitation_scale;
    float refine_gain;
    float refine_hold_bonus;
    float refine_tanh_scale_m;
    float threat_decay_m;
    float threat_squash_mps;
    float trust_cross_track_scale_m;
    float trust_clearance_slope_per_m;
    float trust_clearance_threshold_m;
    float success_position_tolerance_m;
    float success_heading_tolerance_rad;
    float success_speed_tolerance_mps;
    int success_hold_steps;
    int refine_budget_steps;
    int curriculum_ramp_steps;
    float tier_weights[PARK_NUM_TIERS];
    int stagger_first_episode;
} ParkConfig;

typedef struct {
    int status;
    int phase;
    int outcome;
    ParkPose pose;
    float speed_mps;
    float steer_rad;
    float accel_mps2;
    ParkPose start;
    ParkPose goal;
    ParkPose prep;
    float slot_length_m;
    float slot_width_m;
    const ParkRoutePoint *route;
    int route_count;
    int route_segment_idx;
    float route_progress_m;
    float route_length_m;
    int next_subgoal_idx;
    const ParkPlanPoint *plan;
    int plan_count;
    int plan_idx;
    int plan_run_end_idx;
    float plan_progress_m;
    float plan_length_m;
    int plan_stall_steps;
    float plan_run_remaining_m;
    int success_hold_count;
    int refine_step_count;
    float slot_error_m;
    float slot_heading_error_rad;
    float goal_weight;
    float collision_weight;
    int partner_indices[PARK_MAX_PARTNER_SLOTS];
    int partner_count;
    float threat;
    float threat_intensity;
    float static_clearance_m;
    float teacher_accel_mps2;
    float teacher_steer_rad;
    float teacher_cross_track_m;
    float teacher_heading_error_rad;
    float trust_weight;
    float refine_sweep_direction;
    int decision_phase;
    float decision_teacher_accel_mps2;
    float decision_teacher_steer_rad;
    float decision_threat_intensity;
    float decision_trust_weight;
    float decision_slot_error_m;
    float step_route_progress_m;
    int step_subgoals_reached;
    float step_plan_progress_m;
    int step_handoff;
    float episode_return;
    int episode_steps;
    float path_length_m;
    int maneuver_count;
    float last_motion_direction;
    int handed_off;
    int entered_refine;
    float threat_sum;
    float imitation_sum;
} ParkAgent;

typedef struct {
    ParkSegment segment;
    float mid_x;
    float mid_y;
    float half_length_m;
    int source;
} ParkStaticEdge;

typedef struct {
    float *observations;
    int *actions;
    float *rewards;
    unsigned char *terminals;
    unsigned char *truncations;
    unsigned char *masks;
    ParkLog log;
    ParkConfig config;
    ParkSceneBank *bank;
    ParkRng rng;
    int observation_size;
    int partner_offset;
    int lane_offset;
    int boundary_offset;
    int count_offset;
    int scene_idx;
    const ParkScene *scene;
    const ParkLot *lot;
    int timestep;
    int episode_end_step;
    int autoreset_pending;
    int first_reset_pending;
    long long lifetime_steps;
    ParkAgent agents[PARK_MAX_AGENTS_PER_SCENE];
    ParkStaticEdge *static_edges;
    int static_edge_count;
    int static_edge_capacity;
} Park;

static inline int park_observation_size(int partner_slots, int lane_slots, int boundary_slots) {
    return PARK_EGO_FEATURES + PARK_CONDITIONING_FEATURES + PARK_TAIL_FEATURES + partner_slots * PARK_PARTNER_FEATURES
        + lane_slots * PARK_LANE_FEATURES + boundary_slots * PARK_BOUNDARY_FEATURES + PARK_COUNT_FEATURES;
}

static inline void park_to_ego_frame(const ParkPose *ego, float x, float y, float *local_x, float *local_y) {
    float cos_heading = cosf(ego->heading_rad);
    float sin_heading = sinf(ego->heading_rad);
    float delta_x = x - ego->x;
    float delta_y = y - ego->y;
    *local_x = delta_x * cos_heading + delta_y * sin_heading;
    *local_y = -delta_x * sin_heading + delta_y * cos_heading;
}

static inline float park_normalized(float value, float scale) {
    return park_clampf(value / scale, -1.0f, 1.0f);
}

static inline ParkBox park_agent_box(const Park *env, const ParkAgent *agent) {
    return (ParkBox) {agent->pose.x,
                      agent->pose.y,
                      agent->pose.heading_rad,
                      env->bank->vehicle_length_m,
                      env->bank->vehicle_width_m};
}

static inline int park_agent_present(const ParkAgent *agent) {
    return agent->status == PARK_STATUS_ACTIVE || agent->status == PARK_STATUS_PARKED;
}

// Agents that terminate this step without parking leave the lot at the end of the step.
static inline int park_agent_present_after_step(const ParkAgent *agent) {
    if (agent->status == PARK_STATUS_PARKED) {
        return 1;
    }
    return agent->status == PARK_STATUS_ACTIVE
        && (agent->outcome == PARK_OUTCOME_NONE || agent->outcome == PARK_OUTCOME_SUCCESS);
}

static inline int park_agent_running(const ParkAgent *agent) {
    return agent->status == PARK_STATUS_ACTIVE && agent->outcome == PARK_OUTCOME_NONE;
}

static int park_validate_config(const ParkConfig *config, char *error, size_t capacity) {
    int counts_valid = config->episode_steps > 0 && config->max_agents >= 1
        && config->max_agents <= PARK_MAX_AGENTS_PER_SCENE && config->max_controlled_per_scene >= 1
        && config->partner_slots >= 1 && config->partner_slots <= PARK_MAX_PARTNER_SLOTS && config->lane_slots >= 1
        && config->lane_slots <= PARK_MAX_LANE_SLOTS && config->boundary_slots >= 1
        && config->boundary_slots <= PARK_MAX_BOUNDARY_SLOTS && config->success_hold_steps >= 1
        && config->refine_budget_steps >= 1 && config->curriculum_ramp_steps >= 0;
    if (!counts_valid) {
        snprintf(error, capacity, "park config has an invalid count or slot budget");
        return PARK_ERROR_INVALID;
    }
    int positive_valid = isfinite(config->dt_s) && config->dt_s > 0.0f && config->imitation_scale > 0.0f
        && config->refine_tanh_scale_m > 0.0f && config->threat_decay_m > 0.0f && config->threat_squash_mps > 0.0f
        && config->trust_cross_track_scale_m > 0.0f && config->speed_target_band_mps > 0.0f
        && config->success_position_tolerance_m > 0.0f && config->success_heading_tolerance_rad > 0.0f
        && config->success_speed_tolerance_mps > 0.0f;
    if (!positive_valid) {
        snprintf(error, capacity, "park config has a non-positive dt, scale or tolerance");
        return PARK_ERROR_INVALID;
    }
    int weights_valid = config->goal_weight_min > 0.0f && config->goal_weight_min <= config->goal_weight_max
        && config->collision_weight_min > 0.0f && config->collision_weight_min <= config->collision_weight_max
        && config->goal_weight_default > 0.0f && config->collision_weight_default > 0.0f;
    float tier_weight_sum = 0.0f;
    for (int tier_idx = 0; tier_idx < PARK_NUM_TIERS; tier_idx++) {
        weights_valid
            = weights_valid && isfinite(config->tier_weights[tier_idx]) && config->tier_weights[tier_idx] >= 0.0f;
        tier_weight_sum += config->tier_weights[tier_idx];
    }
    if (!weights_valid || tier_weight_sum <= 0.0f) {
        snprintf(error, capacity, "park config has invalid conditioning or tier weights");
        return PARK_ERROR_INVALID;
    }
    return PARK_ERROR_OK;
}

static int park_init(
    Park *env,
    const ParkConfig *config,
    ParkSceneBank *bank,
    uint64_t seed,
    char *error,
    size_t capacity) {
    if (park_validate_config(config, error, capacity) != PARK_ERROR_OK) {
        return PARK_ERROR_INVALID;
    }
    int controlled_capacity = bank->max_scene_agent_count < config->max_controlled_per_scene
        ? bank->max_scene_agent_count
        : config->max_controlled_per_scene;
    if (controlled_capacity > config->max_agents) {
        snprintf(
            error,
            capacity,
            "scenes control up to %d agents but max_agents is %d",
            controlled_capacity,
            config->max_agents);
        return PARK_ERROR_INVALID;
    }
    env->config = *config;
    env->bank = bank;
    park_rng_seed(&env->rng, seed);
    env->partner_offset = PARK_EGO_FEATURES + PARK_CONDITIONING_FEATURES + PARK_TAIL_FEATURES;
    env->lane_offset = env->partner_offset + config->partner_slots * PARK_PARTNER_FEATURES;
    env->boundary_offset = env->lane_offset + config->lane_slots * PARK_LANE_FEATURES;
    env->count_offset = env->boundary_offset + config->boundary_slots * PARK_BOUNDARY_FEATURES;
    env->observation_size = park_observation_size(config->partner_slots, config->lane_slots, config->boundary_slots);
    env->static_edge_capacity = bank->max_scene_static_edge_count;
    env->static_edges = (ParkStaticEdge *) calloc((size_t) env->static_edge_capacity, sizeof(ParkStaticEdge));
    if (env->static_edges == NULL) {
        snprintf(error, capacity, "failed to allocate %d static edges", env->static_edge_capacity);
        return PARK_ERROR_INVALID;
    }
    env->first_reset_pending = config->stagger_first_episode;
    return PARK_ERROR_OK;
}

static void park_close(Park *env) {
    free(env->static_edges);
    env->static_edges = NULL;
}

static float park_curriculum_tier_weight(const Park *env, int tier_idx) {
    float progress = 1.0f;
    if (env->config.curriculum_ramp_steps > 0) {
        progress = park_clampf((float) env->lifetime_steps / (float) env->config.curriculum_ramp_steps, 0.0f, 1.0f);
    }
    float start_weight = tier_idx == 0 ? 1.0f : 0.0f;
    float tier_scene_count
        = (float) (env->bank->tier_scene_offsets[tier_idx + 1] - env->bank->tier_scene_offsets[tier_idx]);
    float weight = start_weight + progress * (env->config.tier_weights[tier_idx] - start_weight);
    return tier_scene_count > 0.0f ? weight : 0.0f;
}

static int park_select_scene(Park *env) {
    float tier_weights[PARK_NUM_TIERS];
    float weight_sum = 0.0f;
    for (int tier_idx = 0; tier_idx < PARK_NUM_TIERS; tier_idx++) {
        tier_weights[tier_idx] = park_curriculum_tier_weight(env, tier_idx);
        weight_sum += tier_weights[tier_idx];
    }
    if (weight_sum <= 0.0f) {
        return park_rng_below(&env->rng, env->bank->scene_count);
    }
    float draw = park_rng_uniform(&env->rng) * weight_sum;
    int chosen_tier = PARK_NUM_TIERS - 1;
    for (int tier_idx = 0; tier_idx < PARK_NUM_TIERS; tier_idx++) {
        if (draw < tier_weights[tier_idx]) {
            chosen_tier = tier_idx;
            break;
        }
        draw -= tier_weights[tier_idx];
    }
    for (int fallback = 0; fallback < PARK_NUM_TIERS && tier_weights[chosen_tier] <= 0.0f; fallback++) {
        chosen_tier = (chosen_tier + PARK_NUM_TIERS - 1) % PARK_NUM_TIERS;
    }
    int tier_start = env->bank->tier_scene_offsets[chosen_tier];
    int tier_count = env->bank->tier_scene_offsets[chosen_tier + 1] - tier_start;
    return env->bank->tier_scene_indices[tier_start + park_rng_below(&env->rng, tier_count)];
}

static void park_add_static_edge(Park *env, ParkSegment segment, int source) {
    assert(env->static_edge_count < env->static_edge_capacity);
    ParkStaticEdge *edge = &env->static_edges[env->static_edge_count++];
    edge->segment = segment;
    edge->mid_x = 0.5f * (segment.x0 + segment.x1);
    edge->mid_y = 0.5f * (segment.y0 + segment.y1);
    edge->half_length_m = 0.5f * hypotf(segment.x1 - segment.x0, segment.y1 - segment.y0);
    edge->source = source;
}

static void park_add_static_box(Park *env, const ParkBox *box, int source) {
    ParkSegment box_edges[PARK_BOX_CORNER_COUNT];
    park_box_edges(box, box_edges);
    for (int edge_idx = 0; edge_idx < PARK_BOX_CORNER_COUNT; edge_idx++) {
        park_add_static_edge(env, box_edges[edge_idx], source);
    }
}

static void park_build_static_edges(Park *env) {
    const ParkSceneBank *bank = env->bank;
    env->static_edge_count = 0;
    for (int curb_idx = 0; curb_idx < env->lot->curb_count; curb_idx++) {
        park_add_static_edge(env, bank->segments[env->lot->curb_offset + curb_idx], PARK_EDGE_SOURCE_CURB);
    }
    for (int obstacle_idx = 0; obstacle_idx < env->lot->obstacle_count; obstacle_idx++) {
        park_add_static_box(env, &bank->boxes[env->lot->obstacle_offset + obstacle_idx], PARK_EDGE_SOURCE_OBSTACLE);
    }
    for (int vehicle_idx = 0; vehicle_idx < env->scene->static_vehicle_count; vehicle_idx++) {
        park_add_static_box(
            env,
            &bank->boxes[env->scene->static_vehicle_offset + vehicle_idx],
            PARK_EDGE_SOURCE_STATIC_VEHICLE);
    }
}

static float park_sample_weight(Park *env, float low, float high) {
    return low + (high - low) * park_rng_uniform(&env->rng);
}

static int park_find_run_end(const ParkPlanPoint *plan, int plan_count, int run_start_idx) {
    if (run_start_idx >= plan_count - 1) {
        return plan_count - 1;
    }
    float run_direction = plan[run_start_idx + 1].direction;
    for (int point_idx = run_start_idx + 2; point_idx < plan_count; point_idx++) {
        if (plan[point_idx].direction != run_direction) {
            return point_idx - 1;
        }
    }
    return plan_count - 1;
}

static void park_spawn_agent(Park *env, ParkAgent *agent, const ParkSceneAgent *scene_agent) {
    const ParkSceneBank *bank = env->bank;
    memset(agent, 0, sizeof(*agent));
    agent->status = PARK_STATUS_ACTIVE;
    agent->phase = PARK_PHASE_NAVIGATE;
    agent->pose = scene_agent->start;
    agent->start = scene_agent->start;
    agent->goal = scene_agent->goal;
    agent->prep = scene_agent->prep;
    agent->slot_length_m = scene_agent->slot_length_m;
    agent->slot_width_m = scene_agent->slot_width_m;
    agent->route = &bank->route_points[scene_agent->route_offset];
    agent->route_count = scene_agent->route_count;
    agent->route_length_m = agent->route[agent->route_count - 1].arc_m;
    agent->plan = &bank->plan_points[scene_agent->plan_offset];
    agent->plan_count = scene_agent->plan_count;
    agent->plan_length_m = agent->plan[agent->plan_count - 1].arc_m;
    agent->goal_weight = env->config.goal_weight_default;
    agent->collision_weight = env->config.collision_weight_default;
    if (env->config.reward_conditioning) {
        agent->goal_weight = park_sample_weight(env, env->config.goal_weight_min, env->config.goal_weight_max);
        agent->collision_weight
            = park_sample_weight(env, env->config.collision_weight_min, env->config.collision_weight_max);
    }
}

static void park_update_slot_error(ParkAgent *agent) {
    float cos_goal = cosf(agent->goal.heading_rad);
    float sin_goal = sinf(agent->goal.heading_rad);
    float delta_x = agent->pose.x - agent->goal.x;
    float delta_y = agent->pose.y - agent->goal.y;
    float slot_long_m = delta_x * cos_goal + delta_y * sin_goal;
    float slot_lat_m = -delta_x * sin_goal + delta_y * cos_goal;
    agent->slot_error_m = sqrtf(slot_long_m * slot_long_m + slot_lat_m * slot_lat_m);
    agent->slot_heading_error_rad = park_wrap_angle_mod_pi(agent->pose.heading_rad - agent->goal.heading_rad);
}

// Keeps the nearest `capacity` candidates sorted by distance; ties keep the earlier index.
static void park_nearest_insert(float *distances, int *indices, int *count, int capacity, float distance, int index) {
    int insert_idx = *count < capacity ? *count : capacity;
    for (; insert_idx > 0 && distances[insert_idx - 1] > distance; insert_idx--) {
        if (insert_idx < capacity) {
            distances[insert_idx] = distances[insert_idx - 1];
            indices[insert_idx] = indices[insert_idx - 1];
        }
    }
    if (insert_idx >= capacity) {
        return;
    }
    distances[insert_idx] = distance;
    indices[insert_idx] = index;
    *count = *count < capacity ? *count + 1 : capacity;
}

static void park_select_partners(Park *env, int agent_idx) {
    ParkAgent *agent = &env->agents[agent_idx];
    float partner_distances[PARK_MAX_PARTNER_SLOTS];
    agent->partner_count = 0;
    for (int other_idx = 0; other_idx < env->config.max_agents; other_idx++) {
        const ParkAgent *other = &env->agents[other_idx];
        if (other_idx == agent_idx || !park_agent_present_after_step(other)) {
            continue;
        }
        float distance_m = hypotf(other->pose.x - agent->pose.x, other->pose.y - agent->pose.y);
        if (distance_m > PARK_OBS_PARTNER_RANGE_M) {
            continue;
        }
        park_nearest_insert(
            partner_distances,
            agent->partner_indices,
            &agent->partner_count,
            env->config.partner_slots,
            distance_m,
            other_idx);
    }
}

static float park_closing_speed(const ParkAgent *agent, const ParkAgent *other, float *distance_out) {
    float delta_x = agent->pose.x - other->pose.x;
    float delta_y = agent->pose.y - other->pose.y;
    float distance_m = fmaxf(sqrtf(delta_x * delta_x + delta_y * delta_y), PARK_GEOMETRY_EPSILON);
    float velocity_dx
        = agent->speed_mps * cosf(agent->pose.heading_rad) - other->speed_mps * cosf(other->pose.heading_rad);
    float velocity_dy
        = agent->speed_mps * sinf(agent->pose.heading_rad) - other->speed_mps * sinf(other->pose.heading_rad);
    *distance_out = distance_m;
    return -(delta_x * velocity_dx + delta_y * velocity_dy) / distance_m;
}

static void park_update_threat(Park *env, ParkAgent *agent) {
    agent->threat = 0.0f;
    for (int slot_idx = 0; slot_idx < agent->partner_count; slot_idx++) {
        float distance_m;
        float closing_mps = park_closing_speed(agent, &env->agents[agent->partner_indices[slot_idx]], &distance_m);
        agent->threat += fmaxf(closing_mps, 0.0f) * expf(-distance_m / env->config.threat_decay_m);
    }
    agent->threat_intensity = 1.0f - expf(-agent->threat / env->config.threat_squash_mps);
}

static void park_update_static_clearance(Park *env, ParkAgent *agent) {
    ParkBox box = park_agent_box(env, agent);
    float box_radius = park_box_bounding_radius(box.length_m, box.width_m);
    agent->static_clearance_m = PARK_CLEARANCE_QUERY_RANGE_M;
    for (int edge_idx = 0; edge_idx < env->static_edge_count; edge_idx++) {
        const ParkStaticEdge *edge = &env->static_edges[edge_idx];
        float reach = box_radius + edge->half_length_m + agent->static_clearance_m;
        float delta_x = edge->mid_x - box.center_x;
        float delta_y = edge->mid_y - box.center_y;
        if (delta_x * delta_x + delta_y * delta_y > reach * reach) {
            continue;
        }
        agent->static_clearance_m = fminf(agent->static_clearance_m, park_box_segment_distance(&box, &edge->segment));
    }
}

static void park_update_perception(Park *env) {
    for (int agent_idx = 0; agent_idx < env->config.max_agents; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (!park_agent_running(agent)) {
            continue;
        }
        park_select_partners(env, agent_idx);
        park_update_threat(env, agent);
        park_update_static_clearance(env, agent);
    }
}

typedef struct {
    float projection_x;
    float projection_y;
    float heading_rad;
    float arc_m;
    int segment_idx;
} ParkPlanProjection;

// Projects a point onto plan segments [first_idx, last_idx), with the path shifted forward by offset_m.
static ParkPlanProjection park_project_on_plan(
    const ParkPlanPoint *plan,
    int first_idx,
    int last_idx,
    float offset_m,
    float point_x,
    float point_y) {
    ParkPlanProjection best = {0};
    float best_distance_sq = INFINITY;
    for (int segment_idx = first_idx; segment_idx < last_idx; segment_idx++) {
        const ParkPlanPoint *from = &plan[segment_idx];
        const ParkPlanPoint *to = &plan[segment_idx + 1];
        float from_x = from->x + offset_m * cosf(from->heading_rad);
        float from_y = from->y + offset_m * sinf(from->heading_rad);
        float to_x = to->x + offset_m * cosf(to->heading_rad);
        float to_y = to->y + offset_m * sinf(to->heading_rad);
        float segment_dx = to_x - from_x;
        float segment_dy = to_y - from_y;
        float length_sq = fmaxf(segment_dx * segment_dx + segment_dy * segment_dy, PARK_GEOMETRY_EPSILON);
        float fraction
            = park_clampf(((point_x - from_x) * segment_dx + (point_y - from_y) * segment_dy) / length_sq, 0.0f, 1.0f);
        float projection_x = from_x + fraction * segment_dx;
        float projection_y = from_y + fraction * segment_dy;
        float distance_sq
            = (projection_x - point_x) * (projection_x - point_x) + (projection_y - point_y) * (projection_y - point_y);
        if (distance_sq >= best_distance_sq) {
            continue;
        }
        best_distance_sq = distance_sq;
        best.projection_x = projection_x;
        best.projection_y = projection_y;
        best.heading_rad = from->heading_rad + fraction * park_wrap_angle(to->heading_rad - from->heading_rad);
        best.arc_m = from->arc_m + fraction * (to->arc_m - from->arc_m);
        best.segment_idx = segment_idx;
    }
    return best;
}

static int park_plan_window_end(const ParkAgent *agent) {
    int window_end = agent->plan_idx + PARK_PLAN_SEARCH_WINDOW_POINTS;
    return window_end < agent->plan_run_end_idx ? window_end : agent->plan_run_end_idx;
}

static int park_plan_window_start(const ParkAgent *agent) {
    return agent->plan_idx < agent->plan_count - 1 ? agent->plan_idx : agent->plan_count - 2;
}

static void park_enter_maneuver(ParkAgent *agent) {
    agent->phase = PARK_PHASE_MANEUVER;
    agent->plan_idx = 0;
    agent->plan_run_end_idx = park_find_run_end(agent->plan, agent->plan_count, 0);
    agent->plan_progress_m = 0.0f;
    agent->plan_stall_steps = 0;
}

static void park_route_vertex(const ParkAgent *agent, int vertex_idx, float *x, float *y, float *arc_m) {
    if (vertex_idx == 0) {
        *x = agent->start.x;
        *y = agent->start.y;
        *arc_m = 0.0f;
        return;
    }
    const ParkRoutePoint *point = &agent->route[vertex_idx - 1];
    *x = point->x;
    *y = point->y;
    *arc_m = point->arc_m;
}

static void park_update_route_progress(ParkAgent *agent) {
    int last_segment = agent->route_segment_idx + PARK_ROUTE_SEARCH_WINDOW_SEGMENTS;
    last_segment = last_segment < agent->route_count ? last_segment : agent->route_count;
    float best_distance_sq = INFINITY;
    float best_progress_m = agent->route_progress_m;
    int best_segment = agent->route_segment_idx;
    for (int segment_idx = agent->route_segment_idx; segment_idx < last_segment; segment_idx++) {
        float from_x, from_y, from_arc, to_x, to_y, to_arc;
        park_route_vertex(agent, segment_idx, &from_x, &from_y, &from_arc);
        park_route_vertex(agent, segment_idx + 1, &to_x, &to_y, &to_arc);
        float segment_dx = to_x - from_x;
        float segment_dy = to_y - from_y;
        float length_sq = fmaxf(segment_dx * segment_dx + segment_dy * segment_dy, PARK_GEOMETRY_EPSILON);
        float fraction = park_clampf(
            ((agent->pose.x - from_x) * segment_dx + (agent->pose.y - from_y) * segment_dy) / length_sq,
            0.0f,
            1.0f);
        float offset_x = from_x + fraction * segment_dx - agent->pose.x;
        float offset_y = from_y + fraction * segment_dy - agent->pose.y;
        float distance_sq = offset_x * offset_x + offset_y * offset_y;
        if (distance_sq >= best_distance_sq) {
            continue;
        }
        best_distance_sq = distance_sq;
        best_progress_m = from_arc + fraction * (to_arc - from_arc);
        best_segment = segment_idx;
    }
    agent->step_route_progress_m = best_progress_m - agent->route_progress_m;
    agent->route_progress_m = best_progress_m;
    agent->route_segment_idx = best_segment;
    for (int subgoal_idx = agent->next_subgoal_idx; subgoal_idx < agent->route_count; subgoal_idx++) {
        if (agent->route_progress_m + PARK_SUBGOAL_REACH_TOLERANCE_M < agent->route[subgoal_idx].arc_m) {
            break;
        }
        agent->step_subgoals_reached++;
        agent->next_subgoal_idx = subgoal_idx + 1;
    }
}

static void park_update_handoff(ParkAgent *agent) {
    float distance_m = hypotf(agent->pose.x - agent->prep.x, agent->pose.y - agent->prep.y);
    float heading_error_rad = fabsf(park_wrap_angle(agent->pose.heading_rad - agent->prep.heading_rad));
    if (distance_m > PARK_PREP_RADIUS_M || heading_error_rad > PARK_PREP_HEADING_TOLERANCE_RAD) {
        return;
    }
    park_enter_maneuver(agent);
    agent->step_handoff = 1;
    agent->handed_off = 1;
}

static void park_update_plan_tracking(const Park *env, ParkAgent *agent) {
    ParkPose rear_axle = park_rear_axle_from_center(agent->pose, env->bank->wheelbase_m);
    ParkPlanProjection projection = park_project_on_plan(
        agent->plan,
        park_plan_window_start(agent),
        park_plan_window_end(agent),
        0.0f,
        rear_axle.x,
        rear_axle.y);
    agent->step_plan_progress_m = projection.arc_m - agent->plan_progress_m;
    agent->plan_progress_m = projection.arc_m;
    agent->plan_idx = projection.segment_idx > agent->plan_idx ? projection.segment_idx : agent->plan_idx;
    agent->plan_stall_steps
        = agent->step_plan_progress_m < PARK_PLAN_STALL_PROGRESS_M ? agent->plan_stall_steps + 1 : 0;
    // Arc-length remaining on the run stays meaningful under lateral offset, unlike distance to the cusp point.
    agent->plan_run_remaining_m = agent->plan[agent->plan_run_end_idx].arc_m - projection.arc_m;
    int at_cusp = agent->plan_run_end_idx < agent->plan_count - 1 && agent->plan_run_remaining_m <= PARK_CUSP_REACH_M
        && fabsf(agent->speed_mps) <= PARK_CUSP_SPEED_MPS;
    if (at_cusp) {
        agent->plan_idx = agent->plan_run_end_idx;
        agent->plan_run_end_idx = park_find_run_end(agent->plan, agent->plan_count, agent->plan_idx);
        agent->plan_run_remaining_m = agent->plan[agent->plan_run_end_idx].arc_m - projection.arc_m;
    }
}

static void park_update_refine_entry(ParkAgent *agent) {
    int on_final_run = agent->plan_run_end_idx == agent->plan_count - 1;
    float progress_fraction = agent->plan_progress_m / agent->plan_length_m;
    int reached_end = on_final_run && agent->plan_run_remaining_m <= PARK_PLAN_END_REACH_M;
    int stalled = agent->plan_stall_steps >= PARK_PLAN_STALL_STEPS && progress_fraction >= PARK_PLAN_STALL_MIN_FRACTION;
    if (!reached_end && !stalled) {
        return;
    }
    agent->phase = PARK_PHASE_REFINE;
    agent->entered_refine = 1;
    agent->refine_step_count = 0;
}

static void park_update_success(const Park *env, ParkAgent *agent) {
    park_update_slot_error(agent);
    int inside_tolerance = agent->slot_error_m <= env->config.success_position_tolerance_m
        && fabsf(agent->slot_heading_error_rad) <= env->config.success_heading_tolerance_rad
        && fabsf(agent->speed_mps) <= env->config.success_speed_tolerance_mps;
    agent->success_hold_count = inside_tolerance ? agent->success_hold_count + 1 : 0;
    if (agent->success_hold_count >= env->config.success_hold_steps) {
        agent->outcome = PARK_OUTCOME_SUCCESS;
    }
}

static void park_update_task_progress(Park *env, ParkAgent *agent) {
    if (agent->phase == PARK_PHASE_NAVIGATE) {
        park_update_route_progress(agent);
        park_update_handoff(agent);
    } else if (agent->phase == PARK_PHASE_MANEUVER) {
        park_update_plan_tracking(env, agent);
        park_update_refine_entry(agent);
    } else {
        agent->refine_step_count++;
    }
    park_update_success(env, agent);
    if (agent->outcome == PARK_OUTCOME_NONE && agent->phase == PARK_PHASE_REFINE
        && agent->refine_step_count >= env->config.refine_budget_steps) {
        agent->outcome = PARK_OUTCOME_REFINE_TIMEOUT;
    }
}

static float park_speed_command(float reference_speed_mps, float speed_mps) {
    return park_clampf(
        PARK_TEACHER_SPEED_GAIN_PER_S * (reference_speed_mps - speed_mps),
        -PARK_MAX_ACCEL_MPS2,
        PARK_MAX_ACCEL_MPS2);
}

// Stanley steering; reverse motion steers a virtual vehicle facing backwards, so the command flips sign.
static float park_stanley_steer(float heading_error_rad, float cross_track_m, float speed_mps, float direction) {
    float virtual_steer = heading_error_rad
        + atanf(PARK_STANLEY_GAIN * cross_track_m / (PARK_STANLEY_SOFTENING_MPS + fabsf(speed_mps)));
    float steer = direction > 0.0f ? virtual_steer : -virtual_steer;
    return park_clampf(steer, -PARK_MAX_STEER_RAD, PARK_MAX_STEER_RAD);
}

static void park_compute_plan_teacher(const Park *env, ParkAgent *agent) {
    float wheelbase_m = env->bank->wheelbase_m;
    float direction = agent->plan[agent->plan_run_end_idx].direction;
    ParkPose rear_axle = park_rear_axle_from_center(agent->pose, wheelbase_m);
    float axle_offset_m = direction > 0.0f ? wheelbase_m : 0.0f;
    float reference_x = rear_axle.x + axle_offset_m * cosf(rear_axle.heading_rad);
    float reference_y = rear_axle.y + axle_offset_m * sinf(rear_axle.heading_rad);
    ParkPlanProjection projection = park_project_on_plan(
        agent->plan,
        park_plan_window_start(agent),
        park_plan_window_end(agent),
        axle_offset_m,
        reference_x,
        reference_y);
    float travel_heading = direction > 0.0f ? projection.heading_rad : projection.heading_rad + PARK_PI;
    float cross_track_m = (projection.projection_x - reference_x) * -sinf(travel_heading)
        + (projection.projection_y - reference_y) * cosf(travel_heading);
    float heading_error_rad = park_wrap_angle(projection.heading_rad - agent->pose.heading_rad);
    float remaining_m = fmaxf(agent->plan[agent->plan_run_end_idx].arc_m - projection.arc_m, 0.0f);
    float reference_speed
        = direction * fminf(PARK_TEACHER_SPEED_MPS, sqrtf(2.0f * PARK_TEACHER_DECEL_MPS2 * remaining_m));
    agent->teacher_accel_mps2 = park_speed_command(reference_speed, agent->speed_mps);
    agent->teacher_steer_rad = park_stanley_steer(heading_error_rad, cross_track_m, agent->speed_mps, direction);
    agent->teacher_cross_track_m = cross_track_m;
    agent->teacher_heading_error_rad = heading_error_rad;
}

// Sweeps back and forth along the slot axis under Stanley until heading and lateral errors settle, then stops.
static void park_compute_refine_teacher(const Park *env, ParkAgent *agent) {
    const ParkConfig *config = &env->config;
    float wheelbase_m = env->bank->wheelbase_m;
    float aligned_heading = agent->goal.heading_rad;
    if (fabsf(park_wrap_angle(agent->pose.heading_rad - aligned_heading)) > PARK_HALF_PI) {
        aligned_heading += PARK_PI;
    }
    float offset_x = agent->pose.x - agent->goal.x;
    float offset_y = agent->pose.y - agent->goal.y;
    float along_error_m = offset_x * cosf(aligned_heading) + offset_y * sinf(aligned_heading);
    float lateral_error_m = -offset_x * sinf(aligned_heading) + offset_y * cosf(aligned_heading);
    int aligned = fabsf(park_wrap_angle(aligned_heading - agent->pose.heading_rad))
            <= PARK_REFINE_ALIGNED_FRACTION * config->success_heading_tolerance_rad
        && fabsf(lateral_error_m) <= PARK_REFINE_ALIGNED_FRACTION * config->success_position_tolerance_m;
    if (agent->refine_sweep_direction == 0.0f) {
        agent->refine_sweep_direction = along_error_m > 0.0f ? -1.0f : 1.0f;
    }
    float target_along_m = aligned ? 0.0f : agent->refine_sweep_direction * PARK_REFINE_SWEEP_M;
    if (!aligned && (along_error_m - target_along_m) * agent->refine_sweep_direction >= 0.0f) {
        agent->refine_sweep_direction = -agent->refine_sweep_direction;
        target_along_m = -target_along_m;
    }
    float reference_speed = park_clampf(
        -PARK_REFINE_POSITION_GAIN_PER_S * (along_error_m - target_along_m),
        -PARK_REFINE_TEACHER_SPEED_MPS,
        PARK_REFINE_TEACHER_SPEED_MPS);
    float direction = reference_speed >= 0.0f ? 1.0f : -1.0f;
    float axle_offset_m = direction > 0.0f ? 0.5f * wheelbase_m : -0.5f * wheelbase_m;
    float reference_x = agent->pose.x + axle_offset_m * cosf(agent->pose.heading_rad);
    float reference_y = agent->pose.y + axle_offset_m * sinf(agent->pose.heading_rad);
    float travel_heading = direction > 0.0f ? aligned_heading : aligned_heading + PARK_PI;
    float cross_track_m = -(
        (reference_x - agent->goal.x) * -sinf(travel_heading) + (reference_y - agent->goal.y) * cosf(travel_heading));
    float heading_error_rad = park_wrap_angle(aligned_heading - agent->pose.heading_rad);
    agent->teacher_accel_mps2 = park_speed_command(reference_speed, agent->speed_mps);
    agent->teacher_steer_rad = park_stanley_steer(heading_error_rad, cross_track_m, agent->speed_mps, direction);
    agent->teacher_cross_track_m = cross_track_m;
    agent->teacher_heading_error_rad = heading_error_rad;
}

static float park_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static void park_update_teacher(Park *env) {
    const ParkConfig *config = &env->config;
    for (int agent_idx = 0; agent_idx < config->max_agents; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (!park_agent_running(agent)) {
            continue;
        }
        agent->teacher_accel_mps2 = 0.0f;
        agent->teacher_steer_rad = 0.0f;
        agent->teacher_cross_track_m = 0.0f;
        agent->teacher_heading_error_rad = 0.0f;
        agent->trust_weight = 0.0f;
        if (agent->phase == PARK_PHASE_REFINE) {
            park_compute_refine_teacher(env, agent);
        }
        if (agent->phase != PARK_PHASE_MANEUVER) {
            continue;
        }
        park_compute_plan_teacher(env, agent);
        float plan_fraction = park_clampf(agent->plan_progress_m / agent->plan_length_m, 0.0f, 1.0f);
        agent->trust_weight = expf(-fabsf(agent->teacher_cross_track_m) / config->trust_cross_track_scale_m)
            * park_sigmoid(config->trust_clearance_slope_per_m
                           * (agent->static_clearance_m - config->trust_clearance_threshold_m))
            * (1.0f - plan_fraction);
    }
}

static void park_snapshot_decision(ParkAgent *agent) {
    agent->decision_phase = agent->phase;
    agent->decision_teacher_accel_mps2 = agent->teacher_accel_mps2;
    agent->decision_teacher_steer_rad = agent->teacher_steer_rad;
    agent->decision_threat_intensity = agent->threat_intensity;
    agent->decision_trust_weight = agent->trust_weight;
    agent->decision_slot_error_m = agent->slot_error_m;
    agent->step_route_progress_m = 0.0f;
    agent->step_subgoals_reached = 0;
    agent->step_plan_progress_m = 0.0f;
    agent->step_handoff = 0;
}

static void park_apply_action(Park *env, ParkAgent *agent, int action) {
    assert(action >= 0 && action < PARK_NUM_ACTIONS);
    float wheelbase_m = env->bank->wheelbase_m;
    agent->accel_mps2 = PARK_ACCEL_VALUES_MPS2[action / PARK_NUM_STEER_ACTIONS];
    agent->steer_rad = PARK_STEER_VALUES_RAD[action % PARK_NUM_STEER_ACTIONS];
    agent->speed_mps = park_clampf(
        agent->speed_mps + agent->accel_mps2 * env->config.dt_s,
        -PARK_MAX_REVERSE_SPEED_MPS,
        PARK_MAX_FORWARD_SPEED_MPS);
    ParkPose rear_axle = park_rear_axle_from_center(agent->pose, wheelbase_m);
    rear_axle
        = park_integrate_arc(rear_axle, tanf(agent->steer_rad) / wheelbase_m, agent->speed_mps * env->config.dt_s);
    agent->pose = park_center_from_rear_axle(rear_axle, wheelbase_m);
}

static void park_update_motion_metrics(Park *env, ParkAgent *agent) {
    agent->path_length_m += fabsf(agent->speed_mps) * env->config.dt_s;
    if (fabsf(agent->speed_mps) < PARK_MOTION_SPEED_EPSILON_MPS) {
        return;
    }
    float motion_direction = agent->speed_mps > 0.0f ? 1.0f : -1.0f;
    agent->maneuver_count += motion_direction != agent->last_motion_direction ? 1 : 0;
    agent->last_motion_direction = motion_direction;
}

static int park_static_contact_source(const Park *env, const ParkBox *box) {
    float box_radius = park_box_bounding_radius(box->length_m, box->width_m);
    for (int edge_idx = 0; edge_idx < env->static_edge_count; edge_idx++) {
        const ParkStaticEdge *edge = &env->static_edges[edge_idx];
        float reach = box_radius + edge->half_length_m;
        float delta_x = edge->mid_x - box->center_x;
        float delta_y = edge->mid_y - box->center_y;
        if (delta_x * delta_x + delta_y * delta_y > reach * reach) {
            continue;
        }
        if (park_box_segment_overlap(box, &edge->segment)) {
            return edge->source;
        }
    }
    return -1;
}

static int park_vehicle_contact(const Park *env, int agent_idx, const ParkBox *box) {
    float reach = 2.0f * park_box_bounding_radius(box->length_m, box->width_m);
    for (int other_idx = 0; other_idx < env->config.max_agents; other_idx++) {
        const ParkAgent *other = &env->agents[other_idx];
        if (other_idx == agent_idx || !park_agent_present(other)) {
            continue;
        }
        if (fabsf(other->pose.x - box->center_x) > reach || fabsf(other->pose.y - box->center_y) > reach) {
            continue;
        }
        ParkBox other_box = park_agent_box(env, other);
        if (park_box_overlap(box, &other_box)) {
            return 1;
        }
    }
    return 0;
}

static void park_detect_contacts(Park *env) {
    for (int agent_idx = 0; agent_idx < env->config.max_agents; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (agent->status != PARK_STATUS_ACTIVE) {
            continue;
        }
        ParkBox box = park_agent_box(env, agent);
        int out_of_lot = box.center_x < env->lot->min_x || box.center_x > env->lot->max_x
            || box.center_y < env->lot->min_y || box.center_y > env->lot->max_y;
        int static_source = park_static_contact_source(env, &box);
        if (out_of_lot || static_source == PARK_EDGE_SOURCE_CURB) {
            agent->outcome = PARK_OUTCOME_OFFROAD;
        } else if (static_source >= 0) {
            agent->outcome = PARK_OUTCOME_COLLISION_STATIC;
        } else if (park_vehicle_contact(env, agent_idx, &box)) {
            agent->outcome = PARK_OUTCOME_COLLISION_VEHICLE;
        }
    }
}

static float park_terminal_reward(const Park *env, const ParkAgent *agent) {
    const ParkConfig *config = &env->config;
    if (agent->outcome == PARK_OUTCOME_SUCCESS) {
        return agent->goal_weight * config->reward_success;
    }
    if (agent->outcome == PARK_OUTCOME_COLLISION_VEHICLE || agent->outcome == PARK_OUTCOME_COLLISION_STATIC) {
        return agent->collision_weight * config->reward_collision;
    }
    if (agent->outcome == PARK_OUTCOME_OFFROAD) {
        return agent->collision_weight * config->reward_offroad;
    }
    return 0.0f;
}

static float park_imitation_reward(const Park *env, const ParkAgent *agent) {
    float accel_gap = (agent->accel_mps2 - agent->decision_teacher_accel_mps2) / PARK_MAX_ACCEL_MPS2;
    float steer_gap = (agent->steer_rad - agent->decision_teacher_steer_rad) / PARK_MAX_STEER_RAD;
    float command_gap = sqrtf(accel_gap * accel_gap + steer_gap * steer_gap);
    return env->config.imitation_weight * (1.0f - agent->decision_threat_intensity) * agent->decision_trust_weight
        * expf(-command_gap / env->config.imitation_scale);
}

static float park_refine_reward(const Park *env, const ParkAgent *agent) {
    const ParkConfig *config = &env->config;
    float improvement_m = agent->decision_slot_error_m - agent->slot_error_m;
    float reward
        = config->refine_gain * config->refine_tanh_scale_m * tanhf(improvement_m / config->refine_tanh_scale_m);
    int holding = agent->slot_error_m <= config->success_position_tolerance_m
        && fabsf(agent->slot_heading_error_rad) <= config->success_heading_tolerance_rad
        && fabsf(agent->speed_mps) <= config->success_speed_tolerance_mps;
    return reward + (holding ? config->refine_hold_bonus : 0.0f);
}

static float park_navigation_reward(const Park *env, const ParkAgent *agent) {
    const ParkConfig *config = &env->config;
    float speed_score = 1.0f - fabsf(agent->speed_mps - config->speed_target_mps) / config->speed_target_band_mps;
    float shaping = config->reward_route_progress_per_m * agent->step_route_progress_m
        + config->reward_subgoal * (float) agent->step_subgoals_reached
        + config->reward_speed_target * fmaxf(speed_score, 0.0f);
    return agent->goal_weight * shaping;
}

static float park_compute_reward(Park *env, ParkAgent *agent) {
    const ParkConfig *config = &env->config;
    float reward = park_terminal_reward(env, agent);
    reward += agent->step_handoff ? agent->goal_weight * config->reward_handoff : 0.0f;
    reward -= agent->collision_weight * config->threat_penalty_weight * agent->threat;
    if (agent->static_clearance_m < config->trust_clearance_threshold_m) {
        reward += agent->collision_weight * config->reward_static_clearance;
    }
    if (agent->decision_phase == PARK_PHASE_NAVIGATE) {
        reward += park_navigation_reward(env, agent);
    } else if (agent->decision_phase == PARK_PHASE_MANEUVER) {
        float imitation = park_imitation_reward(env, agent);
        agent->imitation_sum += imitation;
        reward += agent->goal_weight * config->reward_plan_progress_per_m * agent->step_plan_progress_m + imitation;
    } else {
        reward += park_refine_reward(env, agent);
    }
    return reward;
}

static void park_log_agent_episode(Park *env, const ParkAgent *agent) {
    ParkLog *log = &env->log;
    int success = agent->outcome == PARK_OUTCOME_SUCCESS;
    log->n += 1.0f;
    log->score += (float) success;
    log->success_rate += (float) success;
    log->collision_vehicle_rate += (float) (agent->outcome == PARK_OUTCOME_COLLISION_VEHICLE);
    log->collision_static_rate += (float) (agent->outcome == PARK_OUTCOME_COLLISION_STATIC);
    log->collision_rate += (float) (agent->outcome == PARK_OUTCOME_COLLISION_VEHICLE
                                    || agent->outcome == PARK_OUTCOME_COLLISION_STATIC);
    log->offroad_rate += (float) (agent->outcome == PARK_OUTCOME_OFFROAD);
    log->refine_timeout_rate += (float) (agent->outcome == PARK_OUTCOME_REFINE_TIMEOUT);
    log->horizon_timeout_rate += (float) (agent->outcome == PARK_OUTCOME_HORIZON);
    log->handoff_rate += (float) agent->handed_off;
    log->refine_entered_rate += (float) agent->entered_refine;
    log->success_position_error_m += success ? agent->slot_error_m : 0.0f;
    log->success_heading_error_deg += success ? fabsf(agent->slot_heading_error_rad) * PARK_RADIANS_TO_DEGREES : 0.0f;
    log->path_length_m += agent->path_length_m;
    log->maneuver_count += (float) agent->maneuver_count;
    log->episode_return += agent->episode_return;
    log->episode_length += (float) agent->episode_steps;
    log->mean_threat += agent->episode_steps > 0 ? agent->threat_sum / (float) agent->episode_steps : 0.0f;
    log->imitation_reward += agent->imitation_sum;
    log->scene_agent_count += (float) env->scene->agent_count;
    log->scene_tier += (float) env->scene->tier;
}

static void park_write_ego_obs(const Park *env, const ParkAgent *agent, float *obs) {
    const ParkPose *ego = &agent->pose;
    float local_x, local_y;
    obs[0] = park_normalized(agent->speed_mps, PARK_OBS_SPEED_NORM_MPS);
    obs[1] = agent->steer_rad / PARK_MAX_STEER_RAD;
    obs[2] = agent->accel_mps2 / PARK_MAX_ACCEL_MPS2;
    obs[3] = env->bank->vehicle_length_m / PARK_OBS_LENGTH_NORM_M;
    obs[4] = env->bank->vehicle_width_m / PARK_OBS_WIDTH_NORM_M;
    obs[PARK_EGO_PHASE_OFFSET + agent->phase] = 1.0f;
    park_to_ego_frame(ego, agent->goal.x, agent->goal.y, &local_x, &local_y);
    obs[8] = park_normalized(local_x, PARK_OBS_GOAL_NORM_M);
    obs[9] = park_normalized(local_y, PARK_OBS_GOAL_NORM_M);
    obs[10] = cosf(agent->goal.heading_rad - ego->heading_rad);
    obs[11] = sinf(agent->goal.heading_rad - ego->heading_rad);
    if (agent->phase == PARK_PHASE_NAVIGATE) {
        int subgoal_idx
            = agent->next_subgoal_idx < agent->route_count ? agent->next_subgoal_idx : agent->route_count - 1;
        park_to_ego_frame(ego, agent->route[subgoal_idx].x, agent->route[subgoal_idx].y, &local_x, &local_y);
        obs[12] = park_normalized(local_x, PARK_OBS_GOAL_NORM_M);
        obs[13] = park_normalized(local_y, PARK_OBS_GOAL_NORM_M);
        obs[18] = park_normalized(agent->route_length_m - agent->route_progress_m, PARK_OBS_ROUTE_NORM_M);
    }
    park_to_ego_frame(ego, agent->prep.x, agent->prep.y, &local_x, &local_y);
    obs[14] = park_normalized(local_x, PARK_OBS_GOAL_NORM_M);
    obs[15] = park_normalized(local_y, PARK_OBS_GOAL_NORM_M);
    obs[16] = cosf(agent->prep.heading_rad - ego->heading_rad);
    obs[17] = sinf(agent->prep.heading_rad - ego->heading_rad);
    obs[19] = (float) agent->refine_step_count / (float) env->config.refine_budget_steps;
    obs[20] = (float) env->timestep / (float) env->episode_end_step;
    obs[21] = (float) agent->success_hold_count / (float) env->config.success_hold_steps;
}

static float park_conditioning_feature(float weight, float low, float high) {
    if (high <= low) {
        return 0.0f;
    }
    return (2.0f * weight - low - high) / (high - low);
}

static void park_write_tail_obs(const ParkAgent *agent, float *obs) {
    obs[PARK_TAIL_THREAT_IDX] = agent->threat_intensity;
    obs[PARK_TAIL_CLEARANCE_IDX]
        = fminf(agent->static_clearance_m, PARK_OBS_CLEARANCE_NORM_M) / PARK_OBS_CLEARANCE_NORM_M;
    obs[PARK_TAIL_SPEED_IDX] = park_normalized(agent->speed_mps, PARK_OBS_SPEED_NORM_MPS);
    if (agent->phase == PARK_PHASE_NAVIGATE) {
        return;
    }
    float local_x, local_y;
    ParkPose slot_frame = agent->goal;
    park_to_ego_frame(&slot_frame, agent->pose.x, agent->pose.y, &local_x, &local_y);
    obs[PARK_TAIL_SLOT_LONG_IDX] = park_normalized(local_x, PARK_OBS_SLOT_ERROR_NORM_M);
    obs[PARK_TAIL_SLOT_LAT_IDX] = park_normalized(local_y, PARK_OBS_SLOT_ERROR_NORM_M);
    obs[PARK_TAIL_SLOT_HEADING_IDX] = agent->slot_heading_error_rad / PARK_HALF_PI;
    obs[PARK_TAIL_TEACHER_ACCEL_IDX] = agent->teacher_accel_mps2 / PARK_MAX_ACCEL_MPS2;
    obs[PARK_TAIL_TEACHER_STEER_IDX] = agent->teacher_steer_rad / PARK_MAX_STEER_RAD;
    obs[PARK_TAIL_TEACHER_CROSS_TRACK_IDX] = park_normalized(agent->teacher_cross_track_m, PARK_OBS_CROSS_TRACK_NORM_M);
    obs[PARK_TAIL_TEACHER_HEADING_IDX] = agent->teacher_heading_error_rad / PARK_PI;
}

static void park_write_partner_obs(const Park *env, const ParkAgent *agent, float *obs) {
    float ego_vx = agent->speed_mps * cosf(agent->pose.heading_rad);
    float ego_vy = agent->speed_mps * sinf(agent->pose.heading_rad);
    for (int slot_idx = 0; slot_idx < agent->partner_count; slot_idx++) {
        const ParkAgent *partner = &env->agents[agent->partner_indices[slot_idx]];
        float *features = obs + slot_idx * PARK_PARTNER_FEATURES;
        float local_x, local_y, distance_m;
        park_to_ego_frame(&agent->pose, partner->pose.x, partner->pose.y, &local_x, &local_y);
        float relative_vx = partner->speed_mps * cosf(partner->pose.heading_rad) - ego_vx;
        float relative_vy = partner->speed_mps * sinf(partner->pose.heading_rad) - ego_vy;
        float cos_heading = cosf(agent->pose.heading_rad);
        float sin_heading = sinf(agent->pose.heading_rad);
        float closing_mps = park_closing_speed(agent, partner, &distance_m);
        features[0] = park_normalized(local_x, PARK_OBS_PARTNER_RANGE_M);
        features[1] = park_normalized(local_y, PARK_OBS_PARTNER_RANGE_M);
        features[2] = cosf(partner->pose.heading_rad - agent->pose.heading_rad);
        features[3] = sinf(partner->pose.heading_rad - agent->pose.heading_rad);
        features[4] = env->bank->vehicle_length_m / PARK_OBS_LENGTH_NORM_M;
        features[5] = env->bank->vehicle_width_m / PARK_OBS_WIDTH_NORM_M;
        features[6] = park_normalized(relative_vx * cos_heading + relative_vy * sin_heading, PARK_OBS_SPEED_NORM_MPS);
        features[7] = park_normalized(-relative_vx * sin_heading + relative_vy * cos_heading, PARK_OBS_SPEED_NORM_MPS);
        features[8] = park_normalized(distance_m, PARK_OBS_PARTNER_RANGE_M);
        features[9] = park_normalized(closing_mps, PARK_OBS_SPEED_NORM_MPS);
        features[10] = partner->status == PARK_STATUS_PARKED || partner->outcome == PARK_OUTCOME_SUCCESS ? 1.0f : 0.0f;
    }
}

static void park_write_segment_token(const ParkPose *ego, const ParkSegment *segment, float *features) {
    float local_x, local_y;
    park_to_ego_frame(ego, 0.5f * (segment->x0 + segment->x1), 0.5f * (segment->y0 + segment->y1), &local_x, &local_y);
    float segment_heading = atan2f(segment->y1 - segment->y0, segment->x1 - segment->x0);
    features[0] = park_normalized(local_x, PARK_OBS_ROAD_RANGE_M);
    features[1] = park_normalized(local_y, PARK_OBS_ROAD_RANGE_M);
    features[2] = cosf(segment_heading - ego->heading_rad);
    features[3] = sinf(segment_heading - ego->heading_rad);
    features[4]
        = park_normalized(hypotf(segment->x1 - segment->x0, segment->y1 - segment->y0), PARK_OBS_SEGMENT_NORM_M);
}

static int park_write_lane_obs(const Park *env, const ParkAgent *agent, float *obs) {
    float distances[PARK_MAX_LANE_SLOTS];
    int indices[PARK_MAX_LANE_SLOTS];
    int count = 0;
    const ParkSegment *lanes = &env->bank->segments[env->lot->lane_offset];
    for (int lane_idx = 0; lane_idx < env->lot->lane_count; lane_idx++) {
        float distance_m = park_point_segment_distance(agent->pose.x, agent->pose.y, &lanes[lane_idx]);
        if (distance_m > PARK_OBS_ROAD_RANGE_M) {
            continue;
        }
        park_nearest_insert(distances, indices, &count, env->config.lane_slots, distance_m, lane_idx);
    }
    for (int slot_idx = 0; slot_idx < count; slot_idx++) {
        park_write_segment_token(&agent->pose, &lanes[indices[slot_idx]], obs + slot_idx * PARK_LANE_FEATURES);
    }
    return count;
}

static float park_edge_source_feature(int source) {
    if (source == PARK_EDGE_SOURCE_CURB) {
        return -1.0f;
    }
    return source == PARK_EDGE_SOURCE_STATIC_VEHICLE ? 1.0f : 0.0f;
}

static int park_write_boundary_obs(const Park *env, const ParkAgent *agent, float *obs) {
    float distances[PARK_MAX_BOUNDARY_SLOTS];
    int indices[PARK_MAX_BOUNDARY_SLOTS];
    int count = 0;
    for (int edge_idx = 0; edge_idx < env->static_edge_count; edge_idx++) {
        const ParkStaticEdge *edge = &env->static_edges[edge_idx];
        float distance_m = park_point_segment_distance(agent->pose.x, agent->pose.y, &edge->segment);
        if (distance_m > PARK_OBS_ROAD_RANGE_M) {
            continue;
        }
        park_nearest_insert(distances, indices, &count, env->config.boundary_slots, distance_m, edge_idx);
    }
    for (int slot_idx = 0; slot_idx < count; slot_idx++) {
        const ParkStaticEdge *edge = &env->static_edges[indices[slot_idx]];
        float *features = obs + slot_idx * PARK_BOUNDARY_FEATURES;
        park_write_segment_token(&agent->pose, &edge->segment, features);
        features[5] = park_edge_source_feature(edge->source);
    }
    return count;
}

static void park_write_agent_observation(Park *env, int agent_idx) {
    float *obs = env->observations + (size_t) agent_idx * (size_t) env->observation_size;
    const ParkAgent *agent = &env->agents[agent_idx];
    memset(obs, 0, (size_t) env->observation_size * sizeof(float));
    if (!park_agent_running(agent)) {
        return;
    }
    const ParkConfig *config = &env->config;
    park_write_ego_obs(env, agent, obs);
    obs[PARK_EGO_FEATURES]
        = park_conditioning_feature(agent->goal_weight, config->goal_weight_min, config->goal_weight_max);
    obs[PARK_EGO_FEATURES + 1] = park_conditioning_feature(
        agent->collision_weight,
        config->collision_weight_min,
        config->collision_weight_max);
    park_write_tail_obs(agent, obs + PARK_EGO_FEATURES + PARK_CONDITIONING_FEATURES);
    park_write_partner_obs(env, agent, obs + env->partner_offset);
    int lane_count = park_write_lane_obs(env, agent, obs + env->lane_offset);
    int boundary_count = park_write_boundary_obs(env, agent, obs + env->boundary_offset);
    obs[env->count_offset] = (float) agent->partner_count;
    obs[env->count_offset + 1] = (float) lane_count;
    obs[env->count_offset + 2] = (float) boundary_count;
}

static void park_compute_observations(Park *env) {
    for (int agent_idx = 0; agent_idx < env->config.max_agents; agent_idx++) {
        park_write_agent_observation(env, agent_idx);
    }
}

static void park_update_masks(Park *env) {
    for (int agent_idx = 0; agent_idx < env->config.max_agents; agent_idx++) {
        env->masks[agent_idx] = park_agent_running(&env->agents[agent_idx]);
    }
}

static void c_reset(Park *env) {
    env->autoreset_pending = 0;
    env->timestep = 0;
    env->episode_end_step = env->config.episode_steps;
    if (env->first_reset_pending) {
        env->first_reset_pending = 0;
        env->episode_end_step = 1 + park_rng_below(&env->rng, env->config.episode_steps);
    }
    env->scene_idx = park_select_scene(env);
    env->scene = &env->bank->scenes[env->scene_idx];
    env->lot = &env->bank->lots[env->scene->lot_idx];
    park_build_static_edges(env);
    int controlled_count = env->scene->agent_count < env->config.max_controlled_per_scene
        ? env->scene->agent_count
        : env->config.max_controlled_per_scene;
    for (int agent_idx = 0; agent_idx < env->config.max_agents; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (agent_idx >= controlled_count) {
            memset(agent, 0, sizeof(*agent));
            continue;
        }
        park_spawn_agent(env, agent, &env->bank->agents[env->scene->agent_offset + agent_idx]);
        park_update_slot_error(agent);
    }
    park_update_perception(env);
    park_update_teacher(env);
    park_compute_observations(env);
    park_update_masks(env);
}

static void park_finish_agent(Park *env, int agent_idx) {
    ParkAgent *agent = &env->agents[agent_idx];
    env->terminals[agent_idx] = 1;
    park_log_agent_episode(env, agent);
    if (agent->outcome == PARK_OUTCOME_SUCCESS) {
        agent->status = PARK_STATUS_PARKED;
        agent->speed_mps = 0.0f;
        return;
    }
    agent->status = PARK_STATUS_REMOVED;
}

static void park_end_episode_at_horizon(Park *env) {
    for (int agent_idx = 0; agent_idx < env->config.max_agents; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (!park_agent_running(agent)) {
            continue;
        }
        agent->outcome = PARK_OUTCOME_HORIZON;
        env->truncations[agent_idx] = 1;
        park_log_agent_episode(env, agent);
    }
}

static int park_running_agent_count(const Park *env) {
    int running_count = 0;
    for (int agent_idx = 0; agent_idx < env->config.max_agents; agent_idx++) {
        running_count += park_agent_running(&env->agents[agent_idx]);
    }
    return running_count;
}

static void c_step(Park *env) {
    int agent_slots = env->config.max_agents;
    memset(env->rewards, 0, (size_t) agent_slots * sizeof(float));
    memset(env->terminals, 0, (size_t) agent_slots * sizeof(unsigned char));
    memset(env->truncations, 0, (size_t) agent_slots * sizeof(unsigned char));
    if (env->autoreset_pending) {
        c_reset(env);
        return;
    }
    env->timestep++;
    env->lifetime_steps++;
    for (int agent_idx = 0; agent_idx < agent_slots; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (agent->status != PARK_STATUS_ACTIVE) {
            continue;
        }
        park_snapshot_decision(agent);
        park_apply_action(env, agent, env->actions[agent_idx]);
        park_update_motion_metrics(env, agent);
    }
    park_detect_contacts(env);
    for (int agent_idx = 0; agent_idx < agent_slots; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (park_agent_running(agent)) {
            park_update_task_progress(env, agent);
        }
    }
    park_update_perception(env);
    park_update_teacher(env);
    for (int agent_idx = 0; agent_idx < agent_slots; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (agent->status != PARK_STATUS_ACTIVE) {
            continue;
        }
        float reward = park_compute_reward(env, agent);
        env->rewards[agent_idx] = reward;
        agent->episode_return += reward;
        agent->episode_steps++;
        agent->threat_sum += agent->threat;
    }
    for (int agent_idx = 0; agent_idx < agent_slots; agent_idx++) {
        ParkAgent *agent = &env->agents[agent_idx];
        if (agent->status == PARK_STATUS_ACTIVE && agent->outcome != PARK_OUTCOME_NONE) {
            park_finish_agent(env, agent_idx);
        }
    }
    if (env->timestep >= env->episode_end_step) {
        park_compute_observations(env);
        park_end_episode_at_horizon(env);
        memset(env->masks, 0, (size_t) agent_slots * sizeof(unsigned char));
        env->autoreset_pending = 1;
        return;
    }
    park_compute_observations(env);
    park_update_masks(env);
    env->autoreset_pending = park_running_agent_count(env) == 0;
}

static int park_nearest_value_idx(const float *values, int value_count, float target) {
    int best_idx = 0;
    for (int value_idx = 1; value_idx < value_count; value_idx++) {
        if (fabsf(values[value_idx] - target) < fabsf(values[best_idx] - target)) {
            best_idx = value_idx;
        }
    }
    return best_idx;
}

static int park_action_index(float accel_mps2, float steer_rad) {
    int accel_idx = park_nearest_value_idx(PARK_ACCEL_VALUES_MPS2, PARK_NUM_ACCEL_ACTIONS, accel_mps2);
    int steer_idx = park_nearest_value_idx(PARK_STEER_VALUES_RAD, PARK_NUM_STEER_ACTIONS, steer_rad);
    return accel_idx * PARK_NUM_STEER_ACTIONS + steer_idx;
}

// Scripted route follower plus Stanley tracker; the classical Hybrid A* + Stanley baseline.
static int park_teacher_action(const Park *env, int agent_idx) {
    const ParkAgent *agent = &env->agents[agent_idx];
    if (!park_agent_running(agent)) {
        return park_action_index(0.0f, 0.0f);
    }
    if (agent->phase != PARK_PHASE_NAVIGATE) {
        return park_action_index(agent->teacher_accel_mps2, agent->teacher_steer_rad);
    }
    int subgoal_idx = agent->next_subgoal_idx < agent->route_count ? agent->next_subgoal_idx : agent->route_count - 1;
    float local_x, local_y;
    ParkPose rear_axle = park_rear_axle_from_center(agent->pose, env->bank->wheelbase_m);
    park_to_ego_frame(&rear_axle, agent->route[subgoal_idx].x, agent->route[subgoal_idx].y, &local_x, &local_y);
    float lookahead_sq = fmaxf(local_x * local_x + local_y * local_y, PARK_GEOMETRY_EPSILON);
    float steer_rad = atanf(env->bank->wheelbase_m * 2.0f * local_y / lookahead_sq);
    float remaining_m = fmaxf(agent->route_length_m - agent->route_progress_m, 0.0f);
    float reference_speed = fminf(
        PARK_NAV_TEACHER_SPEED_MPS,
        sqrtf(2.0f * PARK_NAV_TEACHER_DECEL_MPS2 * remaining_m) + PARK_CUSP_SPEED_MPS);
    return park_action_index(
        park_speed_command(reference_speed, agent->speed_mps),
        park_clampf(steer_rad, -PARK_MAX_STEER_RAD, PARK_MAX_STEER_RAD));
}

#endif
