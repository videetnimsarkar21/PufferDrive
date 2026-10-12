#ifndef PUFFERLIB_OCEAN_PARK_PLANNER_H
#define PUFFERLIB_OCEAN_PARK_PLANNER_H

#include "geometry.h"
#include "scene_bank.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PARK_PLAN_OK 0
#define PARK_PLAN_START_IN_COLLISION 1
#define PARK_PLAN_GOAL_IN_COLLISION 2
#define PARK_PLAN_NOT_FOUND 3
#define PARK_PLAN_ALLOCATION_FAILED 4
#define PARK_PLAN_OUTPUT_OVERFLOW 5

#define PARK_RS_SEGMENT_COUNT 5
#define PARK_RS_MAX_CANDIDATES 48
#define PARK_RS_TYPE_COUNT 18
#define PARK_RS_NOP 0
#define PARK_RS_LEFT 1
#define PARK_RS_STRAIGHT 2
#define PARK_RS_RIGHT 3
#define PARK_RS_ZERO 1e-9
#define PARK_RS_PI 3.14159265358979323846
#define PARK_RS_MIN_SEGMENT_M 1e-5f

#define PARK_PLANNER_XY_RESOLUTION_M 0.3f
#define PARK_PLANNER_HEADING_BINS 72
#define PARK_PLANNER_PRIMITIVE_LENGTH_M 0.6f
#define PARK_PLANNER_STEER_LEVELS 5
#define PARK_PLANNER_PRIMITIVE_COUNT (2 * PARK_PLANNER_STEER_LEVELS)
#define PARK_PLANNER_MAX_STEER_RAD 0.5f
#define PARK_PLANNER_COLLISION_MARGIN_M 0.15f
#define PARK_PLANNER_REGION_MARGIN_M 15.0f
#define PARK_PLANNER_MAX_EXPANSIONS 40000
#define PARK_PLANNER_MAX_CELLS 20000000
#define PARK_PLANNER_REVERSE_COST_FACTOR 1.5f
#define PARK_PLANNER_SWITCH_COST_M 4.0f
#define PARK_PLANNER_STEER_COST_PER_RAD 0.5f
#define PARK_PLANNER_STEER_CHANGE_COST_PER_RAD 1.0f
#define PARK_PLANNER_HEURISTIC_WEIGHT 1.5f
#define PARK_PLANNER_ANALYTIC_RANGE_M 25.0f
#define PARK_PLANNER_ANALYTIC_INTERVAL 4
#define PARK_PLANNER_RS_CANDIDATES_CHECKED 10
#define PARK_PLANNER_COLLISION_STEP_M 0.2f
#define PARK_PLAN_SAMPLE_SPACING_M 0.2f
#define PARK_PLANNER_HEAP_MAX_DEPTH 64

static const float PARK_PLANNER_STEER_FRACTIONS[PARK_PLANNER_STEER_LEVELS] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};

// Reeds-Shepp path words: L/R are unit-radius turns, S is straight, following OMPL's table.
static const int PARK_RS_PATH_TYPES[PARK_RS_TYPE_COUNT][PARK_RS_SEGMENT_COUNT] = {
    {PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_NOP, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_NOP, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_STRAIGHT, PARK_RS_LEFT, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_STRAIGHT, PARK_RS_RIGHT, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_STRAIGHT, PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_STRAIGHT, PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_STRAIGHT, PARK_RS_RIGHT, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_STRAIGHT, PARK_RS_LEFT, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_STRAIGHT, PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_STRAIGHT, PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_STRAIGHT, PARK_RS_RIGHT, PARK_RS_NOP, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_STRAIGHT, PARK_RS_LEFT, PARK_RS_NOP, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_STRAIGHT, PARK_RS_LEFT, PARK_RS_NOP, PARK_RS_NOP},
    {PARK_RS_RIGHT, PARK_RS_STRAIGHT, PARK_RS_RIGHT, PARK_RS_NOP, PARK_RS_NOP},
    {PARK_RS_LEFT, PARK_RS_RIGHT, PARK_RS_STRAIGHT, PARK_RS_LEFT, PARK_RS_RIGHT},
    {PARK_RS_RIGHT, PARK_RS_LEFT, PARK_RS_STRAIGHT, PARK_RS_RIGHT, PARK_RS_LEFT},
};

// Segment lengths are in turning radii; a negative length drives that segment in reverse.
typedef struct {
    int type_idx;
    double lengths[PARK_RS_SEGMENT_COUNT];
    double total_length;
} ParkRsPath;

typedef struct {
    ParkRsPath paths[PARK_RS_MAX_CANDIDATES];
    int count;
} ParkRsCandidates;

static double park_rs_mod2pi(double angle_rad) {
    double wrapped = fmod(angle_rad, 2.0 * PARK_RS_PI);
    if (wrapped < -PARK_RS_PI) {
        return wrapped + 2.0 * PARK_RS_PI;
    }
    if (wrapped > PARK_RS_PI) {
        return wrapped - 2.0 * PARK_RS_PI;
    }
    return wrapped;
}

static void park_rs_polar(double x, double y, double *radius, double *angle_rad) {
    *radius = sqrt(x * x + y * y);
    *angle_rad = atan2(y, x);
}

static void park_rs_tau_omega(double u, double v, double xi, double eta, double phi, double *tau, double *omega) {
    double delta = park_rs_mod2pi(u - v);
    double term_a = sin(u) - sin(delta);
    double term_b = cos(u) - cos(delta) - 1.0;
    double angle = atan2(eta * term_a - xi * term_b, xi * term_a + eta * term_b);
    double selector = 2.0 * (cos(delta) - cos(v) - cos(u)) + 3.0;
    *tau = selector < 0.0 ? park_rs_mod2pi(angle + PARK_RS_PI) : park_rs_mod2pi(angle);
    *omega = park_rs_mod2pi(*tau - u + v - phi);
}

// Formula names follow Reeds-Shepp: L/R/S segment, p/m forward/backward, u/up equal-length turns.
static int park_rs_lp_sp_lp(double x, double y, double phi, double *t, double *u, double *v) {
    park_rs_polar(x - sin(phi), y - 1.0 + cos(phi), u, t);
    if (*t < -PARK_RS_ZERO) {
        return 0;
    }
    *v = park_rs_mod2pi(phi - *t);
    return *v >= -PARK_RS_ZERO;
}

static int park_rs_lp_sp_rp(double x, double y, double phi, double *t, double *u, double *v) {
    double radius;
    double angle;
    park_rs_polar(x + sin(phi), y - 1.0 - cos(phi), &radius, &angle);
    double radius_sq = radius * radius;
    if (radius_sq < 4.0) {
        return 0;
    }
    *u = sqrt(radius_sq - 4.0);
    *t = park_rs_mod2pi(angle + atan2(2.0, *u));
    *v = park_rs_mod2pi(*t - phi);
    return *t >= -PARK_RS_ZERO && *v >= -PARK_RS_ZERO;
}

static int park_rs_lp_rm_l(double x, double y, double phi, double *t, double *u, double *v) {
    double radius;
    double angle;
    park_rs_polar(x - sin(phi), y - 1.0 + cos(phi), &radius, &angle);
    if (radius > 4.0) {
        return 0;
    }
    *u = -2.0 * asin(0.25 * radius);
    *t = park_rs_mod2pi(angle + 0.5 * *u + PARK_RS_PI);
    *v = park_rs_mod2pi(phi - *t + *u);
    return *t >= -PARK_RS_ZERO && *u <= PARK_RS_ZERO;
}

static int park_rs_lp_rup_lum_rm(double x, double y, double phi, double *t, double *u, double *v) {
    double xi = x + sin(phi);
    double eta = y - 1.0 - cos(phi);
    double rho = 0.25 * (2.0 + sqrt(xi * xi + eta * eta));
    if (rho > 1.0) {
        return 0;
    }
    *u = acos(rho);
    park_rs_tau_omega(*u, -*u, xi, eta, phi, t, v);
    return *t >= -PARK_RS_ZERO && *v <= PARK_RS_ZERO;
}

static int park_rs_lp_rum_lum_rp(double x, double y, double phi, double *t, double *u, double *v) {
    double xi = x + sin(phi);
    double eta = y - 1.0 - cos(phi);
    double rho = (20.0 - xi * xi - eta * eta) / 16.0;
    if (rho < 0.0 || rho > 1.0) {
        return 0;
    }
    *u = -acos(rho);
    if (*u < -0.5 * PARK_RS_PI) {
        return 0;
    }
    park_rs_tau_omega(*u, *u, xi, eta, phi, t, v);
    return *t >= -PARK_RS_ZERO && *v >= -PARK_RS_ZERO;
}

static int park_rs_lp_rm_sm_lm(double x, double y, double phi, double *t, double *u, double *v) {
    double radius;
    double angle;
    park_rs_polar(x - sin(phi), y - 1.0 + cos(phi), &radius, &angle);
    if (radius < 2.0) {
        return 0;
    }
    double offset = sqrt(radius * radius - 4.0);
    *u = 2.0 - offset;
    *t = park_rs_mod2pi(angle + atan2(offset, -2.0));
    *v = park_rs_mod2pi(phi - 0.5 * PARK_RS_PI - *t);
    return *t >= -PARK_RS_ZERO && *u <= PARK_RS_ZERO && *v <= PARK_RS_ZERO;
}

static int park_rs_lp_rm_sm_rm(double x, double y, double phi, double *t, double *u, double *v) {
    double xi = x + sin(phi);
    double eta = y - 1.0 - cos(phi);
    double radius;
    double angle;
    park_rs_polar(-eta, xi, &radius, &angle);
    if (radius < 2.0) {
        return 0;
    }
    *t = angle;
    *u = 2.0 - radius;
    *v = park_rs_mod2pi(*t + 0.5 * PARK_RS_PI - phi);
    return *t >= -PARK_RS_ZERO && *u <= PARK_RS_ZERO && *v <= PARK_RS_ZERO;
}

static int park_rs_lp_rm_s_lm_rp(double x, double y, double phi, double *t, double *u, double *v) {
    double xi = x + sin(phi);
    double eta = y - 1.0 - cos(phi);
    double radius;
    double angle;
    park_rs_polar(xi, eta, &radius, &angle);
    if (radius < 2.0) {
        return 0;
    }
    *u = 4.0 - sqrt(radius * radius - 4.0);
    if (*u > PARK_RS_ZERO) {
        return 0;
    }
    *t = park_rs_mod2pi(atan2((4.0 - *u) * xi - 2.0 * eta, -2.0 * xi + (*u - 4.0) * eta));
    *v = park_rs_mod2pi(*t - phi);
    return *t >= -PARK_RS_ZERO && *v >= -PARK_RS_ZERO;
}

static void park_rs_add(ParkRsCandidates *candidates, int type_idx, double t, double u, double v, double w, double x) {
    if (candidates->count >= PARK_RS_MAX_CANDIDATES) {
        return;
    }
    ParkRsPath *path = &candidates->paths[candidates->count++];
    path->type_idx = type_idx;
    path->lengths[0] = t;
    path->lengths[1] = u;
    path->lengths[2] = v;
    path->lengths[3] = w;
    path->lengths[4] = x;
    path->total_length = fabs(t) + fabs(u) + fabs(v) + fabs(w) + fabs(x);
}

static void park_rs_csc(double x, double y, double phi, ParkRsCandidates *candidates) {
    double t, u, v;
    if (park_rs_lp_sp_lp(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 14, t, u, v, 0.0, 0.0);
    }
    if (park_rs_lp_sp_lp(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 14, -t, -u, -v, 0.0, 0.0);
    }
    if (park_rs_lp_sp_lp(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 15, t, u, v, 0.0, 0.0);
    }
    if (park_rs_lp_sp_lp(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 15, -t, -u, -v, 0.0, 0.0);
    }
    if (park_rs_lp_sp_rp(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 12, t, u, v, 0.0, 0.0);
    }
    if (park_rs_lp_sp_rp(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 12, -t, -u, -v, 0.0, 0.0);
    }
    if (park_rs_lp_sp_rp(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 13, t, u, v, 0.0, 0.0);
    }
    if (park_rs_lp_sp_rp(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 13, -t, -u, -v, 0.0, 0.0);
    }
}

static void park_rs_ccc(double x, double y, double phi, ParkRsCandidates *candidates) {
    double t, u, v;
    if (park_rs_lp_rm_l(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 0, t, u, v, 0.0, 0.0);
    }
    if (park_rs_lp_rm_l(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 0, -t, -u, -v, 0.0, 0.0);
    }
    if (park_rs_lp_rm_l(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 1, t, u, v, 0.0, 0.0);
    }
    if (park_rs_lp_rm_l(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 1, -t, -u, -v, 0.0, 0.0);
    }
    double x_back = x * cos(phi) + y * sin(phi);
    double y_back = x * sin(phi) - y * cos(phi);
    if (park_rs_lp_rm_l(x_back, y_back, phi, &t, &u, &v)) {
        park_rs_add(candidates, 0, v, u, t, 0.0, 0.0);
    }
    if (park_rs_lp_rm_l(-x_back, y_back, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 0, -v, -u, -t, 0.0, 0.0);
    }
    if (park_rs_lp_rm_l(x_back, -y_back, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 1, v, u, t, 0.0, 0.0);
    }
    if (park_rs_lp_rm_l(-x_back, -y_back, phi, &t, &u, &v)) {
        park_rs_add(candidates, 1, -v, -u, -t, 0.0, 0.0);
    }
}

static void park_rs_cccc(double x, double y, double phi, ParkRsCandidates *candidates) {
    double t, u, v;
    if (park_rs_lp_rup_lum_rm(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 2, t, u, -u, v, 0.0);
    }
    if (park_rs_lp_rup_lum_rm(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 2, -t, -u, u, -v, 0.0);
    }
    if (park_rs_lp_rup_lum_rm(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 3, t, u, -u, v, 0.0);
    }
    if (park_rs_lp_rup_lum_rm(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 3, -t, -u, u, -v, 0.0);
    }
    if (park_rs_lp_rum_lum_rp(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 2, t, u, u, v, 0.0);
    }
    if (park_rs_lp_rum_lum_rp(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 2, -t, -u, -u, -v, 0.0);
    }
    if (park_rs_lp_rum_lum_rp(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 3, t, u, u, v, 0.0);
    }
    if (park_rs_lp_rum_lum_rp(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 3, -t, -u, -u, -v, 0.0);
    }
}

static void park_rs_ccsc(double x, double y, double phi, ParkRsCandidates *candidates) {
    const double half_pi = 0.5 * PARK_RS_PI;
    double t, u, v;
    if (park_rs_lp_rm_sm_lm(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 4, t, -half_pi, u, v, 0.0);
    }
    if (park_rs_lp_rm_sm_lm(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 4, -t, half_pi, -u, -v, 0.0);
    }
    if (park_rs_lp_rm_sm_lm(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 5, t, -half_pi, u, v, 0.0);
    }
    if (park_rs_lp_rm_sm_lm(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 5, -t, half_pi, -u, -v, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 8, t, -half_pi, u, v, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 8, -t, half_pi, -u, -v, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 9, t, -half_pi, u, v, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 9, -t, half_pi, -u, -v, 0.0);
    }
    double x_back = x * cos(phi) + y * sin(phi);
    double y_back = x * sin(phi) - y * cos(phi);
    if (park_rs_lp_rm_sm_lm(x_back, y_back, phi, &t, &u, &v)) {
        park_rs_add(candidates, 6, v, u, -half_pi, t, 0.0);
    }
    if (park_rs_lp_rm_sm_lm(-x_back, y_back, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 6, -v, -u, half_pi, -t, 0.0);
    }
    if (park_rs_lp_rm_sm_lm(x_back, -y_back, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 7, v, u, -half_pi, t, 0.0);
    }
    if (park_rs_lp_rm_sm_lm(-x_back, -y_back, phi, &t, &u, &v)) {
        park_rs_add(candidates, 7, -v, -u, half_pi, -t, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(x_back, y_back, phi, &t, &u, &v)) {
        park_rs_add(candidates, 10, v, u, -half_pi, t, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(-x_back, y_back, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 10, -v, -u, half_pi, -t, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(x_back, -y_back, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 11, v, u, -half_pi, t, 0.0);
    }
    if (park_rs_lp_rm_sm_rm(-x_back, -y_back, phi, &t, &u, &v)) {
        park_rs_add(candidates, 11, -v, -u, half_pi, -t, 0.0);
    }
}

static void park_rs_ccscc(double x, double y, double phi, ParkRsCandidates *candidates) {
    const double half_pi = 0.5 * PARK_RS_PI;
    double t, u, v;
    if (park_rs_lp_rm_s_lm_rp(x, y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 16, t, -half_pi, u, -half_pi, v);
    }
    if (park_rs_lp_rm_s_lm_rp(-x, y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 16, -t, half_pi, -u, half_pi, -v);
    }
    if (park_rs_lp_rm_s_lm_rp(x, -y, -phi, &t, &u, &v)) {
        park_rs_add(candidates, 17, t, -half_pi, u, -half_pi, v);
    }
    if (park_rs_lp_rm_s_lm_rp(-x, -y, phi, &t, &u, &v)) {
        park_rs_add(candidates, 17, -t, half_pi, -u, half_pi, -v);
    }
}

static void park_rs_sort_candidates(ParkRsCandidates *candidates) {
    for (int sorted_end = 1; sorted_end < candidates->count; sorted_end++) {
        ParkRsPath moving = candidates->paths[sorted_end];
        int insert_idx = sorted_end;
        for (; insert_idx > 0 && candidates->paths[insert_idx - 1].total_length > moving.total_length; insert_idx--) {
            candidates->paths[insert_idx] = candidates->paths[insert_idx - 1];
        }
        candidates->paths[insert_idx] = moving;
    }
}

static void park_rs_collect(ParkPose start, ParkPose goal, float turning_radius_m, ParkRsCandidates *candidates) {
    double delta_x = (double) goal.x - (double) start.x;
    double delta_y = (double) goal.y - (double) start.y;
    double cos_heading = cos((double) start.heading_rad);
    double sin_heading = sin((double) start.heading_rad);
    double local_x = (cos_heading * delta_x + sin_heading * delta_y) / turning_radius_m;
    double local_y = (-sin_heading * delta_x + cos_heading * delta_y) / turning_radius_m;
    double phi = park_rs_mod2pi((double) goal.heading_rad - (double) start.heading_rad);
    candidates->count = 0;
    park_rs_csc(local_x, local_y, phi, candidates);
    park_rs_ccc(local_x, local_y, phi, candidates);
    park_rs_cccc(local_x, local_y, phi, candidates);
    park_rs_ccsc(local_x, local_y, phi, candidates);
    park_rs_ccscc(local_x, local_y, phi, candidates);
}

static float park_rs_shortest_length_m(ParkPose start, ParkPose goal, float turning_radius_m) {
    ParkRsCandidates candidates;
    park_rs_collect(start, goal, turning_radius_m, &candidates);
    double shortest_length = INFINITY;
    for (int candidate_idx = 0; candidate_idx < candidates.count; candidate_idx++) {
        shortest_length = fmin(shortest_length, candidates.paths[candidate_idx].total_length);
    }
    return (float) shortest_length * turning_radius_m;
}

static float park_rs_segment_curvature(int segment_type, float turning_radius_m) {
    if (segment_type == PARK_RS_LEFT) {
        return 1.0f / turning_radius_m;
    }
    if (segment_type == PARK_RS_RIGHT) {
        return -1.0f / turning_radius_m;
    }
    return 0.0f;
}

// Pose after driving distance_m along the path; direction_out is the motion arriving there.
static ParkPose park_rs_pose_at(
    ParkPose start,
    const ParkRsPath *path,
    float turning_radius_m,
    float distance_m,
    float *direction_out) {
    ParkPose pose = start;
    float remaining_m = distance_m;
    *direction_out = path->lengths[0] < 0.0 ? -1.0f : 1.0f;
    for (int segment_idx = 0; segment_idx < PARK_RS_SEGMENT_COUNT && remaining_m > 0.0f; segment_idx++) {
        int segment_type = PARK_RS_PATH_TYPES[path->type_idx][segment_idx];
        float segment_length_m = (float) fabs(path->lengths[segment_idx]) * turning_radius_m;
        if (segment_type == PARK_RS_NOP || segment_length_m < PARK_RS_MIN_SEGMENT_M) {
            continue;
        }
        float step_m = fminf(remaining_m, segment_length_m);
        float direction = path->lengths[segment_idx] < 0.0 ? -1.0f : 1.0f;
        pose = park_integrate_arc(pose, park_rs_segment_curvature(segment_type, turning_radius_m), direction * step_m);
        *direction_out = direction;
        remaining_m -= step_m;
    }
    return pose;
}

static float park_rs_path_cost(const ParkRsPath *path, float turning_radius_m, float previous_direction) {
    float cost = 0.0f;
    float direction_so_far = previous_direction;
    for (int segment_idx = 0; segment_idx < PARK_RS_SEGMENT_COUNT; segment_idx++) {
        float segment_length_m = (float) fabs(path->lengths[segment_idx]) * turning_radius_m;
        if (PARK_RS_PATH_TYPES[path->type_idx][segment_idx] == PARK_RS_NOP
            || segment_length_m < PARK_RS_MIN_SEGMENT_M) {
            continue;
        }
        float direction = path->lengths[segment_idx] < 0.0 ? -1.0f : 1.0f;
        cost += segment_length_m * (direction < 0.0f ? PARK_PLANNER_REVERSE_COST_FACTOR : 1.0f);
        cost += (direction_so_far != 0.0f && direction != direction_so_far) ? PARK_PLANNER_SWITCH_COST_M : 0.0f;
        direction_so_far = direction;
    }
    return cost;
}

typedef struct {
    ParkPose rear_axle;
    float cost;
    float priority;
    int parent_idx;
    int primitive_idx;
} ParkPlannerNode;

typedef struct {
    float min_x;
    float min_y;
    float max_x;
    float max_y;
} ParkBounds;

typedef struct {
    float vehicle_length_m;
    float vehicle_width_m;
    float wheelbase_m;
    float turning_radius_m;
    ParkBounds region;
    int cell_count_x;
    int cell_count_y;
    ParkSegment *edges;
    float *edge_mid_x;
    float *edge_mid_y;
    float *edge_half_length_m;
    int edge_count;
    ParkPlannerNode *nodes;
    int node_count;
    int node_capacity;
    int *open_heap;
    int open_count;
    uint8_t *closed_cells;
    int closed_cell_count;
} ParkPlanner;

typedef struct {
    ParkPlanPoint *points;
    int count;
    int capacity;
} ParkPlanOutput;

static float park_planner_primitive_direction(int primitive_idx) {
    return primitive_idx < PARK_PLANNER_STEER_LEVELS ? 1.0f : -1.0f;
}

static float park_planner_primitive_steer(int primitive_idx) {
    return PARK_PLANNER_STEER_FRACTIONS[primitive_idx % PARK_PLANNER_STEER_LEVELS] * PARK_PLANNER_MAX_STEER_RAD;
}

static int park_planner_cell_index(const ParkPlanner *planner, ParkPose rear_axle) {
    int cell_x = (int) floorf((rear_axle.x - planner->region.min_x) / PARK_PLANNER_XY_RESOLUTION_M);
    int cell_y = (int) floorf((rear_axle.y - planner->region.min_y) / PARK_PLANNER_XY_RESOLUTION_M);
    if (cell_x < 0 || cell_y < 0 || cell_x >= planner->cell_count_x || cell_y >= planner->cell_count_y) {
        return -1;
    }
    float heading_positive = rear_axle.heading_rad - PARK_TWO_PI * floorf(rear_axle.heading_rad / PARK_TWO_PI);
    int heading_bin = (int) (heading_positive / PARK_TWO_PI * PARK_PLANNER_HEADING_BINS) % PARK_PLANNER_HEADING_BINS;
    return (cell_y * planner->cell_count_x + cell_x) * PARK_PLANNER_HEADING_BINS + heading_bin;
}

static int park_planner_pose_is_free(const ParkPlanner *planner, ParkPose rear_axle) {
    ParkPose center = park_center_from_rear_axle(rear_axle, planner->wheelbase_m);
    ParkBox box
        = {center.x,
           center.y,
           center.heading_rad,
           planner->vehicle_length_m + 2.0f * PARK_PLANNER_COLLISION_MARGIN_M,
           planner->vehicle_width_m + 2.0f * PARK_PLANNER_COLLISION_MARGIN_M};
    float box_radius = park_box_bounding_radius(box.length_m, box.width_m);
    for (int edge_idx = 0; edge_idx < planner->edge_count; edge_idx++) {
        float delta_x = planner->edge_mid_x[edge_idx] - center.x;
        float delta_y = planner->edge_mid_y[edge_idx] - center.y;
        float reach = box_radius + planner->edge_half_length_m[edge_idx];
        if (delta_x * delta_x + delta_y * delta_y > reach * reach) {
            continue;
        }
        if (park_box_segment_overlap(&box, &planner->edges[edge_idx])) {
            return 0;
        }
    }
    return 1;
}

static int park_planner_arc_is_free(
    const ParkPlanner *planner,
    ParkPose from,
    float curvature_per_m,
    float signed_length_m) {
    int step_count = (int) ceilf(fabsf(signed_length_m) / PARK_PLANNER_COLLISION_STEP_M);
    for (int step_idx = 1; step_idx <= step_count; step_idx++) {
        ParkPose pose
            = park_integrate_arc(from, curvature_per_m, signed_length_m * (float) step_idx / (float) step_count);
        if (!park_planner_pose_is_free(planner, pose)) {
            return 0;
        }
    }
    return 1;
}

static int park_planner_rs_path_is_free(const ParkPlanner *planner, ParkPose start, const ParkRsPath *path) {
    float total_length_m = (float) path->total_length * planner->turning_radius_m;
    int step_count = (int) ceilf(total_length_m / PARK_PLANNER_COLLISION_STEP_M);
    for (int step_idx = 1; step_idx <= step_count; step_idx++) {
        float direction;
        ParkPose pose = park_rs_pose_at(
            start,
            path,
            planner->turning_radius_m,
            total_length_m * (float) step_idx / (float) step_count,
            &direction);
        if (!park_planner_pose_is_free(planner, pose)) {
            return 0;
        }
    }
    return 1;
}

static int park_planner_heap_less(const ParkPlanner *planner, int first_node_idx, int second_node_idx) {
    float first_priority = planner->nodes[first_node_idx].priority;
    float second_priority = planner->nodes[second_node_idx].priority;
    if (first_priority != second_priority) {
        return first_priority < second_priority;
    }
    return first_node_idx < second_node_idx;
}

static void park_planner_heap_push(ParkPlanner *planner, int node_idx) {
    int position = planner->open_count++;
    planner->open_heap[position] = node_idx;
    for (int depth = 0; depth < PARK_PLANNER_HEAP_MAX_DEPTH && position > 0; depth++) {
        int parent = (position - 1) / 2;
        if (!park_planner_heap_less(planner, planner->open_heap[position], planner->open_heap[parent])) {
            return;
        }
        int swapped = planner->open_heap[parent];
        planner->open_heap[parent] = planner->open_heap[position];
        planner->open_heap[position] = swapped;
        position = parent;
    }
}

static int park_planner_heap_pop(ParkPlanner *planner) {
    int top_node_idx = planner->open_heap[0];
    planner->open_heap[0] = planner->open_heap[--planner->open_count];
    int position = 0;
    for (int depth = 0; depth < PARK_PLANNER_HEAP_MAX_DEPTH; depth++) {
        int left = 2 * position + 1;
        int right = left + 1;
        int smallest = position;
        if (left < planner->open_count
            && park_planner_heap_less(planner, planner->open_heap[left], planner->open_heap[smallest])) {
            smallest = left;
        }
        if (right < planner->open_count
            && park_planner_heap_less(planner, planner->open_heap[right], planner->open_heap[smallest])) {
            smallest = right;
        }
        if (smallest == position) {
            break;
        }
        int swapped = planner->open_heap[smallest];
        planner->open_heap[smallest] = planner->open_heap[position];
        planner->open_heap[position] = swapped;
        position = smallest;
    }
    return top_node_idx;
}

static float park_planner_transition_cost(int parent_primitive_idx, int primitive_idx) {
    float direction = park_planner_primitive_direction(primitive_idx);
    float steer_rad = park_planner_primitive_steer(primitive_idx);
    float cost = PARK_PLANNER_PRIMITIVE_LENGTH_M * (direction < 0.0f ? PARK_PLANNER_REVERSE_COST_FACTOR : 1.0f);
    cost += PARK_PLANNER_STEER_COST_PER_RAD * fabsf(steer_rad);
    if (parent_primitive_idx < 0) {
        return cost;
    }
    float parent_direction = park_planner_primitive_direction(parent_primitive_idx);
    float parent_steer_rad = park_planner_primitive_steer(parent_primitive_idx);
    cost += parent_direction != direction ? PARK_PLANNER_SWITCH_COST_M : 0.0f;
    cost += PARK_PLANNER_STEER_CHANGE_COST_PER_RAD * fabsf(steer_rad - parent_steer_rad);
    return cost;
}

static int park_planner_add_node(
    ParkPlanner *planner,
    ParkPose rear_axle,
    float cost,
    float priority,
    int parent_idx,
    int primitive_idx) {
    if (planner->node_count >= planner->node_capacity) {
        return -1;
    }
    int node_idx = planner->node_count++;
    planner->nodes[node_idx] = (ParkPlannerNode) {rear_axle, cost, priority, parent_idx, primitive_idx};
    park_planner_heap_push(planner, node_idx);
    return node_idx;
}

static void park_planner_expand(ParkPlanner *planner, int node_idx, ParkPose goal_rear_axle) {
    ParkPlannerNode node = planner->nodes[node_idx];
    for (int primitive_idx = 0; primitive_idx < PARK_PLANNER_PRIMITIVE_COUNT; primitive_idx++) {
        float curvature_per_m = tanf(park_planner_primitive_steer(primitive_idx)) / planner->wheelbase_m;
        float signed_length_m = park_planner_primitive_direction(primitive_idx) * PARK_PLANNER_PRIMITIVE_LENGTH_M;
        ParkPose child = park_integrate_arc(node.rear_axle, curvature_per_m, signed_length_m);
        int cell_idx = park_planner_cell_index(planner, child);
        if (cell_idx < 0 || planner->closed_cells[cell_idx]) {
            continue;
        }
        if (!park_planner_arc_is_free(planner, node.rear_axle, curvature_per_m, signed_length_m)) {
            continue;
        }
        float cost = node.cost + park_planner_transition_cost(node.primitive_idx, primitive_idx);
        float heuristic_m = park_rs_shortest_length_m(child, goal_rear_axle, planner->turning_radius_m);
        if (park_planner_add_node(
                planner,
                child,
                cost,
                cost + PARK_PLANNER_HEURISTIC_WEIGHT * heuristic_m,
                node_idx,
                primitive_idx)
            < 0) {
            return;
        }
    }
}

static int park_planner_try_analytic(
    const ParkPlanner *planner,
    int node_idx,
    ParkPose goal_rear_axle,
    ParkRsPath *path_out) {
    ParkPose rear_axle = planner->nodes[node_idx].rear_axle;
    float distance_m = hypotf(goal_rear_axle.x - rear_axle.x, goal_rear_axle.y - rear_axle.y);
    if (distance_m > PARK_PLANNER_ANALYTIC_RANGE_M) {
        return 0;
    }
    ParkRsCandidates candidates;
    park_rs_collect(rear_axle, goal_rear_axle, planner->turning_radius_m, &candidates);
    park_rs_sort_candidates(&candidates);
    int checked_count
        = candidates.count < PARK_PLANNER_RS_CANDIDATES_CHECKED ? candidates.count : PARK_PLANNER_RS_CANDIDATES_CHECKED;
    for (int candidate_idx = 0; candidate_idx < checked_count; candidate_idx++) {
        if (park_planner_rs_path_is_free(planner, rear_axle, &candidates.paths[candidate_idx])) {
            *path_out = candidates.paths[candidate_idx];
            return 1;
        }
    }
    return 0;
}

static int park_plan_append(ParkPlanOutput *output, ParkPose rear_axle, float direction) {
    if (output->count >= output->capacity) {
        return PARK_PLAN_OUTPUT_OVERFLOW;
    }
    float arc_m = 0.0f;
    if (output->count > 0) {
        const ParkPlanPoint *previous = &output->points[output->count - 1];
        arc_m = previous->arc_m + hypotf(rear_axle.x - previous->x, rear_axle.y - previous->y);
    }
    output->points[output->count++]
        = (ParkPlanPoint) {rear_axle.x, rear_axle.y, park_wrap_angle(rear_axle.heading_rad), direction, arc_m};
    return PARK_PLAN_OK;
}

static int park_plan_append_arc(ParkPlanOutput *output, ParkPose from, float curvature_per_m, float signed_length_m) {
    int sample_count = (int) ceilf(fabsf(signed_length_m) / PARK_PLAN_SAMPLE_SPACING_M);
    float direction = signed_length_m < 0.0f ? -1.0f : 1.0f;
    for (int sample_idx = 1; sample_idx <= sample_count; sample_idx++) {
        ParkPose pose
            = park_integrate_arc(from, curvature_per_m, signed_length_m * (float) sample_idx / (float) sample_count);
        if (park_plan_append(output, pose, direction) != PARK_PLAN_OK) {
            return PARK_PLAN_OUTPUT_OVERFLOW;
        }
    }
    return PARK_PLAN_OK;
}

static int park_plan_append_rs(ParkPlanOutput *output, ParkPose from, const ParkRsPath *path, float turning_radius_m) {
    float total_length_m = (float) path->total_length * turning_radius_m;
    int sample_count = (int) ceilf(total_length_m / PARK_PLAN_SAMPLE_SPACING_M);
    for (int sample_idx = 1; sample_idx <= sample_count; sample_idx++) {
        float direction;
        ParkPose pose = park_rs_pose_at(
            from,
            path,
            turning_radius_m,
            total_length_m * (float) sample_idx / (float) sample_count,
            &direction);
        if (park_plan_append(output, pose, direction) != PARK_PLAN_OK) {
            return PARK_PLAN_OUTPUT_OVERFLOW;
        }
    }
    return PARK_PLAN_OK;
}

static int park_planner_reconstruct(
    const ParkPlanner *planner,
    int final_node_idx,
    const ParkRsPath *rs_path,
    ParkPlanOutput *output,
    int *chain_buffer) {
    int chain_count = 0;
    int node_idx = final_node_idx;
    for (int hop = 0; hop < planner->node_count && node_idx >= 0; hop++) {
        chain_buffer[chain_count++] = node_idx;
        node_idx = planner->nodes[node_idx].parent_idx;
    }
    output->count = 0;
    ParkPose start = planner->nodes[chain_buffer[chain_count - 1]].rear_axle;
    if (park_plan_append(output, start, 1.0f) != PARK_PLAN_OK) {
        return PARK_PLAN_OUTPUT_OVERFLOW;
    }
    for (int chain_idx = chain_count - 2; chain_idx >= 0; chain_idx--) {
        const ParkPlannerNode *node = &planner->nodes[chain_buffer[chain_idx]];
        const ParkPlannerNode *parent = &planner->nodes[node->parent_idx];
        float curvature_per_m = tanf(park_planner_primitive_steer(node->primitive_idx)) / planner->wheelbase_m;
        float signed_length_m = park_planner_primitive_direction(node->primitive_idx) * PARK_PLANNER_PRIMITIVE_LENGTH_M;
        if (park_plan_append_arc(output, parent->rear_axle, curvature_per_m, signed_length_m) != PARK_PLAN_OK) {
            return PARK_PLAN_OUTPUT_OVERFLOW;
        }
    }
    ParkPose rs_start = planner->nodes[final_node_idx].rear_axle;
    if (park_plan_append_rs(output, rs_start, rs_path, planner->turning_radius_m) != PARK_PLAN_OK) {
        return PARK_PLAN_OUTPUT_OVERFLOW;
    }
    if (output->count < 2) {
        return PARK_PLAN_NOT_FOUND;
    }
    output->points[0].direction = output->points[1].direction;
    return PARK_PLAN_OK;
}

static void park_planner_free(ParkPlanner *planner) {
    free(planner->edges);
    free(planner->edge_mid_x);
    free(planner->edge_mid_y);
    free(planner->edge_half_length_m);
    free(planner->nodes);
    free(planner->open_heap);
    free(planner->closed_cells);
}

static int park_segment_touches_bounds(const ParkSegment *segment, const ParkBounds *bounds) {
    return fmaxf(segment->x0, segment->x1) >= bounds->min_x && fminf(segment->x0, segment->x1) <= bounds->max_x
        && fmaxf(segment->y0, segment->y1) >= bounds->min_y && fminf(segment->y0, segment->y1) <= bounds->max_y;
}

static int park_planner_init(
    ParkPlanner *planner,
    const ParkSegment *edges,
    int edge_count,
    ParkBounds lot_bounds,
    ParkPose start_rear,
    ParkPose goal_rear,
    float vehicle_length_m,
    float vehicle_width_m,
    float wheelbase_m) {
    memset(planner, 0, sizeof(*planner));
    planner->vehicle_length_m = vehicle_length_m;
    planner->vehicle_width_m = vehicle_width_m;
    planner->wheelbase_m = wheelbase_m;
    planner->turning_radius_m = wheelbase_m / tanf(PARK_PLANNER_MAX_STEER_RAD);
    planner->region.min_x = fmaxf(lot_bounds.min_x, fminf(start_rear.x, goal_rear.x) - PARK_PLANNER_REGION_MARGIN_M);
    planner->region.min_y = fmaxf(lot_bounds.min_y, fminf(start_rear.y, goal_rear.y) - PARK_PLANNER_REGION_MARGIN_M);
    planner->region.max_x = fminf(lot_bounds.max_x, fmaxf(start_rear.x, goal_rear.x) + PARK_PLANNER_REGION_MARGIN_M);
    planner->region.max_y = fminf(lot_bounds.max_y, fmaxf(start_rear.y, goal_rear.y) + PARK_PLANNER_REGION_MARGIN_M);
    planner->cell_count_x = (int) ceilf((planner->region.max_x - planner->region.min_x) / PARK_PLANNER_XY_RESOLUTION_M);
    planner->cell_count_y = (int) ceilf((planner->region.max_y - planner->region.min_y) / PARK_PLANNER_XY_RESOLUTION_M);
    int64_t cell_count = (int64_t) planner->cell_count_x * planner->cell_count_y * PARK_PLANNER_HEADING_BINS;
    if (planner->cell_count_x <= 0 || planner->cell_count_y <= 0 || cell_count > PARK_PLANNER_MAX_CELLS) {
        return PARK_PLAN_ALLOCATION_FAILED;
    }
    planner->closed_cell_count = (int) cell_count;
    planner->node_capacity = PARK_PLANNER_MAX_EXPANSIONS * PARK_PLANNER_PRIMITIVE_COUNT + 1;
    planner->edges = (ParkSegment *) calloc((size_t) (edge_count > 0 ? edge_count : 1), sizeof(ParkSegment));
    planner->edge_mid_x = (float *) calloc((size_t) (edge_count > 0 ? edge_count : 1), sizeof(float));
    planner->edge_mid_y = (float *) calloc((size_t) (edge_count > 0 ? edge_count : 1), sizeof(float));
    planner->edge_half_length_m = (float *) calloc((size_t) (edge_count > 0 ? edge_count : 1), sizeof(float));
    planner->nodes = (ParkPlannerNode *) calloc((size_t) planner->node_capacity, sizeof(ParkPlannerNode));
    planner->open_heap = (int *) calloc((size_t) planner->node_capacity, sizeof(int));
    planner->closed_cells = (uint8_t *) calloc((size_t) planner->closed_cell_count, sizeof(uint8_t));
    if (!planner->edges || !planner->edge_mid_x || !planner->edge_mid_y || !planner->edge_half_length_m
        || !planner->nodes || !planner->open_heap || !planner->closed_cells) {
        return PARK_PLAN_ALLOCATION_FAILED;
    }
    float vehicle_reach_m
        = park_box_bounding_radius(vehicle_length_m, vehicle_width_m) + PARK_PLANNER_COLLISION_MARGIN_M;
    ParkBounds query
        = {planner->region.min_x - vehicle_reach_m,
           planner->region.min_y - vehicle_reach_m,
           planner->region.max_x + vehicle_reach_m,
           planner->region.max_y + vehicle_reach_m};
    for (int edge_idx = 0; edge_idx < edge_count; edge_idx++) {
        const ParkSegment *segment = &edges[edge_idx];
        if (!park_segment_touches_bounds(segment, &query)) {
            continue;
        }
        int kept_idx = planner->edge_count++;
        planner->edges[kept_idx] = *segment;
        planner->edge_mid_x[kept_idx] = 0.5f * (segment->x0 + segment->x1);
        planner->edge_mid_y[kept_idx] = 0.5f * (segment->y0 + segment->y1);
        planner->edge_half_length_m[kept_idx] = 0.5f * hypotf(segment->x1 - segment->x0, segment->y1 - segment->y0);
    }
    return PARK_PLAN_OK;
}

static int park_planner_search(
    ParkPlanner *planner,
    ParkPose start_rear,
    ParkPose goal_rear,
    ParkPlanOutput *output,
    float *cost_out) {
    if (!park_planner_pose_is_free(planner, start_rear) || park_planner_cell_index(planner, start_rear) < 0) {
        return PARK_PLAN_START_IN_COLLISION;
    }
    if (!park_planner_pose_is_free(planner, goal_rear)) {
        return PARK_PLAN_GOAL_IN_COLLISION;
    }
    float start_heuristic_m = park_rs_shortest_length_m(start_rear, goal_rear, planner->turning_radius_m);
    park_planner_add_node(planner, start_rear, 0.0f, PARK_PLANNER_HEURISTIC_WEIGHT * start_heuristic_m, -1, -1);
    for (int expansion_idx = 0; expansion_idx < PARK_PLANNER_MAX_EXPANSIONS && planner->open_count > 0;
         expansion_idx++) {
        int node_idx = park_planner_heap_pop(planner);
        int cell_idx = park_planner_cell_index(planner, planner->nodes[node_idx].rear_axle);
        if (planner->closed_cells[cell_idx]) {
            continue;
        }
        planner->closed_cells[cell_idx] = 1;
        ParkRsPath rs_path;
        int try_shot = expansion_idx % PARK_PLANNER_ANALYTIC_INTERVAL == 0;
        if (try_shot && park_planner_try_analytic(planner, node_idx, goal_rear, &rs_path)) {
            int *chain_buffer = (int *) calloc((size_t) planner->node_count, sizeof(int));
            if (chain_buffer == NULL) {
                return PARK_PLAN_ALLOCATION_FAILED;
            }
            int status = park_planner_reconstruct(planner, node_idx, &rs_path, output, chain_buffer);
            free(chain_buffer);
            int primitive_idx = planner->nodes[node_idx].primitive_idx;
            float last_direction = primitive_idx < 0 ? 0.0f : park_planner_primitive_direction(primitive_idx);
            *cost_out = planner->nodes[node_idx].cost
                + park_rs_path_cost(&rs_path, planner->turning_radius_m, last_direction);
            return status;
        }
        park_planner_expand(planner, node_idx, goal_rear);
    }
    return PARK_PLAN_NOT_FOUND;
}

// Plans a rear-axle path between two box-center poses around static edges; offline use only (allocates).
static int park_plan_maneuver(
    const ParkSegment *edges,
    int edge_count,
    ParkBounds lot_bounds,
    ParkPose start_center,
    ParkPose goal_center,
    float vehicle_length_m,
    float vehicle_width_m,
    float wheelbase_m,
    ParkPlanOutput *output,
    float *cost_out) {
    ParkPose start_rear = park_rear_axle_from_center(start_center, wheelbase_m);
    ParkPose goal_rear = park_rear_axle_from_center(goal_center, wheelbase_m);
    ParkPlanner planner;
    int status = park_planner_init(
        &planner,
        edges,
        edge_count,
        lot_bounds,
        start_rear,
        goal_rear,
        vehicle_length_m,
        vehicle_width_m,
        wheelbase_m);
    if (status == PARK_PLAN_OK) {
        status = park_planner_search(&planner, start_rear, goal_rear, output, cost_out);
    }
    park_planner_free(&planner);
    return status;
}

#endif
