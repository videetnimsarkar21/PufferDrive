#ifndef PUFFERLIB_OCEAN_PARK_SCENE_BANK_H
#define PUFFERLIB_OCEAN_PARK_SCENE_BANK_H

#include "geometry.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARK_SCENE_BANK_MAGIC "PARKBANK"
#define PARK_SCENE_BANK_MAGIC_LENGTH 8
#define PARK_SCENE_BANK_VERSION 1
#define PARK_MAX_LOTS 256
#define PARK_MAX_SCENES 1000000
#define PARK_MAX_AGENTS_PER_SCENE 32
#define PARK_MAX_BANK_SEGMENTS 4000000
#define PARK_MAX_BANK_BOXES 4000000
#define PARK_MAX_BANK_AGENTS 8000000
#define PARK_MAX_BANK_ROUTE_POINTS 50000000
#define PARK_MAX_BANK_PLAN_POINTS 100000000
#define PARK_MAX_ROUTE_POINTS_PER_AGENT 512
#define PARK_MAX_PLAN_POINTS_PER_AGENT 4096
#define PARK_MAX_STATIC_VEHICLES_PER_SCENE 512
#define PARK_MAX_LOT_EDGES 8192
#define PARK_NUM_TIERS 3
#define PARK_MAX_PATH_LENGTH 1024
#define PARK_MAX_CACHED_BANKS 8
#define PARK_MAX_COORDINATE_M 10000.0f
#define PARK_MAX_OBJECT_SIZE_M 100.0f
#define PARK_ARC_MONOTONIC_TOLERANCE_M 1e-3f
#define PARK_ERROR_OK 0
#define PARK_ERROR_INVALID 1

typedef struct {
    float min_x;
    float min_y;
    float max_x;
    float max_y;
    int32_t curb_offset;
    int32_t curb_count;
    int32_t obstacle_offset;
    int32_t obstacle_count;
    int32_t lane_offset;
    int32_t lane_count;
    int32_t slot_offset;
    int32_t slot_count;
} ParkLot;

typedef struct {
    int32_t lot_idx;
    int32_t tier;
    int32_t agent_offset;
    int32_t agent_count;
    int32_t static_vehicle_offset;
    int32_t static_vehicle_count;
} ParkScene;

// Start, goal and prep poses are box centers; goal heading is the planned final heading.
typedef struct {
    ParkPose start;
    ParkPose goal;
    ParkPose prep;
    float slot_length_m;
    float slot_width_m;
    int32_t route_offset;
    int32_t route_count;
    int32_t plan_offset;
    int32_t plan_count;
} ParkSceneAgent;

typedef struct {
    float x;
    float y;
    float arc_m;
} ParkRoutePoint;

// Plan points are rear-axle poses; direction is the motion that arrives at the point.
typedef struct {
    float x;
    float y;
    float heading_rad;
    float direction;
    float arc_m;
} ParkPlanPoint;

typedef struct {
    char magic[PARK_SCENE_BANK_MAGIC_LENGTH];
    uint32_t version;
    uint32_t lot_count;
    uint32_t scene_count;
    uint32_t segment_count;
    uint32_t box_count;
    uint32_t agent_count;
    uint32_t route_point_count;
    uint32_t plan_point_count;
    float vehicle_length_m;
    float vehicle_width_m;
    float wheelbase_m;
} ParkSceneBankHeader;

_Static_assert(sizeof(ParkSceneBankHeader) == 52, "scene bank header must be packed");
_Static_assert(sizeof(ParkLot) == 48, "ParkLot must be packed");
_Static_assert(sizeof(ParkScene) == 24, "ParkScene must be packed");
_Static_assert(sizeof(ParkSceneAgent) == 60, "ParkSceneAgent must be packed");
_Static_assert(sizeof(ParkRoutePoint) == 12, "ParkRoutePoint must be packed");
_Static_assert(sizeof(ParkPlanPoint) == 20, "ParkPlanPoint must be packed");
_Static_assert(sizeof(ParkSegment) == 16, "ParkSegment must be packed");
_Static_assert(sizeof(ParkBox) == 20, "ParkBox must be packed");

typedef struct {
    char path[PARK_MAX_PATH_LENGTH];
    int reference_count;
    float vehicle_length_m;
    float vehicle_width_m;
    float wheelbase_m;
    int lot_count;
    int scene_count;
    int segment_count;
    int box_count;
    int agent_count;
    int route_point_count;
    int plan_point_count;
    ParkLot *lots;
    ParkScene *scenes;
    ParkSegment *segments;
    ParkBox *boxes;
    ParkSceneAgent *agents;
    ParkRoutePoint *route_points;
    ParkPlanPoint *plan_points;
    int max_scene_agent_count;
    int max_scene_static_edge_count;
    int tier_scene_offsets[PARK_NUM_TIERS + 1];
    int32_t *tier_scene_indices;
} ParkSceneBank;

static ParkSceneBank *park_scene_bank_cache[PARK_MAX_CACHED_BANKS];

static int park_bank_error(char *error, size_t error_capacity, const char *message, long long detail) {
    snprintf(error, error_capacity, "%s (%lld)", message, detail);
    return PARK_ERROR_INVALID;
}

static int park_range_is_valid(int32_t offset, int32_t count, int total) {
    return offset >= 0 && count >= 0 && (int64_t) offset + (int64_t) count <= (int64_t) total;
}

static int park_coordinate_is_valid(float value) {
    return isfinite(value) && fabsf(value) <= PARK_MAX_COORDINATE_M;
}

static int park_pose_is_valid(const ParkPose *pose) {
    return park_coordinate_is_valid(pose->x) && park_coordinate_is_valid(pose->y) && isfinite(pose->heading_rad)
        && fabsf(pose->heading_rad) <= PARK_TWO_PI;
}

static int park_box_is_valid(const ParkBox *box) {
    ParkPose pose = {box->center_x, box->center_y, box->heading_rad};
    return park_pose_is_valid(&pose) && isfinite(box->length_m) && isfinite(box->width_m) && box->length_m > 0.0f
        && box->width_m > 0.0f && box->length_m <= PARK_MAX_OBJECT_SIZE_M && box->width_m <= PARK_MAX_OBJECT_SIZE_M;
}

static int park_segment_is_valid(const ParkSegment *segment) {
    return park_coordinate_is_valid(segment->x0) && park_coordinate_is_valid(segment->y0)
        && park_coordinate_is_valid(segment->x1) && park_coordinate_is_valid(segment->y1);
}

static void park_scene_bank_free(ParkSceneBank *bank) {
    free(bank->lots);
    free(bank->scenes);
    free(bank->segments);
    free(bank->boxes);
    free(bank->agents);
    free(bank->route_points);
    free(bank->plan_points);
    free(bank->tier_scene_indices);
    free(bank);
}

static int park_validate_header(const ParkSceneBankHeader *header, uint64_t file_size, char *error, size_t capacity) {
    if (memcmp(header->magic, PARK_SCENE_BANK_MAGIC, PARK_SCENE_BANK_MAGIC_LENGTH) != 0) {
        return park_bank_error(error, capacity, "scene bank has wrong magic bytes", 0);
    }
    if (header->version != PARK_SCENE_BANK_VERSION) {
        return park_bank_error(error, capacity, "unsupported scene bank version", header->version);
    }
    if (header->lot_count == 0 || header->lot_count > PARK_MAX_LOTS) {
        return park_bank_error(error, capacity, "invalid lot count", header->lot_count);
    }
    if (header->scene_count == 0 || header->scene_count > PARK_MAX_SCENES) {
        return park_bank_error(error, capacity, "invalid scene count", header->scene_count);
    }
    if (header->segment_count == 0 || header->segment_count > PARK_MAX_BANK_SEGMENTS) {
        return park_bank_error(error, capacity, "invalid segment count", header->segment_count);
    }
    if (header->box_count > PARK_MAX_BANK_BOXES) {
        return park_bank_error(error, capacity, "invalid box count", header->box_count);
    }
    if (header->agent_count == 0 || header->agent_count > PARK_MAX_BANK_AGENTS) {
        return park_bank_error(error, capacity, "invalid agent count", header->agent_count);
    }
    if (header->route_point_count == 0 || header->route_point_count > PARK_MAX_BANK_ROUTE_POINTS) {
        return park_bank_error(error, capacity, "invalid route point count", header->route_point_count);
    }
    if (header->plan_point_count == 0 || header->plan_point_count > PARK_MAX_BANK_PLAN_POINTS) {
        return park_bank_error(error, capacity, "invalid plan point count", header->plan_point_count);
    }
    int vehicle_valid = isfinite(header->vehicle_length_m) && isfinite(header->vehicle_width_m)
        && isfinite(header->wheelbase_m) && header->vehicle_length_m > 0.0f && header->vehicle_width_m > 0.0f
        && header->wheelbase_m > 0.0f && header->wheelbase_m < header->vehicle_length_m
        && header->vehicle_length_m <= PARK_MAX_OBJECT_SIZE_M;
    if (!vehicle_valid) {
        return park_bank_error(error, capacity, "invalid vehicle dimensions", 0);
    }
    uint64_t expected_size = sizeof(ParkSceneBankHeader) + (uint64_t) header->lot_count * sizeof(ParkLot)
        + (uint64_t) header->scene_count * sizeof(ParkScene) + (uint64_t) header->segment_count * sizeof(ParkSegment)
        + (uint64_t) header->box_count * sizeof(ParkBox) + (uint64_t) header->agent_count * sizeof(ParkSceneAgent)
        + (uint64_t) header->route_point_count * sizeof(ParkRoutePoint)
        + (uint64_t) header->plan_point_count * sizeof(ParkPlanPoint);
    if (expected_size != file_size) {
        return park_bank_error(error, capacity, "scene bank size does not match its header", (long long) file_size);
    }
    return PARK_ERROR_OK;
}

static int park_read_array(FILE *file, void *destination, size_t element_size, int count) {
    if (count == 0) {
        return PARK_ERROR_OK;
    }
    return fread(destination, element_size, (size_t) count, file) == (size_t) count ? PARK_ERROR_OK
                                                                                    : PARK_ERROR_INVALID;
}

static int park_validate_lot(const ParkSceneBank *bank, int lot_idx, char *error, size_t capacity) {
    const ParkLot *lot = &bank->lots[lot_idx];
    int bounds_valid = park_coordinate_is_valid(lot->min_x) && park_coordinate_is_valid(lot->min_y)
        && park_coordinate_is_valid(lot->max_x) && park_coordinate_is_valid(lot->max_y) && lot->min_x < lot->max_x
        && lot->min_y < lot->max_y;
    if (!bounds_valid) {
        return park_bank_error(error, capacity, "lot has invalid bounds", lot_idx);
    }
    int ranges_valid = park_range_is_valid(lot->curb_offset, lot->curb_count, bank->segment_count)
        && park_range_is_valid(lot->lane_offset, lot->lane_count, bank->segment_count)
        && park_range_is_valid(lot->obstacle_offset, lot->obstacle_count, bank->box_count)
        && park_range_is_valid(lot->slot_offset, lot->slot_count, bank->box_count);
    if (!ranges_valid || lot->curb_count == 0 || lot->lane_count == 0 || lot->slot_count == 0) {
        return park_bank_error(error, capacity, "lot references missing geometry", lot_idx);
    }
    if (lot->curb_count + 4 * lot->obstacle_count > PARK_MAX_LOT_EDGES) {
        return park_bank_error(error, capacity, "lot has too many static edges", lot_idx);
    }
    return PARK_ERROR_OK;
}

static int park_validate_route(const ParkSceneBank *bank, const ParkSceneAgent *agent) {
    float previous_arc_m = 0.0f;
    for (int point_idx = 0; point_idx < agent->route_count; point_idx++) {
        const ParkRoutePoint *point = &bank->route_points[agent->route_offset + point_idx];
        int point_valid = park_coordinate_is_valid(point->x) && park_coordinate_is_valid(point->y)
            && isfinite(point->arc_m) && point->arc_m + PARK_ARC_MONOTONIC_TOLERANCE_M >= previous_arc_m;
        if (!point_valid) {
            return PARK_ERROR_INVALID;
        }
        previous_arc_m = point->arc_m;
    }
    return PARK_ERROR_OK;
}

static int park_validate_plan(const ParkSceneBank *bank, const ParkSceneAgent *agent) {
    float previous_arc_m = 0.0f;
    for (int point_idx = 0; point_idx < agent->plan_count; point_idx++) {
        const ParkPlanPoint *point = &bank->plan_points[agent->plan_offset + point_idx];
        ParkPose pose = {point->x, point->y, point->heading_rad};
        int point_valid = park_pose_is_valid(&pose) && (point->direction == 1.0f || point->direction == -1.0f)
            && isfinite(point->arc_m) && point->arc_m + PARK_ARC_MONOTONIC_TOLERANCE_M >= previous_arc_m;
        if (!point_valid) {
            return PARK_ERROR_INVALID;
        }
        previous_arc_m = point->arc_m;
    }
    const ParkPlanPoint *last_point = &bank->plan_points[agent->plan_offset + agent->plan_count - 1];
    return last_point->arc_m > 0.0f ? PARK_ERROR_OK : PARK_ERROR_INVALID;
}

static int park_validate_agent(const ParkSceneBank *bank, int agent_idx, char *error, size_t capacity) {
    const ParkSceneAgent *agent = &bank->agents[agent_idx];
    int poses_valid = park_pose_is_valid(&agent->start) && park_pose_is_valid(&agent->goal)
        && park_pose_is_valid(&agent->prep) && isfinite(agent->slot_length_m) && isfinite(agent->slot_width_m)
        && agent->slot_length_m > 0.0f && agent->slot_width_m > 0.0f;
    if (!poses_valid) {
        return park_bank_error(error, capacity, "agent has invalid poses", agent_idx);
    }
    int counts_valid = agent->route_count >= 1 && agent->route_count <= PARK_MAX_ROUTE_POINTS_PER_AGENT
        && agent->plan_count >= 2 && agent->plan_count <= PARK_MAX_PLAN_POINTS_PER_AGENT
        && park_range_is_valid(agent->route_offset, agent->route_count, bank->route_point_count)
        && park_range_is_valid(agent->plan_offset, agent->plan_count, bank->plan_point_count);
    if (!counts_valid) {
        return park_bank_error(error, capacity, "agent has invalid route or plan range", agent_idx);
    }
    if (park_validate_route(bank, agent) != PARK_ERROR_OK) {
        return park_bank_error(error, capacity, "agent route is invalid", agent_idx);
    }
    if (park_validate_plan(bank, agent) != PARK_ERROR_OK) {
        return park_bank_error(error, capacity, "agent plan is invalid", agent_idx);
    }
    return PARK_ERROR_OK;
}

static int park_point_inside_lot(const ParkLot *lot, float x, float y) {
    return x >= lot->min_x && x <= lot->max_x && y >= lot->min_y && y <= lot->max_y;
}

static int park_scene_agents_inside_lot(const ParkSceneBank *bank, const ParkScene *scene) {
    const ParkLot *lot = &bank->lots[scene->lot_idx];
    for (int agent_idx = scene->agent_offset; agent_idx < scene->agent_offset + scene->agent_count; agent_idx++) {
        const ParkSceneAgent *agent = &bank->agents[agent_idx];
        int inside = park_point_inside_lot(lot, agent->start.x, agent->start.y)
            && park_point_inside_lot(lot, agent->goal.x, agent->goal.y)
            && park_point_inside_lot(lot, agent->prep.x, agent->prep.y);
        if (!inside) {
            return 0;
        }
    }
    return 1;
}

static int park_validate_scene(const ParkSceneBank *bank, int scene_idx, char *error, size_t capacity) {
    const ParkScene *scene = &bank->scenes[scene_idx];
    if (scene->lot_idx < 0 || scene->lot_idx >= bank->lot_count) {
        return park_bank_error(error, capacity, "scene references a missing lot", scene_idx);
    }
    if (scene->tier < 0 || scene->tier >= PARK_NUM_TIERS) {
        return park_bank_error(error, capacity, "scene has an invalid density tier", scene_idx);
    }
    if (scene->agent_count < 1 || scene->agent_count > PARK_MAX_AGENTS_PER_SCENE
        || !park_range_is_valid(scene->agent_offset, scene->agent_count, bank->agent_count)) {
        return park_bank_error(error, capacity, "scene has an invalid agent range", scene_idx);
    }
    if (scene->static_vehicle_count > PARK_MAX_STATIC_VEHICLES_PER_SCENE
        || !park_range_is_valid(scene->static_vehicle_offset, scene->static_vehicle_count, bank->box_count)) {
        return park_bank_error(error, capacity, "scene has an invalid static vehicle range", scene_idx);
    }
    if (!park_scene_agents_inside_lot(bank, scene)) {
        return park_bank_error(error, capacity, "scene places an agent outside its lot", scene_idx);
    }
    return PARK_ERROR_OK;
}

static int park_validate_bank_contents(const ParkSceneBank *bank, char *error, size_t capacity) {
    for (int segment_idx = 0; segment_idx < bank->segment_count; segment_idx++) {
        if (!park_segment_is_valid(&bank->segments[segment_idx])) {
            return park_bank_error(error, capacity, "invalid segment", segment_idx);
        }
    }
    for (int box_idx = 0; box_idx < bank->box_count; box_idx++) {
        if (!park_box_is_valid(&bank->boxes[box_idx])) {
            return park_bank_error(error, capacity, "invalid box", box_idx);
        }
    }
    for (int lot_idx = 0; lot_idx < bank->lot_count; lot_idx++) {
        if (park_validate_lot(bank, lot_idx, error, capacity) != PARK_ERROR_OK) {
            return PARK_ERROR_INVALID;
        }
    }
    for (int agent_idx = 0; agent_idx < bank->agent_count; agent_idx++) {
        if (park_validate_agent(bank, agent_idx, error, capacity) != PARK_ERROR_OK) {
            return PARK_ERROR_INVALID;
        }
    }
    for (int scene_idx = 0; scene_idx < bank->scene_count; scene_idx++) {
        if (park_validate_scene(bank, scene_idx, error, capacity) != PARK_ERROR_OK) {
            return PARK_ERROR_INVALID;
        }
    }
    return PARK_ERROR_OK;
}

static int park_index_scene_tiers(ParkSceneBank *bank, char *error, size_t capacity) {
    bank->tier_scene_indices = (int32_t *) calloc((size_t) bank->scene_count, sizeof(int32_t));
    if (bank->tier_scene_indices == NULL) {
        return park_bank_error(error, capacity, "failed to allocate tier index", bank->scene_count);
    }
    int tier_counts[PARK_NUM_TIERS] = {0};
    for (int scene_idx = 0; scene_idx < bank->scene_count; scene_idx++) {
        const ParkScene *scene = &bank->scenes[scene_idx];
        const ParkLot *lot = &bank->lots[scene->lot_idx];
        int static_edge_count = lot->curb_count + 4 * lot->obstacle_count + 4 * scene->static_vehicle_count;
        tier_counts[scene->tier]++;
        bank->max_scene_agent_count
            = bank->max_scene_agent_count > scene->agent_count ? bank->max_scene_agent_count : scene->agent_count;
        bank->max_scene_static_edge_count = bank->max_scene_static_edge_count > static_edge_count
            ? bank->max_scene_static_edge_count
            : static_edge_count;
    }
    int cursors[PARK_NUM_TIERS];
    bank->tier_scene_offsets[0] = 0;
    for (int tier_idx = 0; tier_idx < PARK_NUM_TIERS; tier_idx++) {
        bank->tier_scene_offsets[tier_idx + 1] = bank->tier_scene_offsets[tier_idx] + tier_counts[tier_idx];
        cursors[tier_idx] = bank->tier_scene_offsets[tier_idx];
    }
    for (int scene_idx = 0; scene_idx < bank->scene_count; scene_idx++) {
        int tier = bank->scenes[scene_idx].tier;
        bank->tier_scene_indices[cursors[tier]++] = scene_idx;
    }
    return PARK_ERROR_OK;
}

static int park_allocate_bank_arrays(ParkSceneBank *bank) {
    bank->lots = (ParkLot *) calloc((size_t) bank->lot_count, sizeof(ParkLot));
    bank->scenes = (ParkScene *) calloc((size_t) bank->scene_count, sizeof(ParkScene));
    bank->segments = (ParkSegment *) calloc((size_t) bank->segment_count, sizeof(ParkSegment));
    bank->boxes = (ParkBox *) calloc((size_t) (bank->box_count > 0 ? bank->box_count : 1), sizeof(ParkBox));
    bank->agents = (ParkSceneAgent *) calloc((size_t) bank->agent_count, sizeof(ParkSceneAgent));
    bank->route_points = (ParkRoutePoint *) calloc((size_t) bank->route_point_count, sizeof(ParkRoutePoint));
    bank->plan_points = (ParkPlanPoint *) calloc((size_t) bank->plan_point_count, sizeof(ParkPlanPoint));
    int allocated = bank->lots && bank->scenes && bank->segments && bank->boxes && bank->agents && bank->route_points
        && bank->plan_points;
    return allocated ? PARK_ERROR_OK : PARK_ERROR_INVALID;
}

static int park_read_bank_arrays(ParkSceneBank *bank, FILE *file) {
    int status = park_read_array(file, bank->lots, sizeof(ParkLot), bank->lot_count);
    status |= park_read_array(file, bank->scenes, sizeof(ParkScene), bank->scene_count);
    status |= park_read_array(file, bank->segments, sizeof(ParkSegment), bank->segment_count);
    status |= park_read_array(file, bank->boxes, sizeof(ParkBox), bank->box_count);
    status |= park_read_array(file, bank->agents, sizeof(ParkSceneAgent), bank->agent_count);
    status |= park_read_array(file, bank->route_points, sizeof(ParkRoutePoint), bank->route_point_count);
    status |= park_read_array(file, bank->plan_points, sizeof(ParkPlanPoint), bank->plan_point_count);
    return status;
}

static uint64_t park_file_size(FILE *file) {
    if (fseek(file, 0, SEEK_END) != 0) {
        return 0;
    }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        return 0;
    }
    return (uint64_t) size;
}

static ParkSceneBank *park_scene_bank_load(const char *path, char *error, size_t capacity) {
    if (strlen(path) >= PARK_MAX_PATH_LENGTH) {
        park_bank_error(error, capacity, "scene bank path is too long", (long long) strlen(path));
        return NULL;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        snprintf(error, capacity, "could not open scene bank '%s'", path);
        return NULL;
    }
    uint64_t file_size = park_file_size(file);
    ParkSceneBankHeader header;
    if (file_size < sizeof(header) || fread(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        park_bank_error(error, capacity, "scene bank is truncated", (long long) file_size);
        return NULL;
    }
    if (park_validate_header(&header, file_size, error, capacity) != PARK_ERROR_OK) {
        fclose(file);
        return NULL;
    }
    ParkSceneBank *bank = (ParkSceneBank *) calloc(1, sizeof(ParkSceneBank));
    if (bank == NULL) {
        fclose(file);
        park_bank_error(error, capacity, "failed to allocate scene bank", 0);
        return NULL;
    }
    snprintf(bank->path, sizeof(bank->path), "%s", path);
    bank->vehicle_length_m = header.vehicle_length_m;
    bank->vehicle_width_m = header.vehicle_width_m;
    bank->wheelbase_m = header.wheelbase_m;
    bank->lot_count = (int) header.lot_count;
    bank->scene_count = (int) header.scene_count;
    bank->segment_count = (int) header.segment_count;
    bank->box_count = (int) header.box_count;
    bank->agent_count = (int) header.agent_count;
    bank->route_point_count = (int) header.route_point_count;
    bank->plan_point_count = (int) header.plan_point_count;
    int status = park_allocate_bank_arrays(bank);
    if (status == PARK_ERROR_OK) {
        status = park_read_bank_arrays(bank, file);
    }
    fclose(file);
    if (status != PARK_ERROR_OK) {
        park_scene_bank_free(bank);
        park_bank_error(error, capacity, "failed to read scene bank arrays", 0);
        return NULL;
    }
    if (park_validate_bank_contents(bank, error, capacity) != PARK_ERROR_OK
        || park_index_scene_tiers(bank, error, capacity) != PARK_ERROR_OK) {
        park_scene_bank_free(bank);
        return NULL;
    }
    return bank;
}

static ParkSceneBank *park_scene_bank_acquire(const char *path, char *error, size_t capacity) {
    int free_slot_idx = -1;
    for (int slot_idx = 0; slot_idx < PARK_MAX_CACHED_BANKS; slot_idx++) {
        ParkSceneBank *cached = park_scene_bank_cache[slot_idx];
        if (cached == NULL) {
            free_slot_idx = free_slot_idx < 0 ? slot_idx : free_slot_idx;
            continue;
        }
        if (strcmp(cached->path, path) == 0) {
            cached->reference_count++;
            return cached;
        }
    }
    if (free_slot_idx < 0) {
        park_bank_error(error, capacity, "too many scene banks loaded", PARK_MAX_CACHED_BANKS);
        return NULL;
    }
    ParkSceneBank *bank = park_scene_bank_load(path, error, capacity);
    if (bank == NULL) {
        return NULL;
    }
    bank->reference_count = 1;
    park_scene_bank_cache[free_slot_idx] = bank;
    return bank;
}

static void park_scene_bank_release(ParkSceneBank *bank) {
    bank->reference_count--;
    if (bank->reference_count > 0) {
        return;
    }
    for (int slot_idx = 0; slot_idx < PARK_MAX_CACHED_BANKS; slot_idx++) {
        if (park_scene_bank_cache[slot_idx] == bank) {
            park_scene_bank_cache[slot_idx] = NULL;
        }
    }
    park_scene_bank_free(bank);
}

#endif
