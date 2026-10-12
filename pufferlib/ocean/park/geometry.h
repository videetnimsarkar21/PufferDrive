#ifndef PUFFERLIB_OCEAN_PARK_GEOMETRY_H
#define PUFFERLIB_OCEAN_PARK_GEOMETRY_H

#include <math.h>

#define PARK_PI 3.14159265358979323846f
#define PARK_TWO_PI (2.0f * PARK_PI)
#define PARK_HALF_PI (0.5f * PARK_PI)
#define PARK_GEOMETRY_EPSILON 1e-6f
#define PARK_BOX_CORNER_COUNT 4

// Poses are world frame, meters and radians, heading counter-clockwise from +x.
typedef struct {
    float x;
    float y;
    float heading_rad;
} ParkPose;

typedef struct {
    float center_x;
    float center_y;
    float heading_rad;
    float length_m;
    float width_m;
} ParkBox;

typedef struct {
    float x0;
    float y0;
    float x1;
    float y1;
} ParkSegment;

static inline float park_clampf(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

static inline float park_wrap_angle(float angle_rad) {
    return angle_rad - PARK_TWO_PI * floorf((angle_rad + PARK_PI) / PARK_TWO_PI);
}

// Heading error modulo pi: forward-in and reverse-in slot poses are equivalent.
static inline float park_wrap_angle_mod_pi(float angle_rad) {
    return angle_rad - PARK_PI * floorf((angle_rad + PARK_HALF_PI) / PARK_PI);
}

static inline float park_box_bounding_radius(float length_m, float width_m) {
    return 0.5f * sqrtf(length_m * length_m + width_m * width_m);
}

static inline void park_box_corners(const ParkBox *box, float corners_x[4], float corners_y[4]) {
    float cos_heading = cosf(box->heading_rad);
    float sin_heading = sinf(box->heading_rad);
    float half_length = 0.5f * box->length_m;
    float half_width = 0.5f * box->width_m;
    const float length_signs[PARK_BOX_CORNER_COUNT] = {1.0f, 1.0f, -1.0f, -1.0f};
    const float width_signs[PARK_BOX_CORNER_COUNT] = {1.0f, -1.0f, -1.0f, 1.0f};
    for (int corner_idx = 0; corner_idx < PARK_BOX_CORNER_COUNT; corner_idx++) {
        float along = length_signs[corner_idx] * half_length;
        float across = width_signs[corner_idx] * half_width;
        corners_x[corner_idx] = box->center_x + along * cos_heading - across * sin_heading;
        corners_y[corner_idx] = box->center_y + along * sin_heading + across * cos_heading;
    }
}

static inline void park_box_edges(const ParkBox *box, ParkSegment edges[4]) {
    float corners_x[PARK_BOX_CORNER_COUNT];
    float corners_y[PARK_BOX_CORNER_COUNT];
    park_box_corners(box, corners_x, corners_y);
    for (int corner_idx = 0; corner_idx < PARK_BOX_CORNER_COUNT; corner_idx++) {
        int next_idx = (corner_idx + 1) % PARK_BOX_CORNER_COUNT;
        edges[corner_idx]
            = (ParkSegment) {corners_x[corner_idx], corners_y[corner_idx], corners_x[next_idx], corners_y[next_idx]};
    }
}

// Separating-axis test over the four box face normals.
static inline int park_box_overlap(const ParkBox *first, const ParkBox *second) {
    float first_cos = cosf(first->heading_rad);
    float first_sin = sinf(first->heading_rad);
    float second_cos = cosf(second->heading_rad);
    float second_sin = sinf(second->heading_rad);
    const float axes_x[4] = {first_cos, -first_sin, second_cos, -second_sin};
    const float axes_y[4] = {first_sin, first_cos, second_sin, second_cos};
    float delta_x = second->center_x - first->center_x;
    float delta_y = second->center_y - first->center_y;
    for (int axis_idx = 0; axis_idx < 4; axis_idx++) {
        float axis_x = axes_x[axis_idx];
        float axis_y = axes_y[axis_idx];
        float first_radius = 0.5f * first->length_m * fabsf(first_cos * axis_x + first_sin * axis_y)
            + 0.5f * first->width_m * fabsf(-first_sin * axis_x + first_cos * axis_y);
        float second_radius = 0.5f * second->length_m * fabsf(second_cos * axis_x + second_sin * axis_y)
            + 0.5f * second->width_m * fabsf(-second_sin * axis_x + second_cos * axis_y);
        if (fabsf(delta_x * axis_x + delta_y * axis_y) > first_radius + second_radius) {
            return 0;
        }
    }
    return 1;
}

// Separating axes for a box and a segment: the two box axes plus the segment normal.
static inline int park_box_segment_overlap(const ParkBox *box, const ParkSegment *segment) {
    float cos_heading = cosf(box->heading_rad);
    float sin_heading = sinf(box->heading_rad);
    float start_dx = segment->x0 - box->center_x;
    float start_dy = segment->y0 - box->center_y;
    float end_dx = segment->x1 - box->center_x;
    float end_dy = segment->y1 - box->center_y;
    float start_along = start_dx * cos_heading + start_dy * sin_heading;
    float start_across = -start_dx * sin_heading + start_dy * cos_heading;
    float end_along = end_dx * cos_heading + end_dy * sin_heading;
    float end_across = -end_dx * sin_heading + end_dy * cos_heading;
    float half_length = 0.5f * box->length_m;
    float half_width = 0.5f * box->width_m;
    if (fmaxf(start_along, end_along) < -half_length || fminf(start_along, end_along) > half_length) {
        return 0;
    }
    if (fmaxf(start_across, end_across) < -half_width || fminf(start_across, end_across) > half_width) {
        return 0;
    }
    float normal_along = -(end_across - start_across);
    float normal_across = end_along - start_along;
    float segment_projection = normal_along * start_along + normal_across * start_across;
    float box_radius = half_length * fabsf(normal_along) + half_width * fabsf(normal_across);
    return fabsf(segment_projection) <= box_radius;
}

static inline float park_point_segment_distance(float point_x, float point_y, const ParkSegment *segment) {
    float segment_dx = segment->x1 - segment->x0;
    float segment_dy = segment->y1 - segment->y0;
    float length_sq = segment_dx * segment_dx + segment_dy * segment_dy;
    float fraction = 0.0f;
    if (length_sq > PARK_GEOMETRY_EPSILON) {
        fraction = ((point_x - segment->x0) * segment_dx + (point_y - segment->y0) * segment_dy) / length_sq;
        fraction = park_clampf(fraction, 0.0f, 1.0f);
    }
    float closest_dx = segment->x0 + fraction * segment_dx - point_x;
    float closest_dy = segment->y0 + fraction * segment_dy - point_y;
    return sqrtf(closest_dx * closest_dx + closest_dy * closest_dy);
}

static inline float park_cross_2d(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}

static inline int park_segments_intersect(const ParkSegment *first, const ParkSegment *second) {
    float first_dx = first->x1 - first->x0;
    float first_dy = first->y1 - first->y0;
    float second_dx = second->x1 - second->x0;
    float second_dy = second->y1 - second->y0;
    float side_start = park_cross_2d(first_dx, first_dy, second->x0 - first->x0, second->y0 - first->y0);
    float side_end = park_cross_2d(first_dx, first_dy, second->x1 - first->x0, second->y1 - first->y0);
    float other_side_start = park_cross_2d(second_dx, second_dy, first->x0 - second->x0, first->y0 - second->y0);
    float other_side_end = park_cross_2d(second_dx, second_dy, first->x1 - second->x0, first->y1 - second->y0);
    return side_start * side_end <= 0.0f && other_side_start * other_side_end <= 0.0f;
}

static inline float park_segment_segment_distance(const ParkSegment *first, const ParkSegment *second) {
    if (park_segments_intersect(first, second)) {
        return 0.0f;
    }
    float distance = park_point_segment_distance(first->x0, first->y0, second);
    distance = fminf(distance, park_point_segment_distance(first->x1, first->y1, second));
    distance = fminf(distance, park_point_segment_distance(second->x0, second->y0, first));
    return fminf(distance, park_point_segment_distance(second->x1, second->y1, first));
}

static inline float park_box_segment_distance(const ParkBox *box, const ParkSegment *segment) {
    if (park_box_segment_overlap(box, segment)) {
        return 0.0f;
    }
    ParkSegment box_edges[PARK_BOX_CORNER_COUNT];
    park_box_edges(box, box_edges);
    float distance = INFINITY;
    for (int edge_idx = 0; edge_idx < PARK_BOX_CORNER_COUNT; edge_idx++) {
        distance = fminf(distance, park_segment_segment_distance(&box_edges[edge_idx], segment));
    }
    return distance;
}

// Kinematic bicycle arc of the rear-axle reference point; signed length is negative when reversing.
static inline ParkPose park_integrate_arc(ParkPose rear_axle, float curvature_per_m, float signed_length_m) {
    ParkPose result = rear_axle;
    float heading_change = curvature_per_m * signed_length_m;
    if (fabsf(curvature_per_m) < PARK_GEOMETRY_EPSILON) {
        result.x += signed_length_m * cosf(rear_axle.heading_rad);
        result.y += signed_length_m * sinf(rear_axle.heading_rad);
        return result;
    }
    float end_heading = rear_axle.heading_rad + heading_change;
    result.x += (sinf(end_heading) - sinf(rear_axle.heading_rad)) / curvature_per_m;
    result.y += (cosf(rear_axle.heading_rad) - cosf(end_heading)) / curvature_per_m;
    result.heading_rad = park_wrap_angle(end_heading);
    return result;
}

// Box center sits midway between the axles (symmetric overhangs).
static inline ParkPose park_rear_axle_from_center(ParkPose center, float wheelbase_m) {
    ParkPose rear_axle = center;
    rear_axle.x -= 0.5f * wheelbase_m * cosf(center.heading_rad);
    rear_axle.y -= 0.5f * wheelbase_m * sinf(center.heading_rad);
    return rear_axle;
}

static inline ParkPose park_center_from_rear_axle(ParkPose rear_axle, float wheelbase_m) {
    ParkPose center = rear_axle;
    center.x += 0.5f * wheelbase_m * cosf(rear_axle.heading_rad);
    center.y += 0.5f * wheelbase_m * sinf(rear_axle.heading_rad);
    return center;
}

#endif
