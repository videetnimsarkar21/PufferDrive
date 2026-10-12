"""Offline CoPark scene bank generation.

Every scene samples a density tier, parks static vehicles in random slots, spawns controlled agents on
the aisle graph and gives each an injective slot assignment. An assignment is kept only when a
lane-graph route to a preparation pose exists and Hybrid A* (Reeds-Shepp shots, static-only collision
checks) finds a maneuver from that pose into the slot, so every agent starts kinematically feasible.

    python -m pufferlib.ocean.park.scenes --output pufferlib/resources/park/park_scenes.bin
"""

import argparse
import heapq
import multiprocessing
import os
import struct
from dataclasses import dataclass

import numpy as np

from pufferlib.ocean.park import binding
from pufferlib.ocean.park.lots import LAYOUTS, build_lots

BANK_MAGIC = b"PARKBANK"
HEADER_FORMAT = "<8s8I3f"
VEHICLE_LENGTH_M = 3.2
VEHICLE_WIDTH_M = 1.4
WHEELBASE_M = 2.1
TIER_OCCUPANCIES = (0.25, 0.5, 0.75)
TIER_PROBABILITIES = (0.25, 0.25, 0.5)
CONTROLLED_SHARE = 0.4
SUBGOAL_SPACING_M = 4.0
MIN_SUBGOAL_GAP_M = 1.0
DIRECT_ROUTE_MIN_M = 2.0
SPAWN_EDGE_MARGIN_M = 3.5
SPAWN_MIN_GAP_M = 6.5
SPAWN_LATERAL_JITTER_M = 0.3
SPAWN_HEADING_JITTER_RAD = 0.1
SPAWN_ATTEMPTS_PER_AGENT = 50
PREP_EDGE_MARGIN_M = 1.0
PREP_OFFSETS_M = (-1.0, 5.0)
APPROACH_LENGTH_M = 3.0
PLAN_SAMPLE_SPACING_M = 0.2
SLOT_ASSIGNMENT_ATTEMPTS = 12
MAX_ROUTE_LENGTH_M = 45.0
SCENE_ATTEMPTS = 5
STATIC_LENGTH_RANGE_M = (3.6, 4.6)
STATIC_WIDTH_RANGE_M = (1.55, 1.85)
STATIC_LENGTH_MARGIN_M = 0.4
STATIC_WIDTH_MARGIN_M = 0.5
STATIC_LONG_JITTER_M = 0.25
STATIC_LAT_JITTER_M = 0.12
STATIC_HEADING_JITTER_RAD = 0.04

LOT_DTYPE = np.dtype(
    [("bounds", "<f4", (4,))]
    + [(name, "<i4") for name in ("curb_offset", "curb_count", "obstacle_offset", "obstacle_count")]
    + [(name, "<i4") for name in ("lane_offset", "lane_count", "slot_offset", "slot_count")]
)
SCENE_DTYPE = np.dtype(
    [
        (name, "<i4")
        for name in ("lot_idx", "tier", "agent_offset", "agent_count", "static_vehicle_offset", "static_vehicle_count")
    ]
)
AGENT_DTYPE = np.dtype(
    [
        ("start", "<f4", (3,)),
        ("goal", "<f4", (3,)),
        ("prep", "<f4", (3,)),
        ("slot_length_m", "<f4"),
        ("slot_width_m", "<f4"),
        ("route_offset", "<i4"),
        ("route_count", "<i4"),
        ("plan_offset", "<i4"),
        ("plan_count", "<i4"),
    ]
)


@dataclass
class Spawn:
    edge_idx: int
    along_m: float
    pose: tuple


def wrap_angle(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def box_edges(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    cos_heading, sin_heading = np.cos(boxes[:, 2]), np.sin(boxes[:, 2])
    corner_signs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
    along = corner_signs[None, :, 0] * 0.5 * boxes[:, 3:4]
    across = corner_signs[None, :, 1] * 0.5 * boxes[:, 4:5]
    corner_x = boxes[:, 0:1] + along * cos_heading[:, None] - across * sin_heading[:, None]
    corner_y = boxes[:, 1:2] + along * sin_heading[:, None] + across * cos_heading[:, None]
    next_x, next_y = np.roll(corner_x, -1, axis=1), np.roll(corner_y, -1, axis=1)
    return np.stack([corner_x, corner_y, next_x, next_y], axis=-1).reshape(-1, 4)


def box_hits_segments(box, segments):
    """Separating-axis overlap of one box against many segments, matching geometry.h."""
    center_x, center_y, heading, length_m, width_m = box
    cos_heading, sin_heading = np.cos(heading), np.sin(heading)
    start_dx, start_dy = segments[:, 0] - center_x, segments[:, 1] - center_y
    end_dx, end_dy = segments[:, 2] - center_x, segments[:, 3] - center_y
    start_along = start_dx * cos_heading + start_dy * sin_heading
    start_across = -start_dx * sin_heading + start_dy * cos_heading
    end_along = end_dx * cos_heading + end_dy * sin_heading
    end_across = -end_dx * sin_heading + end_dy * cos_heading
    half_length, half_width = 0.5 * length_m, 0.5 * width_m
    separated = (np.maximum(start_along, end_along) < -half_length) | (np.minimum(start_along, end_along) > half_length)
    separated |= (np.maximum(start_across, end_across) < -half_width) | (
        np.minimum(start_across, end_across) > half_width
    )
    normal_along, normal_across = -(end_across - start_across), end_along - start_along
    projection = normal_along * start_along + normal_across * start_across
    separated |= np.abs(projection) > half_length * np.abs(normal_along) + half_width * np.abs(normal_across)
    return bool(np.any(~separated))


def place_static_vehicle(slot, rng):
    center_x, center_y, heading, slot_length_m, slot_width_m = (float(value) for value in slot)
    length_m = min(rng.uniform(*STATIC_LENGTH_RANGE_M), slot_length_m - STATIC_LENGTH_MARGIN_M)
    width_m = min(rng.uniform(*STATIC_WIDTH_RANGE_M), slot_width_m - STATIC_WIDTH_MARGIN_M)
    along_m = rng.uniform(-1.0, 1.0) * min(STATIC_LONG_JITTER_M, 0.5 * (slot_length_m - length_m))
    across_m = rng.uniform(-1.0, 1.0) * min(STATIC_LAT_JITTER_M, 0.5 * (slot_width_m - width_m))
    center_x += along_m * np.cos(heading) - across_m * np.sin(heading)
    center_y += along_m * np.sin(heading) + across_m * np.cos(heading)
    vehicle_heading = (
        heading + (np.pi if rng.random() < 0.5 else 0.0) + rng.uniform(-1.0, 1.0) * STATIC_HEADING_JITTER_RAD
    )
    return (center_x, center_y, float(wrap_angle(vehicle_heading)), length_m, width_m)


class LaneGraph:
    def __init__(self, lot):
        self.nodes = lot.nodes
        self.edges = lot.edges
        starts, ends = self.nodes[self.edges[:, 0]], self.nodes[self.edges[:, 1]]
        self.lengths = np.linalg.norm(ends - starts, axis=1)
        self.directions = (ends - starts) / self.lengths[:, None]
        self.successors = []
        for edge_start, edge_end in self.edges:
            following = [idx for idx, (start, end) in enumerate(self.edges) if start == edge_end and end != edge_start]
            if not following:
                following = [idx for idx, (start, _) in enumerate(self.edges) if start == edge_end]
            self.successors.append(following)

    def point(self, edge_idx, along_m):
        return self.nodes[self.edges[edge_idx, 0]] + self.directions[edge_idx] * along_m

    def route(self, start_edge, start_along_m, goal_edge, goal_along_m):
        """Shortest polyline from a spawn to a goal on directed edges, never U-turning mid-aisle."""
        start_point = self.point(start_edge, start_along_m)
        goal_point = self.point(goal_edge, goal_along_m)
        if start_edge == goal_edge and goal_along_m - start_along_m >= DIRECT_ROUTE_MIN_M:
            return [start_point, goal_point], goal_along_m - start_along_m
        to_start_end_m = self.lengths[start_edge] - start_along_m
        distances = {}
        parents = {}
        frontier = []
        for successor in self.successors[start_edge]:
            heapq.heappush(frontier, (to_start_end_m + self.lengths[successor], successor, -1))
        while frontier:
            distance_m, edge_idx, parent = heapq.heappop(frontier)
            if edge_idx in distances:
                continue
            distances[edge_idx] = distance_m
            parents[edge_idx] = parent
            if edge_idx == goal_edge:
                break
            for successor in self.successors[edge_idx]:
                if successor not in distances:
                    heapq.heappush(frontier, (distance_m + self.lengths[successor], successor, edge_idx))
        if goal_edge not in distances:
            return None, None
        chain = [goal_edge]
        while parents[chain[-1]] != -1:
            chain.append(parents[chain[-1]])
        polyline = [start_point, self.nodes[self.edges[start_edge, 1]]]
        polyline += [self.nodes[self.edges[edge_idx, 1]] for edge_idx in reversed(chain[1:])]
        polyline.append(goal_point)
        return polyline, distances[goal_edge] - self.lengths[goal_edge] + goal_along_m


def sample_route_points(polyline):
    points = np.asarray(polyline, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_m = cumulative[-1]
    arcs = list(np.arange(SUBGOAL_SPACING_M, total_m - MIN_SUBGOAL_GAP_M, SUBGOAL_SPACING_M)) + [total_m]
    route = []
    for arc_m in arcs:
        segment_idx = min(int(np.searchsorted(cumulative, arc_m, side="right")) - 1, len(segment_lengths) - 1)
        fraction = (arc_m - cumulative[segment_idx]) / max(segment_lengths[segment_idx], 1e-9)
        x, y = points[segment_idx] + fraction * (points[segment_idx + 1] - points[segment_idx])
        route.append((x, y, arc_m))
    return np.array(route, dtype=np.float32)


def sample_spawns(lot, graph, static_edges, agent_count, rng):
    usable_m = np.maximum(graph.lengths - 2.0 * SPAWN_EDGE_MARGIN_M, 0.0)
    if usable_m.sum() <= 0.0:
        return []
    edge_probabilities = usable_m / usable_m.sum()
    spawns = []
    for _ in range(SPAWN_ATTEMPTS_PER_AGENT * agent_count):
        if len(spawns) == agent_count:
            break
        edge_idx = int(rng.choice(len(usable_m), p=edge_probabilities))
        along_m = SPAWN_EDGE_MARGIN_M + rng.random() * usable_m[edge_idx]
        direction = graph.directions[edge_idx]
        lateral_m = rng.uniform(-1.0, 1.0) * SPAWN_LATERAL_JITTER_M
        x, y = graph.point(edge_idx, along_m) + lateral_m * np.array([-direction[1], direction[0]])
        heading = float(np.arctan2(direction[1], direction[0]) + rng.uniform(-1.0, 1.0) * SPAWN_HEADING_JITTER_RAD)
        if any(np.hypot(x - spawn.pose[0], y - spawn.pose[1]) < SPAWN_MIN_GAP_M for spawn in spawns):
            continue
        if box_hits_segments((x, y, heading, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M), static_edges):
            continue
        spawns.append(Spawn(edge_idx, along_m, (float(x), float(y), heading)))
    return spawns


def prep_candidates(lot, graph, static_edges, slot_idx):
    """Aisle poses just before (forward-in) and past (reverse-in) the slot, per allowed travel direction.

    Each candidate also carries its approach start, APPROACH_LENGTH_M back along the aisle: agents hand off
    there and roll straight into the preparation pose, so a plan that begins in reverse starts from rest.
    """
    aisle = lot.aisles[int(lot.slot_aisles[slot_idx])]
    slot_center = lot.slots[slot_idx, :2].astype(np.float64)
    candidates = []
    for edge_idx, (edge_start, edge_end) in enumerate(graph.edges):
        if {edge_start, edge_end} != {aisle.start_node, aisle.end_node}:
            continue
        slot_along_m = float(np.dot(slot_center - graph.nodes[edge_start], graph.directions[edge_idx]))
        heading = float(np.arctan2(graph.directions[edge_idx][1], graph.directions[edge_idx][0]))
        for offset_m in PREP_OFFSETS_M:
            along_m = slot_along_m + offset_m
            approach_along_m = along_m - APPROACH_LENGTH_M
            if approach_along_m < PREP_EDGE_MARGIN_M or along_m > graph.lengths[edge_idx] - PREP_EDGE_MARGIN_M:
                continue
            approach_pose = (*(float(value) for value in graph.point(edge_idx, approach_along_m)), heading)
            if box_hits_segments((*approach_pose, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M), static_edges):
                continue
            prep_pose = (*(float(value) for value in graph.point(edge_idx, along_m)), heading)
            candidates.append((edge_idx, approach_along_m, approach_pose, prep_pose))
    return candidates


def prepend_approach(approach_pose, maneuver_plan):
    """Forward rear-axle samples from the approach start up to the maneuver's first point."""
    x, y, heading = approach_pose
    direction = np.array([np.cos(heading), np.sin(heading)])
    rear_start = np.array([x, y]) - 0.5 * WHEELBASE_M * direction
    sample_count = int(np.ceil(APPROACH_LENGTH_M / PLAN_SAMPLE_SPACING_M))
    arcs_m = np.arange(sample_count) * (APPROACH_LENGTH_M / sample_count)
    approach = np.zeros((sample_count, 5), dtype=np.float32)
    approach[:, :2] = rear_start[None, :] + arcs_m[:, None] * direction[None, :]
    approach[:, 2] = heading
    approach[:, 3] = 1.0
    approach[:, 4] = arcs_m
    maneuver = maneuver_plan.copy()
    maneuver[:, 4] += APPROACH_LENGTH_M
    maneuver[0, 3] = 1.0
    return np.concatenate([approach, maneuver])


def plan_assignment(lot, graph, static_edges, spawn, slot_idx):
    slot = lot.slots[slot_idx]
    best = None
    for edge_idx, approach_along_m, approach_pose, prep_pose in prep_candidates(lot, graph, static_edges, slot_idx):
        polyline, route_length_m = graph.route(spawn.edge_idx, spawn.along_m, edge_idx, approach_along_m)
        if polyline is None or route_length_m > MAX_ROUTE_LENGTH_M:
            continue
        for goal_heading in (float(slot[2]), float(wrap_angle(slot[2] + np.pi))):
            goal_pose = (float(slot[0]), float(slot[1]), goal_heading)
            plan, cost, status = binding.plan_maneuver(
                static_edges, lot.bounds, prep_pose, goal_pose, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M, WHEELBASE_M
            )
            if status != binding.PLAN_OK:
                continue
            total_cost = route_length_m + cost
            if best is None or total_cost < best["cost"]:
                best = {
                    "cost": total_cost,
                    "approach": approach_pose,
                    "goal": goal_pose,
                    "plan": prepend_approach(approach_pose, plan),
                    "polyline": polyline,
                }
    if best is None:
        return None
    return {
        "start": spawn.pose,
        "goal": best["goal"],
        "prep": best["approach"],
        "slot_length_m": float(slot[3]),
        "slot_width_m": float(slot[4]),
        "route": sample_route_points(best["polyline"]),
        "plan": best["plan"],
    }


def lot_static_edges(lot):
    edges = [lot.curbs.astype(np.float64)]
    if len(lot.obstacles):
        edges.append(box_edges(lot.obstacles))
    return np.concatenate(edges)


def generate_scene(lot, tier, rng):
    slot_count = len(lot.slots)
    occupied_count = int(round(TIER_OCCUPANCIES[tier] * slot_count))
    controlled_count = int(np.clip(round(occupied_count * CONTROLLED_SHARE), 1, binding.MAX_AGENTS_PER_SCENE))
    static_count = max(occupied_count - controlled_count, 0)
    slot_order = rng.permutation(slot_count)
    static_vehicles = np.array([place_static_vehicle(lot.slots[idx], rng) for idx in slot_order[:static_count]])
    static_vehicles = static_vehicles.reshape(-1, 5).astype(np.float32)
    edge_parts = [lot_static_edges(lot)] + ([box_edges(static_vehicles)] if static_count else [])
    static_edges = np.ascontiguousarray(np.concatenate(edge_parts), dtype=np.float32)
    graph = LaneGraph(lot)
    free_slots = [int(idx) for idx in slot_order[static_count:]]
    agents = []
    for spawn in sample_spawns(lot, graph, static_edges, controlled_count, rng):
        for _ in range(min(SLOT_ASSIGNMENT_ATTEMPTS, len(free_slots))):
            slot_idx = free_slots[int(rng.integers(len(free_slots)))]
            assignment = plan_assignment(lot, graph, static_edges, spawn, slot_idx)
            if assignment is None:
                continue
            agents.append(assignment)
            free_slots.remove(slot_idx)
            break
    return {"tier": tier, "static_vehicles": static_vehicles, "agents": agents}


def generate_scene_job(job):
    scene_idx, lot_idx, tier, layout_names, seed = job
    lot = build_lots(layout_names)[lot_idx]
    seeds = np.random.SeedSequence([seed, scene_idx]).spawn(SCENE_ATTEMPTS)
    for attempt_seed in seeds:
        scene = generate_scene(lot, tier, np.random.default_rng(attempt_seed))
        if scene["agents"]:
            scene["lot_idx"] = lot_idx
            return scene
    raise RuntimeError(f"scene {scene_idx} on lot {lot.name} found no feasible agent in {SCENE_ATTEMPTS} attempts")


def write_scene_bank(path, lots, scenes):
    segments, boxes, agent_rows, route_points, plan_points = [], [], [], [], []
    segment_count = box_count = route_count = plan_count = 0
    lot_rows = np.zeros(len(lots), dtype=LOT_DTYPE)
    for lot_idx, lot in enumerate(lots):
        lot_rows[lot_idx]["bounds"] = lot.bounds
        for key, array, store in (("curb", lot.curbs, segments), ("lane", lot.lanes, segments)):
            offset = segment_count
            store.append(array.astype(np.float32))
            segment_count += len(array)
            lot_rows[lot_idx][f"{key}_offset"], lot_rows[lot_idx][f"{key}_count"] = offset, len(array)
        for key, array in (("obstacle", lot.obstacles), ("slot", lot.slots)):
            lot_rows[lot_idx][f"{key}_offset"], lot_rows[lot_idx][f"{key}_count"] = box_count, len(array)
            boxes.append(array.astype(np.float32))
            box_count += len(array)
    scene_rows = np.zeros(len(scenes), dtype=SCENE_DTYPE)
    for scene_idx, scene in enumerate(scenes):
        row = scene_rows[scene_idx]
        row["lot_idx"], row["tier"] = scene["lot_idx"], scene["tier"]
        row["agent_offset"], row["agent_count"] = len(agent_rows), len(scene["agents"])
        row["static_vehicle_offset"], row["static_vehicle_count"] = box_count, len(scene["static_vehicles"])
        boxes.append(scene["static_vehicles"])
        box_count += len(scene["static_vehicles"])
        for agent in scene["agents"]:
            agent_row = np.zeros((), dtype=AGENT_DTYPE)
            for key in ("start", "goal", "prep", "slot_length_m", "slot_width_m"):
                agent_row[key] = agent[key]
            agent_row["route_offset"], agent_row["route_count"] = route_count, len(agent["route"])
            agent_row["plan_offset"], agent_row["plan_count"] = plan_count, len(agent["plan"])
            route_points.append(agent["route"])
            plan_points.append(agent["plan"].astype(np.float32))
            route_count += len(agent["route"])
            plan_count += len(agent["plan"])
            agent_rows.append(agent_row)
    header = struct.pack(
        HEADER_FORMAT,
        BANK_MAGIC,
        binding.SCENE_BANK_VERSION,
        len(lots),
        len(scenes),
        segment_count,
        box_count,
        len(agent_rows),
        route_count,
        plan_count,
        VEHICLE_LENGTH_M,
        VEHICLE_WIDTH_M,
        WHEELBASE_M,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as bank_file:
        bank_file.write(header)
        bank_file.write(lot_rows.tobytes())
        bank_file.write(scene_rows.tobytes())
        bank_file.write(np.concatenate(segments).astype("<f4").tobytes())
        bank_file.write(np.concatenate(boxes).astype("<f4").tobytes() if box_count else b"")
        bank_file.write(np.array(agent_rows, dtype=AGENT_DTYPE).tobytes())
        bank_file.write(np.concatenate(route_points).astype("<f4").tobytes())
        bank_file.write(np.concatenate(plan_points).astype("<f4").tobytes())


def read_scene_bank(path):
    with open(path, "rb") as bank_file:
        data = bank_file.read()
    header_size = struct.calcsize(HEADER_FORMAT)
    magic, version, *counts = struct.unpack(HEADER_FORMAT, data[:header_size])
    lot_count, scene_count, segment_count, box_count, agent_count, route_count, plan_count = counts[:7]
    vehicle_length_m, vehicle_width_m, wheelbase_m = counts[7:]
    if magic != BANK_MAGIC or version != binding.SCENE_BANK_VERSION:
        raise ValueError(f"{path} is not a version {binding.SCENE_BANK_VERSION} park scene bank")
    layout = [
        ("lots", LOT_DTYPE, lot_count),
        ("scenes", SCENE_DTYPE, scene_count),
        ("segments", np.dtype(("<f4", (4,))), segment_count),
        ("boxes", np.dtype(("<f4", (5,))), box_count),
        ("agents", AGENT_DTYPE, agent_count),
        ("route_points", np.dtype(("<f4", (3,))), route_count),
        ("plan_points", np.dtype(("<f4", (5,))), plan_count),
    ]
    bank = {"vehicle_length_m": vehicle_length_m, "vehicle_width_m": vehicle_width_m, "wheelbase_m": wheelbase_m}
    offset = header_size
    for name, dtype, count in layout:
        bank[name] = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += dtype.itemsize * count
    if offset != len(data):
        raise ValueError(f"{path} has {len(data) - offset} trailing bytes")
    return bank


def save_scene_preview(bank, scene_idx, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    scene = bank["scenes"][scene_idx]
    lot = bank["lots"][scene["lot_idx"]]
    figure, axis = plt.subplots(figsize=(10, 6))

    def draw_boxes(rows, **style):
        for row in rows:
            corners = box_edges(row)[:, :2]
            axis.add_patch(Polygon(corners, closed=True, **style))

    for start, count, color in (
        (lot["curb_offset"], lot["curb_count"], "black"),
        (lot["lane_offset"], lot["lane_count"], "0.8"),
    ):
        for x0, y0, x1, y1 in bank["segments"][start : start + count]:
            axis.plot([x0, x1], [y0, y1], color=color, linewidth=1)
    draw_boxes(bank["boxes"][lot["slot_offset"] : lot["slot_offset"] + lot["slot_count"]], fill=False, edgecolor="0.6")
    draw_boxes(bank["boxes"][lot["obstacle_offset"] : lot["obstacle_offset"] + lot["obstacle_count"]], color="brown")
    static_start = scene["static_vehicle_offset"]
    draw_boxes(bank["boxes"][static_start : static_start + scene["static_vehicle_count"]], color="slategray")
    for agent in bank["agents"][scene["agent_offset"] : scene["agent_offset"] + scene["agent_count"]]:
        start, goal = agent["start"], agent["goal"]
        draw_boxes([(*start, bank["vehicle_length_m"], bank["vehicle_width_m"])], color="tab:blue", alpha=0.8)
        draw_boxes(
            [(*goal, agent["slot_length_m"], agent["slot_width_m"])], fill=False, edgecolor="gold", linestyle="--"
        )
        route = bank["route_points"][agent["route_offset"] : agent["route_offset"] + agent["route_count"]]
        plan = bank["plan_points"][agent["plan_offset"] : agent["plan_offset"] + agent["plan_count"]]
        axis.plot([start[0], *route[:, 0]], [start[1], *route[:, 1]], color="tab:blue", linewidth=0.8, alpha=0.6)
        axis.plot(plan[:, 0], plan[:, 1], color="tab:orange", linewidth=1.2)
    axis.set_aspect("equal")
    axis.set_title(f"scene {scene_idx}: lot {scene['lot_idx']} tier {scene['tier']} agents {scene['agent_count']}")
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def generate_scene_bank(output_path, scene_count, seed, worker_count, layout_names):
    lots = build_lots(layout_names)
    tier_rng = np.random.default_rng(seed)
    tiers = tier_rng.choice(len(TIER_OCCUPANCIES), size=scene_count, p=TIER_PROBABILITIES)
    jobs = [
        (scene_idx, scene_idx % len(lots), int(tiers[scene_idx]), layout_names, seed)
        for scene_idx in range(scene_count)
    ]
    if worker_count > 1:
        with multiprocessing.get_context("spawn").Pool(worker_count) as pool:
            scenes = pool.map(generate_scene_job, jobs, chunksize=4)
    else:
        scenes = [generate_scene_job(job) for job in jobs]
    write_scene_bank(output_path, lots, scenes)
    return scenes


def main():
    parser = argparse.ArgumentParser(description="Generate a CoPark parking scene bank")
    parser.add_argument("--output", default="pufferlib/resources/park/park_scenes.bin")
    parser.add_argument("--scene-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--layouts", default=",".join(LAYOUTS), help="comma-separated layout names")
    parser.add_argument("--preview-dir", default=None, help="write one PNG per lot for visual inspection")
    args = parser.parse_args()
    layout_names = args.layouts.split(",")
    scenes = generate_scene_bank(args.output, args.scene_count, args.seed, args.workers, layout_names)
    info = binding.scene_bank_info(args.output)
    agent_total = sum(len(scene["agents"]) for scene in scenes)
    print(f"wrote {args.output}: {info['scene_count']} scenes, {agent_total} agents, tiers {info['tier_scene_counts']}")
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)
        bank = read_scene_bank(args.output)
        for lot_idx, name in enumerate(layout_names):
            save_scene_preview(bank, lot_idx, os.path.join(args.preview_dir, f"{name}.png"))


if __name__ == "__main__":
    main()
