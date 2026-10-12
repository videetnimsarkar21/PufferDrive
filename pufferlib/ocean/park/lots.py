"""Parametric parking-lot layouts in the clean-map form used by the CoPark scene generator.

Each lot declares curbs (lot outline), fixed obstacle boxes, slot boxes, and a directed lane graph
along aisle centerlines. Slot headings point into the slot; boxes are (center_x, center_y, heading,
length, width) in meters and radians.
"""

from dataclasses import dataclass

import numpy as np

SLOT_WIDTH_M = 2.5
SLOT_LENGTH_M = 5.0
PARALLEL_SLOT_LENGTH_M = 6.0
TWO_WAY_AISLE_WIDTH_M = 6.0
ONE_WAY_AISLE_WIDTH_M = 4.5
COMPACT_AISLE_WIDTH_M = 5.5
COMPACT_SLOT_WIDTH_M = 2.4
PILLAR_SIZE_M = 0.6
PILLAR_DEPTH_M = 1.0
CURB_MARGIN_M = 0.4
TOKEN_MAX_LENGTH_M = 4.0
NODE_MERGE_TOLERANCE_M = 1e-3


@dataclass
class Lot:
    name: str
    bounds: tuple
    curbs: np.ndarray
    lanes: np.ndarray
    obstacles: np.ndarray
    slots: np.ndarray
    slot_aisles: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    aisles: list


@dataclass
class Aisle:
    start_node: int
    end_node: int
    two_way: bool
    half_width_m: float


def chop_segment(start, end, max_length_m=TOKEN_MAX_LENGTH_M):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    piece_count = max(1, int(np.ceil(np.linalg.norm(end - start) / max_length_m)))
    fractions = np.linspace(0.0, 1.0, piece_count + 1)
    points = start[None, :] + fractions[:, None] * (end - start)[None, :]
    return np.concatenate([points[:-1], points[1:]], axis=1)


class LotBuilder:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.aisles = []
        self.slots = []
        self.slot_aisles = []
        self.obstacles = []

    def node(self, x, y):
        for node_idx, (node_x, node_y) in enumerate(self.nodes):
            if abs(node_x - x) < NODE_MERGE_TOLERANCE_M and abs(node_y - y) < NODE_MERGE_TOLERANCE_M:
                return node_idx
        self.nodes.append((float(x), float(y)))
        return len(self.nodes) - 1

    def aisle(self, start, end, width_m=TWO_WAY_AISLE_WIDTH_M, two_way=True):
        aisle = Aisle(self.node(*start), self.node(*end), two_way, 0.5 * width_m)
        self.aisles.append(aisle)
        return len(self.aisles) - 1

    def aisle_frame(self, aisle_idx):
        aisle = self.aisles[aisle_idx]
        start = np.array(self.nodes[aisle.start_node])
        end = np.array(self.nodes[aisle.end_node])
        direction = (end - start) / np.linalg.norm(end - start)
        left_normal = np.array([-direction[1], direction[0]])
        return start, direction, left_normal, aisle.half_width_m

    def slot_row(
        self,
        aisle_idx,
        side,
        start_m,
        count,
        angle_deg=90.0,
        slot_width_m=SLOT_WIDTH_M,
        slot_length_m=SLOT_LENGTH_M,
        group_size=0,
        group_gap_m=0.0,
    ):
        """Slots opening onto one side (+1 left, -1 right) of an aisle, starting start_m from its start node."""
        start, direction, left_normal, half_width_m = self.aisle_frame(aisle_idx)
        outward = side * left_normal
        angle_rad = np.deg2rad(angle_deg)
        into_slot = np.cos(angle_rad) * direction + np.sin(angle_rad) * outward
        pitch_m = slot_width_m / np.sin(angle_rad)
        heading = float(np.arctan2(into_slot[1], into_slot[0]))
        along_m = start_m
        for slot_idx in range(count):
            if group_size and slot_idx > 0 and slot_idx % group_size == 0:
                pillar_center = start + direction * (along_m + 0.5 * group_gap_m)
                pillar_center = pillar_center + outward * (half_width_m + PILLAR_DEPTH_M)
                self.obstacles.append((*pillar_center, heading, PILLAR_SIZE_M, PILLAR_SIZE_M))
                along_m += group_gap_m
            entry = start + direction * (along_m + 0.5 * pitch_m) + outward * half_width_m
            center = entry + into_slot * 0.5 * slot_length_m
            self.slots.append((*center, heading, slot_length_m, slot_width_m))
            self.slot_aisles.append(aisle_idx)
            along_m += pitch_m

    def parallel_row(self, aisle_idx, side, start_m, count, slot_length_m=PARALLEL_SLOT_LENGTH_M):
        start, direction, left_normal, half_width_m = self.aisle_frame(aisle_idx)
        outward = side * left_normal
        heading = float(np.arctan2(direction[1], direction[0]))
        for slot_idx in range(count):
            center = start + direction * (start_m + (slot_idx + 0.5) * slot_length_m)
            center = center + outward * (half_width_m + 0.5 * SLOT_WIDTH_M)
            self.slots.append((*center, heading, slot_length_m, SLOT_WIDTH_M))
            self.slot_aisles.append(aisle_idx)

    def obstacle(self, center_x, center_y, heading, length_m, width_m):
        self.obstacles.append((center_x, center_y, heading, length_m, width_m))

    def build(self):
        points = [np.array(node) for node in self.nodes]
        for aisle in self.aisles:
            for node_idx in (aisle.start_node, aisle.end_node):
                node = np.array(self.nodes[node_idx])
                points.extend([node - aisle.half_width_m, node + aisle.half_width_m])
        for center_x, center_y, heading, length_m, width_m in self.slots + self.obstacles:
            radius_m = 0.5 * np.hypot(length_m, width_m)
            points.extend(
                [
                    np.array([center_x - radius_m, center_y - radius_m]),
                    np.array([center_x + radius_m, center_y + radius_m]),
                ]
            )
        points = np.stack(points)
        min_x, min_y = points.min(axis=0) - CURB_MARGIN_M
        max_x, max_y = points.max(axis=0) + CURB_MARGIN_M
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        curbs = np.concatenate([chop_segment(corners[i], corners[(i + 1) % 4]) for i in range(4)])
        lanes = np.concatenate(
            [chop_segment(self.nodes[aisle.start_node], self.nodes[aisle.end_node]) for aisle in self.aisles]
        )
        edges = []
        for aisle in self.aisles:
            edges.append((aisle.start_node, aisle.end_node))
            if aisle.two_way:
                edges.append((aisle.end_node, aisle.start_node))
        return Lot(
            name=self.name,
            bounds=(float(min_x), float(min_y), float(max_x), float(max_y)),
            curbs=curbs.astype(np.float32),
            lanes=lanes.astype(np.float32),
            obstacles=np.array(self.obstacles, dtype=np.float32).reshape(-1, 5),
            slots=np.array(self.slots, dtype=np.float32).reshape(-1, 5),
            slot_aisles=np.array(self.slot_aisles, dtype=np.int32),
            nodes=np.array(self.nodes, dtype=np.float64),
            edges=np.array(edges, dtype=np.int32),
            aisles=list(self.aisles),
        )


def ring_double_row():
    builder = LotBuilder("ring_double_row")
    width, height, row_count = 36.0, 16.0, 12
    bottom = builder.aisle((0, 0), (width, 0))
    builder.aisle((width, 0), (width, height))
    top = builder.aisle((width, height), (0, height))
    builder.aisle((0, height), (0, 0))
    edge_offset = 0.5 * TWO_WAY_AISLE_WIDTH_M
    builder.slot_row(bottom, +1, edge_offset, row_count)
    builder.slot_row(bottom, -1, edge_offset, row_count)
    builder.slot_row(top, +1, edge_offset, row_count)
    builder.slot_row(top, -1, edge_offset, row_count)
    return builder.build()


def two_islands():
    builder = LotBuilder("two_islands")
    island_pitch, height, row_count = 26.0, 16.0, 8
    edge_offset = 0.5 * TWO_WAY_AISLE_WIDTH_M
    for column in range(2):
        x0, x1 = column * island_pitch, (column + 1) * island_pitch
        bottom = builder.aisle((x0, 0), (x1, 0))
        top = builder.aisle((x1, height), (x0, height))
        for aisle in (bottom, top):
            builder.slot_row(aisle, +1, edge_offset, row_count)
            builder.slot_row(aisle, -1, edge_offset, row_count)
    for column in range(3):
        builder.aisle((column * island_pitch, 0), (column * island_pitch, height))
    return builder.build()


def angled_one_way():
    builder = LotBuilder("angled_one_way")
    width, height, row_count = 40.0, 18.0, 11
    edge_offset = 0.5 * ONE_WAY_AISLE_WIDTH_M + 1.0
    bottom = builder.aisle((0, 0), (width, 0), ONE_WAY_AISLE_WIDTH_M, two_way=False)
    builder.aisle((width, 0), (width, height), ONE_WAY_AISLE_WIDTH_M, two_way=False)
    top = builder.aisle((width, height), (0, height), ONE_WAY_AISLE_WIDTH_M, two_way=False)
    builder.aisle((0, height), (0, 0), ONE_WAY_AISLE_WIDTH_M, two_way=False)
    for aisle in (bottom, top):
        builder.slot_row(aisle, +1, edge_offset, row_count, angle_deg=60.0)
        builder.slot_row(aisle, -1, edge_offset, row_count, angle_deg=60.0)
    return builder.build()


def parallel_ring():
    builder = LotBuilder("parallel_ring")
    width, height, row_count = 52.0, 14.0, 7
    edge_offset = 0.5 * TWO_WAY_AISLE_WIDTH_M + 2.0
    bottom = builder.aisle((0, 0), (width, 0))
    builder.aisle((width, 0), (width, height))
    top = builder.aisle((width, height), (0, height))
    builder.aisle((0, height), (0, 0))
    for aisle in (bottom, top):
        builder.parallel_row(aisle, +1, edge_offset, row_count)
        builder.parallel_row(aisle, -1, edge_offset, row_count)
    return builder.build()


def garage_pillars():
    builder = LotBuilder("garage_pillars")
    width, row_pitch, row_count, group_size = 38.0, 17.0, 12, 3
    edge_offset = 0.5 * TWO_WAY_AISLE_WIDTH_M
    horizontal = [builder.aisle((0, level * row_pitch), (width, level * row_pitch)) for level in range(3)]
    for level in range(2):
        builder.aisle((0, level * row_pitch), (0, (level + 1) * row_pitch))
        builder.aisle((width, level * row_pitch), (width, (level + 1) * row_pitch))
    sides = [(-1, +1), (-1, +1), (-1, +1)]
    for aisle, aisle_sides in zip(horizontal, sides):
        for side in aisle_sides:
            builder.slot_row(aisle, side, edge_offset, row_count, group_size=group_size, group_gap_m=PILLAR_SIZE_M)
    return builder.build()


def compact_ladder():
    builder = LotBuilder("compact_ladder")
    island_pitch, height, row_count = 27.0, 15.5, 8
    planter_length_m = 2.0
    edge_offset = 0.5 * COMPACT_AISLE_WIDTH_M + planter_length_m
    for column in range(2):
        x0, x1 = column * island_pitch, (column + 1) * island_pitch
        bottom = builder.aisle((x0, 0), (x1, 0), COMPACT_AISLE_WIDTH_M)
        top = builder.aisle((x1, height), (x0, height), COMPACT_AISLE_WIDTH_M)
        for aisle in (bottom, top):
            builder.slot_row(aisle, +1, edge_offset, row_count, slot_width_m=COMPACT_SLOT_WIDTH_M)
            builder.slot_row(aisle, -1, edge_offset, row_count, slot_width_m=COMPACT_SLOT_WIDTH_M)
        planter_x = x0 + 0.5 * COMPACT_AISLE_WIDTH_M + 0.5 * planter_length_m + 0.2
        builder.obstacle(planter_x, 0.5 * height, 0.0, planter_length_m - 0.4, 1.5)
    for column in range(3):
        builder.aisle((column * island_pitch, 0), (column * island_pitch, height), COMPACT_AISLE_WIDTH_M)
    return builder.build()


LAYOUTS = {
    "ring_double_row": ring_double_row,
    "two_islands": two_islands,
    "angled_one_way": angled_one_way,
    "parallel_ring": parallel_ring,
    "garage_pillars": garage_pillars,
    "compact_ladder": compact_ladder,
}


def build_lots(names=None):
    names = list(LAYOUTS) if names is None else names
    return [LAYOUTS[name]() for name in names]
