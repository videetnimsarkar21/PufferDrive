"""Top-down GIF replay of one park episode.

    python -m pufferlib.ocean.park.render --scene-bank pufferlib/resources/park/park_scenes.bin \
        --output experiments/park/teacher.gif

Agents are colored by phase (blue navigate, orange maneuver, purple refine) and turn green once parked; the dashed
outline is each agent's assigned slot and the thin orange line its planned maneuver.
"""

import argparse
import os

import numpy as np

from pufferlib.ocean.park import binding
from pufferlib.ocean.park.park import Park
from pufferlib.ocean.park.scenes import box_edges, read_scene_bank

STATE_X, STATE_Y, STATE_HEADING = 0, 1, 2
STATE_STATUS, STATE_PHASE, STATE_OUTCOME = 5, 6, 7
PHASE_COLORS = ("tab:blue", "tab:orange", "tab:purple")
PARKED_COLOR = "tab:green"
FRAME_DPI = 80
FRAME_SIZE_INCHES = (10, 7)
OUTCOME_NAMES = {1: "parked", 2: "hit vehicle", 3: "hit static", 4: "off-road", 5: "refine timeout", 6: "horizon"}


def record_episode(env, select_actions, max_steps):
    """Steps every env with select_actions(env) and keeps env 0's agent states until its episode ends."""
    agent_slots = env.max_agents_per_env
    scene_idx = env.scene_indices()[0]
    frames = [env.agent_states()[:agent_slots].copy()]
    for _ in range(max_steps):
        env.step(select_actions(env))
        states = env.agent_states()[:agent_slots].copy()
        frames.append(states)
        running = (states[:, STATE_STATUS] == binding.STATUS_ACTIVE) & (states[:, STATE_OUTCOME] == 0)
        if not np.any(running):
            break
    return scene_idx, frames


def draw_boxes(axis, polygon_cls, boxes, **style):
    for box in np.asarray(boxes).reshape(-1, 5):
        axis.add_patch(polygon_cls(box_edges(box)[:, :2], closed=True, **style))


def draw_lot(axis, polygon_cls, bank, scene):
    lot = bank["lots"][scene["lot_idx"]]
    for start, count, color, width in (
        (lot["curb_offset"], lot["curb_count"], "black", 1.5),
        (lot["lane_offset"], lot["lane_count"], "0.85", 1.0),
    ):
        for x0, y0, x1, y1 in bank["segments"][start : start + count]:
            axis.plot([x0, x1], [y0, y1], color=color, linewidth=width, zorder=1)
    draw_boxes(
        axis,
        polygon_cls,
        bank["boxes"][lot["slot_offset"] : lot["slot_offset"] + lot["slot_count"]],
        fill=False,
        edgecolor="0.75",
        linewidth=0.8,
        zorder=1,
    )
    draw_boxes(
        axis,
        polygon_cls,
        bank["boxes"][lot["obstacle_offset"] : lot["obstacle_offset"] + lot["obstacle_count"]],
        color="saddlebrown",
        zorder=2,
    )
    static_start = scene["static_vehicle_offset"]
    draw_boxes(
        axis,
        polygon_cls,
        bank["boxes"][static_start : static_start + scene["static_vehicle_count"]],
        color="slategray",
        zorder=2,
    )
    bounds = lot["bounds"]
    axis.set_xlim(bounds[0] - 1.0, bounds[2] + 1.0)
    axis.set_ylim(bounds[1] - 1.0, bounds[3] + 1.0)


def draw_agents(axis, polygon_cls, bank, scene, states, vehicle_size):
    scene_agents = bank["agents"][scene["agent_offset"] : scene["agent_offset"] + scene["agent_count"]]
    for agent_idx, agent in enumerate(scene_agents):
        state = states[agent_idx]
        status = int(state[STATE_STATUS])
        if status not in (binding.STATUS_ACTIVE, binding.STATUS_PARKED):
            continue
        color = PARKED_COLOR if status == binding.STATUS_PARKED else PHASE_COLORS[int(state[STATE_PHASE])]
        if status == binding.STATUS_ACTIVE:
            draw_boxes(
                axis,
                polygon_cls,
                [(*agent["goal"], agent["slot_length_m"], agent["slot_width_m"])],
                fill=False,
                edgecolor=color,
                linestyle="--",
                linewidth=1.2,
                zorder=3,
            )
            plan = bank["plan_points"][agent["plan_offset"] : agent["plan_offset"] + agent["plan_count"]]
            axis.plot(plan[:, 0], plan[:, 1], color="tab:orange", linewidth=0.6, alpha=0.5, zorder=3)
        pose = (state[STATE_X], state[STATE_Y], state[STATE_HEADING], *vehicle_size)
        draw_boxes(axis, polygon_cls, [pose], color=color, zorder=4)
        nose_x = state[STATE_X] + 0.5 * vehicle_size[0] * np.cos(state[STATE_HEADING])
        nose_y = state[STATE_Y] + 0.5 * vehicle_size[0] * np.sin(state[STATE_HEADING])
        axis.plot([state[STATE_X], nose_x], [state[STATE_Y], nose_y], color="white", linewidth=1.0, zorder=5)


def outcome_summary(final_states, agent_count):
    outcomes = final_states[:agent_count, STATE_OUTCOME].astype(int)
    parts = [f"{np.sum(outcomes == code)} {name}" for code, name in OUTCOME_NAMES.items() if np.any(outcomes == code)]
    return ", ".join(parts) if parts else "in progress"


def save_gif(bank, scene_idx, frames, output_path, dt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image

    scene = bank["scenes"][scene_idx]
    vehicle_size = (bank["vehicle_length_m"], bank["vehicle_width_m"])
    figure, axis = plt.subplots(figsize=FRAME_SIZE_INCHES, dpi=FRAME_DPI)
    images = []
    for frame_idx, states in enumerate(frames):
        axis.clear()
        draw_lot(axis, Polygon, bank, scene)
        draw_agents(axis, Polygon, bank, scene, states, vehicle_size)
        axis.set_aspect("equal")
        axis.set_title(
            f"scene {scene_idx}  tier {scene['tier']}  t={frame_idx * dt:4.1f}s  "
            f"{outcome_summary(states, scene['agent_count'])}"
        )
        figure.canvas.draw()
        images.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]))
    plt.close(figure)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=int(1000 * dt), loop=0)


def main():
    parser = argparse.ArgumentParser(description="Render one park episode as a GIF")
    parser.add_argument("--scene-bank", default="pufferlib/resources/park/park_scenes.bin")
    parser.add_argument("--output", default="experiments/park/teacher.gif")
    parser.add_argument("--controller", choices=["teacher"], default="teacher")
    parser.add_argument("--max-controlled", type=int, default=32, help="agents per scene to control (1 = solo)")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = Park(
        num_agents=binding.MAX_AGENTS_PER_SCENE,
        max_agents_per_env=binding.MAX_AGENTS_PER_SCENE,
        scene_bank_path=args.scene_bank,
        max_controlled_per_scene=args.max_controlled,
        episode_steps=args.max_steps,
        stagger_first_episode=False,
    )
    env.reset(seed=args.seed)
    scene_idx, frames = record_episode(env, lambda park_env: park_env.teacher_actions(), args.max_steps)
    env.close()
    bank = read_scene_bank(args.scene_bank)
    save_gif(bank, scene_idx, frames, args.output, dt=0.1)
    agent_count = min(bank["scenes"][scene_idx]["agent_count"], args.max_controlled)
    print(f"wrote {args.output}: scene {scene_idx}, {len(frames)} frames, {outcome_summary(frames[-1], agent_count)}")


if __name__ == "__main__":
    main()
