"""Household tasks: loading a dishwasher and folding a towel.

Both reuse the motion primitives from the planner (smooth joint moves,
straight in-plane tool moves, 3D tool paths) and add task logic on top.

Dishwasher
----------
Plates stand in a holder on the counter, each in a vertical plane that
passes through the robot's waist axis, so the arm can reach straight down,
pinch the top rim and lift the plate out without twisting it. They go into
the slots of the bottom rack the same way.

Cups stand upright on the counter. The arm comes in from the side with the
gripper level, pinches the cup, lifts it, then turns its wrist (through
pointing straight down) to flip the cup mouth-down (cups go in upside down so they don't
fill with water) and sets it on the top rack.

Towel
-----
The towel is folded in half (grab the far edge, swing it over in an arc and
lay it down on the near half), then in half again across the first fold, and
the folded towel goes into a laundry basket. The cloth shape itself is drawn
by the front end from the fold lines and the gripper path recorded here.
"""
from __future__ import annotations

import math

import numpy as np

from .kinematics import HOME_Q4, fk, fk_world
from .planner import CREEP_SPEED, FPS, QUINTIC_PEAK, TOOL_DOWN, TrajectoryBuilder, Unreachable, pose_at, roll_to_close_along

PLATE_R = 0.055
PLATE_T = 0.01
CUP_R = 0.026
CUP_H = 0.075
TOWEL_HALF = 0.13
CLOTH_EPS = 0.004  # thickness of one layer of towel

# Gripper opening (1 = fully open, 0 = closed on a 40 mm block). Thinner
# things close further.
GRIP_PLATE = -1.8
GRIP_CUP = 0.6
GRIP_CLOTH = -2.2

CARRY_Y = 0.22  # tool height while carrying dishes between counter and racks

# What the force sensors read (N): object weights and squeeze forces.
WEIGHT_PLATE, FORCE_PLATE = 3.4, 15.0
WEIGHT_CUP, FORCE_CUP = 2.2, 12.0
WEIGHT_TOWEL_EDGE, WEIGHT_TOWEL = 0.8, 2.9
FORCE_CLOTH = 6.0
PLATE_HOLDER_H = 0.015
PLATE_RACK_H = 0.012
CUP_RACK_H = 0.07
# Cloth is pinched with the gripper tilted, not stabbing straight down: the
# lower finger slides under the edge at a shallow angle, like a hand does.
PINCH_TILT = math.radians(25)
PINCH_PHI = TOOL_DOWN + PINCH_TILT  # tool leans out, away from the base
RIM_GRIP = 0.012  # fingers overlap the plate rim by this much

TASKS = ("dishes", "laundry")


def _polar(r: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)


# ---- Scenes ---------------------------------------------------------------

def dishes_scene() -> dict:
    plate_colors = ["#f8fafc", "#e0f2fe", "#fef3c7"]
    cup_colors = ["#f97316", "#14b8a6", "#a855f7"]
    objects = []
    holder_slots = []
    for i, deg in enumerate((-14, 0, 14)):
        x, z = _polar(0.40, deg)
        holder_slots.append({"x": round(x, 4), "z": round(z, 4), "yaw": round(math.radians(deg), 5)})
        objects.append({
            "id": f"plate-{i + 1}", "kind": "plate", "color": plate_colors[i],
            "x": round(x, 4), "y": PLATE_HOLDER_H + PLATE_R, "z": round(z, 4),
            "yaw": round(math.radians(deg), 5), "tilt": 0.0, "size": PLATE_R, "binId": None,
        })
    for i, deg in enumerate((42, -42, 64)):
        x, z = _polar(0.37, deg)  # far enough out that the side approach has elbow room
        objects.append({
            "id": f"cup-{i + 1}", "kind": "cup", "color": cup_colors[i],
            "x": round(x, 4), "y": CUP_H / 2, "z": round(z, 4),
            "yaw": round(math.radians(deg), 5), "tilt": 0.0, "size": CUP_R, "binId": None,
        })
    plate_slots = []
    for deg in (180 - 12, 180, 180 + 12):
        x, z = _polar(0.50, deg)
        plate_slots.append({"x": round(x, 4), "z": round(z, 4), "yaw": round(math.radians(deg), 5)})
    cup_slots = []
    for deg in (180 - 22, 180, 180 + 22):
        x, z = _polar(0.25, deg)
        cup_slots.append({"x": round(x, 4), "z": round(z, 4), "yaw": round(math.radians(deg), 5)})
    fixtures = [
        {"id": "plate-holder", "kind": "plate_holder", "x": 0.40, "z": 0.0, "width": 0.14, "depth": 0.26,
         "height": PLATE_HOLDER_H, "slots": holder_slots, "label": "Plate holder"},
        {"id": "plate-rack", "kind": "plate_rack", "x": -0.50, "z": 0.0, "width": 0.24, "depth": 0.32,
         "height": PLATE_RACK_H, "slots": plate_slots, "label": "Bottom rack (plates)"},
        {"id": "cup-rack", "kind": "cup_rack", "x": -0.25, "z": 0.0, "width": 0.14, "depth": 0.30,
         "height": CUP_RACK_H, "slots": cup_slots, "label": "Top rack (cups)"},
    ]
    return {"objects": objects, "fixtures": fixtures}


def laundry_scene() -> dict:
    objects = [{
        "id": "towel", "kind": "towel", "color": "#0ea5e9", "x": 0.40, "y": 0.0, "z": 0.0,
        "yaw": 0.0, "tilt": 0.0, "size": TOWEL_HALF, "binId": None,
    }]
    fixtures = [
        {"id": "basket", "kind": "basket", "x": -0.42, "z": 0.0, "width": 0.26, "depth": 0.26,
         "height": 0.08, "label": "Laundry basket"},
    ]
    return {"objects": objects, "fixtures": fixtures}


# ---- Helpers ----------------------------------------------------------------

def _world(u: float, y: float, waist: float) -> np.ndarray:
    return np.array([u * math.cos(waist), y, u * math.sin(waist)])


def _require(q, what: str):
    if q is None:
        raise Unreachable(what)
    return q


def _free_slot(fixture: dict, objects: list[dict]):
    for i, slot in enumerate(fixture["slots"]):
        if not any(o.get("binId") == fixture["id"] and math.hypot(o["x"] - slot["x"], o["z"] - slot["z"]) < 0.01
                   for o in objects):
            return i, slot
    return None, None


def _result(b: TrajectoryBuilder, done: int, skipped: int, extra: dict | None = None) -> dict:
    out = {
        "fps": FPS,
        "duration": round(b.t, 3),
        "frames": b.frames,
        "events": b.events,
        "timeline": b.timeline,
        "placements": [],
        "warnings": [] if b.min_clearance >= 0 else ["A joint dipped below the table top during the run."],
        "sensors": b.sensors,
        "perception": b.perception,
        "stats": {
            "sorted": done,
            "skipped": skipped,
            "slips": 0,
            "drops": 0,
            "recovered": 0,
            "durationS": round(b.t, 2),
            "toolPathM": round(b.path_length, 3),
            "maxJointSpeedDegS": round(math.degrees(b.max_joint_speed), 1),
            "minClearanceMm": round(b.min_clearance * 1000, 1),
            "frames": len(b.frames),
        },
    }
    if extra:
        out.update(extra)
    return out


# ---- Dishwasher ---------------------------------------------------------------

def _move_plate(b: TrajectoryBuilder, plate: dict, slot: dict, rack: dict):
    L = b.lengths
    center = np.array([plate["x"], plate["y"], plate["z"]])
    rim = center + np.array([0.0, PLATE_R - RIM_GRIP, 0.0])  # where the fingertips pinch
    size = [2 * PLATE_R, 2 * PLATE_R, PLATE_T + 0.002]
    b.event("Pick a plate from the holder (pinch the top rim)", "pick")
    est = b.look(plate["id"], "plate", center, size, yaw=plate.get("yaw", 0.0), height=0.17)
    est_rim = est + np.array([0.0, PLATE_R - RIM_GRIP, 0.0])
    b.joint_move(_require(pose_at(est_rim + [0, 0.1, 0], TOOL_DOWN, L, prefer=b.q), "above the plate"))
    b.approach_down(rim[1] + 0.03, slow=0.0)
    b.align(plate["id"], "plate", center, est, size, yaw=plate.get("yaw", 0.0))
    b.approach_down(rim[1])
    b.squeeze(GRIP_PLATE, FORCE_PLATE)
    b.grab(plate["id"], center, weight=WEIGHT_PLATE)
    b.lift_off(CARRY_Y)  # lift straight up out of the holder

    b.event("Slide the plate into the bottom rack", "place")
    place_center = np.array([slot["x"], rack["height"] + PLATE_R, slot["z"]])
    place_rim = place_center + np.array([0.0, PLATE_R - RIM_GRIP, 0.0])
    b.joint_move(_require(pose_at([place_rim[0], CARRY_Y, place_rim[2]], TOOL_DOWN, L, prefer=b.q), "above the rack"))
    b.touch_down(place_rim[1], WEIGHT_PLATE)
    b.let_go()
    b.put(plate["id"], place_center, yaw=slot["yaw"], tilt=0.0, mode="bin", binId=rack["id"])
    b.lift_off(CARRY_Y)
    b.verify("plate", "its rack slot")
    plate.update({"x": slot["x"], "y": float(place_center[1]), "z": slot["z"], "binId": rack["id"]})


def _move_cup(b: TrajectoryBuilder, cup: dict, slot: dict, rack: dict):
    L = b.lengths
    w = math.atan2(cup["z"], cup["x"])
    u = math.hypot(cup["x"], cup["z"])
    mid = CUP_H / 2
    center = np.array([cup["x"], mid, cup["z"]])
    size = [2 * CUP_R, CUP_H, 2 * CUP_R]
    b.event("Pick a cup (gripper level, pinch it from the side)", "pick")
    est = b.look(cup["id"], "cup", center, size, yaw=w)
    we, ue = math.atan2(est[2], est[0]), math.hypot(est[0], est[2])
    # Come in low from the base side, level with the cup's middle.
    b.joint_move(_require(pose_at(_world(ue - 0.1, 0.15, we), 0.0, L, prefer=b.q), "beside the cup"))
    b.plane_move(ue - 0.1, mid, 0.0)
    b.align(cup["id"], "cup", center, est, size, yaw=w)
    # Slide in slowly until the fingers straddle the cup.
    b.path3d([b.tool(), center], phi=0.0, speed=CREEP_SPEED, min_dur=0.5)
    b.squeeze(GRIP_CUP, FORCE_CUP)
    b.grab(cup["id"], center, tilt=0.0, weight=WEIGHT_CUP)
    b.lift_off(CARRY_Y)
    b.event("Flip the cup upside down (wrist turn)", "info")
    # Turn through pointing straight down: pointing straight up would need
    # the wrist past its 160 degree limit.
    b.plane_move(u, CARRY_Y, -math.pi, phi_from=0.0)

    b.event("Set the cup mouth-down on the top rack", "place")
    us = math.hypot(slot["x"], slot["z"])
    place_y = rack["height"] + CUP_H / 2
    b.joint_move(_require(pose_at([slot["x"], CARRY_Y, slot["z"]], -math.pi, L, prefer=b.q), "above the top rack"))
    b.touch_down(place_y, WEIGHT_CUP)
    b.let_go()
    b.put(cup["id"], [slot["x"], place_y, slot["z"]], yaw=slot["yaw"], tilt=-math.pi, mode="bin", binId=rack["id"])
    # Back out away from the base (the fingers slide off the cup's sides), then up.
    b.plane_move(us + 0.09, place_y, -math.pi)
    b.plane_move(us + 0.09, CARRY_Y, -math.pi)
    b.verify("cup", "the top rack, mouth down")
    cup.update({"x": slot["x"], "y": place_y, "z": slot["z"], "binId": rack["id"], "tilt": -math.pi})


def plan_dishes(lengths, q0, objects: list[dict], fixtures: list[dict], park_q=None, tuck_u: float | None = None) -> dict:
    """Load the dishwasher. With park_q the arm ends parked there (two-arm
    mode) instead of going home."""
    b = TrajectoryBuilder(lengths, np.asarray(q0, dtype=float))
    world = [dict(o) for o in objects]
    racks = {f["kind"]: f for f in fixtures}
    plate_rack, cup_rack = racks.get("plate_rack"), racks.get("cup_rack")
    done, skipped = 0, 0
    # Bottom rack first (plates), then the top rack (cups), like a person would.
    todo = [o for o in world if o["kind"] == "plate" and not o.get("binId")] + \
           [o for o in world if o["kind"] == "cup" and not o.get("binId")]
    if not todo:
        b.event("The dishwasher is already loaded. Press Reset scene to unload it.", "warn")
    for obj in todo:
        rack = plate_rack if obj["kind"] == "plate" else cup_rack
        name = obj["kind"]
        if rack is None:
            b.event(f"Skipped a {name}: no rack for it.", "warn")
            skipped += 1
            continue
        _, slot = _free_slot(rack, world)
        if slot is None:
            b.event(f"Skipped a {name}: the {rack['label'].lower()} is full.", "warn")
            skipped += 1
            continue
        saved = b.snapshot()
        try:
            (_move_plate if name == "plate" else _move_cup)(b, obj, slot, rack)
        except Unreachable as exc:
            b.restore(saved)
            b.event(f"Skipped a {name}: path leaves the workspace near {exc}.", "warn")
            skipped += 1
            continue
        done += 1
    if park_q is None:
        b.event("Return home", "info")
        b.joint_move(HOME_Q4, roll_to=0.0)
        b.pause(0.3)
    else:
        b.event("Park on its own side", "info")
        b.transfer(park_q, tuck_u, roll_to=0.0)
    b.event(f"Done: {done} loaded ({sum(o['kind'] == 'plate' and bool(o.get('binId')) for o in world)} plates, "
            f"{sum(o['kind'] == 'cup' and bool(o.get('binId')) for o in world)} cups), {skipped} skipped.", "done")
    return _result(b, done, skipped)


# ---- Towel folding ------------------------------------------------------------

def _fold(b: TrajectoryBuilder, F: np.ndarray, n: np.ndarray, half: float, base_y: float, label: str) -> dict:
    """Grab the edge midpoint on the +n side, swing it over the fold line in
    an arc, lay it down on the -n side. Returns the fold record for the
    front end's cloth model."""
    L = b.lengths
    edge = F + n * half
    grip_y = base_y + 0.004
    b.event(f"Grab the {label} edge of the towel", "pick")
    q_above = _require(pose_at([edge[0], base_y + 0.08, edge[1]], PINCH_PHI, L, prefer=b.q), "above the edge")
    # Turn the wrist so the fingers close straight across the edge (along n):
    # one finger slides under the hem, the other pinches from above.
    roll = roll_to_close_along(math.atan2(n[1], n[0]), float(q_above[0]))
    b.event(f"Wrist rolls {math.degrees(roll):+.0f}° so the fingers close across the {label} edge", "info")
    b.joint_move(q_above, roll_to=roll)
    b.approach_down(grip_y)
    b.squeeze(GRIP_CLOTH, FORCE_CLOTH)
    b.held_weight = WEIGHT_TOWEL_EDGE  # half the towel hangs from the gripper
    t_grasp = b.t
    b.event(f"Fold: swing the {label} edge over", "place")
    lift = 0.6 * half + 0.03
    arc = []
    for i in range(41):
        s = i / 40
        a = half * math.cos(math.pi * s)
        p = F + n * a
        arc.append([p[0], grip_y + lift * math.sin(math.pi * s) + 0.004 * s, p[1]])
    b.path3d(arc, phi=PINCH_PHI, speed=0.28)
    b.let_go()
    t_release = b.t
    end = np.array(arc[-1])
    b.path3d([end, end + [0, 0.07, 0]], phi=PINCH_PHI, min_dur=0.3)
    return {"F": [round(float(F[0]), 5), round(float(F[1]), 5)], "n": [round(float(n[0]), 5), round(float(n[1]), 5)],
            "w": half, "tGrasp": round(t_grasp, 4), "tRelease": round(t_release, 4)}


def plan_towel(lengths, q0, objects: list[dict], fixtures: list[dict]) -> dict:
    b = TrajectoryBuilder(lengths, np.asarray(q0, dtype=float))
    towel = next((o for o in objects if o["kind"] == "towel"), None)
    basket = next((f for f in fixtures if f["kind"] == "basket"), None)
    if towel is None or basket is None:
        b.event("Nothing to fold: reset the laundry scene.", "warn")
        b.pause(0.3)
        return _result(b, 0, 1)
    if towel.get("binId"):
        b.event("The towel is already folded and in the basket. Press Reset scene to start again.", "warn")
        b.pause(0.3)
        return _result(b, 0, 0)

    half = towel["size"]
    c = np.array([towel["x"], towel["z"]])
    n1 = c / np.linalg.norm(c)  # fold toward the robot: the motion stays in the arm's plane
    n2 = np.array([-n1[1], n1[0]])  # second fold is across the first
    saved = b.snapshot()
    try:
        b.look(towel["id"], "towel", [c[0], 0.002, c[1]], [2 * half, 0.004, 2 * half], yaw=math.atan2(n1[1], n1[0]), height=0.25)
        f1 = _fold(b, c, n1, half, 0.0, "far")
        F2 = c - n1 * half / 2
        f2 = _fold(b, F2, n2, half, 2 * CLOTH_EPS, "side")

        # Pick up the folded square (a quarter of the towel) and drop it in the basket.
        sq = F2 - n2 * half / 2
        stack = 4 * CLOTH_EPS
        b.event("Pick up the folded towel", "pick")
        fold_center = [sq[0], stack / 2, sq[1]]
        fold_size = [half, stack, half]
        est = b.look(towel["id"], "folded towel", fold_center, fold_size, yaw=math.atan2(n1[1], n1[0]))
        q_above = _require(pose_at([est[0], 0.1, est[2]], PINCH_PHI, b.lengths, prefer=b.q), "above the towel")
        b.joint_move(q_above, roll_to=roll_to_close_along(math.atan2(n1[1], n1[0]), float(q_above[0]), math.pi / 2))
        b.approach_down(stack + 0.03, slow=0.0)
        b.align(towel["id"], "folded towel", fold_center, est, fold_size, yaw=math.atan2(n1[1], n1[0]))
        b.approach_down(stack)
        b.squeeze(GRIP_CLOTH, FORCE_CLOTH)
        b.held_weight = WEIGHT_TOWEL
        t_pick = b.t
        b.path3d([[sq[0], stack, sq[1]], [sq[0], CARRY_Y, sq[1]]], phi=PINCH_PHI, min_dur=0.4)
        b.event("Drop it in the laundry basket", "place")
        floor = 0.012
        b.joint_move(_require(pose_at([basket["x"], CARRY_Y, basket["z"]], PINCH_PHI, b.lengths, prefer=b.q), "above the basket"))
        b.approach_down(floor + 0.04)
        b.let_go()
        t_drop = b.t
        b.path3d([[basket["x"], floor + 0.04, basket["z"]], [basket["x"], CARRY_Y, basket["z"]]], phi=PINCH_PHI, min_dur=0.4)
        b.verify("towel", "the basket")
    except Unreachable as exc:
        b.restore(saved)
        b.event(f"Can't fold this towel: the path leaves the workspace near {exc}.", "warn")
        b.pause(0.3)
        return _result(b, 0, 1)
    b.event("Return home", "info")
    b.joint_move(HOME_Q4, roll_to=0.0)
    b.pause(0.3)
    b.event("Done: towel folded in quarters and put in the basket.", "done")
    cloth = {
        "towelId": towel["id"],
        "center": [float(c[0]), float(c[1])],
        "half": half,
        "eps": CLOTH_EPS,
        "folds": [f1, f2],
        "carry": {"tGrasp": round(t_pick, 4), "tRelease": round(t_drop, 4), "floorY": floor, "basketId": basket["id"]},
    }
    return _result(b, 1, 0, {"cloth": cloth})


def plan_task(task: str, lengths, q0, objects, fixtures) -> dict:
    if task == "dishes":
        return plan_dishes(lengths, q0, objects, fixtures)
    if task == "laundry":
        return plan_towel(lengths, q0, objects, fixtures)
    raise ValueError(f"unknown task {task!r}")


__all__ = ["plan_task", "plan_dishes", "plan_towel", "dishes_scene", "laundry_scene", "TASKS"]
