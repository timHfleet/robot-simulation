"""Two arms on the household tasks: loading the dishwasher and folding a towel.

Dishes: the work is split by kind. Arm A loads the plates into the bottom
rack while arm B loads the cups into the top rack, at the same time. Every
dish is one job, planned by the same single-arm code (camera scan, careful
contact, force sensing) in that arm's own coordinates, then fitted into the
shared schedule with the same frame-by-frame collision check the two-arm
block sorting uses.

Towel: a real two-handed fold. Both arms pinch the two far corners and fold
the towel in half together, in lockstep so the edge stays straight (one arm
alone has to pinch the middle of the edge and the corners droop). Arm A then
folds it across, and arm B, which is nearer the basket's side, picks up the
folded towel and drops it in the basket while arm A parks.
"""
from __future__ import annotations

import math

import numpy as np

from .bimanual import _arc, _assemble, _sync
from .dual import ARM_B_Z, LINE_Z, PARK_A, PARK_B, TUCK_U, Schedule, _to_local
from .kinematics import HOME_Q4, fk_world
from .planner import TOOL_DOWN, TrajectoryBuilder, Unreachable, pose_at, roll_to_close_along
from .tasks import CARRY_Y, CLOTH_EPS, CUP_H, CUP_R, CUP_RACK_H, PLATE_RACK_H, FORCE_CLOTH, GRIP_CLOTH, TOWEL_HALF, \
    WEIGHT_TOWEL, WEIGHT_TOWEL_EDGE, _polar, dishes_scene, plan_dishes


def _arms() -> list[dict]:
    return [
        {"id": "A", "baseZ": 0.0, "park": [float(v) for v in PARK_A]},
        {"id": "B", "baseZ": ARM_B_Z, "park": [float(v) for v in PARK_B]},
    ]


# ---- Dishes -------------------------------------------------------------------

def dishes2_scene() -> dict:
    """Arm A's side as in the one-arm scene (plate holder and bottom rack);
    the cups and the top rack sit on arm B's side, laid out the same way
    around arm B's base."""
    one = dishes_scene()
    objects = [o for o in one["objects"] if o["kind"] == "plate"]
    cup_colors = ["#f97316", "#14b8a6", "#a855f7"]
    for i, deg in enumerate((-42, -20, -64)):
        x, z = _polar(0.37, deg)
        objects.append({
            "id": f"cup-{i + 1}", "kind": "cup", "color": cup_colors[i],
            "x": round(x, 4), "y": CUP_H / 2, "z": round(z + ARM_B_Z, 4),
            "yaw": round(math.radians(deg), 5), "tilt": 0.0, "size": CUP_R, "binId": None,
        })
    fixtures = [f for f in one["fixtures"] if f["kind"] not in ("cup_rack", "plate_rack")]
    # The bottom rack sits round on arm A's own side (+z), so arm A swings the
    # plates that way, away from arm B, instead of across the shared line.
    plate_slots = []
    for deg in (150 - 12, 150, 150 + 12):
        x, z = _polar(0.50, deg)
        plate_slots.append({"x": round(x, 4), "z": round(z, 4), "yaw": round(math.radians(deg), 5)})
    rx, rz = _polar(0.50, 150)
    fixtures.append({"id": "plate-rack", "kind": "plate_rack", "x": round(rx, 4), "z": round(rz, 4), "width": 0.24,
                     "depth": 0.32, "height": PLATE_RACK_H, "slots": plate_slots, "label": "Bottom rack (plates)"})
    cup_slots = []
    for deg in (180 - 22, 180, 180 + 22):
        x, z = _polar(0.25, deg)
        cup_slots.append({"x": round(x, 4), "z": round(z + ARM_B_Z, 4), "yaw": round(math.radians(deg), 5)})
    fixtures.append({"id": "cup-rack", "kind": "cup_rack", "x": -0.25, "z": ARM_B_Z, "width": 0.14, "depth": 0.30,
                     "height": CUP_RACK_H, "slots": cup_slots, "label": "Top rack (cups)"})
    return {"objects": objects, "bins": [], "fixtures": fixtures, "arms": _arms()}


def _fixture_local(f: dict, base_z: float) -> dict:
    f = _to_local(f, base_z)
    if "slots" in f:
        f["slots"] = [_to_local(s, base_z) for s in f["slots"]]
    return f


def plan_dishes2(lengths, objects: list[dict], fixtures: list[dict]) -> dict:
    s = Schedule(lengths)
    world = {o["id"]: dict(o) for o in objects}
    # Arm A loads the plates, arm B the cups.
    queues = {
        "A": [o["id"] for o in objects if o["kind"] == "plate" and not o.get("binId")],
        "B": [o["id"] for o in objects if o["kind"] == "cup" and not o.get("binId")],
    }
    if not queues["A"] and not queues["B"]:
        s.events.append({"t": 0.0, "text": "The dishwasher is already loaded. Press Reset scene to unload it.", "kind": "warn"})
    while queues["A"] or queues["B"]:
        arm = min((a for a in s.arms if queues[a.id]), key=lambda a: a.end_index)
        oid = queues[arm.id].pop(0)
        kind = world[oid]["kind"]
        # Plan this one dish; the arm also sees the dishes already in its rack so it picks a free slot.
        mine = [_to_local(o, arm.base_z) for o in world.values() if o["kind"] == kind and (o["id"] == oid or o.get("binId"))]
        job = plan_dishes(lengths, arm.q, mine, [_fixture_local(f, arm.base_z) for f in fixtures],
                          park_q=PARK_A if arm.id == "A" else PARK_B, tuck_u=TUCK_U)
        if job["stats"]["sorted"] != 1:
            s.warn(arm, next((e["text"] for e in job["events"] if e["kind"] == "warn"), f"Could not load the {kind}."))
            continue
        if not s.commit(arm, job, kind):
            continue
        # Where it ended up: the last keyframe of its timeline (in world coordinates now).
        kf = s.timeline[oid][-1]
        world[oid].update({"x": kf["pos"][0], "y": kf["pos"][1], "z": kf["pos"][2], "binId": kf.get("binId")})
    plates = sum(1 for o in world.values() if o["kind"] == "plate" and o.get("binId"))
    cups = sum(1 for o in world.values() if o["kind"] == "cup" and o.get("binId"))
    single_s = None
    if s.jobs:
        # The same six dishes loaded by one arm alone (the one-arm kitchen layout), for the speed-up figure.
        one = dishes_scene()
        solo = plan_dishes(lengths, HOME_Q4, one["objects"], one["fixtures"])
        single_s = solo["stats"]["durationS"] if solo["stats"]["sorted"] else None
    out = s.finish("loaded", single_s)
    out["events"][-1]["text"] = out["events"][-1]["text"].replace("loaded", f"loaded ({plates} plates, {cups} cups)", 1)
    return out


# ---- Towel ----------------------------------------------------------------------

TOWEL_X = 0.36  # towel centre, on the shared work line
BASKET_X = -0.40
CORNER_IN = 0.035  # pinch just inside each corner


def laundry2_scene() -> dict:
    return {
        "objects": [{
            "id": "towel", "kind": "towel", "color": "#0ea5e9", "x": TOWEL_X, "y": 0.0, "z": LINE_Z,
            "yaw": 0.0, "tilt": 0.0, "size": TOWEL_HALF, "binId": None,
        }],
        "bins": [],
        "fixtures": [{"id": "basket", "kind": "basket", "x": BASKET_X, "z": LINE_Z, "width": 0.26, "depth": 0.26,
                      "height": 0.08, "label": "Laundry basket"}],
        "arms": _arms(),
    }


def plan_towel2(lengths, objects: list[dict], fixtures: list[dict]) -> dict:
    towel = next((o for o in objects if o.get("kind") == "towel"), None)
    basket = next((f for f in fixtures if f.get("kind") == "basket"), None)
    A = TrajectoryBuilder(lengths, PARK_A, seed=21)
    B = TrajectoryBuilder(lengths, PARK_B, seed=22)
    arms = {"A": A, "B": B}
    base_z = {"A": 0.0, "B": ARM_B_Z}
    events: list[dict] = []

    def local(arm: str, p):
        p = np.asarray(p, dtype=float).copy()
        p[2] -= base_z[arm]
        return p

    def say(text: str, kind: str = "info", at: TrajectoryBuilder | None = None):
        events.append({"t": round((at or A).t, 3), "text": text, "kind": kind})

    def go_above(arm: str, p_world, height: float = 0.09, close_along: float | None = None):
        b = arms[arm]
        q = pose_at(local(arm, [p_world[0], p_world[1] + height, p_world[2]]), TOOL_DOWN, lengths, prefer=b.q)
        if q is None:
            raise Unreachable(f"above ({p_world[0]:.2f}, {p_world[2]:.2f}) for arm {arm}")
        roll = None if close_along is None else roll_to_close_along(close_along, float(q[0]))
        b.joint_move(q, roll_to=roll)

    def tip(arm: str):
        return local(arm, fk_world(arms[arm].q, lengths)[0][-1] + np.array([0, 0, base_z[arm]]))

    if towel is None or basket is None:
        say("Nothing to fold: reset the laundry scene.", "warn")
        A.pause(0.3)
        _sync(A, B)
        return _assemble(lengths, A, B, events, [], 0, None)
    if towel.get("binId"):
        say("The towel is already folded and in the basket. Press Reset scene to start again.", "warn")
        A.pause(0.3)
        _sync(A, B)
        return _assemble(lengths, A, B, events, [], 0, None)

    half = float(towel["size"])
    cx, cz = float(towel["x"]), float(towel.get("z") or 0.0)
    folds = []
    try:
        # 1. Scan ------------------------------------------------------------
        say("Both cameras scan the towel and find its corners", "pick")
        A.look(towel["id"], "towel (near half)", [cx, 0.002, cz + half / 2], [2 * half, 0.004, half], height=0.28)
        B.look(towel["id"], "towel (far half)", local("B", [cx, 0.002, cz - half / 2]), [2 * half, 0.004, half], height=0.28)
        _sync(A, B)

        # 2. Fold in half together: both far corners over to the near edge --
        say("Fold in half: each arm pinches one far corner", "place")
        side = {"A": +1, "B": -1}
        corner = {k: np.array([cx + half - CORNER_IN, 0.0, cz + side[k] * (half - CORNER_IN)]) for k in arms}
        for k in arms:
            go_above(k, corner[k], close_along=0.0)  # the far edge runs along z: close across it, along x
        _sync(A, B)
        say("Wrists roll so both grippers close straight across the edge", "info")
        for k in arms:
            arms[k].approach_down(0.004)
        _sync(A, B)
        for k in arms:
            arms[k].squeeze(GRIP_CLOTH, FORCE_CLOTH)
            arms[k].held_weight = WEIGHT_TOWEL_EDGE / 2  # half the weight each
        _sync(A, B)
        t_grasp = A.t
        s_grab = half - CORNER_IN
        arcs = {k: _arc(np.array([cx, corner[k][2]]), np.array([1.0, 0.0]), s_grab, 0.0) for k in arms}
        dur = max(1.6, float(np.linalg.norm(np.diff(arcs["A"], axis=0), axis=1).sum()) / 0.25 * 1.875)
        say("Lockstep: both arcs share the same timing so the edge stays straight", "sense")
        for k in arms:
            arms[k].path3d([local(k, p) for p in arcs[k]], dur=dur)
        _sync(A, B)
        for k in arms:
            arms[k].let_go()
        _sync(A, B)
        folds.append({"F": [round(cx, 5), round(cz, 5)], "n": [1.0, 0.0], "w": round(half, 5), "arms": ["A", "B"],
                      "tGrasp": round(t_grasp, 4), "tRelease": round(A.t, 4)})
        for k in arms:
            p = tip(k)
            arms[k].path3d([p, p + [0, 0.07, 0]], min_dur=0.3)
        _sync(A, B)

        # 3. Arm A folds it across; arm B pulls back out of the way ---------
        say("Arm A folds it across toward arm B; arm B moves clear", "place")
        B.transfer(PARK_B, tuck_u=TUCK_U, roll_to=0.0)
        F2 = np.array([cx - half / 2, cz])
        edge = np.array([F2[0], 2 * CLOTH_EPS, cz + half])
        go_above("A", edge, close_along=math.pi / 2)  # this edge runs along x: close across it, along z
        A.approach_down(2 * CLOTH_EPS + 0.004)
        A.squeeze(GRIP_CLOTH, FORCE_CLOTH)
        A.held_weight = WEIGHT_TOWEL_EDGE
        t_grasp = A.t
        arc = _arc(F2, np.array([0.0, 1.0]), half, 2 * CLOTH_EPS)
        A.path3d([local("A", p) for p in arc], speed=0.28)
        A.let_go()
        folds.append({"F": [round(float(F2[0]), 5), round(cz, 5)], "n": [0.0, 1.0], "w": round(half, 5), "arms": ["A"],
                      "tGrasp": round(t_grasp, 4), "tRelease": round(A.t, 4)})
        p = tip("A")
        A.path3d([p, p + [0, 0.07, 0]], min_dur=0.3)
        _sync(A, B)

        # 4. Hand-off: A parks, B takes the folded towel to the basket ------
        say("Arm A parks; arm B takes the folded towel to the basket", "pick")
        A.transfer(PARK_A, tuck_u=TUCK_U, roll_to=0.0)
        stack = 4 * CLOTH_EPS
        sq = np.array([F2[0], stack / 2, cz - half / 2])
        est = B.look(towel["id"], "folded towel", local("B", sq), [half, stack, half], yaw=0.0, height=0.2)
        est_w = est + np.array([0, 0, ARM_B_Z])
        go_above("B", [est_w[0], stack, est_w[2]], height=0.09, close_along=0.0)
        B.approach_down(stack + 0.03, slow=0.0)
        B.align(towel["id"], "folded towel", local("B", sq), est, [half, stack, half], yaw=0.0)
        B.approach_down(stack)
        B.squeeze(GRIP_CLOTH, FORCE_CLOTH)
        B.held_weight = WEIGHT_TOWEL
        t_pick = B.t
        p = local("B", [sq[0], stack, sq[2]])
        B.path3d([p, p + [0, CARRY_Y - stack, 0]], min_dur=0.4)
        say("Arm B drops it in the laundry basket", "place", at=B)
        floor = 0.012
        q = pose_at(local("B", [basket["x"], CARRY_Y, basket["z"]]), TOOL_DOWN, lengths, prefer=B.q)
        if q is None:
            raise Unreachable("above the basket for arm B")
        B.transfer(q, tuck_u=TUCK_U)
        B.approach_down(floor + 0.04)
        B.let_go()
        t_drop = B.t
        p = tip("B")
        B.path3d([p, p + [0, CARRY_Y - p[1], 0]], min_dur=0.4)
        B.verify("towel", "the basket")
        B.transfer(PARK_B, tuck_u=TUCK_U, roll_to=0.0)
        _sync(A, B)
        A.pause(0.3)
        _sync(A, B)
    except Unreachable as exc:
        say(f"Can't fold this towel: the path leaves the workspace near {exc}.", "warn")
        _sync(A, B)
        return _assemble(lengths, A, B, events, [], 0, None)

    say("Done: towel folded in quarters by two arms and put in the basket.", "done")
    cloth = {
        "towelId": towel["id"],
        "center": [cx, cz],
        "half": half,
        "yaw": 0.0,
        "eps": CLOTH_EPS,
        "folds": folds,
        "carry": {"tGrasp": round(t_pick, 4), "tRelease": round(t_drop, 4), "floorY": floor, "basketId": basket["id"], "arm": "B"},
    }
    out = _assemble(lengths, A, B, events, folds, 1, cloth)
    # The two-handed half fold is easier on the cloth, but needs more care: use a looser clearance like the shirt.
    return out


__all__ = ["dishes2_scene", "laundry2_scene", "plan_dishes2", "plan_towel2"]
