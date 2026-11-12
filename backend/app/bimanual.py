"""Two arms folding a T-shirt together.

The shirt lies flat on the shared work line between the two arms, collar
toward the robots' bases (small x), hem away from them (large x). The fold
is the classic three-step one, done the way two-armed folding robots do it:

1. Scan: both wrist cameras look at the shirt and find its key points
   (collar, sleeve tips, hem corners).
2. Side folds, at the same time: each arm pinches its side seam and swings
   that side of the body (sleeve and all) in toward the middle.
3. Bottom fold, together: both arms pinch the hem at the two corners of the
   folded stack, lift in lockstep and lay the bottom half over the top half.
   Both arcs use exactly the same timing so the hem stays straight.
4. Press the crease: both arms press down along the new fold; the wrist
   force sensors feel the contact.

Both arms are planned with the same motion and sensing code as everything
else (TrajectoryBuilder), each in its own coordinates, and kept in lockstep
by padding whichever arm finishes a step first. The cloth itself is drawn by
the front end from the fold lines and the grippers' paths recorded here.
"""
from __future__ import annotations

import math

import numpy as np

from .dual import ARM_B_Z, LINE_Z, PARK_A, PARK_B, _link_points
from .kinematics import fk_world
from .planner import FPS, TOOL_DOWN, TrajectoryBuilder, Unreachable, pose_at, roll_to_close_along
from .tasks import CLOTH_EPS, FORCE_CLOTH, GRIP_CLOTH

# Shirt shape (metres). u runs from the collar (0) to the hem (LENGTH);
# v runs across the body (+v toward arm A).
LENGTH = 0.36
HALF = 0.15  # half the body width
SLEEVE = 0.25  # sleeve tip, from the centre line
X_COLLAR = 0.15  # where the collar sits on the table

SIDE_FOLD_V = 0.07  # side fold lines at v = +/-0.07
HEM_FOLD_U = 0.18  # bottom fold line, half way up the body
GRAB_U = {"A": 0.26, "B": 0.16}  # where each arm pinches its side seam (staggered so the grippers don't meet)
HEM_GRAB_V = 0.06  # pinch the hem just inside the folded edges
COOP_CLEARANCE = 0.06  # m: the arms work close together on purpose

WEIGHT_SIDE = 0.5  # N of shirt hanging from each gripper during the side folds
WEIGHT_HEM = 0.7


def shirt_scene() -> dict:
    return {
        "objects": [{
            "id": "shirt", "kind": "shirt", "color": "#e11d48", "x": X_COLLAR, "y": 0.0, "z": LINE_Z,
            "yaw": 0.0, "tilt": 0.0, "size": LENGTH, "binId": None,
        }],
        "bins": [],
        "fixtures": [],
        "arms": [
            {"id": "A", "baseZ": 0.0, "park": [float(v) for v in PARK_A]},
            {"id": "B", "baseZ": ARM_B_Z, "park": [float(v) for v in PARK_B]},
        ],
    }


def _world(u: float, v: float, y: float = 0.0) -> np.ndarray:
    return np.array([X_COLLAR + u, y, LINE_Z + v])


def _sync(*arms: TrajectoryBuilder):
    """Keep the arms in lockstep: whoever finished first holds still."""
    n = max(len(a.frames) for a in arms)
    for a in arms:
        while len(a.frames) < n:
            a._emit(a.q)


def _arc(F_xz, n_xz, s_grab: float, base_y: float):
    """The pinched edge's path for a fold: up and over the fold line in an
    arc, laid down just as far on the other side."""
    lift = 0.6 * s_grab + 0.03
    grip_y = base_y + 0.004
    pts = []
    for i in range(41):
        s = i / 40
        a = s_grab * math.cos(math.pi * s)
        pts.append([F_xz[0] + n_xz[0] * a, grip_y + lift * math.sin(math.pi * s) + 0.004 * s, F_xz[1] + n_xz[1] * a])
    return np.array(pts)


def plan_shirt(lengths, objects: list[dict] | None = None) -> dict:
    shirt = next((o for o in (objects or []) if o.get("kind") == "shirt"), None)
    A = TrajectoryBuilder(lengths, PARK_A, seed=11)
    B = TrajectoryBuilder(lengths, PARK_B, seed=12)
    arms = {"A": A, "B": B}
    base_z = {"A": 0.0, "B": ARM_B_Z}
    events: list[dict] = []

    def local(arm: str, p):
        p = np.asarray(p, dtype=float).copy()
        p[2] -= base_z[arm]
        return p

    def say(text: str, kind: str = "info"):
        events.append({"t": round(A.t, 3), "text": text, "kind": kind})

    def go_above(arm: str, p_world, height: float = 0.09, close_along: float | None = None):
        """Move above a point. With close_along (an angle in the table plane),
        roll the wrist on the way so the fingers will close in that direction:
        straight across a seam or hem, which is how you pinch cloth cleanly."""
        b = arms[arm]
        q = pose_at(local(arm, [p_world[0], p_world[1] + height, p_world[2]]), TOOL_DOWN, lengths, prefer=b.q)
        if q is None:
            raise Unreachable(f"above ({p_world[0]:.2f}, {p_world[2]:.2f}) for arm {arm}")
        roll = None
        if close_along is not None:
            # Both arms use the same angle conventions (arm B is only shifted
            # across the table), so a world direction works for either.
            roll = roll_to_close_along(close_along, float(q[0]))
        b.joint_move(q, roll_to=roll)

    if shirt is None or shirt.get("binId"):
        say("The shirt is already folded. Press Reset scene to lay out a fresh one.", "warn")
        A.pause(0.3)
        _sync(A, B)
        return _assemble(lengths, A, B, events, [], 0, None)

    folds = []
    try:
        # 1. Scan ----------------------------------------------------------
        say("Both cameras scan the shirt", "pick")
        A.look("shirt", "shirt (near half)", _world(0.18, 0.08, 0.003), [LENGTH, 0.006, HALF + 0.1], height=0.3)
        B.look("shirt", "shirt (far half)", local("B", _world(0.18, -0.08, 0.003)), [LENGTH, 0.006, HALF + 0.1], height=0.3)
        _sync(A, B)
        t0 = A.t
        for label, (u, v) in {
            "collar": (0.0, 0.0), "sleeve tip": (0.07, SLEEVE), "sleeve tip ": (0.07, -SLEEVE),
            "hem corner": (LENGTH, HALF), "hem corner ": (LENGTH, -HALF),
        }.items():
            p = _world(u, v, 0.003)
            A.perception.append({
                "tStart": round(t0 - 0.3, 4), "tEnd": round(t0 + 0.9, 4), "id": "shirt", "label": label.strip(),
                "confidence": round(A.rng.uniform(0.92, 0.99), 2), "center": [round(float(x), 4) for x in p],
                "size": [0.03, 0.01, 0.03], "yaw": 0.0, "phase": "align",
            })
        say("Key points found: collar, both sleeve tips, both hem corners", "sense")

        # 2. Side folds, both arms at once ---------------------------------
        say("Side folds: each arm folds its side and sleeve in toward the middle", "place")
        side = {"A": +1, "B": -1}
        s_grab = HALF - SIDE_FOLD_V
        grabs = {k: _world(GRAB_U[k], side[k] * HALF, 0.0) for k in arms}
        for k in arms:
            go_above(k, grabs[k], close_along=math.pi / 2)  # side seam runs along x: close across it, along z
        _sync(A, B)
        say("Wrists roll so the fingers close straight across each side seam", "info")
        for k in arms:
            arms[k].approach_down(0.004)
        _sync(A, B)
        for k in arms:
            arms[k].squeeze(GRIP_CLOTH, FORCE_CLOTH)
            arms[k].held_weight = WEIGHT_SIDE
        _sync(A, B)
        t_grasp = A.t
        arcs = {}
        for k in arms:
            F = np.array([X_COLLAR + GRAB_U[k], LINE_Z + side[k] * SIDE_FOLD_V])
            n = np.array([0.0, float(side[k])])
            arcs[k] = (F, n, _arc(F, n, s_grab, 0.0))
        dur = max(1.4, float(np.linalg.norm(np.diff(arcs["A"][2], axis=0), axis=1).sum()) / 0.25 * 1.875)
        for k in arms:
            arms[k].path3d([local(k, p) for p in arcs[k][2]], dur=dur)
        _sync(A, B)
        for k in arms:
            arms[k].let_go()
        _sync(A, B)
        t_release = A.t
        for k in arms:
            F, n, _ = arcs[k]
            folds.append({"F": [round(float(F[0]), 5), round(float(F[1]), 5)], "n": [0.0, float(n[1])], "w": round(s_grab, 5),
                          "arms": [k], "tGrasp": round(t_grasp, 4), "tRelease": round(t_release, 4)})
        for k in arms:
            tip = local(k, fk_world(arms[k].q, lengths)[0][-1] + np.array([0, 0, base_z[k]]))
            arms[k].path3d([tip, tip + [0, 0.07, 0]], min_dur=0.3)
        _sync(A, B)

        # Both grippers are now over the middle: back out to their own sides
        # before heading for the hem, so they never cross paths.
        for k in arms:
            go_above(k, _world(GRAB_U[k], side[k] * (HALF + 0.03)), height=0.1)
        _sync(A, B)

        # 3. Bottom fold, together -----------------------------------------
        say("Bottom fold: both arms pinch the hem corners and fold it up together", "place")
        hem = {"A": _world(LENGTH, +HEM_GRAB_V), "B": _world(LENGTH, -HEM_GRAB_V)}
        base_y = 2 * CLOTH_EPS
        for k in arms:
            go_above(k, hem[k] + [0, base_y, 0], close_along=0.0)  # hem runs along z: close across it, along x
        _sync(A, B)
        say("Wrists roll about 90° so the fingers close across the hem and pinch the corners cleanly", "info")
        for k in arms:
            arms[k].approach_down(base_y + 0.004)
        _sync(A, B)
        for k in arms:
            arms[k].squeeze(GRIP_CLOTH, FORCE_CLOTH)
            arms[k].held_weight = WEIGHT_HEM
        _sync(A, B)
        t_grasp = A.t
        s_hem = LENGTH - HEM_FOLD_U
        F_hem = {k: np.array([X_COLLAR + HEM_FOLD_U, hem[k][2]]) for k in arms}
        n_hem = np.array([1.0, 0.0])
        hem_arcs = {k: _arc(F_hem[k], n_hem, s_hem, base_y) for k in arms}
        dur = max(1.8, float(np.linalg.norm(np.diff(hem_arcs["A"], axis=0), axis=1).sum()) / 0.25 * 1.875)
        say("Lockstep: both arcs share the exact same timing so the hem stays straight", "sense")
        for k in arms:
            arms[k].path3d([local(k, p) for p in hem_arcs[k]], dur=dur)
        _sync(A, B)
        for k in arms:
            arms[k].let_go()
        _sync(A, B)
        t_release = A.t
        folds.append({"F": [round(X_COLLAR + HEM_FOLD_U, 5), LINE_Z], "n": [1.0, 0.0], "w": round(s_hem, 5),
                      "arms": ["A", "B"], "tGrasp": round(t_grasp, 4), "tRelease": round(t_release, 4)})
        for k in arms:
            tip = local(k, fk_world(arms[k].q, lengths)[0][-1] + np.array([0, 0, base_z[k]]))
            arms[k].path3d([tip, tip + [0, 0.07, 0]], min_dur=0.3)
        _sync(A, B)

        # 4. Press the crease ----------------------------------------------
        say("Press the crease along the new fold", "place")
        stack = 4 * CLOTH_EPS
        press = {"A": _world(HEM_FOLD_U - 0.02, +0.05), "B": _world(HEM_FOLD_U - 0.02, -0.05)}
        for k in arms:
            go_above(k, press[k] + [0, stack, 0], height=0.06)
        _sync(A, B)
        for k in arms:
            arms[k].approach_down(stack + 0.003)
            arms[k].contact = 4.0  # the wrist sensor feels the press
            arms[k].pause(0.35)
        _sync(A, B)
        say("Wrist sensors felt the press on both sides", "sense")
        for k in arms:
            arms[k].lift_off(0.12)
        _sync(A, B)

        # 5. Park ------------------------------------------------------------
        say("Both arms park", "info")
        for k in arms:
            arms[k].transfer(PARK_A if k == "A" else PARK_B, tuck_u=0.15, roll_to=0.0)
        _sync(A, B)
        A.pause(0.3)
        _sync(A, B)
    except Unreachable as exc:
        say(f"Can't fold this shirt: the path leaves the workspace near {exc}.", "warn")
        _sync(A, B)
        return _assemble(lengths, A, B, events, [], 0, None)

    say("Done: shirt folded by two arms (sides in, bottom up, crease pressed).", "done")
    cloth = {
        "shape": "shirt",
        "towelId": "shirt",
        "center": [X_COLLAR, LINE_Z],
        "half": HALF,
        "dims": {"length": LENGTH, "half": HALF, "sleeve": SLEEVE},
        "eps": CLOTH_EPS,
        "folds": folds,
        "carry": None,
    }
    return _assemble(lengths, A, B, events, folds, 1, cloth)


def _assemble(lengths, A: TrajectoryBuilder, B: TrajectoryBuilder, events: list[dict], folds, done: int, cloth) -> dict:
    for e in A.events:
        events.append({"t": e["t"], "text": f"[Arm A] {e['text']}", "kind": e["kind"]})
    for e in B.events:
        events.append({"t": e["t"], "text": f"[Arm B] {e['text']}", "kind": e["kind"]})
    events.sort(key=lambda e: e["t"])
    perception = [dict(p, arm="A") for p in A.perception]
    for p in B.perception:
        p = dict(p, arm="B")
        p["center"] = [p["center"][0], p["center"][1], round(p["center"][2] + ARM_B_Z, 4)]
        perception.append(p)
    pa = [_link_points(f["q"], lengths, 0.0) for f in A.frames]
    pb = [_link_points(f["q"], lengths, ARM_B_Z) for f in B.frames]
    gap = min(float(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).min()) for x, y in zip(pa, pb))
    n = len(A.frames)
    grasps = {k: sum(1 for i in range(1, len(b.sensors["grip"])) if b.sensors["grip"][i - 1] == 0 and b.sensors["grip"][i] > 0)
              for k, b in (("A", A), ("B", B))}
    out = {
        "fps": FPS,
        "duration": round(A.t, 3),
        "frames": A.frames,
        "sensors": A.sensors,
        "armB": {"baseZ": ARM_B_Z, "frames": B.frames, "sensors": B.sensors},
        "events": events,
        "timeline": {},
        "perception": perception,
        "placements": [],
        "warnings": [] if gap >= COOP_CLEARANCE - 1e-6 else ["The arms came closer than the safety clearance."],
        "stats": {
            "sorted": done, "skipped": 0 if done else 1, "slips": 0, "drops": 0, "recovered": 0,
            "durationS": round(A.t, 2),
            "toolPathM": round(A.path_length + B.path_length, 2),
            "maxJointSpeedDegS": round(math.degrees(max(A.max_joint_speed, B.max_joint_speed)), 1),
            "minClearanceMm": round(min(A.min_clearance, B.min_clearance) * 1000, 1),
            "frames": n,
        },
        "dual": {
            "singleArmS": None, "speedup": None,
            "minArmGapMm": round(gap * 1000, 1), "clearanceMm": COOP_CLEARANCE * 1000,
            "arms": [
                {"id": k, "blocks": grasps[k], "busyS": round(A.t, 1), "waitS": 0.0, "waits": 0, "utilization": 1.0}
                for k in ("A", "B")
            ],
        },
    }
    if cloth:
        out["cloth"] = cloth
    return out


__all__ = ["plan_shirt", "shirt_scene", "LENGTH", "HALF", "SLEEVE", "X_COLLAR"]
