"""Two arms sharing one job: sorting blocks across a shared work line.

Layout (seen from above)

      z = 0      arm A's base
      z = -0.22  work line: blocks on the right (+x), bins on the left (-x)
      z = -0.44  arm B's base

Each arm plans its own pick-and-place jobs with the same planner the single
arm uses (camera scan, close-up correction, careful contact, force sensing),
in its own coordinates. A scheduler then decides when each job may start:

* Dispatch: whichever arm is free first takes the remaining block closest
  to its base.
* Collision check: before a job is committed, every frame of it (plus any
  waiting before it) is checked against the other arm's committed motion.
  Both arms are modelled as points spread along their links; if any pair
  comes closer than CLEARANCE the job is delayed a little and checked again.
  This is prioritised planning: jobs already committed keep their timing,
  the newcomer waits.
* Every job ends by parking the arm on its own side, so an idle arm never
  sits in the shared space blocking the other.
"""
from __future__ import annotations

import math

import numpy as np

from . import planner
from .kinematics import fk_world
from .planner import FPS, plan_commands, plan_sort

ARM_B_Z = -0.44
LINE_Z = -0.22
CLEARANCE = 0.085  # m between any two link points of the two arms
WAIT_STEP = 0.25  # s
MAX_WAIT = 20.0  # s
PTS_PER_LINK = 7
TUCK_U = 0.15  # m: how close to its base an arm pulls in before a big waist turn

# Park poses: folded up, turned away from the work line (arm A faces +z, B faces -z).
PARK_A = np.array([math.pi / 2, math.radians(95), math.radians(-115), math.radians(-60)])
PARK_B = np.array([-math.pi / 2, math.radians(95), math.radians(-115), math.radians(-60)])


def dual_bins() -> list[dict]:
    return [
        {"id": "bin-red", "color": "red", "x": -0.50, "z": LINE_Z, "width": planner.BIN_WIDTH, "wall": planner.BIN_WALL},
        {"id": "bin-blue", "color": "blue", "x": -0.31, "z": LINE_Z, "width": planner.BIN_WIDTH, "wall": planner.BIN_WALL},
        {"id": "bin-green", "color": "green", "x": -0.12, "z": LINE_Z, "width": planner.BIN_WIDTH, "wall": planner.BIN_WALL},
    ]


def dual_scene(seed: int = 7) -> dict:
    return {
        "objects": planner.random_scene(seed, count=6, x_lo=0.16, x_hi=0.5, z=LINE_Z),
        "bins": dual_bins(),
        "fixtures": [],
        "arms": [
            {"id": "A", "baseZ": 0.0, "park": [float(v) for v in PARK_A]},
            {"id": "B", "baseZ": ARM_B_Z, "park": [float(v) for v in PARK_B]},
        ],
    }


def _link_points(q, lengths, base_z: float) -> np.ndarray:
    """Points spread along the arm's three links, in world coordinates."""
    pts = fk_world(q, lengths)[0] + np.array([0.0, 0.0, base_z])
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0.0, 1.0, PTS_PER_LINK):
            out.append(a + s * (b - a))
    return np.array(out)


class Arm:
    def __init__(self, arm_id: str, base_z: float, q0, lengths):
        self.id = arm_id
        self.base_z = base_z
        self.lengths = lengths
        self.q = np.asarray(q0, dtype=float)
        self.frames: list[dict] = []
        self.points: list[np.ndarray] = []  # link points per frame, for collision checks
        self.sensors: dict[str, list[float]] = {"fz": [], "grip": []}
        self.blocks = 0
        self.busy_frames = 0
        self.wait_s = 0.0
        self.waits = 0
        self._append({"q": [float(v) for v in self.q], "grip": 1.0, "roll": 0.0, "holding": None}, 0.0, 0.0)

    def _append(self, frame: dict, fz: float, grip: float):
        f = dict(frame)
        f["t"] = round(len(self.frames) / FPS, 4)
        self.frames.append(f)
        self.points.append(_link_points(f["q"], self.lengths, self.base_z))
        self.sensors["fz"].append(fz)
        self.sensors["grip"].append(grip)

    @property
    def end_index(self) -> int:
        return len(self.frames) - 1


def _clear(candidate: list[np.ndarray], start_index: int, other: Arm) -> bool:
    """Is a candidate motion (link points per frame, from start_index on)
    clear of the other arm's committed motion?"""
    n = len(candidate)
    idx = np.minimum(np.arange(start_index, start_index + n), other.end_index)
    mine = np.stack(candidate)  # (n, P, 3)
    theirs = np.stack([other.points[i] for i in idx])  # (n, P, 3)
    d = np.linalg.norm(mine[:, :, None, :] - theirs[:, None, :, :], axis=-1)
    return float(d.min()) >= CLEARANCE


def _to_local(o: dict, base_z: float) -> dict:
    o = dict(o)
    o["z"] = (o.get("z") or 0.0) - base_z
    return o


class Schedule:
    """Two arms, their committed motion, and everything shown alongside it."""

    def __init__(self, lengths):
        self.lengths = lengths
        self.arms = [Arm("A", 0.0, PARK_A, lengths), Arm("B", ARM_B_Z, PARK_B, lengths)]
        self.events: list[dict] = []
        self.timeline: dict[str, list[dict]] = {}
        self.perception: list[dict] = []
        self.skipped = 0
        self.jobs = 0
        self.min_table_clearance = float("inf")
        self.peak_speed = 0.0

    def other(self, arm: Arm) -> Arm:
        return self.arms[1] if arm is self.arms[0] else self.arms[0]

    def warn(self, arm: Arm, text: str):
        self.events.append({"t": round(arm.end_index / FPS, 3), "text": f"[Arm {arm.id}] {text}", "kind": "warn"})
        self.skipped += 1

    def commit(self, arm: Arm, job: dict, what: str) -> bool:
        """Fit a job (planned in the arm's own coordinates, starting from its
        current pose) into the schedule: wait until it is clear of the other
        arm, then append it. Returns False if it never got clear."""
        other = self.other(arm)
        jf = job["frames"][1:]  # the first frame is the arm's current pose
        if not jf:
            return False
        job_pts = [_link_points(f["q"], self.lengths, arm.base_z) for f in jf]
        hold_pts = arm.points[-1]
        start = arm.end_index + 1
        delay = None
        for k in range(int(MAX_WAIT / WAIT_STEP) + 1):
            hold = [hold_pts] * int(round(k * WAIT_STEP * FPS))
            if _clear(hold + job_pts, start, other):
                delay = len(hold)
                break
        if delay is None:
            self.warn(arm, f"Gave up on the {what}: the other arm kept the space busy.")
            return False

        self.jobs += 1
        self.min_table_clearance = min(self.min_table_clearance, job["stats"]["minClearanceMm"])
        self.peak_speed = max(self.peak_speed, job["stats"]["maxJointSpeedDegS"])
        if delay:
            arm.waits += 1
            arm.wait_s += delay / FPS
            self.events.append({"t": round(arm.end_index / FPS, 3),
                                "text": f"[Arm {arm.id}] Waits {delay / FPS:.2f} s: Arm {other.id} is in the shared space",
                                "kind": "wait"})
            last = dict(arm.frames[-1])
            for _ in range(delay):
                arm._append(last, arm.sensors["fz"][-1], 0.0)
        t0 = arm.end_index / FPS  # job time 0 maps here
        for i, f in enumerate(jf):
            arm._append(f, job["sensors"]["fz"][i + 1], job["sensors"]["grip"][i + 1])
        arm.busy_frames += len(jf)
        arm.q = np.asarray(jf[-1]["q"], dtype=float)
        arm.blocks += 1

        for e in job["events"]:
            if e["text"].startswith("Done:") or e["text"] == "Return home":
                continue
            self.events.append({"t": round(e["t"] + t0, 3), "text": f"[Arm {arm.id}] {e['text']}", "kind": e["kind"]})
        for oid, kfs in job["timeline"].items():
            for kf in kfs:
                kf = dict(kf)
                kf["t"] = round(kf["t"] + t0, 4)
                for key in ("pos", "to"):
                    if key in kf:
                        kf[key] = [kf[key][0], kf[key][1], round(kf[key][2] + arm.base_z, 4)]
                if kf["mode"] == "held":
                    kf["arm"] = arm.id
                self.timeline.setdefault(oid, []).append(kf)
        for p in job["perception"]:
            p = dict(p)
            p["tStart"] = round(p["tStart"] + t0, 4)
            p["tEnd"] = round(p["tEnd"] + t0, 4)
            p["center"] = [p["center"][0], p["center"][1], round(p["center"][2] + arm.base_z, 4)]
            p["arm"] = arm.id
            self.perception.append(p)
        return True

    def finish(self, summary: str, single_s: float | None = None, extra: dict | None = None) -> dict:
        arms = self.arms
        # Both arms end parked; pad the shorter timeline so they finish together.
        n = max(len(a.frames) for a in arms)
        if n < 2:
            n = 2  # nothing was done: still return a (short) playable plan
        for a in arms:
            last = dict(a.frames[-1])
            while len(a.frames) < n:
                a._append(last, a.sensors["fz"][-1], 0.0)
        duration = (n - 1) / FPS
        self.events.sort(key=lambda e: e["t"])
        done = sum(a.blocks for a in arms)

        a, b = arms
        min_gap = min(
            float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1).min())
            for pa, pb in zip(a.points[::3], b.points[::3])
        )
        text = f"Done: {done} {summary} by two arms in {duration:.1f} s"
        if single_s:
            text += f" (one arm alone: {single_s:.1f} s)"
        if self.skipped:
            text += f", {self.skipped} skipped"
        self.events.append({"t": round(duration, 3), "text": text + ".", "kind": "done"})
        tool_path = sum(
            float(np.linalg.norm(np.diff(np.array([fk_world(f["q"], self.lengths)[0][-1] for f in arm.frames[::2]]), axis=0), axis=1).sum())
            for arm in arms
        )
        out = {
            "fps": FPS,
            "duration": round(duration, 3),
            "frames": a.frames,
            "sensors": a.sensors,
            "armB": {"baseZ": ARM_B_Z, "frames": b.frames, "sensors": b.sensors},
            "events": self.events,
            "timeline": self.timeline,
            "perception": self.perception,
            "placements": [],
            "warnings": [] if min_gap >= CLEARANCE - 1e-6 else ["The arms came closer than the safety clearance."],
            "stats": {
                "sorted": done,
                "skipped": self.skipped,
                "slips": 0,
                "drops": 0,
                "recovered": 0,
                "durationS": round(duration, 2),
                "toolPathM": round(tool_path, 2),
                "maxJointSpeedDegS": round(self.peak_speed, 1),
                "minClearanceMm": round(self.min_table_clearance, 1) if self.jobs else 0.0,
                "frames": n,
            },
            "dual": {
                "singleArmS": single_s,
                "speedup": round(single_s / duration, 2) if single_s and done else None,
                "minArmGapMm": round(min_gap * 1000, 1),
                "clearanceMm": CLEARANCE * 1000,
                "arms": [
                    {"id": x.id, "blocks": x.blocks, "busyS": round(x.busy_frames / FPS, 1), "waitS": round(x.wait_s, 1),
                     "waits": x.waits, "utilization": round(x.busy_frames / max(1, n - 1), 3)}
                    for x in arms
                ],
            },
        }
        if extra:
            out.update(extra)
        return out


def plan_dual(lengths, objects: list[dict], bins: list[dict], seed: int | None = None, compare: bool = True) -> dict:
    s = Schedule(lengths)
    world = {o["id"]: dict(o) for o in objects}
    remaining = [o["id"] for o in objects if not o.get("binId")]
    bins_by_color = {b["color"]: b for b in bins}
    if not remaining:
        s.events.append({"t": 0.0, "text": "Every block is already sorted. Press Reset scene to start again.", "kind": "warn"})

    while remaining:
        arm = min(s.arms, key=lambda a: a.end_index)
        # The free arm takes the remaining block closest to its own base.
        bid = min(remaining, key=lambda i: math.hypot(world[i]["x"], (world[i].get("z") or 0.0) - arm.base_z))
        remaining.remove(bid)
        obj = world[bid]
        bin_ = bins_by_color.get(obj["color"])
        if bin_ is None:
            s.warn(arm, f"Skipped {obj['color']} block: no bin.")
            continue

        park = PARK_A if arm.id == "A" else PARK_B
        job = plan_commands(
            lengths, arm.q, [_to_local(o, arm.base_z) for o in world.values()], [_to_local(b, arm.base_z) for b in bins],
            [{"blockId": bid, "binId": bin_["id"]}], faults="none", seed=(seed or 0) * 100 + s.jobs, park_q=park,
            tuck_u=TUCK_U,
        )
        if job["stats"]["sorted"] != 1:
            warn = next((e["text"] for e in job["events"] if e["kind"] == "warn"), "could not plan it")
            s.warn(arm, warn)
            continue
        if not s.commit(arm, job, f"{obj['color']} block"):
            continue
        pl = job["placements"][0]
        world[bid].update({"x": pl["x"], "z": round(pl["z"] + arm.base_z, 4), "binId": bin_["id"]})

    single_s = None
    if compare and s.jobs:
        # The same blocks sorted by arm A alone, for the speed-up figure.
        solo = plan_sort(lengths, PARK_A, objects, bins, faults="none", seed=seed)
        single_s = solo["stats"]["durationS"] if solo["stats"]["sorted"] else None
    return s.finish("sorted", single_s)


__all__ = ["plan_dual", "dual_scene", "Schedule", "Arm", "ARM_B_Z", "LINE_Z", "PARK_A", "PARK_B", "CLEARANCE", "TUCK_U"]
