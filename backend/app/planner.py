"""Task and motion planning for the color-sorting demo, with failure recovery.

For every block whose color has a matching bin the arm runs a classic
pick-and-place sequence:

    move above block -> descend -> close gripper -> lift
    -> move above bin slot -> descend -> open gripper -> lift

Big moves are joint-space moves with a quintic time profile (smooth start
and stop: zero velocity and acceleration at both ends). Approach and lift
are straight vertical moves, solved with analytic IK every frame, so the
gripper goes straight down and up instead of swinging sideways.

The blocks sit on the +x side and the bins on the -x side. The waist turns
the arm around between them while the elbow stays bent the same way, so the
arm never flips over the top.

Failures and recovery
---------------------
Real grippers drop things. The planner can inject two faults and plans the
recovery for each:

* slip:  the block slides out of the fingers while lifting. The grip-force
         sensor sees the load vanish, the block falls back onto the table,
         and the arm re-grasps with more force.
* drop:  the block falls out mid-swing and lands somewhere new on the table.
         The arm brakes to a stop, the camera relocates the block, and the
         arm turns its waist to wherever it landed, picks it up again and
         finishes the job.

Every block's whereabouts over time are recorded as a timeline of keyframes
(on the table, held, falling, in a bin) so the front end can draw it.
"""
from __future__ import annotations

import copy
import math
import random

import numpy as np

from .kinematics import BASE, HOME_Q4, fk, fk_world, ik_analytic

FPS = 60
GRAVITY = 9.81
TOOL_DOWN = -math.pi / 2
MAX_JOINT_SPEED = 2.5  # rad/s peak (about 143 deg/s)
LINEAR_SPEED = 0.3  # m/s peak
WRIST_TURN_SPEED = 2.2  # rad/s peak for in-place tool turns
ROLL_SPEED = 3.0  # rad/s peak for the wrist roll joint


def _wrap(a: float, period: float) -> float:
    """Wrap an angle into (-period/2, period/2]."""
    return -((-a + period / 2) % period - period / 2)


def roll_to_close_along(direction: float, waist: float, period: float = math.pi) -> float:
    """Wrist roll that makes a downward-pointing gripper close its fingers
    along a horizontal direction (angle in the x-z plane, same convention as
    the waist). With roll 0 the fingers close across the arm's plane, i.e.
    along waist + 90 degrees; rolling turns that direction by the roll angle.
    Use period pi for an edge (either way across it is fine) and pi/2 for a
    square block (any pair of faces will do), so the wrist turns as little
    as possible."""
    return _wrap(direction - waist - math.pi / 2, period)
QUINTIC_PEAK = 1.875  # peak / average speed for the quintic profile
CLEARANCE = 0.08  # m above the block (or bin wall) when approaching
GRASP_FRACTION = 0.3  # fingertips close at 30% of the block height
BIN_WALL = 0.05
BIN_WIDTH = 0.17
SLOT_OFFSETS = (-0.05, 0.0, 0.05)
BLOCK_SIZE = 0.04

COLORS = ("red", "blue", "green")
FAULT_MODES = ("none", "demo", "random")


def quintic(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Unreachable(Exception):
    pass


def pose_at(p3, phi: float, lengths, prefer=None):
    """Full 4-joint pose that puts the tool point at 3D point p3 = (x, y, z)
    with in-plane tool angle phi (-pi/2 = pointing straight down, 0 = pointing
    away from the base, pi = pointing back toward the base). The waist turns
    toward the point; the elbow stays up. Returns None when out of reach."""
    x, y, z = (float(v) + 0.0 for v in p3)  # + 0.0 turns -0.0 into 0.0
    u = math.hypot(x, z)
    if u < 0.02:
        waist = float(prefer[0]) if prefer is not None else 0.0
    else:
        waist = math.atan2(z, x)
        if prefer is not None:  # stay on the same side of the +/-180 seam
            waist += 2 * math.pi * round((float(prefer[0]) - waist) / (2 * math.pi))
    res = ik_analytic((u, y), phi, lengths, elbow="up",
                      prefer=None if prefer is None else np.asarray(prefer[1:]))
    return np.array([waist, *res.q]) if res.reachable else None


def pose_down(p3, lengths, prefer=None):
    """pose_at with the gripper pointing straight down."""
    return pose_at(p3, TOOL_DOWN, lengths, prefer)


# Weights of the things the robot carries, as the wrist force sensor sees them (N).
WEIGHT_BLOCK = 0.6
GRIP_FORCE_BLOCK = 10.0  # N squeeze on a block
SLOW_APPROACH = 0.03  # m: the last few cm before contact are done slowly
CREEP_SPEED = 0.04  # m/s for that last bit


class TrajectoryBuilder:
    def __init__(self, lengths, q0, seed: int | None = None):
        # Sensor model: the wrist force/torque sensor (vertical load, N) and
        # the gripper's force sensor (N), sampled every frame with a little noise.
        self.rng = random.Random(1 if seed is None else seed)
        self.sensors: dict[str, list[float]] = {"fz": [], "grip": []}
        self.perception: list[dict] = []
        self.held_weight = 0.0
        self.grip_force = 0.0
        self.contact = 0.0  # short force spike when something touches down
        self.lengths = lengths
        self.q = np.asarray(q0, dtype=float)
        self.grip = 1.0  # 1 open, 0 closed
        self.roll = 0.0  # wrist roll (rad): turns the gripper about its own axis
        self.held_yaw0 = 0.0
        self.holding: str | None = None
        self.t = 0.0
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self.timeline: dict[str, list[dict]] = {}
        self.path_length = 0.0
        self.max_joint_speed = 0.0
        self.min_clearance = math.inf
        self._emit()

    # -- bookkeeping ---------------------------------------------------
    def snapshot(self):
        return (self.q.copy(), self.t, self.grip, self.holding, len(self.frames), len(self.events),
                copy.deepcopy(self.timeline), self.path_length, len(self.perception),
                self.held_weight, self.grip_force, self.roll, self.held_yaw0)

    def restore(self, s):
        (self.q, self.t, self.grip, self.holding, n_frames, n_events, self.timeline, self.path_length,
         n_perc, self.held_weight, self.grip_force, self.roll, self.held_yaw0) = s
        del self.frames[n_frames:]
        del self.events[n_events:]
        del self.perception[n_perc:]
        for v in self.sensors.values():
            del v[n_frames:]

    def _emit(self, q=None):
        if q is not None:
            if self.frames:
                dq = np.abs(np.asarray(q) - self.q)
                self.max_joint_speed = max(self.max_joint_speed, float(np.max(dq)) * FPS)
                prev_tool = fk_world(self.q, self.lengths)[0][-1]
                self.path_length += float(np.linalg.norm(fk_world(q, self.lengths)[0][-1] - prev_tool))
            self.q = np.asarray(q, dtype=float)
            self.t += 1 / FPS
        pts, _ = fk_world(self.q, self.lengths)
        self.min_clearance = min(self.min_clearance, float(np.min(pts[1:, 1])))
        self.frames.append({
            "t": round(self.t, 4),
            "q": [round(float(a), 5) for a in self.q],
            "grip": round(self.grip, 3),
            "roll": round(self.roll, 5),
            "holding": self.holding,
        })
        self.contact *= 0.8
        self.sensors["fz"].append(round(self.held_weight + self.contact + self.rng.gauss(0.0, 0.03), 3))
        noise = self.rng.gauss(0.0, 0.08) if self.grip_force > 0 else 0.0
        self.sensors["grip"].append(round(max(0.0, self.grip_force + noise), 3))

    def event(self, text: str, kind: str = "info"):
        self.events.append({"t": round(self.t, 3), "text": text, "kind": kind})

    def keyframe(self, block_id: str, mode: str, pos=None, **extra):
        kf = {"t": round(self.t, 4), "mode": mode}
        if pos is not None:
            kf["pos"] = [round(float(v), 4) for v in pos]
        kf.update(extra)
        self.timeline.setdefault(block_id, []).append(kf)

    def tool(self) -> np.ndarray:
        return fk_world(self.q, self.lengths)[0][-1]

    # -- motions -------------------------------------------------------
    def joint_move(self, q_to, min_dur: float = 0.5, stop_when=None, roll_to: float | None = None) -> bool:
        """Smooth joint-space move. If `stop_when(q, progress)` becomes true part way,
        stop there (without braking) and return True."""
        q_from = self.q.copy()
        dq = np.asarray(q_to) - q_from
        r0 = self.roll
        dr = 0.0 if roll_to is None else roll_to - r0
        dur = max(min_dur, float(np.max(np.abs(dq))) / MAX_JOINT_SPEED * QUINTIC_PEAK, abs(dr) / ROLL_SPEED * QUINTIC_PEAK)
        n = max(1, math.ceil(dur * FPS))
        for i in range(1, n + 1):
            self.roll = r0 + quintic(i / n) * dr
            self._emit(q_from + quintic(i / n) * dq)
            if stop_when is not None and stop_when(self.q, i / n):
                return True
        return False

    def twist(self, roll_to: float, min_dur: float = 0.25):
        """Turn only the wrist roll joint."""
        r0, dr = self.roll, roll_to - self.roll
        n = max(1, math.ceil(max(min_dur, abs(dr) / ROLL_SPEED * QUINTIC_PEAK) * FPS))
        for i in range(1, n + 1):
            self.roll = r0 + quintic(i / n) * dr
            self._emit(self.q)

    def transfer(self, q_to, tuck_u: float | None = None, tuck_y: float = 0.2, roll_to: float | None = None):
        """Joint move to q_to. With tuck_u, a big waist turn is done with the
        arm pulled in close to the base (tool tuck_u from the waist axis),
        so it sweeps a small circle instead of a long arm through the space
        around it. That's what keeps two arms out of each other's way."""
        q_to = np.asarray(q_to, dtype=float)
        if tuck_u is None or abs(q_to[0] - self.q[0]) < math.radians(50):
            self.joint_move(q_to, roll_to=roll_to)
            return
        res = ik_analytic((tuck_u, tuck_y), TOOL_DOWN, self.lengths, prefer=self.q[1:])
        if not res.reachable:
            self.joint_move(q_to, roll_to=roll_to)
            return
        self.joint_move(np.array([self.q[0], *res.q]), min_dur=0.3)
        self.joint_move(np.array([q_to[0], *res.q]), min_dur=0.3)
        self.joint_move(q_to, min_dur=0.3, roll_to=roll_to)

    def brake(self, dur: float = 0.35):
        """Emergency stop: keep the last joint velocity and ramp it to zero."""
        if len(self.frames) < 2:
            return
        v = np.asarray(self.frames[-1]["q"]) - np.asarray(self.frames[-2]["q"])
        n = max(1, math.ceil(dur * FPS))
        for k in range(1, n + 1):
            self._emit(self.q + v * (1 - k / n))

    def vertical_move(self, y_to: float, min_dur: float = 0.3):
        """Move the tool straight up or down, gripper pointing down."""
        pts, _ = fk(self.q[1:], self.lengths)
        u, y_from = pts[-1]
        dist = abs(y_to - y_from)
        dur = max(min_dur, dist / LINEAR_SPEED * QUINTIC_PEAK)
        n = max(1, math.ceil(dur * FPS))
        waist = float(self.q[0])
        for i in range(1, n + 1):
            y = y_from + quintic(i / n) * (y_to - y_from)
            res = ik_analytic((u, y), TOOL_DOWN, self.lengths, prefer=self.q[1:])
            if not res.reachable:
                raise Unreachable(f"height {y:.2f} m")
            self._emit(np.array([waist, *res.q]))

    def plane_move(self, u_to: float, y_to: float, phi_to: float, phi_from: float | None = None,
                   min_dur: float = 0.35, speed: float = LINEAR_SPEED):
        """Straight-line tool move inside the arm's current plane (waist fixed),
        optionally turning the tool angle on the way. With the same start and
        end point this is a pure wrist turn, e.g. flipping a cup over.
        Angles are interpolated as given (not wrapped), so 0 -> pi turns
        through pointing straight up."""
        pts, phi_now = fk(self.q[1:], self.lengths)
        u0, y0 = pts[-1]
        phi0 = phi_now if phi_from is None else phi_from
        dist = math.hypot(u_to - u0, y_to - y0)
        dur = max(min_dur, dist / speed * QUINTIC_PEAK, abs(phi_to - phi0) / WRIST_TURN_SPEED * QUINTIC_PEAK)
        n = max(1, math.ceil(dur * FPS))
        waist = float(self.q[0])
        for i in range(1, n + 1):
            s = quintic(i / n)
            u, y, phi = u0 + s * (u_to - u0), y0 + s * (y_to - y0), phi0 + s * (phi_to - phi0)
            res = ik_analytic((u, y), phi, self.lengths, prefer=self.q[1:])
            if not res.reachable:
                raise Unreachable(f"u={u:.2f}, y={y:.2f} m")
            self._emit(np.array([waist, *res.q]))

    def path3d(self, points, phi: float = TOOL_DOWN, speed: float = LINEAR_SPEED, min_dur: float = 0.5,
               dur: float | None = None):
        """Move the tool along a 3D polyline (smooth start and stop), turning
        the waist as needed. Used for the cloth-folding arcs. Pass `dur` to
        fix the timing exactly (two arms moving in lockstep)."""
        pts = np.asarray(points, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if dur is None:
            dur = max(min_dur, total / speed * QUINTIC_PEAK)
        n = max(1, math.ceil(dur * FPS))
        for i in range(1, n + 1):
            d = quintic(i / n) * total
            k = int(np.searchsorted(cum, d, side="right") - 1)
            k = min(max(k, 0), len(seg) - 1)
            f = 0.0 if seg[k] == 0 else (d - cum[k]) / seg[k]
            p = pts[k] + f * (pts[k + 1] - pts[k])
            q = pose_at(p, phi, self.lengths, prefer=self.q)
            if q is None:
                raise Unreachable(f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
            self._emit(q)

    # -- sensing and careful contact -------------------------------------
    def phi(self) -> float:
        return fk(self.q[1:], self.lengths)[1]

    def look(self, obj_id: str, label: str, center, size, yaw: float = 0.0, height: float = 0.2) -> np.ndarray:
        """Hover the wrist camera over an object and scan it. Returns the
        camera's estimate of where it is, which is a few mm off, like real
        vision."""
        c = np.asarray(center, dtype=float)
        q = pose_at([c[0], c[1] + height, c[2]], TOOL_DOWN, self.lengths, prefer=self.q)
        if q is not None:
            self.joint_move(q)
        t0 = self.t
        ang = self.rng.uniform(0, 2 * math.pi)
        mag = self.rng.uniform(0.003, 0.009)
        est = c + np.array([mag * math.cos(ang), 0.0, mag * math.sin(ang)])
        conf = round(self.rng.uniform(0.9, 0.99), 2)
        self.event(f"Camera: {label} found ({conf * 100:.0f}% sure), position uncertain by about {mag * 1000:.0f} mm", "sense")
        self.pause(0.6)
        self.perception.append({
            "tStart": round(t0, 4), "tEnd": round(self.t + 0.8, 4), "id": obj_id, "label": label,
            "confidence": conf, "center": [round(float(v), 4) for v in est],
            "size": [round(float(v), 4) for v in size], "yaw": round(float(yaw), 4), "phase": "scan",
        })
        return est

    def align(self, obj_id: str, label: str, true_center, est, size, yaw: float = 0.0):
        """Close-up look from just above the object, then shift sideways to
        fix the camera's earlier error."""
        t0 = self.t
        d = np.asarray(true_center, dtype=float) - np.asarray(est, dtype=float)
        d[1] = 0.0
        self.event(f"Close-up camera: {label} is {np.linalg.norm(d) * 1000:.0f} mm off the first estimate, correcting", "sense")
        self.pause(0.3)
        self.perception.append({
            "tStart": round(t0, 4), "tEnd": round(self.t + 0.4, 4), "id": obj_id, "label": label,
            "confidence": 0.99, "center": [round(float(v), 4) for v in true_center],
            "size": [round(float(v), 4) for v in size], "yaw": round(float(yaw), 4), "phase": "align",
        })
        tool = self.tool()
        self.path3d([tool, tool + d], phi=self.phi(), speed=0.05, min_dur=0.35)

    def approach_down(self, y_to: float, slow: float = SLOW_APPROACH):
        """Go down fast through open space, then creep the last few cm."""
        tool, phi = self.tool(), self.phi()
        if tool[1] - y_to > slow + 0.004:
            self.path3d([tool, [tool[0], y_to + slow, tool[2]]], phi=phi, min_dur=0.3)
        tool = self.tool()
        if abs(tool[1] - y_to) > 1e-4:
            self.path3d([tool, [tool[0], y_to, tool[2]]], phi=phi, speed=CREEP_SPEED, min_dur=0.35)

    def lift_off(self, y_to: float, slow: float = SLOW_APPROACH):
        """Lift gently for the first few cm (so nothing jerks or snags), then fast."""
        tool, phi = self.tool(), self.phi()
        first = min(y_to, tool[1] + slow)
        self.path3d([tool, [tool[0], first, tool[2]]], phi=phi, speed=CREEP_SPEED, min_dur=0.3)
        if y_to - first > 1e-4:
            tool = self.tool()
            self.path3d([tool, [tool[0], y_to, tool[2]]], phi=phi, min_dur=0.3)

    def touch_down(self, y_to: float, weight: float):
        """Lower a held object onto a surface. The wrist force sensor feels
        the contact (a spike) and the load disappears as the surface takes it."""
        self.approach_down(y_to)
        self.contact = 2.0 + weight
        self.held_weight = 0.0
        self.pause(0.12)

    def squeeze(self, g_contact: float, force: float):
        """Close fast until near the object, slowly until the fingers touch,
        then build up grip force."""
        near = g_contact + 0.25
        if self.grip > near:
            self.gripper(near, dur=0.18)
        self.gripper(g_contact, dur=0.22)
        n = 12
        for i in range(1, n + 1):
            self.grip_force = force * i / n
            self._emit(self.q)

    def let_go(self):
        """Release grip force, then open the fingers."""
        self.held_weight = 0.0
        f0, n = self.grip_force, 6
        for i in range(1, n + 1):
            self.grip_force = f0 * (1 - i / n)
            self._emit(self.q)
        self.gripper(1.0)

    def verify(self, label: str, where: str):
        """Quick look to confirm the object ended up where it should."""
        self.pause(0.25)
        self.event(f"Check: camera confirms the {label} is in {where}", "sense")

    def held_yaw(self) -> float:
        """Yaw of the held object now: it turns with the waist and the wrist roll."""
        return float(self.q[0]) + self.roll + self.held_yaw0

    def grab(self, obj_id: str, center, tilt: float = 0.0, weight: float | None = None, yaw: float | None = None):
        """Start holding an object. Records where the object sits in the
        tool's frame so the front end can carry it along exactly, including
        any wrist turn (a cup flipping over)."""
        self.holding = obj_id
        if weight is not None:
            self.held_weight = weight
        w = float(self.q[0])
        pts, phi = fk(self.q[1:], self.lengths)
        ut, yt = pts[-1]
        cx, cy, cz = (float(v) for v in center)
        uc = cx * math.cos(w) + cz * math.sin(w)
        nc = -cx * math.sin(w) + cz * math.cos(w)
        du, dy = uc - ut, cy - yt
        a = du * math.cos(phi) + dy * math.sin(phi)
        b = -du * math.sin(phi) + dy * math.cos(phi)
        self.held_yaw0 = 0.0 if yaw is None else float(yaw) - w - self.roll
        self.keyframe(obj_id, "held", local=[round(float(a), 5), round(float(b), 5), round(float(nc), 5) + 0.0],
                      tilt0=round(float(tilt - phi), 5), yaw0=round(self.held_yaw0, 5))

    def put(self, obj_id: str, center, yaw: float = 0.0, tilt: float = 0.0, mode: str = "table", **extra):
        """Let go of an object, leaving it at rest at `center`."""
        self.holding = None
        self.keyframe(obj_id, mode, center, yaw=round(float(yaw), 5), tilt=round(float(tilt), 5), **extra)

    def gripper(self, to: float, dur: float = 0.25):
        start = self.grip
        n = max(1, math.ceil(dur * FPS))
        for i in range(1, n + 1):
            self.grip = start + (to - start) * (i / n)
            self._emit(self.q)

    def pause(self, dur: float):
        for _ in range(max(1, math.ceil(dur * FPS))):
            self._emit(self.q)

    # -- block helpers -------------------------------------------------
    def held_block_center(self, size: float) -> np.ndarray:
        return self.tool() + np.array([0.0, size * (0.5 - GRASP_FRACTION), 0.0])

    def release_falling(self, block_id: str, size: float) -> tuple[np.ndarray, float]:
        """The held block falls straight down onto the table. Returns where it
        lands and how long the fall takes."""
        start = self.held_block_center(size)
        land = np.array([start[0], size / 2, start[2]])
        fall_time = math.sqrt(2 * max(0.0, start[1] - land[1]) / GRAVITY)
        self.holding = None
        self.held_weight = 0.0  # the load cell suddenly reads nothing...
        self.grip_force = 0.0  # ...and the fingers close on air
        self.keyframe(block_id, "falling", start, to=[round(float(v), 4) for v in land],
                      yaw=round(self.held_yaw(), 4))
        return land, fall_time


def plan_sort(lengths, q0, objects: list[dict], bins: list[dict], faults: str = "demo",
              seed: int | None = None) -> dict:
    """Plan a trajectory that sorts every block into the bin of its color.

    faults: "none" for a clean run, "demo" for a fixed showcase (one slip and
    one drop), or "random" (each pick has a chance of either fault).
    """
    bins_by_color = {bin_["color"]: bin_ for bin_ in bins}
    todo = sorted((o for o in objects if not o.get("binId")), key=lambda o: o["x"])
    commands = []
    for o in todo:
        bin_ = bins_by_color.get(o["color"])
        commands.append({
            "blockId": o["id"],
            "binId": bin_["id"] if bin_ else None,
            "reason": None if bin_ else f"no {o['color']} bin",
        })
    out = plan_commands(lengths, q0, objects, bins, commands, faults=faults, seed=seed, return_home=True)
    if not todo:
        out["events"].insert(0, {"t": 0.0, "text": "All blocks are already sorted. Shuffle the blocks to plan a new run.", "kind": "warn"})
    return out


# ---- Command execution -------------------------------------------------------

TABLE_X = 0.76
TABLE_Z = 0.42
MIN_BLOCK_GAP = 0.065  # centre-to-centre, so the open fingers fit between blocks
BASE_KEEP_OUT = 0.13


def table_spot_problem(x: float, z: float, objects: list[dict], bins: list[dict], moving_id: str | None = None) -> str | None:
    """Why a block can't be put down at (x, z) on the table, or None if it can."""
    if abs(x) > TABLE_X or abs(z) > TABLE_Z:
        return "that spot is off the table"
    if math.hypot(x, z) < BASE_KEEP_OUT:
        return "that spot is on the robot's base"
    for bin_ in bins:
        if abs(x - bin_["x"]) < bin_["width"] / 2 + 0.03 and abs(z) < 0.1:
            return f"that spot is on the {bin_['color']} bin (pick the bin instead)"
    for o in objects:
        if o["id"] != moving_id and not o.get("binId") and math.hypot(x - o["x"], z - (o.get("z") or 0.0)) < MIN_BLOCK_GAP:
            return f"that spot is taken by the {o['color']} block"
    return None


def free_slot(bin_: dict, objects: list[dict], moving_id: str | None = None) -> int | None:
    taken = {
        i for i, off in enumerate(SLOT_OFFSETS) for o in objects
        if o.get("binId") == bin_["id"] and o["id"] != moving_id and abs(o["x"] - (bin_["x"] + off)) < 0.01
    }
    return next((i for i in range(len(SLOT_OFFSETS)) if i not in taken), None)


def plan_commands(lengths, q0, objects: list[dict], bins: list[dict], commands: list[dict],
                  faults: str = "none", seed: int | None = None, fault_offset: int = 0,
                  return_home: bool = False, park_q=None, tuck_u: float | None = None) -> dict:
    """Run a list of pick-and-place commands in order.

    Each command is {"blockId": ..., "binId": ...} to drop the block in a bin,
    or {"blockId": ..., "x": ..., "z": ...} to set it down on the table.
    Blocks are tracked as they move, so later commands see the new layout.
    `fault_offset` is how many commands already ran this session, so the
    "demo" fault pattern (2nd command slips, 4th drops) carries across calls.
    """
    if faults not in FAULT_MODES:
        raise ValueError(f"faults must be one of {FAULT_MODES}")
    q0 = np.asarray(q0, dtype=float)
    if q0.shape == (3,):  # older clients sent only the 3 arm joints
        q0 = np.array([0.0, *q0])
    b = TrajectoryBuilder(lengths, q0, seed=seed)
    rng = random.Random(seed)
    world = {o["id"]: dict(o) for o in objects}
    bins_by_id = {bin_["id"]: bin_ for bin_ in bins}

    placements = []
    done_count, skipped = 0, 0
    counts = {"slips": 0, "drops": 0, "recovered": 0}

    for index, cmd in enumerate(commands):
        obj = world.get(cmd.get("blockId", ""))
        if obj is None:
            b.event(f"Skipped command {index + 1}: unknown block.", "warn")
            skipped += 1
            continue
        label = f"{obj['color']} block"
        h = obj.get("size", BLOCK_SIZE)
        others = list(world.values())

        # Work out where it goes.
        if cmd.get("reason"):
            b.event(f"Skipped {label}: {cmd['reason']}.", "warn")
            skipped += 1
            continue
        if cmd.get("binId"):
            bin_ = bins_by_id.get(cmd["binId"])
            if bin_ is None:
                b.event(f"Skipped {label}: unknown bin.", "warn")
                skipped += 1
                continue
            if obj.get("binId") == bin_["id"]:
                b.event(f"The {label} is already in the {bin_['color']} bin.", "warn")
                skipped += 1
                continue
            slot = free_slot(bin_, others, obj["id"])
            if slot is None:
                b.event(f"Skipped {label}: the {bin_['color']} bin is full.", "warn")
                skipped += 1
                continue
            dest = np.array([bin_["x"] + SLOT_OFFSETS[slot], bin_.get("z", 0.0)])
            dest_name = f"the {bin_['color']} bin"
            approach_h = max(BIN_WALL, h) + h + CLEARANCE
            dest_bin = bin_["id"]
        else:
            x, z = float(cmd.get("x", 0.0)), float(cmd.get("z", 0.0))
            problem = table_spot_problem(x, z, others, bins, obj["id"])
            if problem:
                b.event(f"Skipped {label}: {problem}.", "warn")
                skipped += 1
                continue
            dest = np.array([x, z])
            dest_name = f"the table at x={x:.2f}, z={z:.2f} m"
            approach_h = h + CLEARANCE
            dest_bin = None

        pos = np.array([obj["x"], h / 2, obj.get("z") or 0.0])
        lift_h = (max(BIN_WALL, h) + h + CLEARANCE) if obj.get("binId") else h + CLEARANCE
        place = np.array([dest[0], h * GRASP_FRACTION, dest[1]])
        above_place = np.array([dest[0], approach_h, dest[1]])

        # Check every waypoint before moving so a bad command never leaves
        # the arm half way through.
        waypoints = ([pos[0], h * GRASP_FRACTION, pos[2]], [pos[0], lift_h, pos[2]], place, above_place)
        if any(pose_down(p, lengths) is None for p in waypoints):
            where = "it is" if pose_down(waypoints[0], lengths) is None else "the destination is"
            b.event(f"Skipped {label}: {where} outside the arm's reach.", "warn")
            skipped += 1
            continue

        # Which fault (if any) hits this command.
        fault = None
        n = fault_offset + index
        if faults == "demo":
            fault = {1: "slip", 3: "drop"}.get(n)
        elif faults == "random":
            r = rng.random()
            fault = "slip" if r < 0.2 else "drop" if r < 0.4 else None

        saved = b.snapshot()
        try:
            b.event(f"Pick {label} at x={pos[0]:.2f}, z={pos[2]:.2f} m", "pick")
            _pick(b, obj["id"], pos, h, label, slip=fault == "slip", counts=counts, lift_h=lift_h,
                  yaw=float(obj.get("yaw") or 0.0))

            b.event(f"Carry {label} to {dest_name}", "place")
            target = pose_down(above_place, lengths, prefer=b.q)
            # Turn the wrist on the way so the block lands square (faces lined up with the bin / table).
            square = _wrap(-(float(target[0]) + b.held_yaw0), math.pi / 2)
            if abs(square - b.roll) > math.radians(2):
                b.event(f"Wrist rolls {math.degrees(square - b.roll):+.0f}° on the way so the {label} lands square", "info")
            dropped = False
            if fault == "drop":
                start_waist = float(b.q[0])

                def drop_here(q, progress):
                    # Drop part way through the move, but only where it can land safely.
                    if progress < 0.35 or (abs(q[0] - start_waist) < math.radians(20) and progress < 0.6):
                        return False
                    tip = fk_world(q, lengths)[0][-1]
                    return table_spot_problem(float(tip[0]), float(tip[2]), list(world.values()), bins, obj["id"]) is None

                dropped = b.joint_move(target, stop_when=drop_here, roll_to=square)
            else:
                b.transfer(target, tuck_u, roll_to=square)
            if dropped:
                land, fall_time = b.release_falling(obj["id"], h)
                counts["drops"] += 1
                b.event(
                    f"FAULT: {label} dropped mid-move (gripper lost contact at waist "
                    f"{math.degrees(b.q[0]):.0f}°). Emergency stop.", "fault",
                )
                fall_kf = b.timeline[obj["id"]][-1]
                b.brake()
                if b.t < fall_kf["t"] + fall_time:
                    b.pause(fall_kf["t"] + fall_time - b.t)
                b.keyframe(obj["id"], "table", land, yaw=fall_kf.get("yaw", 0))
                b.pause(0.5)
                if pose_down([land[0], h + CLEARANCE, land[2]], lengths) is None:
                    raise Unreachable("the drop spot")
                b.event(f"RECOVER: re-planning the pick from where the {label} landed (x={land[0]:.2f}, z={land[2]:.2f} m).", "recover")
                _pick(b, obj["id"], land, h, label, slip=False, counts=counts, lift_h=h + CLEARANCE,
                      yaw=float(fall_kf.get("yaw", 0.0)))
                counts["recovered"] += 1
                b.event(f"Carry {label} to {dest_name}", "place")
                again = pose_down(above_place, lengths, prefer=b.q)
                b.joint_move(again, roll_to=_wrap(-(float(again[0]) + b.held_yaw0), math.pi / 2))

            b.touch_down(place[1], WEIGHT_BLOCK)
            b.let_go()
            b.holding = None
            final = [float(dest[0]), h / 2, float(dest[1])]
            yaw = b.held_yaw()
            b.keyframe(obj["id"], "bin" if dest_bin else "table", final, binId=dest_bin, yaw=round(yaw, 4))
            placements.append({"id": obj["id"], "binId": dest_bin, "t": round(b.t, 4),
                               "x": round(final[0], 4), "y": round(final[1], 4), "z": round(final[2], 4)})
            b.lift_off(above_place[1])
            b.verify(label, dest_name)
        except Unreachable as exc:
            b.restore(saved)
            b.event(f"Skipped {label}: path leaves the workspace near {exc}. Flagged for an operator.", "warn")
            skipped += 1
            continue
        obj.update({"x": float(dest[0]), "y": h / 2, "z": float(dest[1]), "binId": dest_bin})
        done_count += 1

    if return_home:
        b.event("Return home", "info")
        b.joint_move(HOME_Q4, roll_to=0.0)
    if park_q is not None:
        b.event("Clear the shared space (park)", "info")
        b.transfer(park_q, tuck_u, roll_to=0.0)
    b.pause(0.3)
    faults_total = counts["slips"] + counts["drops"]
    verb = "sorted" if return_home else "done"
    summary = f"Done: {done_count} {verb}, {skipped} skipped"
    if faults_total:
        summary += f", {counts['recovered']}/{faults_total} faults recovered"
    b.event(summary + ".", "done")

    warnings = []
    if b.min_clearance < 0:
        warnings.append("A joint dipped below the table top during the run.")
    return {
        "fps": FPS,
        "duration": round(b.t, 3),
        "frames": b.frames,
        "events": b.events,
        "timeline": b.timeline,
        "placements": placements,
        "warnings": warnings,
        "sensors": b.sensors,
        "perception": b.perception,
        "stats": {
            "sorted": done_count,
            "skipped": skipped,
            "slips": counts["slips"],
            "drops": counts["drops"],
            "recovered": counts["recovered"],
            "durationS": round(b.t, 2),
            "toolPathM": round(b.path_length, 3),
            "maxJointSpeedDegS": round(math.degrees(b.max_joint_speed), 1),
            "minClearanceMm": round(b.min_clearance * 1000, 1),
            "frames": len(b.frames),
        },
    }


def _pick(b: TrajectoryBuilder, block_id: str, pos, h: float, label: str, slip: bool, counts: dict,
          lift_h: float | None = None, yaw: float = 0.0):
    """Approach, grasp and lift one block. With slip=True the first grasp
    fails during the lift and the arm re-grasps with more force."""
    lengths = b.lengths
    lift_h = h + CLEARANCE if lift_h is None else lift_h
    grasp_y = h * GRASP_FRACTION
    center = [pos[0], h / 2, pos[2]]
    est_yaw = yaw + b.rng.gauss(0.0, math.radians(3))  # the camera's angle estimate is a little off too
    est = b.look(block_id, label, center, [h, h, h], yaw=est_yaw)
    q_above = pose_down([est[0], lift_h, est[2]], lengths, prefer=b.q)
    roll = roll_to_close_along(est_yaw, float(q_above[0]), math.pi / 2)
    b.event(f"Wrist rolls {math.degrees(roll):+.0f}° so the fingers line up with the {label}'s faces", "info")
    b.joint_move(q_above, roll_to=roll)
    b.approach_down(grasp_y + SLOW_APPROACH, slow=0.0)
    b.align(block_id, label, center, est, [h, h, h], yaw=yaw)
    fine = roll_to_close_along(yaw, float(b.q[0]), math.pi / 2)
    if abs(fine - b.roll) > math.radians(0.5):
        b.twist(fine, min_dur=0.2)
    b.approach_down(grasp_y)
    b.squeeze(0.0, GRIP_FORCE_BLOCK)
    b.grab(block_id, center, weight=WEIGHT_BLOCK, yaw=yaw)
    if slip:
        b.vertical_move(grasp_y + 0.035, min_dur=0.25)
        land, fall_time = b.release_falling(block_id, h)
        b.gripper(-0.6, dur=0.1)  # the fingers snap shut on nothing
        counts["slips"] += 1
        b.event(f"FAULT: {label} slipped while lifting (grip-force sensor reads 0 N).", "fault")
        b.pause(max(fall_time, 0.25))
        b.keyframe(block_id, "table", land, yaw=b.timeline[block_id][-1].get("yaw", 0))
        b.gripper(1.0)
        b.event(f"RECOVER: re-grasping the {label} with 40% more grip force.", "recover")
        land_yaw = b.timeline[block_id][-1].get("yaw", 0.0)
        b.twist(roll_to_close_along(land_yaw, float(b.q[0]), math.pi / 2), min_dur=0.2)
        b.approach_down(grasp_y)
        b.squeeze(0.0, GRIP_FORCE_BLOCK * 1.4)
        b.grab(block_id, [land[0], h / 2, land[2]], weight=WEIGHT_BLOCK, yaw=land_yaw)
        counts["recovered"] += 1
    b.lift_off(lift_h)


def default_bins() -> list[dict]:
    return [
        {"id": "bin-red", "color": "red", "x": -0.56, "width": BIN_WIDTH, "wall": BIN_WALL},
        {"id": "bin-blue", "color": "blue", "x": -0.37, "width": BIN_WIDTH, "wall": BIN_WALL},
        {"id": "bin-green", "color": "green", "x": -0.18, "width": BIN_WIDTH, "wall": BIN_WALL},
    ]


def random_scene(seed: int | None = None, count: int = 5, x_lo: float = 0.2, x_hi: float = 0.56,
                 z: float = 0.0) -> list[dict]:
    """Random blocks on the right side of the table, spaced so the open
    gripper fits between neighbours."""
    rng = random.Random(seed)
    xs: list[float] = []
    for _ in range(5000):
        if len(xs) == count:
            break
        x = round(rng.uniform(x_lo, x_hi), 3)
        if all(abs(x - other) >= 0.075 for other in xs):
            xs.append(x)
    if len(xs) < count:  # fall back to even spacing
        xs = [round(x_lo + i * (x_hi - x_lo) / max(1, count - 1), 3) for i in range(count)]
    colors = [COLORS[i % len(COLORS)] for i in range(count)]
    rng.shuffle(colors)
    yaw_rng = random.Random(None if seed is None else seed + 1000)  # separate, so positions don't change
    return [
        {"id": f"block-{i + 1}", "color": c, "x": x, "y": BLOCK_SIZE / 2, "z": z, "size": BLOCK_SIZE, "binId": None,
         "yaw": round(math.radians(yaw_rng.uniform(-40, 40)), 4)}
        for i, (x, c) in enumerate(zip(sorted(xs), colors))
    ]


def default_scene() -> list[dict]:
    return random_scene(seed=7)


__all__ = ["roll_to_close_along", "SLOW_APPROACH", "pose_at", "TrajectoryBuilder", "Unreachable", "plan_sort", "plan_commands", "table_spot_problem", "default_bins", "random_scene", "default_scene", "FPS", "BASE", "FAULT_MODES"]
