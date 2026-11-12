"""4-joint robot arm: a rotating waist plus a 3-joint arm in a vertical plane.

Conventions
-----------
* World frame: x to the right, y up, z toward the viewer, metres. The table
  top is y = 0 and the waist axis is the vertical line x = z = 0.
* q = [waist, shoulder, elbow, wrist] in radians.
  - The waist (yaw) turns the whole arm plane about the vertical axis.
    0 faces +x (the blocks), pi faces -x (the bins).
  - Shoulder, elbow and wrist work inside that plane, in plane coordinates
    (u, y): u is the horizontal distance out from the waist axis. Each angle
    is relative to the previous link; the shoulder is measured from +u.
* World point of a plane point (u, y): x = u cos(waist), z = u sin(waist).
* The tool point is the centre of the gripper fingertips. The tool angle
  phi = q1 + q2 + q3 is the direction the gripper points in the plane.

Having the waist means the arm turns around to reach the other side instead
of flipping its elbow over the top. The planar functions below (fk, jacobian,
ik_*) take the 3 in-plane joints; the *_world helpers add the waist.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

BASE = np.array([0.0, 0.12])
DEFAULT_LENGTHS = (0.34, 0.30, 0.12)
# Joint limits in degrees: [min, max] for shoulder, elbow, wrist.
LIMITS_DEG = np.array([[-10.0, 190.0], [-160.0, 160.0], [-160.0, 160.0]])
LIMITS = np.radians(LIMITS_DEG)
HOME_Q = np.radians([95.0, -115.0, -60.0])
# Extra start poses for the Jacobian solver when the first try gets stuck.
RESTART_SEEDS = tuple(np.radians(s) for s in (
    [95.0, -115.0, -60.0], [45.0, 60.0, 60.0], [135.0, -60.0, -60.0],
    [20.0, -90.0, 120.0], [160.0, 90.0, -120.0], [90.0, 0.0, 0.0],
))

METHODS = ("analytic", "dls", "ccd")


def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def normalize(q: np.ndarray) -> np.ndarray:
    """Put angles in the range the joint limits are expressed in.

    The shoulder lives in [-90, 270) degrees so that reaching to the left
    (about 180 degrees) does not jump across the +/-180 seam.
    """
    q = np.array([wrap(float(a)) for a in q])
    if q[0] < -math.pi / 2:
        q[0] += 2 * math.pi
    return q


def clamp(q: np.ndarray) -> np.ndarray:
    return np.clip(q, LIMITS[:, 0], LIMITS[:, 1])


def within_limits(q: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.all(q >= LIMITS[:, 0] - tol) and np.all(q <= LIMITS[:, 1] + tol))


def fk(q: np.ndarray, lengths) -> tuple[np.ndarray, float]:
    """Joint positions (base, elbow, wrist, tool) as a (4, 2) array, and the tool angle."""
    pts = [BASE.copy()]
    angle = 0.0
    p = BASE.copy()
    for qi, li in zip(q, lengths):
        angle += float(qi)
        p = p + li * np.array([math.cos(angle), math.sin(angle)])
        pts.append(p)
    return np.array(pts), angle


def jacobian(q: np.ndarray, lengths) -> np.ndarray:
    """3x3 Jacobian mapping joint velocities to (x_dot, y_dot, phi_dot)."""
    pts, _ = fk(q, lengths)
    tool = pts[-1]
    J = np.zeros((3, 3))
    for i in range(3):
        d = tool - pts[i]
        J[0, i] = -d[1]
        J[1, i] = d[0]
        J[2, i] = 1.0
    return J


@dataclass
class IKResult:
    q: np.ndarray
    method: str
    reachable: bool
    within_limits: bool
    iterations: int
    position_error: float  # metres
    angle_error: float  # radians (0 when the tool angle was not requested)
    solve_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self, lengths) -> dict:
        pts, phi = fk(self.q, lengths)
        return {
            "q": [float(a) for a in self.q],
            "qDeg": [round(math.degrees(a), 2) for a in self.q],
            "joints": pts.round(5).tolist(),
            "tool": {"x": float(pts[-1][0]), "y": float(pts[-1][1]), "phiDeg": round(math.degrees(wrap(phi)), 2)},
            "method": self.method,
            "reachable": self.reachable,
            "withinLimits": self.within_limits,
            "iterations": self.iterations,
            "positionErrorMm": round(self.position_error * 1000, 3),
            "angleErrorDeg": round(math.degrees(self.angle_error), 3),
            "solveMs": round(self.solve_ms, 3),
            "notes": self.notes,
        }


def _errors(q, lengths, target, phi):
    pts, a = fk(q, lengths)
    pos = float(np.linalg.norm(np.asarray(target) - pts[-1]))
    ang = abs(wrap(phi - a)) if phi is not None else 0.0
    return pos, ang


def ik_analytic(target, phi: float, lengths, elbow: str = "up", prefer: np.ndarray | None = None) -> IKResult:
    """Closed-form IK for a planar 3R arm with a fixed tool angle.

    Subtract the last link to get the wrist point, then solve the 2-link
    problem with the law of cosines. There are two answers (elbow up and
    elbow down); pick one that respects the joint limits, then either the
    one closest to `prefer` (for smooth paths) or the requested elbow.
    """
    L1, L2, L3 = lengths
    w = np.asarray(target, dtype=float) - L3 * np.array([math.cos(phi), math.sin(phi)]) - BASE
    c2 = (float(w @ w) - L1**2 - L2**2) / (2 * L1 * L2)
    reachable = -1.0 <= c2 <= 1.0
    c2 = max(-1.0, min(1.0, c2))

    candidates = []
    for sign in (1.0, -1.0):
        q2 = sign * math.acos(c2)
        q1 = math.atan2(w[1], w[0]) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
        q = normalize(np.array([q1, q2, phi - q1 - q2]))
        elbow_y = fk(q, lengths)[0][1][1]
        candidates.append((q, elbow_y))

    def score(c):
        q, elbow_y = c
        ok = within_limits(q)
        if prefer is not None:
            return (ok, -float(np.max(np.abs(q - prefer))))
        return (ok, elbow_y if elbow == "up" else -elbow_y)

    q = max(candidates, key=score)[0]
    notes = []
    ok_limits = within_limits(q)
    if not ok_limits:
        notes.append("No solution inside the joint limits; clamped to the nearest limit.")
        q = clamp(q)
    if not reachable:
        notes.append("Target is outside the workspace for this tool angle; arm stretches toward it.")
    pos, ang = _errors(q, lengths, target, phi)
    return IKResult(q, "analytic", reachable and ok_limits, ok_limits, 1, pos, ang, notes=notes)


def ik_dls(target, lengths, q0, phi: float | None = None, damping: float = 0.06,
           max_iter: int = 300, tol: float = 1e-4, orient_weight: float = 0.25) -> IKResult:
    """Damped least squares (Levenberg-Marquardt) Jacobian IK.

    dq = J^T (J J^T + lambda^2 I)^-1 e. Damping keeps it stable near
    singularities (arm fully stretched) and when the target is out of reach.
    The tool angle is optional; when given it is a weighted third task row.
    """
    target = np.asarray(target, dtype=float)
    q = clamp(normalize(np.asarray(q0, dtype=float)))
    it = 0
    for it in range(1, max_iter + 1):
        pts, a = fk(q, lengths)
        e = np.array([*(target - pts[-1]), 0.0])
        J = jacobian(q, lengths)
        if phi is None:
            e, J = e[:2], J[:2]
            done = np.linalg.norm(e) < tol
        else:
            e[2] = orient_weight * wrap(phi - a)
            J[2] *= orient_weight
            done = np.linalg.norm(e[:2]) < tol and abs(e[2]) < 1e-3 * orient_weight
        if done:
            break
        # Joints sitting on a limit and being pushed further out are frozen
        # (their Jacobian column is zeroed) so the others take up the motion.
        active = np.ones(3, dtype=bool)
        for _ in range(3):
            Ja = J * active
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + damping**2 * np.eye(len(e)), e)
            blocked = ((q <= LIMITS[:, 0] + 1e-9) & (dq < 0)) | ((q >= LIMITS[:, 1] - 1e-9) & (dq > 0))
            if not np.any(blocked & active):
                break
            active &= ~blocked
        step = float(np.linalg.norm(dq))
        if step > 0.3:
            dq *= 0.3 / step
        q_next = clamp(normalize(q + dq))
        if np.allclose(q_next, q, atol=1e-9):
            break  # stuck: out of reach or every useful joint is at a limit
        q = q_next
    pos, ang = _errors(q, lengths, target, phi)
    reachable = pos < 1e-3 and (phi is None or ang < math.radians(0.5))
    notes = [] if reachable else ["Did not converge: target is out of reach or blocked by joint limits."]
    return IKResult(q, "dls", reachable, within_limits(q), it, pos, ang, notes=notes)


def ik_ccd(target, lengths, q0, max_iter: int = 150, tol: float = 1e-4) -> IKResult:
    """Cyclic coordinate descent: rotate one joint at a time, wrist first,
    so the line from that joint to the tool points at the target."""
    target = np.asarray(target, dtype=float)
    q = clamp(normalize(np.asarray(q0, dtype=float)))
    it = 0
    for it in range(1, max_iter + 1):
        for i in (2, 1, 0):
            pts, _ = fk(q, lengths)
            to_tool = pts[-1] - pts[i]
            to_target = target - pts[i]
            delta = math.atan2(to_target[1], to_target[0]) - math.atan2(to_tool[1], to_tool[0])
            q[i] += wrap(delta)
            q = clamp(normalize(q))
        if np.linalg.norm(target - fk(q, lengths)[0][-1]) < tol:
            break
    pos, _ = _errors(q, lengths, target, None)
    reachable = pos < 1e-3
    notes = [] if reachable else ["Did not converge: target is out of reach or blocked by joint limits."]
    return IKResult(q, "ccd", reachable, within_limits(q), it, pos, 0.0, notes=notes)


def solve(method: str, target, lengths, q0, phi: float | None, elbow: str = "up") -> IKResult:
    start = time.perf_counter()
    if method == "analytic":
        res = ik_analytic(target, phi if phi is not None else -math.pi / 2, lengths, elbow)
    elif method == "dls":
        # Jacobian IK is a local method: from a bad starting pose it can get
        # trapped against a joint limit. Retry from a few spread-out seeds.
        res, total, attempts = None, 0, 0
        for seed in (q0, *RESTART_SEEDS):
            attempt = ik_dls(target, lengths, seed, phi)
            attempts += 1
            total += attempt.iterations
            if res is None or attempt.position_error + attempt.angle_error < res.position_error + res.angle_error:
                res = attempt
            if attempt.reachable:
                break
        res.iterations = total
        if attempts > 1 and res.reachable:
            res.notes = res.notes + [f"Converged after {attempts - 1} restart(s) from other start poses."]
    elif method == "ccd":
        res = ik_ccd(target, lengths, q0)
    else:
        raise ValueError(f"Unknown IK method: {method}")
    res.solve_ms = (time.perf_counter() - start) * 1000
    return res


# ---- Waist (base yaw) -------------------------------------------------------
WAIST_LIMITS_DEG = (-180.0, 180.0)
HOME_Q4 = np.array([0.0, *HOME_Q])
FRONT, BACK = 0.0, math.pi


def facing_for(x: float, current_waist: float) -> float:
    """Waist angle that points the arm plane at world x: front (+x) or back (-x).
    Targets right above the base keep the current facing."""
    if abs(x) < 0.02:
        return FRONT if math.cos(current_waist) >= 0 else BACK
    return FRONT if x > 0 else BACK


def side(waist: float) -> float:
    """+1 when the arm faces +x, -1 when it faces -x."""
    return 1.0 if math.cos(waist) >= 0 else -1.0


def phi_world_to_plane(phi: float, waist: float) -> float:
    """A side-view tool angle seen from the arm's own plane (mirrored when facing -x)."""
    return phi if side(waist) > 0 else wrap(math.pi - phi)


def fk_world(q4, lengths) -> tuple[np.ndarray, float]:
    """3D joint positions (base, elbow, wrist, tool) as (4, 3) [x, y, z] and the in-plane tool angle."""
    pts, phi = fk(np.asarray(q4[1:]), lengths)
    c, s = math.cos(q4[0]), math.sin(q4[0])
    return np.array([[u * c, y, u * s] for u, y in pts]), phi


def solve_world(method: str, target, lengths, q0_4, phi_world: float | None, elbow: str = "up"):
    """Turn the waist toward the target's side, then solve the in-plane IK."""
    waist = facing_for(target[0], float(q0_4[0]))
    u = side(waist) * target[0]
    phi = None if phi_world is None else phi_world_to_plane(phi_world, waist)
    res = solve(method, (u, target[1]), lengths, np.asarray(q0_4[1:], dtype=float), phi, elbow)
    return waist, res


def world_result(waist: float, res: IKResult, lengths) -> dict:
    out = res.as_dict(lengths)
    q4 = [waist, *[float(a) for a in res.q]]
    pts, phi = fk_world(q4, lengths)
    phi_world = phi if side(waist) > 0 else math.pi - phi
    out.update({
        "q": q4,
        "qDeg": [round(math.degrees(a), 2) for a in q4],
        "joints": pts.round(5).tolist(),
        "tool": {"x": float(pts[-1][0]), "y": float(pts[-1][1]), "phiDeg": round(math.degrees(wrap(phi_world)), 2)},
    })
    return out
