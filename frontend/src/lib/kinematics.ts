/**
 * Kinematics of the humanoid's 7-joint arms.
 *
 * One table (ARM) describes the chain: for each joint, where it sits in the
 * previous joint's frame and which axis it turns about. The 3D model, forward
 * kinematics, the Jacobian, inverse kinematics and the gravity torques all read
 * this same table, so what you see is exactly what is computed.
 *
 * Frames: x forward, y up, z to the robot's right. Zero pose = arm hanging
 * straight down. The left arm is the right arm mirrored in the x-y plane.
 */
import * as THREE from 'three'
import { clamp, rad } from './math'

export type Side = 'L' | 'R'
export type V3 = [number, number, number]

export const UPPER = 0.31 // shoulder to elbow (m)
export const FORE = 0.28 // elbow to wrist (m)
/** Wrist-yaw frame to the point between the fingertips (the tool centre point). */
export const TOOL: V3 = [0, -0.14, 0]
export const GRAVITY = 9.81

export interface JointDef {
  key: string
  label: string
  /** Position of this joint in the previous joint's frame (right arm). */
  offset: V3
  /** Rotation axis in this joint's frame (right arm). */
  axis: V3
  /** Limits in degrees. */
  min: number
  max: number
  /** Motor limits: rad/s and rad/s². */
  vmax: number
  amax: number
  /** Rated continuous torque (N·m). */
  rated: number
  /** Mass carried by the link after this joint, at a point in its frame. */
  mass: { m: number; at: V3 }
  color: string
}

export const ARM: JointDef[] = [
  { key: 'sp', label: 'Shoulder pitch', offset: [0, 0, 0.2], axis: [0, 0, 1], min: -60, max: 180, vmax: 1.8, amax: 6, rated: 90, mass: { m: 0.4, at: [0, 0, 0.04] }, color: '#e2e8f0' },
  { key: 'sr', label: 'Shoulder roll', offset: [0, 0, 0.065], axis: [-1, 0, 0], min: -15, max: 100, vmax: 1.8, amax: 6, rated: 90, mass: { m: 0.5, at: [0, -0.04, 0] }, color: '#38bdf8' },
  { key: 'sy', label: 'Upper-arm twist', offset: [0, -0.075, 0], axis: [0, 1, 0], min: -100, max: 100, vmax: 2.2, amax: 8, rated: 45, mass: { m: 1.6, at: [0, -0.11, 0] }, color: '#a78bfa' },
  { key: 'el', label: 'Elbow', offset: [0, -(UPPER - 0.075), 0], axis: [0, 0, 1], min: 0, max: 145, vmax: 2.2, amax: 8, rated: 60, mass: { m: 0.5, at: [0, -0.03, 0] }, color: '#34d399' },
  { key: 'fy', label: 'Forearm twist', offset: [0, -0.07, 0], axis: [0, 1, 0], min: -120, max: 120, vmax: 3, amax: 10, rated: 18, mass: { m: 1.0, at: [0, -0.1, 0] }, color: '#f472b6' },
  { key: 'wp', label: 'Wrist pitch', offset: [0, -(FORE - 0.07), 0], axis: [0, 0, 1], min: -80, max: 80, vmax: 3.5, amax: 12, rated: 12, mass: { m: 0.15, at: [0, -0.015, 0] }, color: '#facc15' },
  { key: 'wy', label: 'Wrist yaw', offset: [0, -0.035, 0], axis: [-1, 0, 0], min: -45, max: 45, vmax: 3.5, amax: 12, rated: 12, mass: { m: 0.6, at: [0, -0.07, 0] }, color: '#fb7185' },
]
export const DOF = ARM.length

const sgn = (side: Side) => (side === 'R' ? 1 : -1)
/** Mirror a point for the left arm (flip z). */
export const mirrorPoint = (side: Side, v: V3): V3 => [v[0], v[1], sgn(side) * v[2]]
/** Mirror a rotation axis for the left arm (an axis is a pseudovector: x and y flip, z stays). */
export const mirrorAxis = (side: Side, v: V3): V3 => (side === 'R' ? v : [-v[0], -v[1], v[2]])

export const jointOffset = (side: Side, i: number) => new THREE.Vector3(...mirrorPoint(side, ARM[i].offset))
export const jointAxis = (side: Side, i: number) => new THREE.Vector3(...mirrorAxis(side, ARM[i].axis))
export const limitsRad = ARM.map((j) => [rad(j.min), rad(j.max)] as [number, number])
export const clampQ = (q: number[]) => q.map((v, i) => clamp(v, limitsRad[i][0], limitsRad[i][1]))

export interface ArmFK {
  /** World position of each joint. */
  origins: THREE.Vector3[]
  /** World rotation axis of each joint. */
  axes: THREE.Vector3[]
  /** Frame of each link (after its joint has turned). */
  frames: THREE.Matrix4[]
  tool: THREE.Matrix4
  /** Tool centre point. */
  tcp: THREE.Vector3
  /** Direction the fingers point (tool -y axis). */
  dir: THREE.Vector3
}

const _m = new THREE.Matrix4()

/** Forward kinematics: joint angles to every joint frame and the tool pose, starting from `base` (the shoulder carriage). */
export function fk(side: Side, q: number[], base: THREE.Matrix4): ArmFK {
  const T = base.clone()
  const origins: THREE.Vector3[] = []
  const axes: THREE.Vector3[] = []
  const frames: THREE.Matrix4[] = []
  for (let i = 0; i < DOF; i++) {
    const off = jointOffset(side, i)
    T.multiply(_m.makeTranslation(off.x, off.y, off.z))
    origins.push(new THREE.Vector3().setFromMatrixPosition(T))
    const a = jointAxis(side, i)
    axes.push(a.clone().transformDirection(T))
    T.multiply(_m.makeRotationAxis(a, q[i]))
    frames.push(T.clone())
  }
  const tool = T.clone().multiply(_m.makeTranslation(...TOOL))
  const tcp = new THREE.Vector3().setFromMatrixPosition(tool)
  const dir = new THREE.Vector3(0, -1, 0).transformDirection(tool)
  return { origins, axes, frames, tool, tcp, dir }
}

/**
 * Torque each motor must hold against gravity (N·m, about its own axis), from the
 * masses of all the links further down the arm: tau_i = -a_i · Σ (p_m - p_i) × m g.
 */
export function gravityTorques(side: Side, q: number[], base: THREE.Matrix4, payload = 0): number[] {
  const f = fk(side, q, base)
  const pts = ARM.map((j, k) => ({ m: j.mass.m, p: new THREE.Vector3(...mirrorPoint(side, j.mass.at)).applyMatrix4(f.frames[k]) }))
  if (payload > 0) pts.push({ m: payload, p: f.tcp.clone() })
  const tmp = new THREE.Vector3()
  return f.axes.map((a, i) => {
    let tau = 0
    for (let k = 0; k < pts.length; k++) {
      if (k < i) continue // links before this joint don't load it
      const { m, p } = pts[k]
      tmp.subVectors(p, f.origins[i]).cross(new THREE.Vector3(0, -m * GRAVITY, 0))
      tau += a.dot(tmp)
    }
    return -tau
  })
}

// ---------------------------------------------------------------------------
// Small dense linear algebra for the IK (systems are at most 6x6).
// ---------------------------------------------------------------------------
/** Solve A X = B (A n×n, B n×k) by Gauss-Jordan elimination with partial pivoting. */
function solve(A: number[][], B: number[][]): number[][] {
  const n = A.length
  const k = B[0].length
  const M = A.map((row, i) => [...row, ...B[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    ;[M[c], M[p]] = [M[p], M[c]]
    const d = M[c][c] || 1e-12
    for (let j = c; j < n + k; j++) M[c][j] /= d
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = M[r][c]
      if (f !== 0) for (let j = c; j < n + k; j++) M[r][j] -= f * M[c][j]
    }
  }
  return M.map((row) => row.slice(n))
}

export type DirGoal = 'free' | 'forward' | 'down'
export const DIR_VECTORS: Record<Exclude<DirGoal, 'free'>, THREE.Vector3> = {
  forward: new THREE.Vector3(1, 0, 0),
  down: new THREE.Vector3(0, -1, 0),
}

export interface IKResult {
  q: number[]
  /** Remaining position error (m). */
  posErr: number
  /** Remaining pointing error (rad), 0 when the direction is free. */
  dirErr: number
  iters: number
  ms: number
  reached: boolean
}

/**
 * Inverse kinematics by damped least squares:  dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e.
 * The task is the tool position (3 rows), plus optionally the pointing direction
 * of the fingers (3 rows, weighted). The arm has 7 joints, more than the task
 * needs, so the spare freedom (the null space) is used to stay close to a natural
 * rest pose: dq += (I - J⁺J) k (q_rest - q). Joints are clamped to their limits.
 */
export function solveIK(
  side: Side,
  q0: number[],
  base: THREE.Matrix4,
  target: THREE.Vector3,
  opts: { dir?: THREE.Vector3 | null; rest: number[]; maxIters?: number },
): IKResult {
  const t0 = performance.now()
  const dirT = opts.dir ?? null
  const ok = (r: Attempt) => r.posErr < 0.005 && (!dirT || r.dirErr < rad(3))
  const score = (r: Attempt) => r.posErr + 0.05 * r.dirErr
  // DLS is a local method and can get stuck against a joint limit, so if the
  // first try fails, retry from the rest pose and a few scattered start poses.
  const seeds = [q0, opts.rest]
  let rng = 12345
  const rand = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648)
  for (let k = 0; k < 6; k++) seeds.push(limitsRad.map(([lo, hi]) => lo + (0.15 + 0.7 * rand()) * (hi - lo)))
  let iters = 0
  const run = (goal: THREE.Vector3) => {
    let best: Attempt | null = null
    for (const seed of seeds) {
      const r = dlsAttempt(side, seed, base, goal, dirT, opts.rest, opts.maxIters ?? 150)
      iters += r.iters
      if (!best || score(r) < score(best)) best = r
      if (ok(best)) break
    }
    return best!
  }
  let b = run(target)
  // Beyond the arm's length: aim for the nearest point it can reach, on the line
  // from the shoulder to the target, so the arm stretches straight toward it.
  const shoulder = fk(side, q0, base).origins[1]
  const d = target.clone().sub(shoulder)
  if (!ok(b) && d.length() > 0.97 * ARM_REACH) {
    const near = run(shoulder.clone().add(d.setLength(0.97 * ARM_REACH)))
    const f = fk(side, near.q, base)
    const posErr = f.tcp.distanceTo(target)
    if (posErr < b.posErr) b = { ...near, posErr }
  }
  return { q: b.q, posErr: b.posErr, dirErr: dirT ? b.dirErr : 0, iters, ms: performance.now() - t0, reached: ok(b) }
}

/** Distance from the shoulder-roll joint to the fingertips with the arm straight. */
export const ARM_REACH = ARM.slice(2).reduce((s, j) => s + Math.hypot(...j.offset), 0) + Math.hypot(...TOOL)

interface Attempt {
  q: number[]
  posErr: number
  dirErr: number
  iters: number
}

function dlsAttempt(
  side: Side,
  q0: number[],
  base: THREE.Matrix4,
  target: THREE.Vector3,
  dirT: THREE.Vector3 | null,
  rest: number[],
  maxIters: number,
): Attempt {
  const lambda = 0.03
  const wDir = 0.3
  const kNull = 0.05
  let q = clampQ(q0)
  let posErr = Infinity
  let dirErr = 0
  let it = 0
  const tmp = new THREE.Vector3()
  for (; it < maxIters; it++) {
    const f = fk(side, q, base)
    const ep = tmp.subVectors(target, f.tcp).clone()
    posErr = ep.length()
    const J: number[][] = [[], [], []]
    const e: number[] = [ep.x, ep.y, ep.z]
    for (let i = 0; i < DOF; i++) {
      const c = new THREE.Vector3().subVectors(f.tcp, f.origins[i])
      const col = new THREE.Vector3().crossVectors(f.axes[i], c)
      J[0].push(col.x)
      J[1].push(col.y)
      J[2].push(col.z)
    }
    if (dirT) {
      const d = f.dir
      dirErr = d.angleTo(dirT)
      const eo = new THREE.Vector3().crossVectors(d, dirT).multiplyScalar(wDir)
      e.push(eo.x, eo.y, eo.z)
      // Only turns that tilt the fingers count (turning about their own axis doesn't).
      const rows: number[][] = [[], [], []]
      for (let i = 0; i < DOF; i++) {
        const a = f.axes[i].clone()
        a.sub(d.clone().multiplyScalar(a.dot(d))).multiplyScalar(wDir)
        rows[0].push(a.x)
        rows[1].push(a.y)
        rows[2].push(a.z)
      }
      J.push(...rows)
    }
    if (posErr < 2e-4 && (!dirT || dirErr < 2e-3)) break

    // A joint pinned at a limit and pushed further out is frozen (its column
    // zeroed) and the step re-solved, so the other joints take over its share.
    const free = new Array(DOF).fill(true)
    let dq: number[] = []
    for (let pass = 0; pass < 3; pass++) {
      const Jf = J.map((row) => row.map((v, i) => (free[i] ? v : 0)))
      dq = dlsStep(Jf, e, lambda, q.map((v, i) => (free[i] ? kNull * (rest[i] - v) : 0)))
      let changed = false
      for (let i = 0; i < DOF; i++) {
        const [lo, hi] = limitsRad[i]
        if (free[i] && ((q[i] <= lo + 1e-6 && dq[i] < 0) || (q[i] >= hi - 1e-6 && dq[i] > 0))) {
          free[i] = false
          changed = true
        }
      }
      if (!changed) break
    }
    const norm = Math.hypot(...dq)
    const scale = norm > 0.25 ? 0.25 / norm : 1
    q = clampQ(q.map((v, i) => v + dq[i] * scale))
  }
  // Report the error of the final q (the loop may have stopped after a step).
  const f = fk(side, q, base)
  posErr = f.tcp.distanceTo(target)
  dirErr = dirT ? f.dir.angleTo(dirT) : 0
  return { q, posErr, dirErr, iters: it }
}

/**
 * One damped-least-squares step plus a null-space motion z:
 *   dq = Jᵀ A⁻¹ e + (I - Jᵀ A⁻¹ J) z,   A = J Jᵀ + λ² I
 */
function dlsStep(J: number[][], e: number[], lambda: number, z: number[]): number[] {
  const m = J.length
  const n = J[0].length
  const A = J.map((ri) => J.map((rj) => ri.reduce((s, v, k) => s + v * rj[k], 0)))
  for (let i = 0; i < m; i++) A[i][i] += lambda * lambda
  // One solve gives both A⁻¹e and A⁻¹J.
  const X = solve(A, J.map((row, i) => [e[i], ...row]))
  const dq = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    let v = z[i]
    for (let r = 0; r < m; r++) {
      v += J[r][i] * X[r][0]
      let jy = 0
      for (let j = 0; j < n; j++) jy += X[r][j + 1] * z[j]
      v -= J[r][i] * jy
    }
    dq[i] = v
  }
  return dq
}
