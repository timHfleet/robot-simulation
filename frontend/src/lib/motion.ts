/**
 * Hand motion between two poses as a chain of small straight-line steps.
 *
 * Moving each joint straight from its old angle to its new one makes the hand
 * swing along a curve (through a table, say) and lets the IK jump to a different
 * arm configuration. Instead the hand moves along a straight line in small steps
 * (about 3 cm and 9 degrees each), each step solved starting from the previous one,
 * so the arm changes smoothly and never flips. The waist turns gradually over the
 * same steps.
 */
import * as THREE from 'three'
import { WAIST_X, fk, solveIK, torsoBase, waistFor, type IKResult, type KeepOut, type Side } from './kinematics'
import { DOCKS, KEEP_OUT, toWorld, type StationId } from './stations'
import type { ArmGoal, Target } from './tasks'

export interface HandGoal {
  /** Robot frame. */
  pos: THREE.Vector3
  dir: THREE.Vector3
  close: THREE.Vector3 | null
}

const STEP = 0.03
const TURN = 0.15

function startPose(side: Side, q0: number[], lift: number, w0: number) {
  const f = fk(side, q0, torsoBase(lift, w0))
  return { p: f.tcp, d: f.dir, c: new THREE.Vector3(0, 0, 1).transformDirection(f.tool) }
}

/** How many steps a move needs. */
export function pathSteps(side: Side, q0: number[], lift: number, w0: number, w1: number, goal: HandGoal) {
  const s = startPose(side, q0, lift, w0)
  const n = Math.max(s.p.distanceTo(goal.pos) / STEP, s.d.angleTo(goal.dir) / TURN, Math.abs(w1 - w0) / 0.1)
  return THREE.MathUtils.clamp(Math.ceil(n), 1, 40)
}

/** Joint angles for each step from the current pose to the goal (k = 1..steps). */
export function handPath(
  side: Side,
  q0: number[],
  lift: number,
  w0: number,
  w1: number,
  goal: HandGoal,
  steps: number,
  opts: { rest: number[]; keepOut?: KeepOut[] | null; posture?: number[] },
): { qs: number[][]; res: IKResult } {
  const s = startPose(side, q0, lift, w0)
  // The gripper is symmetric: closing along c or -c is the same grasp.
  const c0 = goal.close && s.c.dot(goal.close) < 0 ? s.c.clone().negate() : s.c
  const close = (r: IKResult) => r.reached || (r.posErr < 0.05 && r.dirErr < 0.35)
  /**
   * Walk the hand along the line. The arm's spare freedom changes as little as possible
   * from step to step (not drifting toward a rest pose), so the forearm and wrist never
   * twist against each other while the hand barely moves.
   */
  const walk = () => {
    const qs: number[][] = []
    let q = q0
    let res!: IKResult
    for (let k = 1; k <= steps; k++) {
      const t = k / steps
      const last = k === steps
      const p = last ? goal.pos : s.p.clone().lerp(goal.pos, t)
      const d = last ? goal.dir : s.d.clone().lerp(goal.dir, t).normalize()
      const c = goal.close ? (last ? goal.close : c0.clone().lerp(goal.close, t).normalize()) : null
      // With a posture to settle into, each step starts a little nearer it (all the way by the
      // last step), so the elbow comes down to it smoothly as the hand travels; otherwise the
      // spare freedom stays as it is.
      const P = opts.posture
      const seed = P ? q.map((v, i) => v + (P[i] - v) / (steps - k + 1)) : q
      res = solveIK(side, seed, torsoBase(lift, w0 + (w1 - w0) * t), p, { dir: d, close: c, rest: P ?? q, keepOut: opts.keepOut, local: true })
      q = res.q
      qs.push(q)
    }
    return { qs, res }
  }
  const first = walk()
  if (close(first.res)) return first
  // No smooth way along the line: find a pose that reaches the goal and blend every
  // joint toward it over the whole move, rather than snapping at the end.
  const wide = solveIK(side, q0, torsoBase(lift, w1), goal.pos, { dir: goal.dir, close: goal.close, rest: opts.posture ?? opts.rest, keepOut: opts.keepOut })
  if (!wide.reached && wide.posErr >= first.res.posErr) return first
  const qs = Array.from({ length: steps }, (_, k) => q0.map((v, i) => v + ((wide.q[i] - v) * (k + 1)) / steps))
  return { qs, res: wide }
}

const Y = new THREE.Vector3(0, 1, 0)
/** Turn a robot-frame vector or point by the waist angle about the waist axis. */
function aboutWaist(v: THREE.Vector3, waist: number, point: boolean) {
  if (!point) return v.clone().applyAxisAngle(Y, waist)
  return v.clone().sub(new THREE.Vector3(WAIST_X, 0, 0)).applyAxisAngle(Y, waist).add(new THREE.Vector3(WAIST_X, 0, 0))
}

export interface RobotPose {
  x: number
  z: number
  yaw: number
}

/** A task target in the robot frame, given where the robot stands, its lift and waist. */
export function toRobotGoal(g: Target, robot: RobotPose, lift: number, waist: number): HandGoal {
  if (g.frame === 'torso') {
    return {
      pos: aboutWaist(g.pos.clone().add(new THREE.Vector3(0, lift, 0)), waist, true),
      dir: aboutWaist(g.approach, waist, false).normalize(),
      close: g.close ? aboutWaist(g.close, waist, false).normalize() : null,
    }
  }
  const inv = (v: THREE.Vector3) => v.clone().applyAxisAngle(Y, -robot.yaw)
  return {
    pos: inv(g.pos.clone().sub(new THREE.Vector3(robot.x, 0, robot.z))),
    dir: inv(g.approach).normalize(),
    close: g.close ? inv(g.close).normalize() : null,
  }
}

/** Waist angle for a move: turn toward the work, keep it for tuck moves, square up for two-arm poses. */
export function waistForMove(arms: Partial<Record<Side, ArmGoal>>, robot: RobotPose, lift: number, current: number) {
  const work = (Object.entries(arms) as [Side, ArmGoal | undefined][]).flatMap(([side, g]) =>
    g && !Array.isArray(g) && !g.keepWaist ? [{ side, p: toRobotGoal(g, robot, lift, current).pos }] : [],
  )
  if (work.length) return waistFor(work)
  if (Array.isArray(arms.L) && Array.isArray(arms.R)) return 0
  return current
}

/** A station's obstacles in the robot frame, for where the robot stands now. */
/**
 * Every station's surfaces, exactly where they are relative to the robot however it stands
 * (each box keeps its station's orientation), plus round containers on the floor. Used when
 * the robot works around the room rather than docked square to one station.
 */
/** Extra room round every keep-out box: the solver keeps the arm's centre line clear, and its links are ~5 cm thick. */
const PAD = 0.03

export function keepOutsAll(robot: RobotPose, round: { x: number; z: number; r: number; h: number }[]): KeepOut[] {
  const out: KeepOut[] = []
  for (const st of Object.keys(KEEP_OUT) as StationId[]) {
    const rot = DOCKS[st].yaw - robot.yaw
    // Station-frame offsets from the robot: world offset turned into the station's axes.
    const rel = (a: number, b: number) => toWorld(st, a, 0, b).sub(new THREE.Vector3(robot.x, 0, robot.z)).applyAxisAngle(Y, -DOCKS[st].yaw)
    for (const o of KEEP_OUT[st]) {
      const p = rel(o.a0, o.b0)
      const q = rel(o.a1, o.b1)
      // The hand too: working on the floor next to furniture, it mustn't tip into it.
      // 3 cm above the surface too: the solver keeps the arm's centre line clear, and the links are ~5 cm thick.
      out.push({ x0: Math.min(p.x, q.x) - PAD, x1: Math.max(p.x, q.x) + PAD, z0: Math.min(p.z, q.z) - PAD, z1: Math.max(p.z, q.z) + PAD, y: o.y + PAD, hand: true, rot })
    }
  }
  for (const c of round) {
    const p = new THREE.Vector3(c.x - robot.x, 0, c.z - robot.z).applyAxisAngle(Y, -robot.yaw)
    out.push({ x0: p.x - c.r - PAD, x1: p.x + c.r + PAD, z0: p.z - c.r - PAD, z1: p.z + c.r + PAD, y: c.h + PAD, hand: true })
  }
  return out
}

/** Extra keep-outs: round things and world-aligned boxes (x/z in world), each with a top the arm stays above. */
export function keepOutsRound(robot: RobotPose, round: { x: number; z: number; r: number; h: number }[], boxes: { rect: { x0: number; x1: number; z0: number; z1: number }; y: number }[] = []): KeepOut[] {
  const out: KeepOut[] = []
  for (const c of round) {
    const p = new THREE.Vector3(c.x - robot.x, 0, c.z - robot.z).applyAxisAngle(Y, -robot.yaw)
    out.push({ x0: p.x - c.r - PAD, x1: p.x + c.r + PAD, z0: p.z - c.r - PAD, z1: p.z + c.r + PAD, y: c.h + PAD, hand: true })
  }
  // A world-aligned box, in a frame turned by the robot's heading (rot = -yaw), measured from the robot.
  for (const b of boxes)
    out.push({ x0: b.rect.x0 - robot.x - PAD, x1: b.rect.x1 - robot.x + PAD, z0: b.rect.z0 - robot.z - PAD, z1: b.rect.z1 - robot.z + PAD, y: b.y + PAD, hand: true, rot: -robot.yaw })
  return out
}

export function keepOutsFor(station: StationId, robot: RobotPose): KeepOut[] {
  const toRobot = (a: number, b: number) =>
    toWorld(station, a, 0, b).sub(new THREE.Vector3(robot.x, 0, robot.z)).applyAxisAngle(Y, -robot.yaw)
  return KEEP_OUT[station].map((o) => {
    const p = toRobot(o.a0, o.b0)
    const q = toRobot(o.a1, o.b1)
    // 3 cm above the surface: the solver keeps the arm's centre line clear, and the links are ~5 cm thick.
    // ...and 3 cm round the sides, for the same reason.
    return { x0: Math.min(p.x, q.x) - PAD, x1: Math.max(p.x, q.x) + PAD, z0: Math.min(p.z, q.z) - PAD, z1: Math.max(p.z, q.z) + PAD, y: o.y + PAD, hand: o.hand }
  })
}
