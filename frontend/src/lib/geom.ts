import type { Block, ClothPlan, Keyframe, Lengths, Plan, Q, Vec } from './types'

export type Vec3 = [number, number, number] // x (right), y (up), z (toward the viewer)

export interface Pose {
  /** Side-view (x, y) of base, elbow, wrist, tool. */
  pts: Vec[]
  /** 3D positions of the same joints. */
  pts3: Vec3[]
  /** In-plane tool angle (relative to the arm's own plane). */
  phi: number
  /** Tool angle as seen in the side view. */
  phiSide: number
  /** cos(waist): 1 facing right, -1 facing left, 0 facing the viewer. */
  c: number
}

/**
 * Forward kinematics, mirrored from the Python engine, used for drawing.
 * q = [waist, shoulder, elbow, wrist]. The 3 arm joints work in a vertical
 * plane (u, y); the waist turns that plane: x = u cos(waist), z = u sin(waist).
 */
export function fk(q: Q, lengths: Lengths, base: Vec): Pose {
  const c = Math.cos(q[0])
  const s = Math.sin(q[0])
  const plane: Vec[] = [base]
  let angle = 0
  let [u, y] = base
  for (let i = 0; i < 3; i++) {
    angle += q[i + 1]
    u += lengths[i] * Math.cos(angle)
    y += lengths[i] * Math.sin(angle)
    plane.push([u, y])
  }
  const pts3 = plane.map(([pu, py]) => [pu * c, py, pu * s] as Vec3)
  return {
    pts: pts3.map(([x, py]) => [x, py] as Vec),
    pts3,
    phi: angle,
    phiSide: c >= 0 ? angle : Math.PI - angle,
    c,
  }
}

export const deg = (r: number) => (r * 180) / Math.PI
export const rad = (d: number) => (d * Math.PI) / 180
export const wrapDeg = (d: number) => ((((d + 180) % 360) + 360) % 360) - 180

/** Interpolated arm state at time t (seconds) along a planned trajectory. */
export function frameAt(plan: Plan, t: number) {
  return frameAtFrames(plan.frames, plan.fps, t)
}

/** Same, for any list of frames (e.g. the second arm's). */
export function frameAtFrames(frames: Plan['frames'], fps: number, t: number) {
  const n = frames.length
  const f = Math.max(0, Math.min(n - 1, t * fps))
  const i0 = Math.floor(f)
  const i1 = Math.min(n - 1, i0 + 1)
  const a = frames[i0]
  const b = frames[i1]
  const k = f - i0
  const q = a.q.map((v, i) => v + (b.q[i] - v) * k) as Q
  const roll = (a.roll ?? 0) + ((b.roll ?? 0) - (a.roll ?? 0)) * k
  return { q, grip: a.grip + (b.grip - a.grip) * k, roll, holding: a.holding }
}

/** Where each block is at time t: on the table, in the gripper, or in its bin. */
export function blocksAt(
  plan: Plan,
  blocks: Block[],
  t: number,
  lengths: Lengths,
  base: Vec,
  graspFraction: number,
): Block[] {
  const { q, roll } = frameAt(plan, t)
  const pose = fk(q, lengths, base)
  const tip = pose.pts3[3]
  return blocks.map((b) => {
    const kfs = plan.timeline[b.id] ?? []
    let kf: Keyframe | undefined
    for (const k of kfs) if (k.t <= t + 1e-9) kf = k
    if (!kf) return b
    switch (kf.mode) {
      case 'held': {
        if (kf.local && kf.arm === 'B' && plan.armB) {
          // Held by the second arm: use its pose, then shift to where it stands.
          const fb = frameAtFrames(plan.armB.frames, plan.fps, t)
          const p = heldPose(kf.local, kf.tilt0 ?? 0, fb.q, lengths, base, fb.roll, kf.yaw0 ?? 0)
          return { ...b, ...p, z: p.z + plan.armB.baseZ, binId: null }
        }
        if (kf.local) return { ...b, ...heldPose(kf.local, kf.tilt0 ?? 0, q, lengths, base, roll, kf.yaw0 ?? 0), binId: null }
        // Older plans: the block hangs between the fingers, centre a little behind the tip.
        const back = b.size * (0.5 - graspFraction)
        const along = Math.cos(pose.phi) * back
        return {
          ...b,
          x: tip[0] - along * Math.cos(q[0]),
          y: tip[1] - Math.sin(pose.phi) * back,
          z: tip[2] - along * Math.sin(q[0]),
          yaw: q[0],
          binId: null,
        }
      }
      case 'falling': {
        // Free fall under gravity until it hits the table.
        const [x0, y0, z0] = kf.pos!
        const dt = t - kf.t
        const y = Math.max(kf.to![1], y0 - 0.5 * 9.81 * dt * dt)
        return { ...b, x: x0, y, z: z0, yaw: kf.yaw ?? 0, binId: null }
      }
      case 'table':
      case 'bin': {
        const [x, y, z] = kf.pos!
        return { ...b, x, y, z, yaw: kf.yaw ?? 0, tilt: kf.tilt ?? 0, binId: kf.mode === 'bin' ? (kf.binId ?? null) : null }
      }
    }
  })
}

export const fmt = (v: number, digits = 1) => (Number.isFinite(v) ? v.toFixed(digits) : '–')

/** The tool point in the arm's own plane: distance out from the waist axis, height, tool angle. */
function toolInPlane(q: Q, lengths: Lengths, base: Vec) {
  const pose = fk(q, lengths, base)
  const [x, y, z] = pose.pts3[3]
  return { u: x * Math.cos(q[0]) + z * Math.sin(q[0]), y, phi: pose.phi, w: q[0] }
}

/**
 * Where a held object is, from its offset in the tool frame (recorded by the
 * planner at the moment of the grasp). Handles wrist turns: a cup being
 * flipped rotates with the tool.
 */
export function heldPose(local: [number, number, number], tilt0: number, q: Q, lengths: Lengths, base: Vec, roll = 0, yaw0 = 0) {
  const { u, y, phi, w } = toolInPlane(q, lengths, base)
  const [a, b, n] = local
  const uc = u + a * Math.cos(phi) - b * Math.sin(phi)
  const yc = y + a * Math.sin(phi) + b * Math.cos(phi)
  return {
    x: uc * Math.cos(w) - n * Math.sin(w),
    y: yc,
    z: uc * Math.sin(w) + n * Math.cos(w),
    yaw: w + roll + yaw0, // turns with the waist and the wrist roll
    tilt: phi + tilt0,
  }
}

// ---- Cloth (towel and T-shirt) --------------------------------------------

export const CLOTH_N = 24 // towel grid cells per side

/** A cloth mesh: a (nu+1) x (nv+1) vertex grid; mask says which cells are cloth. */
export interface ClothGrid {
  nu: number
  nv: number
  mask: Uint8Array | null
  colors: Float32Array
  key: string
  /** Textured cloth (the shirt): drawn with UVs, a fabric texture and an outline. */
  texture?: 'shirt'
  color?: string
}

/** A flat towel: (N+1) x (N+1) vertices, square, edges along n1 and across it. */
export function flatCloth(center: [number, number], half: number): Float32Array {
  const len = Math.hypot(center[0], center[1]) || 1
  const n1 = [center[0] / len, center[1] / len]
  const n2 = [-n1[1], n1[0]]
  const out = new Float32Array((CLOTH_N + 1) * (CLOTH_N + 1) * 3)
  for (let i = 0; i <= CLOTH_N; i++) {
    for (let j = 0; j <= CLOTH_N; j++) {
      const s = -half + (2 * half * i) / CLOTH_N
      const r = -half + (2 * half * j) / CLOTH_N
      const k = (i * (CLOTH_N + 1) + j) * 3
      out[k] = center[0] + n1[0] * s + n2[0] * r
      out[k + 1] = 0.002
      out[k + 2] = center[1] + n1[1] * s + n2[1] * r
    }
  }
  return out
}

function hexRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16)
  // three.js expects vertex colours in linear space, so undo the sRGB curve.
  const lin = (c: number) => Math.pow(c / 255, 2.2)
  return [lin((n >> 16) & 255), lin((n >> 8) & 255), lin(n & 255)]
}

export function towelGrid(color: string): ClothGrid {
  const [r, g, b] = hexRgb(color)
  const colors = new Float32Array((CLOTH_N + 1) * (CLOTH_N + 1) * 3)
  for (let i = 0; i <= CLOTH_N; i++) {
    for (let j = 0; j <= CLOTH_N; j++) {
      // White stripes near two opposite edges, like a bath towel.
      const stripe = (j >= 2 && j <= 3) || (j >= CLOTH_N - 3 && j <= CLOTH_N - 2)
      colors.set(stripe ? [1, 1, 1] : [r, g, b], (i * (CLOTH_N + 1) + j) * 3)
    }
  }
  return { nu: CLOTH_N, nv: CLOTH_N, mask: null, colors, key: `towel-${color}`, color }
}

// T-shirt: u runs from the collar (0) to the hem (length), v across the body.
// The mesh is a fine rectangular grid covering the whole shirt; the shirt's
// actual outline (sleeves, neckline, curved hem) comes from a smooth alpha
// texture, so the edges are clean curves instead of grid steps.
export const SHIRT_DIMS = { length: 0.36, half: 0.15, sleeve: 0.25 }
export const SHIRT_NU = 64
export const SHIRT_NV = 88
export const SHIRT_V = 0.26 // grid half-width (a little past the sleeve tips)

/** Small, smooth wrinkles so a flat shirt doesn't look like a sheet of paper. */
function wrinkle(u: number, v: number) {
  return (
    0.0012 * Math.sin(u * 37 + v * 11) * Math.sin(v * 29 - u * 7) +
    0.0007 * Math.sin(u * 83 - v * 41) +
    0.0005 * Math.cos(v * 67 + u * 19)
  )
}

/** Flat shirt on the table: collar at x = center[0], centred on z = center[1]. */
export function flatShirt(center: [number, number], d = SHIRT_DIMS): Float32Array {
  const out = new Float32Array((SHIRT_NU + 1) * (SHIRT_NV + 1) * 3)
  for (let i = 0; i <= SHIRT_NU; i++) {
    for (let j = 0; j <= SHIRT_NV; j++) {
      const k = (i * (SHIRT_NV + 1) + j) * 3
      const u = (d.length * i) / SHIRT_NU
      const v = -SHIRT_V + (2 * SHIRT_V * j) / SHIRT_NV
      out[k] = center[0] + u
      out[k + 1] = 0.0035 + wrinkle(u, v)
      out[k + 2] = center[1] + v
    }
  }
  return out
}

export function shirtGrid(color: string): ClothGrid {
  return { nu: SHIRT_NU, nv: SHIRT_NV, mask: null, colors: new Float32Array(0), key: `shirt-${color}`, texture: 'shirt', color }
}

/**
 * Fold the cloth over the line through F (normal n points at the moving
 * side). The pinched point, w out from the fold line, follows the gripper
 * E = (a along n, height b). Cloth between the fold line and the pinch
 * follows a curve: flat when E is still out at +w, a loop when E is
 * overhead, laid flat on the other side when E is at -w. Cloth beyond the
 * pinch (a sleeve past the side seam) hangs from it: down while lifted,
 * flat once laid down.
 */
function applyFold(v: Float32Array, F: [number, number], n: [number, number], w: number, a: number, b: number) {
  let top = 0
  for (let k = 0; k < v.length; k += 3) {
    const s = (v[k] - F[0]) * n[0] + (v[k + 2] - F[1]) * n[1]
    if (s > 1e-6) top = Math.max(top, v[k + 1])
  }
  const done = Math.min(1, Math.max(0, (w - a) / (2 * w))) // 0 flat ... 1 folded over
  // Direction the cloth beyond the pinch hangs in, in (along n, up) coordinates.
  let hx = a / w
  let hy = -(1 - Math.min(1, Math.abs(a) / w))
  const hl = Math.hypot(hx, hy) || 1
  hx /= hl
  hy /= hl
  for (let k = 0; k < v.length; k += 3) {
    const s = (v[k] - F[0]) * n[0] + (v[k + 2] - F[1]) * n[1]
    if (s <= 1e-6) continue
    const lx = v[k] - F[0] - s * n[0]
    const lz = v[k + 2] - F[1] - s * n[1]
    const h0 = v[k + 1]
    let bx: number
    let by: number
    if (s <= w) {
      const r = s / w
      // Quadratic Bezier from the fold line (0, h0) through (a/2, b) to the pinch (a, b).
      bx = 2 * r * (1 - r) * (a / 2) + r * r * a
      by = (1 - r) * (1 - r) * h0 + 2 * r * (1 - r) * b + r * r * b
    } else {
      const extra = s - w
      bx = a + hx * extra
      by = b + hy * extra
    }
    v[k] = F[0] + n[0] * bx + lx
    v[k + 2] = F[1] + n[1] * bx + lz
    // Once it lies flat again the layer order is flipped: stack it on top.
    v[k + 1] = Math.max(0.002, by + (top - h0 + 0.004) * done * done)
  }
}

/** Gripper tip of arm A or B at time t, in world coordinates. */
function tipAt(plan: Plan, arm: 'A' | 'B', t: number, lengths: Lengths, base: Vec): Vec3 {
  if (arm === 'B' && plan.armB) {
    const p = fk(frameAtFrames(plan.armB.frames, plan.fps, t).q, lengths, base).pts3[3]
    return [p[0], p[1], p[2] + plan.armB.baseZ]
  }
  return fk(frameAt(plan, t).q, lengths, base).pts3[3]
}

/** The cloth's shape at time t of a folding plan (towel or shirt). */
export function clothAt(plan: Plan, cloth: ClothPlan, t: number, lengths: Lengths, base: Vec): Float32Array {
  const v = cloth.shape === 'shirt' ? flatShirt(cloth.center, cloth.dims ?? SHIRT_DIMS) : flatCloth(cloth.center, cloth.half)
  // Folds that happen at the same time (one per arm) work on separate sides.
  for (const f of cloth.folds) {
    if (t < f.tGrasp) break
    const arms = f.arms ?? ['A']
    const tips = arms.map((arm) => tipAt(plan, arm, Math.min(t, f.tRelease), lengths, base))
    const tip: Vec3 = [0, 1, 2].map((i) => tips.reduce((acc, p) => acc + p[i], 0) / tips.length) as Vec3
    const a = (tip[0] - f.F[0]) * f.n[0] + (tip[2] - f.F[1]) * f.n[1]
    applyFold(v, f.F, f.n, f.w, a, tip[1])
  }
  const c = cloth.carry
  if (c && t >= c.tGrasp) {
    // Carried as a rigid bundle: keep each vertex fixed in the arm's frame.
    const g = toolInPlane(frameAt(plan, c.tGrasp).q, lengths, base)
    const now = toolInPlane(frameAt(plan, Math.min(t, c.tRelease)).q, lengths, base)
    let minY = Infinity
    for (let k = 0; k < v.length; k += 3) {
      const u = v[k] * Math.cos(g.w) + v[k + 2] * Math.sin(g.w)
      const n = -v[k] * Math.sin(g.w) + v[k + 2] * Math.cos(g.w)
      const u2 = now.u + (u - g.u)
      v[k + 1] = now.y + (v[k + 1] - g.y)
      v[k] = u2 * Math.cos(now.w) - n * Math.sin(now.w)
      v[k + 2] = u2 * Math.sin(now.w) + n * Math.cos(now.w)
      minY = Math.min(minY, v[k + 1])
    }
    if (t >= c.tRelease) {
      // Let go: it drops onto the basket floor.
      const dy = c.floorY + 0.002 - minY
      for (let k = 1; k < v.length; k += 3) v[k] += dy
    }
  }
  return v
}
