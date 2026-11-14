/**
 * Cloth physics for the T-shirt: position-based dynamics on a particle grid.
 *
 * The shirt is a grid of particles (every second vertex of the drawn mesh), joined by
 * distance constraints: stretch (to the next particle), shear (diagonals) and bending
 * (two particles away, softer). Each 1/120 s step moves the particles under gravity
 * with a little air damping, then repeatedly nudges every constrained pair back toward
 * its rest length. Particles can't go through the table (with friction, so cloth
 * doesn't skate) or through each other (self-collision, so a folded layer lies on top
 * of the one below). Where a robot hand pinches, the particles under its fingertips
 * are pinned to the fingertips and the rest of the shirt follows.
 *
 * Only particles inside the T-shirt outline are simulated; the drawn mesh is filled in
 * between particles, with a few millimetres of wrinkle added back on top.
 */
import * as THREE from 'three'
import { CLOTH_NU, CLOTH_NV, CLOTH_Y, LAUNDRY } from '../lib/stations'

const STEP = 1 / 120
/** Most cloth steps per frame; beyond that each step gets longer, so the cloth keeps time with the robot. */
const MAX_STEPS = 16
const ITERS = 10
const GRAVITY = -9.81
const DAMPING = 0.012
/** Particles stay this far apart (cloth thickness, both sides). */
const MIN_DIST = 0.0085
/** Cloth rests this far above the table. */
const REST_UP = 0.0045
/** Below this much sliding per step, cloth on the table sticks. */
const STICK = 0.003
/** How strongly touching layers of cloth are made to move together (0 to 1). */
const LAYER_GRIP = 0.8
const PINCH_R = 0.045
/** How far from the fingertips the hand can still catch an edge of cloth. */
const REACH = 0.09
/** A named-point grasp catches the cloth only if the fingertips are this close to it. */
const GRASP_TOL = 0.04
/** Radius of cloth a smoothing hand carries along (fingertips and the pads between them). */
const SWEEP_R = 0.035
/** A stack grasp holds every layer within this distance of the point (the width of the gripper's hold on an edge). */
const STACK_R = 0.03

export interface ClothParts {
  mesh: THREE.Mesh
  flat: Float32Array
  verts: Float32Array
  /** Is texture point (u, v) inside the shirt's outline? */
  mask: (u: number, v: number) => boolean
}

/** Named points on the shirt, as (distance up from the hem, across from the centre) in metres. */
export const FEATURES = {
  shoulderL: [0.255, -0.14],
  shoulderR: [0.255, 0.14],
  sleeveL: [0.19, -0.28],
  sleeveR: [0.19, 0.28],
  sideL: [0.12, -0.15],
  sideR: [0.12, 0.15],
  sideTopL: [0.2, -0.15],
  sideTopR: [0.2, 0.15],
  sideLowL: [0.04, -0.14],
  sideLowR: [0.04, 0.14],
  hemL: [0.012, -0.14],
  hemR: [0.012, 0.14],
  hemInL: [0.015, -0.065],
  hemInR: [0.015, 0.065],
  collar: [0.27, 0],
} as const
export type Feature = keyof typeof FEATURES

export interface Cloth {
  /** Advance by dt seconds. `hands`: station-local fingertip points of hands holding the cloth. */
  step(dt: number, hands: Partial<Record<'L' | 'R', THREE.Vector3>>): void
  /** Pinch the cloth under a hand's fingertips (station-local); returns how many particles it holds. */
  pinch(side: 'L' | 'R', tcp: THREE.Vector3): number
  /** Let go. */
  release(side: 'L' | 'R'): void
  /** Pinch whatever cloth is nearest the fingertips (within reach), through every layer there. */
  grabAt(side: 'L' | 'R', tcp: THREE.Vector3): number
  /**
   * Smoothing: while on, the hand's fingertips press on the cloth and slide it along with
   * the hand (cloth further away is held by the table), which pulls out wrinkles.
   */
  sweep(side: 'L' | 'R', on: boolean): void
  /** Put the shirt back: flat, or (by default) dropped in a crumpled heap, different each time. */
  reset(crumpled?: boolean): void
  /** Station-local position of a named point on the shirt (shoulders, sleeves, hem, collar). */
  feature(name: Feature): THREE.Vector3
  /**
   * Pinch a named point with the fingertips (station-local hand position). `stack`: pinch
   * through every layer lying at that spot, as fingers closing on a folded edge do.
   */
  grabFeature(side: 'L' | 'R', name: Feature, tcp: THREE.Vector3, stack?: boolean): number
  /** Is the cloth at rest (nothing held, nothing moving)? */
  sleeping(): boolean
  /** Debugging: the nearest cloth particle to a station-local point. */
  probe(x: number, z: number): { active: number; nearest: number; at: number[] }
}

export function createCloth(parts: ClothParts): Cloth {
  const SU = CLOTH_NU / 2 // particle rows (along a, hem to collar)
  const SV = CLOTH_NV / 2 // particle columns (along b, sleeve to sleeve)
  const RW = CLOTH_NV + 1 // drawn vertices per row
  const W = SV + 1
  const N = (SU + 1) * W
  const id = (r: number, c: number) => r * W + c
  const rid = (i: number, j: number) => i * RW + j

  // Which particles are cloth: any drawn vertex within one particle spacing inside the outline.
  const active = new Uint8Array(N)
  for (let r = 0; r <= SU; r++)
    for (let c = 0; c <= SV; c++) {
      let on = false
      for (let di = -2; di <= 2 && !on; di++)
        for (let dj = -2; dj <= 2 && !on; dj++) {
          const i = 2 * r + di
          const j = 2 * c + dj
          if (i < 0 || j < 0 || i > CLOTH_NU || j > CLOTH_NV) continue
          on = parts.mask(j / CLOTH_NV, i / CLOTH_NU)
        }
      active[id(r, c)] = on ? 1 : 0
    }

  const rest = new Float32Array(N * 3)
  const pos = new Float32Array(N * 3)
  const prev = new Float32Array(N * 3)
  const inv = new Float32Array(N)
  for (let r = 0; r <= SU; r++)
    for (let c = 0; c <= SV; c++) {
      const k = id(r, c)
      const f = rid(2 * r, 2 * c) * 3
      rest[k * 3] = parts.flat[f]
      rest[k * 3 + 1] = CLOTH_Y + REST_UP
      rest[k * 3 + 2] = parts.flat[f + 2]
    }

  // The particle nearest each named point.
  const featureIdx = {} as Record<Feature, number>
  for (const name of Object.keys(FEATURES) as Feature[]) {
    const [fa, fb] = FEATURES[name]
    const x = LAUNDRY.shirt.hemA + fa
    let best = -1
    let bd = Infinity
    for (let k = 0; k < N; k++) {
      if (!active[k]) continue
      const d = Math.hypot(rest[k * 3] - x, rest[k * 3 + 2] - fb)
      if (d < bd) {
        bd = d
        best = k
      }
    }
    featureIdx[name] = best
  }

  // Distance constraints: [a, b, rest length, stiffness].
  const ca: number[] = []
  const cb: number[] = []
  const cl: number[] = []
  const ck: number[] = []
  const cc: number[] = []
  /** k: stiffness against stretching; squash: fraction of it against being pushed closer. */
  const link = (r0: number, c0: number, r1: number, c1: number, k: number, squash = 0.7) => {
    if (r1 > SU || c1 < 0 || c1 > SV) return
    const a = id(r0, c0)
    const b = id(r1, c1)
    if (!active[a] || !active[b]) return
    ca.push(a)
    cb.push(b)
    cl.push(Math.hypot(rest[a * 3] - rest[b * 3], rest[a * 3 + 2] - rest[b * 3 + 2]))
    ck.push(k)
    cc.push(squash)
  }
  for (let r = 0; r <= SU; r++)
    for (let c = 0; c <= SV; c++) {
      link(r, c, r, c + 1, 1) // stretch
      link(r, c, r + 1, c, 1)
      link(r, c, r + 1, c + 1, 0.6) // shear
      link(r, c, r + 1, c - 1, 0.6)
      // Bending: keeps the cloth from crumpling when it's pulled, but never pushes a fold
      // back open (cloth lies folded flat without springing back).
      link(r, c, r, c + 2, 0.3, 0.12)
      link(r, c, r + 2, c, 0.3, 0.12)
    }
  const A = Int32Array.from(ca)
  const B = Int32Array.from(cb)
  const L = Float32Array.from(cl)
  const K = Float32Array.from(ck)
  const Cq = Float32Array.from(cc)

  const pins: Record<'L' | 'R', { k: number; off: [number, number, number] }[]> = { L: [], R: [] }
  let acc = 0
  let still = 0
  const lastHands: Partial<Record<'L' | 'R', THREE.Vector3>> = {}
  const sweeping = { L: false, R: false }
  const lastSweep: Partial<Record<'L' | 'R', THREE.Vector3>> = {}
  const _h2 = new THREE.Vector3()
  const _h = new THREE.Vector3()
  let asleep = false

  let drops = 0
  const reset = (crumpled = true) => {
    pos.set(rest)
    prev.set(rest)
    for (let k = 0; k < N; k++) inv[k] = active[k] ? 1 : 0
    pins.L = []
    pins.R = []
    asleep = false
    still = 0
    if (crumpled) crumple(drops++)
    writeMesh()
  }

  /**
   * Make a heap the way a shirt really lands in one: pick it up by one point, let it
   * hang, swing it a little and let go. Simulated before the scene is shown; each
   * reset picks a different point and swing.
   */
  const crumple = (n: number) => {
    const holds: Feature[] = ['collar', 'hemR', 'sleeveL', 'shoulderR', 'hemL', 'sleeveR']
    const hold = holds[n % holds.length]
    const k0 = featureIdx[hold]
    const start = new THREE.Vector3(pos[k0 * 3], pos[k0 * 3 + 1], pos[k0 * 3 + 2])
    const ang = n * 2.4
    const top = new THREE.Vector3(0.58, LAUNDRY.table.h + 0.42, 0)
    const swing = top.clone().add(new THREE.Vector3(Math.cos(ang) * 0.06, -0.25, Math.sin(ang) * 0.1))
    const hand = new THREE.Vector3()
    pins.R = neighbours(k0).map((k) => ({ k, off: [pos[k * 3] - start.x, pos[k * 3 + 1] - start.y, pos[k * 3 + 2] - start.z] as [number, number, number] }))
    for (const p of pins.R) inv[p.k] = 0
    const run = (from: THREE.Vector3, to: THREE.Vector3, steps: number) => {
      for (let i = 1; i <= steps; i++) {
        const t = i / steps
        hand.lerpVectors(from, to, t * t * (3 - 2 * t))
        lastHands.R = hand.clone()
        substep({ R: hand }, 1, STEP)
      }
    }
    lastHands.R = start.clone()
    run(start, top, 70) // lift: the shirt hangs from that point
    run(top, top, 50)
    run(top, swing, 35) // swing down and drop
    for (const p of pins.R) inv[p.k] = 1
    pins.R = []
    for (let i = 0; i < 480; i++) substep({}, 1, STEP)
    asleep = true
  }

  /** A particle and its neighbours in the fabric (a fingertip-sized patch). */
  const neighbours = (k0: number) => {
    const r0 = Math.floor(k0 / W)
    const c0 = k0 % W
    const out: number[] = []
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const r = r0 + dr
        const c = c0 + dc
        if (r < 0 || r > SU || c < 0 || c > SV) continue
        if (active[id(r, c)]) out.push(id(r, c))
      }
    return out
  }

  // Spatial hash for self-collision.
  const CELL = MIN_DIST
  const BUCKETS = 8192
  const head = new Int32Array(BUCKETS)
  const next = new Int32Array(N)
  const key = (x: number, y: number, z: number) =>
    ((Math.floor(x / CELL) * 73856093) ^ (Math.floor(y / CELL) * 19349663) ^ (Math.floor(z / CELL) * 83492791)) & (BUCKETS - 1)

  const t = LAUNDRY.table
  const floorY = t.h + REST_UP
  let stick = STICK
  const collideTable = (k: number) => {
    const x = pos[k * 3]
    const z = pos[k * 3 + 2]
    const onTable = x > t.a0 && x < t.a1 && z > t.b0 && z < t.b1
    const lim = onTable ? floorY : 0.003
    if (pos[k * 3 + 1] < lim) {
      pos[k * 3 + 1] = lim
      // Friction, applied to position (as position-based dynamics does): cotton on wood stays
      // exactly where it lies unless it's pulled firmly, then slides with the excess.
      const sx = x - prev[k * 3]
      const sz = z - prev[k * 3 + 2]
      const d = Math.hypot(sx, sz)
      if (d < stick) {
        pos[k * 3] = prev[k * 3]
        pos[k * 3 + 2] = prev[k * 3 + 2]
      } else {
        const keep = 1 - stick / d
        pos[k * 3] = prev[k * 3] + sx * keep
        pos[k * 3 + 2] = prev[k * 3 + 2] + sz * keep
      }
    }
  }

  const selfCollide = () => {
    head.fill(-1)
    for (let k = 0; k < N; k++) {
      if (!active[k]) continue
      const h = key(pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2])
      next[k] = head[h]
      head[h] = k
    }
    const d2min = MIN_DIST * MIN_DIST
    for (let k = 0; k < N; k++) {
      if (!active[k]) continue
      const x = pos[k * 3]
      const y = pos[k * 3 + 1]
      const z = pos[k * 3 + 2]
      const rk = Math.floor(k / W)
      const ck_ = k % W
      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            for (let m = head[key(x + dx * CELL, y + dy * CELL, z + dz * CELL)]; m >= 0; m = next[m]) {
              if (m <= k) continue
              // Neighbours in the fabric are held apart by the constraints already.
              if (Math.abs(Math.floor(m / W) - rk) <= 2 && Math.abs((m % W) - ck_) <= 2) continue
              const ex = pos[m * 3] - x
              const ey = pos[m * 3 + 1] - y
              const ez = pos[m * 3 + 2] - z
              const d2 = ex * ex + ey * ey + ez * ez
              if (d2 >= d2min || d2 < 1e-12) continue
              const w = inv[k] + inv[m]
              if (!w) continue
              const d = Math.sqrt(d2)
              const s = (MIN_DIST - d) / d / w
              pos[k * 3] -= ex * s * inv[k]
              pos[k * 3 + 1] -= ey * s * inv[k]
              pos[k * 3 + 2] -= ez * s * inv[k]
              pos[m * 3] += ex * s * inv[m]
              pos[m * 3 + 1] += ey * s * inv[m]
              pos[m * 3 + 2] += ez * s * inv[m]
              // Cotton grips cotton: layers in contact mostly move together.
              for (let e = 0; e < 3; e++) {
                const vk = pos[k * 3 + e] - prev[k * 3 + e]
                const vm = pos[m * 3 + e] - prev[m * 3 + e]
                const dv = (vm - vk) * LAYER_GRIP * 0.5
                prev[k * 3 + e] -= dv * (inv[k] / w) * 2
                prev[m * 3 + e] += dv * (inv[m] / w) * 2
              }
            }
          }
    }
  }

  let hPrev = STEP
  const substep = (hands: Partial<Record<'L' | 'R', THREE.Vector3>>, t: number, h: number) => {
    const g = GRAVITY * h * h
    const vs = (1 - DAMPING) * (h / hPrev)
    hPrev = h
    stick = STICK * (h / STEP)
    let moved = 0
    for (let k = 0; k < N; k++) {
      if (!inv[k]) continue
      const i = k * 3
      for (let a = 0; a < 3; a++) {
        const p = pos[i + a]
        const v = (p - prev[i + a]) * vs
        prev[i + a] = p
        pos[i + a] = p + v + (a === 1 ? g : 0)
      }
    }
    for (const side of ['L', 'R'] as const) {
      const cur = hands[side]
      if (!cur) continue
      // Where the hand is part-way through this frame (the robot moves once per frame,
      // the cloth several times), so held cloth is carried smoothly instead of jerked.
      const was = lastHands[side] ?? cur
      const hp = _h.lerpVectors(was, cur, t)
      for (const pin of pins[side]) {
        const i = pin.k * 3
        prev[i] = pos[i]
        prev[i + 1] = pos[i + 1]
        prev[i + 2] = pos[i + 2]
        pos[i] = hp.x + pin.off[0]
        pos[i + 1] = hp.y + pin.off[1]
        pos[i + 2] = hp.z + pin.off[2]
      }
    }
    for (let it = 0; it < ITERS; it++) {
      for (let c = 0; c < A.length; c++) {
        const a = A[c]
        const b = B[c]
        const w = inv[a] + inv[b]
        if (!w) continue
        const ex = pos[b * 3] - pos[a * 3]
        const ey = pos[b * 3 + 1] - pos[a * 3 + 1]
        const ez = pos[b * 3 + 2] - pos[a * 3 + 2]
        const d = Math.sqrt(ex * ex + ey * ey + ez * ez)
        if (d < 1e-9) continue
        // Cloth resists stretching more than compressing (it buckles instead).
        const diff = d - L[c]
        const k = diff > 0 ? K[c] : K[c] * Cq[c]
        if (!k) continue
        const s = (diff / d / w) * k
        pos[a * 3] += ex * s * inv[a]
        pos[a * 3 + 1] += ey * s * inv[a]
        pos[a * 3 + 2] += ez * s * inv[a]
        pos[b * 3] -= ex * s * inv[b]
        pos[b * 3 + 1] -= ey * s * inv[b]
        pos[b * 3 + 2] -= ez * s * inv[b]
      }
      for (let k = 0; k < N; k++) if (active[k]) collideTable(k)
    }
    selfCollide()
    for (let k = 0; k < N; k++) if (active[k]) collideTable(k)
    // Smoothing hands: everything under the fingertips moves with the hand.
    for (const side of ['L', 'R'] as const) {
      const cur = hands[side]
      if (!sweeping[side] || !cur) continue
      const hp = _h2.lerpVectors(lastHands[side] ?? cur, cur, t)
      const last = lastSweep[side] ?? hp.clone()
      const dx = hp.x - last.x
      const dz = hp.z - last.z
      lastSweep[side] = hp.clone()
      for (let k = 0; k < N; k++) {
        if (!inv[k]) continue
        const ex = pos[k * 3] - hp.x
        const ez = pos[k * 3 + 2] - hp.z
        if (ex * ex + ez * ez > SWEEP_R * SWEEP_R || pos[k * 3 + 1] > hp.y + 0.004) continue
        pos[k * 3] += dx
        pos[k * 3 + 2] += dz
        prev[k * 3] = pos[k * 3]
        prev[k * 3 + 1] = pos[k * 3 + 1]
        prev[k * 3 + 2] = pos[k * 3 + 2]
      }
    }
    for (let k = 0; k < N; k++) {
      if (!active[k]) continue
      const i = k * 3
      moved = Math.max(moved, Math.abs(pos[i] - prev[i]) + Math.abs(pos[i + 1] - prev[i + 1]) + Math.abs(pos[i + 2] - prev[i + 2]))
    }
    return moved
  }

  /** Fill the drawn mesh from the particles, adding the fine wrinkles back. */
  const writeMesh = () => {
    const out = parts.verts
    for (let i = 0; i <= CLOTH_NU; i++) {
      const r0 = i >> 1
      const r1 = Math.min(SU, r0 + (i & 1))
      for (let j = 0; j <= CLOTH_NV; j++) {
        const c0 = j >> 1
        const c1 = Math.min(SV, c0 + (j & 1))
        const q = rid(i, j) * 3
        const a = id(r0, c0) * 3
        const b = id(r0, c1) * 3
        const c = id(r1, c0) * 3
        const d = id(r1, c1) * 3
        const wrinkle = parts.flat[q + 1] - CLOTH_Y - 0.0015
        for (let e = 0; e < 3; e++) out[q + e] = (pos[a + e] + pos[b + e] + pos[c + e] + pos[d + e]) / 4
        out[q + 1] += wrinkle
      }
    }
    const attr = parts.mesh.geometry.getAttribute('position') as THREE.BufferAttribute
    attr.needsUpdate = true
    parts.mesh.geometry.computeVertexNormals()
  }

  reset()

  return {
    step: (dt, hands) => {
      const held = pins.L.length + pins.R.length > 0 || sweeping.L || sweeping.R
      if (asleep && !held) {
        for (const side of ['L', 'R'] as const) if (hands[side]) lastHands[side] = hands[side]!.clone()
        return
      }
      acc = Math.min(acc + dt, 0.5)
      if (acc < STEP) return
      const n = Math.min(MAX_STEPS, Math.floor(acc / STEP))
      const h = acc / n
      acc = 0
      let moved = 0
      for (let k = 1; k <= n; k++) moved = Math.max(moved, substep(hands, k / n, h))
      for (const side of ['L', 'R'] as const) if (hands[side]) lastHands[side] = hands[side]!.clone()
      // Sleep once everything has stopped, so a folded shirt costs nothing.
      still = !held && moved < 2e-5 ? still + 1 : 0
      if (still > 30) asleep = true
      writeMesh()
    },
    pinch: (side, tcp) => {
      pins[side] = []
      // The fingers close where the cloth is: if earlier folds have pulled its edge in a
      // little, they catch the nearest cloth within reach of the hand.
      let cx = tcp.x
      let cz = tcp.z
      let best = REACH * REACH
      for (let k = 0; k < N; k++) {
        if (!active[k] || Math.abs(pos[k * 3 + 1] - tcp.y) > 0.04) continue
        const dx = pos[k * 3] - tcp.x
        const dz = pos[k * 3 + 2] - tcp.z
        const d2 = dx * dx + dz * dz
        if (d2 < PINCH_R * PINCH_R) {
          cx = tcp.x
          cz = tcp.z
          break
        }
        if (d2 < best) {
          best = d2
          cx = pos[k * 3]
          cz = pos[k * 3 + 2]
        }
      }
      for (let k = 0; k < N; k++) {
        if (!active[k]) continue
        const dx = pos[k * 3] - tcp.x
        const dy = pos[k * 3 + 1] - tcp.y
        const dz = pos[k * 3 + 2] - tcp.z
        const ex = pos[k * 3] - cx
        const ez = pos[k * 3 + 2] - cz
        if (ex * ex + ez * ez < PINCH_R * PINCH_R && Math.abs(dy) < 0.04) {
          pins[side].push({ k, off: [dx, dy, dz] })
          inv[k] = 0
        }
      }
      asleep = false
      still = 0
      if (import.meta.env.DEV) console.debug(`[cloth] ${side} pinch at ${tcp.toArray().map((v) => v.toFixed(3))}: ${pins[side].length} particles`)
      return pins[side].length
    },
    grabAt: (side, tcp) => {
      for (const p of pins[side]) inv[p.k] = 1
      pins[side] = []
      let k0 = -1
      let bd = GRASP_TOL
      for (let k = 0; k < N; k++) {
        if (!active[k]) continue
        const d = Math.hypot(pos[k * 3] - tcp.x, pos[k * 3 + 1] - tcp.y, pos[k * 3 + 2] - tcp.z)
        if (d < bd) {
          bd = d
          k0 = k
        }
      }
      if (k0 < 0) return 0
      for (let k = 0; k < N; k++) {
        if (!active[k]) continue
        if (Math.hypot(pos[k * 3] - pos[k0 * 3], pos[k * 3 + 2] - pos[k0 * 3 + 2]) < STACK_R && Math.abs(pos[k * 3 + 1] - pos[k0 * 3 + 1]) < 0.04)
          pins[side].push({ k, off: [pos[k * 3] - tcp.x, pos[k * 3 + 1] - tcp.y, pos[k * 3 + 2] - tcp.z] })
      }
      for (const p of pins[side]) inv[p.k] = 0
      asleep = false
      still = 0
      return pins[side].length
    },
    sweep: (side, on) => {
      sweeping[side] = on
      delete lastSweep[side]
      asleep = false
      still = 0
    },
    release: (side) => {
      for (const p of pins[side]) inv[p.k] = 1
      pins[side] = []
      asleep = false
      still = 0
    },
    reset,
    sleeping: () => asleep,
    feature: (name) => {
      const k = featureIdx[name]
      return new THREE.Vector3(pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2])
    },
    grabFeature: (side, name, tcp, stack = false) => {
      for (const p of pins[side]) inv[p.k] = 1
      pins[side] = []
      const k0 = featureIdx[name]
      // The fingers only catch the cloth if they are actually at it.
      if (Math.hypot(pos[k0 * 3] - tcp.x, pos[k0 * 3 + 1] - tcp.y, pos[k0 * 3 + 2] - tcp.z) > GRASP_TOL) return 0
      const set = new Set(neighbours(k0))
      if (stack) {
        // Every layer lying under or over that spot (a folded edge is several layers thick).
        const x = pos[k0 * 3]
        const z = pos[k0 * 3 + 2]
        for (let k = 0; k < N; k++) {
          if (!active[k]) continue
          if (Math.hypot(pos[k * 3] - x, pos[k * 3 + 2] - z) < STACK_R && Math.abs(pos[k * 3 + 1] - pos[k0 * 3 + 1]) < 0.04) set.add(k)
        }
      }
      pins[side] = [...set].map((k) => ({ k, off: [pos[k * 3] - tcp.x, pos[k * 3 + 1] - tcp.y, pos[k * 3 + 2] - tcp.z] as [number, number, number] }))
      for (const p of pins[side]) inv[p.k] = 0
      asleep = false
      still = 0
      if (import.meta.env.DEV)
        console.debug(`[cloth] ${side} grabs ${name}: ${pins[side].length} particles, ${(Math.hypot(pos[k0 * 3] - tcp.x, pos[k0 * 3 + 1] - tcp.y, pos[k0 * 3 + 2] - tcp.z) * 1000).toFixed(0)} mm from the fingertips`)
      return pins[side].length
    },
    probe: (x: number, z: number) => {
      let best = 1e9
      let at: number[] = []
      let n = 0
      for (let k = 0; k < N; k++) {
        if (!active[k]) continue
        n++
        const d = Math.hypot(pos[k * 3] - x, pos[k * 3 + 2] - z)
        if (d < best) {
          best = d
          at = [pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2]]
        }
      }
      return { active: n, nearest: best, at }
    },
  }
}
