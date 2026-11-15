/**
 * Cloth physics for the towels dropped on the floor, the same way as the folding towel:
 * position-based dynamics on a particle grid.
 *
 * Each towel is a grid of particles joined by distance constraints (stretch to the next
 * particle, shear across the diagonals, a softer bend two particles away). Every 1/120 s
 * step moves the particles under gravity with a little air damping, then nudges every
 * constrained pair back toward its rest length. Particles can't go through the floor (with
 * friction, so a towel doesn't skate), furniture, the robot's base, the hamper's walls and
 * floor, or each other (self- and towel-to-towel collision, so towels bunch up and pile).
 * A hand pinches the particles under its fingertips; they follow the fingertips and the
 * rest of the towel hangs from them. A towel nobody touches settles and goes to sleep.
 *
 * Positions are in world coordinates; the drawn mesh uses the particles as its vertices.
 */
import * as THREE from 'three'

const STEP = 1 / 120
const ITERS = 8
const GRAVITY = -9.81
const DAMPING = 0.015
/** Particles keep this far apart (cloth thickness), within a towel and between towels. */
const MIN_DIST = 0.012
/** Cloth rests this far above any surface. */
const REST_UP = 0.004
/** Cotton on the floor: static and sliding friction. */
const MU_STATIC = 1.0
const MU_KINETIC = 0.7
/** Particles within this distance of the fingertips are pinched. */
const PINCH_R = 0.05
/** Sleep this long (s) after a towel was last moving. */
const SLEEP_AFTER = 1.5
const MAX_STEPS = 12

export interface Box {
  x0: number
  x1: number
  z0: number
  z1: number
  y0: number
  y1: number
}
export interface Tub {
  x: number
  z: number
  r: number
  h: number
  /** Height of the inside floor. */
  floor: number
}

export interface FloorTowel {
  readonly mesh: THREE.Mesh
  /** Where to grab it: the particle nearest its middle, from above. */
  grabPoint(): THREE.Vector3
  /** Middle of the towel (mean of its particles). */
  centroid(): THREE.Vector3
  /** Horizontal radius it covers round its centroid. */
  spread(): number
  /** Pinch the particles under a hand's fingertips: how many were caught (0 = missed). */
  pinch(side: 'L' | 'R', tcp: THREE.Vector3): number
  release(side: 'L' | 'R'): void
  held(): boolean
  reset(): void
}

/** A loose object on the floor (a mug, a can) the cloth drapes over, as a sphere. */
export interface Ball {
  x: number
  y: number
  z: number
  r: number
}

export interface FloorCloth {
  towels: FloorTowel[]
  /**
   * Advance by dt seconds; hands: fingertip positions; base: the robot base as a box in world
   * space (turned by yaw); balls: loose things on the floor where they are now.
   */
  step(dt: number, hands: Partial<Record<'L' | 'R', THREE.Vector3>>, base: { x: number; z: number; yaw: number; half: [number, number]; top: number } | null, balls?: Ball[]): void
  sleeping(): boolean
}

interface T {
  n: number
  nu: number
  nv: number
  p: Float32Array
  prev: Float32Array
  rest: Float32Array
  cons: { a: number; b: number; len: number; k: number }[]
  pins: Record<'L' | 'R', { idx: number[]; off: Float32Array } | null>
  mesh: THREE.Mesh
  awake: boolean
  quietFor: number
}

/**
 * Make the towels. Each starts as a flat grid (w x l metres, nu x nv particles) at a place and
 * heading, is scrunched up, then settled on the floor so it lies crumpled.
 */
export function createFloorCloth(
  starts: { x: number; z: number; yaw: number; material: THREE.Material; seed: number }[],
  size: { w: number; l: number; nu: number; nv: number },
  boxes: Box[],
  tubs: Tub[],
  startBalls: Ball[] = [],
): FloorCloth {
  const all: T[] = []
  let balls: Ball[] = startBalls

  const make = (s: (typeof starts)[number]): T => {
    const { w, l, nu, nv } = size
    const n = nu * nv
    const p = new Float32Array(n * 3)
    const rand = rng(s.seed)
    // Scrunched: a grid squeezed toward its middle, with a few soft ridges standing up.
    const ridges = Array.from({ length: 4 }, () => ({ u: (rand() - 0.5) * w, v: (rand() - 0.5) * l, a: rand() * Math.PI, h: 0.03 + rand() * 0.05 }))
    const c = Math.cos(s.yaw)
    const sn = Math.sin(s.yaw)
    for (let j = 0; j < nv; j++)
      for (let i = 0; i < nu; i++) {
        const u = (i / (nu - 1) - 0.5) * w
        const v = (j / (nv - 1) - 0.5) * l
        let y = 0.02
        for (const r of ridges) {
          const d = (u - r.u) * Math.cos(r.a) + (v - r.v) * Math.sin(r.a)
          const along = (u - r.u) * -Math.sin(r.a) + (v - r.v) * Math.cos(r.a)
          y += r.h * Math.exp(-(d * d) / 0.002) * Math.exp(-(along * along) / 0.03)
        }
        const su = u * 0.72
        const sv = v * 0.78
        const k = (j * nu + i) * 3
        p[k] = s.x + su * c + sv * sn
        p[k + 1] = y
        p[k + 2] = s.z - su * sn + sv * c
      }
    // Constraints use the flat (unscrunched) spacing, so the cloth wants to spread back out
    // but friction on the floor stops it: it settles bunched, the way a dropped towel lies.
    const du = w / (nu - 1)
    const dv = l / (nv - 1)
    const cons: T['cons'] = []
    const at = (i: number, j: number) => j * nu + i
    for (let j = 0; j < nv; j++)
      for (let i = 0; i < nu; i++) {
        if (i + 1 < nu) cons.push({ a: at(i, j), b: at(i + 1, j), len: du, k: 1 })
        if (j + 1 < nv) cons.push({ a: at(i, j), b: at(i, j + 1), len: dv, k: 1 })
        if (i + 1 < nu && j + 1 < nv) {
          const dd = Math.hypot(du, dv)
          cons.push({ a: at(i, j), b: at(i + 1, j + 1), len: dd, k: 0.6 }, { a: at(i + 1, j), b: at(i, j + 1), len: dd, k: 0.6 })
        }
        if (i + 2 < nu) cons.push({ a: at(i, j), b: at(i + 2, j), len: 2 * du, k: 0.3 })
        if (j + 2 < nv) cons.push({ a: at(i, j), b: at(i, j + 2), len: 2 * dv, k: 0.3 })
      }
    const geo = new THREE.PlaneGeometry(w, l, nu - 1, nv - 1)
    // PlaneGeometry orders vertices row by row from the top (+y): flip rows so index = j * nu + i.
    const uv = geo.attributes.uv
    const pos = new THREE.BufferAttribute(new Float32Array(n * 3), 3)
    const uvs = new Float32Array(n * 2)
    for (let j = 0; j < nv; j++)
      for (let i = 0; i < nu; i++) {
        const src = (nv - 1 - j) * nu + i
        uvs[(j * nu + i) * 2] = uv.getX(src)
        uvs[(j * nu + i) * 2 + 1] = uv.getY(src)
      }
    const index: number[] = []
    for (let j = 0; j < nv - 1; j++)
      for (let i = 0; i < nu - 1; i++) {
        const a = at(i, j)
        const b = at(i + 1, j)
        const c2 = at(i, j + 1)
        const d = at(i + 1, j + 1)
        index.push(a, c2, b, b, c2, d)
      }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', pos)
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
    g.setIndex(index)
    geo.dispose()
    const mesh = new THREE.Mesh(g, s.material)
    mesh.castShadow = true
    mesh.receiveShadow = true
    mesh.frustumCulled = false
    const t: T = { n, nu, nv, p, prev: p.slice(), rest: p.slice(), cons, pins: { L: null, R: null }, mesh, awake: true, quietFor: 0 }
    return t
  }

  for (const s of starts) all.push(make(s))

  const draw = (t: T) => {
    const pos = t.mesh.geometry.attributes.position as THREE.BufferAttribute
    ;(pos.array as Float32Array).set(t.p)
    pos.needsUpdate = true
    t.mesh.geometry.computeVertexNormals()
    t.mesh.geometry.computeBoundingSphere()
  }

  // --- collisions with the world ---------------------------------------------------------
  const collide = (t: T, i: number, base: Parameters<FloorCloth['step']>[2]) => {
    const k = i * 3
    const p = t.p
    const q = t.prev
    let x = p[k]
    let y = p[k + 1]
    let z = p[k + 2]
    let ground = REST_UP
    // Hamper and bin: inside, the floor is raised and the wall keeps it in; outside, the wall keeps it out.
    for (const b of tubs) {
      const d = Math.hypot(x - b.x, z - b.z)
      const dPrev = Math.hypot(q[k] - b.x, q[k + 2] - b.z)
      if (y > b.h + 0.01) continue
      const inside = dPrev < b.r
      if (inside) {
        ground = Math.max(ground, b.floor + REST_UP)
        const lim = b.r - 0.016
        if (d > lim) {
          x = b.x + ((x - b.x) / d) * lim
          z = b.z + ((z - b.z) / d) * lim
        }
      } else if (d < b.r + 0.012) {
        const s = (b.r + 0.012) / Math.max(d, 1e-6)
        x = b.x + (x - b.x) * s
        z = b.z + (z - b.z) * s
      }
    }
    // Furniture (tables, counters, the washer): pushed out of the nearest face.
    for (const b of boxes) {
      if (x < b.x0 || x > b.x1 || z < b.z0 || z > b.z1 || y < b.y0 || y > b.y1 + REST_UP) continue
      const opts: [number, number][] = [
        [b.y1 + REST_UP - y, 1],
        [x - b.x0, 0],
        [b.x1 - x, 2],
        [z - b.z0, 3],
        [b.z1 - z, 4],
      ]
      const [, face] = opts.reduce((m, o) => (o[0] < m[0] ? o : m))
      if (face === 1) y = b.y1 + REST_UP
      else if (face === 0) x = b.x0
      else if (face === 2) x = b.x1
      else if (face === 3) z = b.z0
      else z = b.z1
    }
    // Loose things on the floor: out to their surface.
    for (const b of balls) {
      const dx = x - b.x
      const dy = y - b.y
      const dz = z - b.z
      const d = Math.hypot(dx, dy, dz)
      const lim = b.r + REST_UP
      if (d < lim && d > 1e-6) {
        x = b.x + (dx / d) * lim
        y = b.y + (dy / d) * lim
        z = b.z + (dz / d) * lim
      }
    }
    // The robot's base (a turned box): out over its top, or out of its nearest side.
    if (base) {
      const c = Math.cos(base.yaw)
      const sn = Math.sin(base.yaw)
      const dx = x - base.x
      const dz = z - base.z
      const u = dx * c - dz * sn
      const v = dx * sn + dz * c
      const [hu, hv] = base.half
      if (Math.abs(u) < hu && Math.abs(v) < hv && y < base.top) {
        const pens = [base.top - y, hu - Math.abs(u), hv - Math.abs(v)]
        const m = pens.indexOf(Math.min(...pens))
        if (m === 0) y = base.top
        else {
          const nu = m === 1 ? Math.sign(u) * hu : u
          const nv = m === 2 ? Math.sign(v) * hv : v
          x = base.x + nu * c + nv * sn
          z = base.z - nu * sn + nv * c
        }
      }
    }
    // The ground (floor, or the hamper's floor), with friction.
    if (y < ground) {
      const pen = ground - y
      y = ground
      const mx = x - q[k]
      const mz = z - q[k + 2]
      const slide = Math.hypot(mx, mz)
      if (slide < MU_STATIC * pen) {
        x = q[k]
        z = q[k + 2]
      } else if (slide > 0) {
        const keep = Math.max(0, 1 - (MU_KINETIC * pen) / slide)
        x = q[k] + mx * keep
        z = q[k + 2] + mz * keep
      }
    }
    p[k] = x
    p[k + 1] = y
    p[k + 2] = z
  }

  // --- self and towel-to-towel collision (spatial hash) ------------------------------------
  const cell = MIN_DIST
  const hash = new Map<number, number[]>()
  const key = (x: number, y: number, z: number) => ((Math.floor(x / cell) * 73856093) ^ (Math.floor(y / cell) * 19349663) ^ (Math.floor(z / cell) * 83492791)) | 0
  const separate = (awake: T[]) => {
    hash.clear()
    const ref: [T, number][] = []
    for (const t of awake)
      for (let i = 0; i < t.n; i++) {
        const id = ref.length
        ref.push([t, i])
        const k = key(t.p[i * 3], t.p[i * 3 + 1], t.p[i * 3 + 2])
        let b = hash.get(k)
        if (!b) hash.set(k, (b = []))
        b.push(id)
      }
    for (let id = 0; id < ref.length; id++) {
      const [ta, ia] = ref[id]
      const ax = ta.p[ia * 3]
      const ay = ta.p[ia * 3 + 1]
      const az = ta.p[ia * 3 + 2]
      const cx = Math.floor(ax / cell)
      const cy = Math.floor(ay / cell)
      const cz = Math.floor(az / cell)
      for (let ox = -1; ox <= 1; ox++)
        for (let oy = -1; oy <= 1; oy++)
          for (let oz = -1; oz <= 1; oz++) {
            const b = hash.get((((cx + ox) * 73856093) ^ ((cy + oy) * 19349663) ^ ((cz + oz) * 83492791)) | 0)
            if (!b) continue
            for (const other of b) {
              if (other <= id) continue
              const [tb, ib] = ref[other]
              // Grid neighbours are already held apart by the constraints.
              if (ta === tb) {
                const ua = ia % ta.nu
                const va = Math.floor(ia / ta.nu)
                const ub = ib % ta.nu
                const vb = Math.floor(ib / ta.nu)
                if (Math.abs(ua - ub) <= 1 && Math.abs(va - vb) <= 1) continue
              }
              const dx = tb.p[ib * 3] - ax
              const dy = tb.p[ib * 3 + 1] - ay
              const dz = tb.p[ib * 3 + 2] - az
              const d = Math.hypot(dx, dy, dz)
              if (d >= MIN_DIST || d < 1e-7) continue
              const f = (MIN_DIST - d) / d / 2
              ta.p[ia * 3] -= dx * f
              ta.p[ia * 3 + 1] -= dy * f
              ta.p[ia * 3 + 2] -= dz * f
              tb.p[ib * 3] += dx * f
              tb.p[ib * 3 + 1] += dy * f
              tb.p[ib * 3 + 2] += dz * f
            }
          }
    }
  }

  const stepOnce = (awake: T[], hands: Partial<Record<'L' | 'R', THREE.Vector3>>, base: Parameters<FloorCloth['step']>[2]) => {
    for (const t of awake) {
      const p = t.p
      const q = t.prev
      for (let i = 0; i < t.n; i++) {
        const k = i * 3
        for (let a = 0; a < 3; a++) {
          const v = (p[k + a] - q[k + a]) * (1 - DAMPING)
          q[k + a] = p[k + a]
          p[k + a] += v + (a === 1 ? GRAVITY * STEP * STEP : 0)
        }
      }
    }
    for (let it = 0; it < ITERS; it++) {
      for (const t of awake) {
        const p = t.p
        for (const c of t.cons) {
          const a = c.a * 3
          const b = c.b * 3
          const dx = p[b] - p[a]
          const dy = p[b + 1] - p[a + 1]
          const dz = p[b + 2] - p[a + 2]
          const d = Math.hypot(dx, dy, dz) || 1e-9
          // Cloth hardly stretches, but squashes and bunches up almost freely: a dropped towel
          // stays crumpled instead of springing back out flat.
          const diff = (d - c.len) / d
          const s = (diff > 0 ? c.k : c.k * 0.12) * 0.5 * diff
          p[a] += dx * s
          p[a + 1] += dy * s
          p[a + 2] += dz * s
          p[b] -= dx * s
          p[b + 1] -= dy * s
          p[b + 2] -= dz * s
        }
        // Pinched particles follow the fingertips.
        for (const side of ['L', 'R'] as const) {
          const pin = t.pins[side]
          const h = hands[side]
          if (!pin || !h) continue
          pin.idx.forEach((i, m) => {
            p[i * 3] = h.x + pin.off[m * 3]
            p[i * 3 + 1] = h.y + pin.off[m * 3 + 1]
            p[i * 3 + 2] = h.z + pin.off[m * 3 + 2]
          })
        }
        for (let i = 0; i < t.n; i++) collide(t, i, base)
      }
      if (it % 2 === 1) separate(awake)
    }
  }

  const reset = (t: T) => {
    t.p.set(t.rest)
    t.prev.set(t.rest)
    t.pins = { L: null, R: null }
    t.awake = true
    t.quietFor = 0
  }

  // Drop each towel, the way it got onto the floor: held up by one point so it hangs, carried
  // a little sideways, then let go. It falls in folds and crumples where it lands.
  all.forEach((t, k) => {
    const rand = rng(starts[k].seed + 1)
    const pick = Math.floor(rand() * t.n)
    const c = new THREE.Vector3(starts[k].x, 0, starts[k].z)
    const drift = new THREE.Vector3(rand() - 0.5, 0, rand() - 0.5).normalize().multiplyScalar(0.3)
    // Start flat in the air round the pinch point.
    const px = t.p[pick * 3] - c.x
    const pz = t.p[pick * 3 + 2] - c.z
    for (let i = 0; i < t.n; i++) {
      t.p[i * 3] -= px
      t.p[i * 3 + 1] = 0.32
      t.p[i * 3 + 2] -= pz
    }
    t.prev.set(t.p)
    t.pins.L = { idx: [pick], off: new Float32Array(3) }
    const hold = new THREE.Vector3()
    for (let s = 0; s < 150; s++) {
      hold.copy(c).addScaledVector(drift, s / 150)
      hold.y = 0.32
      stepOnce([t], { L: hold }, null)
    }
    t.pins.L = null
    for (let s = 0; s < 360; s++) stepOnce([t], {}, null)
  })
  for (const t of all) {
    t.rest = t.p.slice()
    t.prev.set(t.p)
    t.awake = false
    draw(t)
  }

  let acc = 0
  const step: FloorCloth['step'] = (dt, hands, base, now) => {
    if (now) balls = now
    const awake = all.filter((t) => t.awake || t.pins.L || t.pins.R)
    if (!awake.length) {
      acc = 0
      return
    }
    acc = Math.min(acc + dt, STEP * MAX_STEPS)
    let n = 0
    while (acc >= STEP) {
      stepOnce(awake, hands, base)
      acc -= STEP
      n++
    }
    if (!n) return
    for (const t of awake) {
      // Asleep once nothing holds it and it has stopped moving for a while.
      let maxV = 0
      for (let i = 0; i < t.p.length; i++) maxV = Math.max(maxV, Math.abs(t.p[i] - t.prev[i]))
      const quiet = !t.pins.L && !t.pins.R && maxV < 0.0004
      t.quietFor = quiet ? t.quietFor + n * STEP : 0
      t.awake = t.quietFor < SLEEP_AFTER
      if (!t.awake) t.prev.set(t.p)
      draw(t)
    }
  }

  const api = (t: T): FloorTowel => ({
    mesh: t.mesh,
    centroid: () => {
      const c = new THREE.Vector3()
      for (let i = 0; i < t.n; i++) c.add(new THREE.Vector3(t.p[i * 3], t.p[i * 3 + 1], t.p[i * 3 + 2]))
      return c.multiplyScalar(1 / t.n)
    },
    grabPoint: () => {
      // The highest particle near the middle: what fingers coming down would meet first.
      const c = new THREE.Vector3()
      for (let i = 0; i < t.n; i++) c.add(new THREE.Vector3(t.p[i * 3], 0, t.p[i * 3 + 2]))
      c.multiplyScalar(1 / t.n)
      let best = -1
      let bestScore = Infinity
      for (let i = 0; i < t.n; i++) {
        const d = Math.hypot(t.p[i * 3] - c.x, t.p[i * 3 + 2] - c.z)
        const score = d - t.p[i * 3 + 1] * 0.5
        if (d < 0.06 && score < bestScore) {
          bestScore = score
          best = i
        }
      }
      if (best < 0) best = Math.floor(t.n / 2)
      return new THREE.Vector3(t.p[best * 3], t.p[best * 3 + 1], t.p[best * 3 + 2])
    },
    spread: () => {
      let cx = 0
      let cz = 0
      for (let i = 0; i < t.n; i++) {
        cx += t.p[i * 3]
        cz += t.p[i * 3 + 2]
      }
      cx /= t.n
      cz /= t.n
      let r = 0
      for (let i = 0; i < t.n; i++) r = Math.max(r, Math.hypot(t.p[i * 3] - cx, t.p[i * 3 + 2] - cz))
      return r
    },
    pinch: (side, tcp) => {
      const idx: number[] = []
      for (let i = 0; i < t.n; i++) if (Math.hypot(t.p[i * 3] - tcp.x, t.p[i * 3 + 1] - tcp.y, t.p[i * 3 + 2] - tcp.z) < PINCH_R) idx.push(i)
      if (!idx.length) return 0
      // Gathered into the hand: the pinched cloth bunches up between the fingers.
      const off = new Float32Array(idx.length * 3)
      idx.forEach((i, m) => {
        off[m * 3] = (t.p[i * 3] - tcp.x) * 0.25
        off[m * 3 + 1] = (t.p[i * 3 + 1] - tcp.y) * 0.25
        off[m * 3 + 2] = (t.p[i * 3 + 2] - tcp.z) * 0.25
      })
      t.pins[side] = { idx, off }
      t.awake = true
      t.quietFor = 0
      return idx.length
    },
    release: (side) => {
      t.pins[side] = null
      t.awake = true
      t.quietFor = 0
    },
    held: () => !!(t.pins.L || t.pins.R),
    reset: () => {
      reset(t)
      t.awake = false
      draw(t)
    },
  })

  return {
    towels: all.map(api),
    step,
    sleeping: () => all.every((t) => !t.awake && !t.pins.L && !t.pins.R),
  }
}

function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}
