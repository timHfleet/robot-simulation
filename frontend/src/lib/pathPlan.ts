/**
 * Path planning for the base: A* on a 5 cm grid over the floor.
 *
 * - Hard: furniture, crates, barrels and the room's walls, grown by the base's turning
 *   radius, so the base can turn on the spot anywhere along the path without touching them.
 * - Soft: litter and other loose things on the floor. Cells near them cost a lot, so the
 *   path goes round them when it can, and only squeezes past when there is no other way.
 *
 * The grid path is then pulled tight into a few straight runs (a shortcut is taken only
 * if it is as clear as the stretch of path it replaces).
 */
export interface Rect {
  x0: number
  x1: number
  z0: number
  z1: number
}
export interface Disc {
  x: number
  z: number
  r: number
}
export interface Pt {
  x: number
  z: number
}

const RES = 0.05

export class FloorGrid {
  readonly n: number
  readonly half: number
  /** 1 where the base's centre must not go. */
  readonly hard: Uint8Array

  /** `bounds`: half the room's size; `turnR`: how far the base's corners reach from its centre. */
  constructor(bounds: number, rects: Rect[], discs: Disc[], turnR: number, wallMargin: number) {
    this.half = bounds
    this.n = Math.round((2 * bounds) / RES)
    this.hard = new Uint8Array(this.n * this.n)
    for (let i = 0; i < this.n; i++)
      for (let j = 0; j < this.n; j++) {
        const x = this.x(i)
        const z = this.z(j)
        let bad = Math.abs(x) > bounds - wallMargin || Math.abs(z) > bounds - wallMargin
        for (const r of rects) {
          if (bad) break
          const dx = Math.max(r.x0 - x, 0, x - r.x1)
          const dz = Math.max(r.z0 - z, 0, z - r.z1)
          if (dx * dx + dz * dz < turnR * turnR) bad = true
        }
        for (const d of discs) {
          if (bad) break
          if (Math.hypot(x - d.x, z - d.z) < d.r + turnR) bad = true
        }
        this.hard[i * this.n + j] = bad ? 1 : 0
      }
  }
  x(i: number) {
    return -this.half + (i + 0.5) * RES
  }
  z(j: number) {
    return -this.half + (j + 0.5) * RES
  }
  i(x: number) {
    return Math.max(0, Math.min(this.n - 1, Math.floor((x + this.half) / RES)))
  }
  j(z: number) {
    return Math.max(0, Math.min(this.n - 1, Math.floor((z + this.half) / RES)))
  }
  isHard(x: number, z: number) {
    return this.hard[this.i(x) * this.n + this.j(z)] === 1
  }
}

/** Driving straight, the base's sides pass this close to its centre line. */
const HALF_WIDTH = 0.36 // half its width (0.29) plus a little room
/** Cost of a gap `g` between the base's centre and a loose thing's edge, for a base that turns within `body`. */
const gapCost = (g: number, body: number) => (g < HALF_WIDTH ? 60 : g < body ? 15 : g < body + 0.12 ? 3 : 0)
/** Extra cost of passing a point because of loose things nearby (0 = clear). */
function softCost(x: number, z: number, soft: Disc[], body: number) {
  let c = 0
  for (const d of soft) c += gapCost(Math.hypot(x - d.x, z - d.z) - d.r, body)
  return c
}
/** Room to turn on the spot here without the corners touching anything? */
const turnClear = (p: Pt, soft: Disc[], body: number) => soft.every((d) => Math.hypot(p.x - d.x, p.z - d.z) - d.r >= body)

/**
 * Plan from `start` to `goal`. Returns the waypoints after the start (the last one is the
 * goal), or null if the goal can't be reached. If the start itself is inside a hard zone
 * (parked at a dock, say), the path first heads straight for the nearest clear cell.
 */
/** `body`: how close litter may come to the base's centre, the radius its corners sweep turning on the spot. */
export function planPath(grid: FloorGrid, start: Pt, goal: Pt, soft: Disc[], body = 0.46): Pt[] | null {
  const n = grid.n
  if (grid.isHard(goal.x, goal.z)) return null
  const gi = grid.i(goal.x)
  const gj = grid.j(goal.z)
  let si = grid.i(start.x)
  let sj = grid.j(start.z)
  const prefix: Pt[] = []
  if (grid.hard[si * n + sj]) {
    // Nearest clear cell (breadth-first, up to 1.2 m away).
    const seen = new Uint8Array(n * n)
    const q: number[] = [si * n + sj]
    seen[q[0]] = 1
    let found = -1
    for (let h = 0; h < q.length && found < 0; h++) {
      const c = q[h]
      const ci = Math.floor(c / n)
      const cj = c % n
      if (Math.hypot(ci - si, cj - sj) * RES > 1.2) continue
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ni = ci + di
        const nj = cj + dj
        if (ni < 0 || nj < 0 || ni >= n || nj >= n) continue
        const k = ni * n + nj
        if (seen[k]) continue
        seen[k] = 1
        if (!grid.hard[k]) {
          found = k
          break
        }
        q.push(k)
      }
    }
    if (found < 0) return null
    si = Math.floor(found / n)
    sj = found % n
    prefix.push({ x: grid.x(si), z: grid.z(sj) })
  }

  // Soft cost per cell, computed only near loose things.
  const cost = new Float32Array(n * n)
  for (const d of soft) {
    const reach = d.r + body + 0.12
    for (let i = grid.i(d.x - reach); i <= grid.i(d.x + reach); i++)
      for (let j = grid.j(d.z - reach); j <= grid.j(d.z + reach); j++) cost[i * n + j] += gapCost(Math.hypot(grid.x(i) - d.x, grid.z(j) - d.z) - d.r, body)
  }

  // A* (8-connected), octile distance as the heuristic.
  const gScore = new Float32Array(n * n).fill(Infinity)
  const from = new Int32Array(n * n).fill(-1)
  const closed = new Uint8Array(n * n)
  const open = new Heap()
  const s0 = si * n + sj
  const goalK = gi * n + gj
  gScore[s0] = 0
  const hEst = (i: number, j: number) => {
    const di = Math.abs(i - gi)
    const dj = Math.abs(j - gj)
    return (Math.max(di, dj) + 0.4142 * Math.min(di, dj)) * RES
  }
  open.push(s0, hEst(si, sj))
  const steps: [number, number, number][] = [
    [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
    [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2],
  ]
  let done = false
  while (open.size) {
    const c = open.pop()
    if (closed[c]) continue
    closed[c] = 1
    if (c === goalK) {
      done = true
      break
    }
    const ci = Math.floor(c / n)
    const cj = c % n
    for (const [di, dj, len] of steps) {
      const ni = ci + di
      const nj = cj + dj
      if (ni < 0 || nj < 0 || ni >= n || nj >= n) continue
      const k = ni * n + nj
      if (grid.hard[k] || closed[k]) continue
      const g = gScore[c] + len * RES * (1 + cost[k])
      if (g < gScore[k]) {
        gScore[k] = g
        from[k] = c
        open.push(k, g + hEst(ni, nj))
      }
    }
  }
  if (!done) return null

  const cells: number[] = []
  for (let c = goalK; c !== -1; c = from[c]) cells.push(c)
  cells.reverse()
  const pts = cells.map((c) => ({ x: grid.x(Math.floor(c / n)), z: grid.z(c % n) }))
  pts[0] = prefix.length ? prefix[0] : { x: start.x, z: start.z }
  pts[pts.length - 1] = { x: goal.x, z: goal.z }

  // Pull tight: from each kept point, jump to the farthest later point whose straight line is
  // no harder and no costlier than the path it replaces.
  const lineCost = (a: Pt, b: Pt) => {
    const d = Math.hypot(b.x - a.x, b.z - a.z)
    const m = Math.max(1, Math.ceil(d / (RES / 2)))
    let c = 0
    for (let k = 0; k <= m; k++) {
      const x = a.x + ((b.x - a.x) * k) / m
      const z = a.z + ((b.z - a.z) * k) / m
      if (grid.isHard(x, z)) return Infinity
      c += softCost(x, z, soft, body) * (d / m)
    }
    return c
  }
  const segCost: number[] = [0]
  for (let k = 1; k < pts.length; k++) segCost.push(segCost[k - 1] + softCost(pts[k].x, pts[k].z, soft, body) * Math.hypot(pts[k].x - pts[k - 1].x, pts[k].z - pts[k - 1].z))
  const out: Pt[] = []
  let a = 0
  while (a < pts.length - 1) {
    let b = pts.length - 1
    // The robot turns on the spot at every kept point, so those need room for its corners.
    while (b > a + 1 && (lineCost(pts[a], pts[b]) > (segCost[b] - segCost[a]) * 1.15 + 0.3 || (b < pts.length - 1 && !turnClear(pts[b], soft, body)))) b--
    out.push(pts[b])
    a = b
  }
  // Parked inside a hard zone: first a straight move out to the clear cell, then the path.
  const all = prefix.length ? [prefix[0], ...out] : out
  return all.filter((p, k) => Math.hypot(p.x - (k ? all[k - 1].x : start.x), p.z - (k ? all[k - 1].z : start.z)) > 0.01)
}

/** Total length of a path from `start` through `pts`. */
export const pathLength = (start: Pt, pts: Pt[]) => pts.reduce((s, p, k) => s + Math.hypot(p.x - (k ? pts[k - 1].x : start.x), p.z - (k ? pts[k - 1].z : start.z)), 0)

/** Minimal binary heap keyed by priority. */
class Heap {
  private k: number[] = []
  private p: number[] = []
  get size() {
    return this.k.length
  }
  push(key: number, pri: number) {
    this.k.push(key)
    this.p.push(pri)
    let i = this.k.length - 1
    while (i > 0) {
      const par = (i - 1) >> 1
      if (this.p[par] <= this.p[i]) break
      ;[this.k[par], this.k[i]] = [this.k[i], this.k[par]]
      ;[this.p[par], this.p[i]] = [this.p[i], this.p[par]]
      i = par
    }
  }
  pop() {
    const top = this.k[0]
    const lk = this.k.pop()!
    const lp = this.p.pop()!
    if (this.k.length) {
      this.k[0] = lk
      this.p[0] = lp
      let i = 0
      for (;;) {
        const l = 2 * i + 1
        const r = l + 1
        let m = i
        if (l < this.k.length && this.p[l] < this.p[m]) m = l
        if (r < this.k.length && this.p[r] < this.p[m]) m = r
        if (m === i) break
        ;[this.k[m], this.k[i]] = [this.k[i], this.k[m]]
        ;[this.p[m], this.p[i]] = [this.p[i], this.p[m]]
        i = m
      }
    }
    return top
  }
}
