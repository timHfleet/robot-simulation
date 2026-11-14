/**
 * The four work stations: where they stand, their furniture, the items on them and
 * the moving parts (dishwasher door and racks, washer lid, shirt folds).
 *
 * Each station has a dock: the pose the robot parks in to work there. Everything at a
 * station is given in that dock's frame (a = forward from the dock, y = up,
 * b = to the robot's right), which is also the station group's local frame in the
 * 3D scene (x = a, y = y, z = b).
 */
import * as THREE from 'three'

export type StationId = 'sort' | 'kitchen' | 'laundry' | 'washer'
export interface Dock {
  x: number
  z: number
  yaw: number
}

export const DOCKS: Record<StationId, Dock> = {
  sort: { x: 1.3, z: 0, yaw: 0 },
  kitchen: { x: -1.3, z: 0, yaw: Math.PI },
  laundry: { x: 0, z: -1.3, yaw: Math.PI / 2 },
  washer: { x: 0, z: 1.3, yaw: -Math.PI / 2 },
}
/** Shoulder height used at each station. */
export const LIFTS: Record<StationId, number> = { sort: 0.95, kitchen: 1.0, laundry: 0.85, washer: 1.15 }
export const STATION_NAMES: Record<StationId, string> = {
  sort: 'sorting table',
  kitchen: 'kitchen',
  laundry: 'laundry table',
  washer: 'washing machine',
}
/** How far in front of a dock the robot lines up before driving straight in. */
export const STAGING = 0.4

export const UP = new THREE.Vector3(0, 1, 0)
export const stationQuat = (id: StationId) => new THREE.Quaternion().setFromAxisAngle(UP, DOCKS[id].yaw)
export const stationMatrix = (id: StationId) =>
  new THREE.Matrix4().compose(new THREE.Vector3(DOCKS[id].x, 0, DOCKS[id].z), stationQuat(id), new THREE.Vector3(1, 1, 1))
/** Station-local point (a, y, b) to world. */
export const toWorld = (id: StationId, a: number, y: number, b: number) => new THREE.Vector3(a, y, b).applyMatrix4(stationMatrix(id))
/** Station-local direction to world. */
export const dirWorld = (id: StationId, a: number, y: number, b: number) => new THREE.Vector3(a, y, b).applyQuaternion(stationQuat(id))
export const toLocal = (id: StationId, p: THREE.Vector3) => p.clone().applyMatrix4(stationMatrix(id).invert())

/** A furniture box in station coordinates, used for drawing and for collisions. */
export interface Box {
  a0: number
  a1: number
  y0: number
  y1: number
  b0: number
  b1: number
}

// ---------------------------------------------------------------------------
// Sorting table
// ---------------------------------------------------------------------------
export const SORT = {
  table: { a0: 0.36, a1: 0.96, b0: -0.75, b1: 0.75, h: 0.72 },
  block: 0.05,
  bins: [
    { color: 'red', a: 0.44, b: -0.5 },
    { color: 'green', a: 0.44, b: -0.33 },
    { color: 'blue', a: 0.44, b: 0.42 },
  ],
  /** Where one hand sets a block down for the other hand to take. */
  handover: { a: 0.45, b: 0 },
  binSize: 0.15,
  binWall: 0.07,
  binFloor: 0.006,
  /** Where blocks land inside a bin (offsets from its centre). */
  slots: [
    [-0.035, -0.035],
    [0.035, 0.035],
    [-0.035, 0.035],
    [0.035, -0.035],
  ] as [number, number][],
  blocks: [
    { color: 'blue', a: 0.42, b: -0.17, yaw: 0.3 },
    { color: 'red', a: 0.42, b: 0.12, yaw: -0.5 },
    { color: 'green', a: 0.42, b: 0.24, yaw: 0.1 },
    { color: 'green', a: 0.5, b: -0.12, yaw: 0.6 },
    { color: 'red', a: 0.5, b: 0.14, yaw: -0.2 },
    { color: 'blue', a: 0.5, b: -0.22, yaw: 0.45 },
  ],
}

// ---------------------------------------------------------------------------
// Kitchen: counters either side of a raised dishwasher (drop-down door, two racks)
// ---------------------------------------------------------------------------
export const KITCHEN = {
  counterH: 0.86,
  counters: [
    { a0: 0.38, a1: 0.98, b0: -0.8, b1: -0.28 },
    { a0: 0.38, a1: 0.98, b0: 0.28, b1: 0.8 },
  ],
  /** Tall cabinet around the dishwasher. */
  cabinet: { a0: 0.62, a1: 1.12, b0: -0.28, b1: 0.28, top: 1.32 },
  door: { hingeA: 0.62, hingeY: 0.55, height: 0.6, width: 0.5, thick: 0.04 },
  /** Door-local points (u = out of the outer face, s = up the door from the hinge). */
  handle: { u: -0.045, s: 0.56 },
  edge: { u: 0, s: 0.6 },
  push: { u: -0.035, s: 0.45 },
  openByHand: THREE.MathUtils.degToRad(30),
  liftTo: THREE.MathUtils.degToRad(40),
  racks: {
    lower: { floor: 0.6, frontA: 0.66, depth: 0.42, width: 0.46, wall: 0.05, travel: 0.32 },
    upper: { floor: 0.88, frontA: 0.66, depth: 0.42, width: 0.46, wall: 0.04, travel: 0.32 },
  },
  button: { a: 0.62, y: 1.2, b: 0.16 },
  plate: { r: 0.1, t: 0.012 },
  cup: { r: 0.04, h: 0.09 },
  /** Drying rack on the left counter: plates stand with their faces along b. */
  plateRack: { a: 0.47, b0: -0.56, b1: -0.32, slots: [-0.37, -0.44, -0.51] },
  cups: [
    { a: 0.46, b: 0.37 },
    { a: 0.46, b: 0.48 },
  ],
  /** Slots in rack-local coordinates (a from the rack front). */
  plateSlots: [-0.16, -0.08, 0],
  plateSlotA: 0.18,
  cupSlots: [
    { a: 0.14, b: 0.08 },
    { a: 0.14, b: 0.17 },
  ],
}
export type RackId = 'lower' | 'upper'
/** Rack origin (front-centre, on its floor) in station coordinates for a given pull-out. */
export const rackOrigin = (rack: RackId, out: number) => {
  const r = KITCHEN.racks[rack]
  return new THREE.Vector3(r.frontA - out, r.floor, 0)
}
/** Station-local (a, y) of a door-local point when the door is open by phi (0 closed, pi/2 flat). */
export function doorPoint(u: number, s: number, phi: number): [number, number] {
  const { hingeA, hingeY } = KITCHEN.door
  return [hingeA + u * Math.cos(phi) - s * Math.sin(phi), hingeY + u * Math.sin(phi) + s * Math.cos(phi)]
}
/** Door angle that puts door-local point (u, s) at station-local (a, y). */
export function doorAngleFor(u: number, s: number, a: number, y: number) {
  const { hingeA, hingeY } = KITCHEN.door
  return Math.atan2(y - hingeY, a - hingeA) - Math.atan2(s, u)
}

// ---------------------------------------------------------------------------
// Laundry table and the shirt to fold
// ---------------------------------------------------------------------------
export const LAUNDRY = {
  table: { a0: 0.36, a1: 0.84, b0: -0.65, b1: 0.65, h: 0.74 },
  shirt: { hemA: 0.38, length: 0.28, half: 0.16, sleeve: 0.32, sleeveA0: 0.52, sleeveA1: 0.65, neck: 0.05 },
  /** Folds in order: left side, right side, then bottom half up. */
  folds: [
    { point: [0, 0.746, -0.08], n: [0, 0, -1], pinch: [0.55, -0.26] },
    { point: [0, 0.75, 0.08], n: [0, 0, 1], pinch: [0.55, 0.26] },
    { point: [0.51, 0.755, 0], n: [-1, 0, 0], pinch: [0.39, 0.065] },
  ] as { point: [number, number, number]; n: [number, number, number]; pinch: [number, number] }[],
}
export const CLOTH_Y = LAUNDRY.table.h + 0.003
export const CLOTH_NU = 30 // cells along a (hem to collar)
export const CLOTH_NV = 68 // cells along b (sleeve to sleeve)

/** Flat shirt: vertex positions (station-local) and which grid cells are fabric. */
export function shirtGrid() {
  const { hemA, length, sleeve } = LAUNDRY.shirt
  const verts = new Float32Array((CLOTH_NU + 1) * (CLOTH_NV + 1) * 3)
  for (let i = 0; i <= CLOTH_NU; i++) {
    for (let j = 0; j <= CLOTH_NV; j++) {
      const k = (i * (CLOTH_NV + 1) + j) * 3
      verts[k] = hemA + (i / CLOTH_NU) * length
      verts[k + 1] = CLOTH_Y
      verts[k + 2] = -sleeve + (j / CLOTH_NV) * 2 * sleeve
    }
  }
  const mask = new Uint8Array(CLOTH_NU * CLOTH_NV)
  for (let i = 0; i < CLOTH_NU; i++) {
    for (let j = 0; j < CLOTH_NV; j++) {
      const a = hemA + ((i + 0.5) / CLOTH_NU) * length
      const b = -sleeve + ((j + 0.5) / CLOTH_NV) * 2 * sleeve
      mask[i * CLOTH_NV + j] = shirtCell(a, b) ? 1 : 0
    }
  }
  return { verts, mask }
}

function shirtCell(a: number, b: number) {
  const { hemA, length, half, sleeve, sleeveA0, sleeveA1, neck } = LAUNDRY.shirt
  const top = hemA + length
  const ab = Math.abs(b)
  if (a > top - 0.02 && ab < neck) return false // neck opening
  if (ab <= half) return true // body
  if (ab > sleeve) return false
  // Sleeves taper toward the cuff.
  const t = (ab - half) / (sleeve - half)
  const s0 = sleeveA0 + 0.035 * t
  const s1 = sleeveA1 - 0.01 * t
  return a >= s0 && a <= s1
}

/** Apply the folds (angles 0 = flat, pi = folded over) to the flat shirt. */
export function foldShirt(flat: Float32Array, angles: number[], out: Float32Array) {
  out.set(flat)
  const p = new THREE.Vector3()
  const axis = new THREE.Vector3()
  const q = new THREE.Quaternion()
  LAUNDRY.folds.forEach((f, k) => {
    const th = angles[k]
    if (th <= 0) return
    const n = new THREE.Vector3(...f.n)
    const pivot = new THREE.Vector3(...f.point)
    axis.crossVectors(n, UP).normalize()
    q.setFromAxisAngle(axis, th)
    for (let i = 0; i < out.length; i += 3) {
      p.set(out[i], out[i + 1], out[i + 2])
      // Fold the part on the moving side of the line (judged on the table plane).
      if ((p.x - pivot.x) * n.x + (p.z - pivot.z) * n.z <= 0) continue
      p.sub(pivot).applyQuaternion(q).add(pivot)
      out[i] = p.x
      out[i + 1] = p.y
      out[i + 2] = p.z
    }
  })
}
/** Fold angle implied by a hand at station-local p holding the fold's flap. */
export function foldAngleFor(k: number, p: THREE.Vector3) {
  const f = LAUNDRY.folds[k]
  const n = new THREE.Vector3(...f.n)
  const d = (p.x - f.point[0]) * n.x + (p.z - f.point[2]) * n.z
  return THREE.MathUtils.clamp(Math.atan2(p.y - f.point[1], d), 0, Math.PI)
}

// ---------------------------------------------------------------------------
// Washing machine (top loader) with a clothes basket on a stand
// ---------------------------------------------------------------------------
export const WASHER = {
  body: { a0: 0.42, a1: 0.92, b0: -0.3, b1: 0.3, top: 0.88 },
  backsplash: { a0: 0.84, a1: 0.92, top: 1.02 },
  lid: { hingeA: 0.8, hingeY: 0.895, depth: 0.3, width: 0.5, open: THREE.MathUtils.degToRad(70) },
  drum: { a: 0.64, r: 0.16, y0: 0.46, y1: 0.86 },
  button: { a: 0.46, y: 0.885, b: 0.18 },
  stand: { a0: 0.4, a1: 0.64, b0: -0.68, b1: -0.36, top: 0.76 },
  basketWall: 0.13,
  clothes: [
    { a: 0.47, b: -0.45, color: '#2563eb' },
    { a: 0.56, b: -0.58, color: '#f59e0b' },
    { a: 0.47, b: -0.58, color: '#16a34a' },
    { a: 0.56, b: -0.45, color: '#db2777' },
  ],
  /** Where clothes settle inside the drum (drum-local). */
  drumRest: [
    [0.05, 0.05, 0.04],
    [-0.05, 0.06, -0.03],
    [0.03, 0.09, -0.06],
    [-0.04, 0.1, 0.05],
  ] as [number, number, number][],
}
/** Station-local (a, y) of the lid's front edge at opening angle phi. */
export function lidEdge(phi: number, inset = 0): [number, number] {
  const { hingeA, hingeY, depth } = WASHER.lid
  const r = depth - inset
  return [hingeA - r * Math.cos(phi), hingeY + 0.01 + r * Math.sin(phi)]
}
export function lidAngleFor(a: number, y: number) {
  const { hingeA, hingeY } = WASHER.lid
  return THREE.MathUtils.clamp(Math.atan2(y - hingeY - 0.01, hingeA - a), 0, WASHER.lid.open)
}

// ---------------------------------------------------------------------------
// Items
// ---------------------------------------------------------------------------
export type ItemKind = 'block' | 'plate' | 'cup' | 'clothes'
export interface ItemDef {
  id: string
  kind: ItemKind
  color: string
  station: StationId
  /** Home pose, station-local. */
  pos: [number, number, number]
  yaw: number
}

export const ITEMS: ItemDef[] = [
  ...SORT.blocks.map((b, i) => ({
    id: `block${i}`,
    kind: 'block' as const,
    color: b.color,
    station: 'sort' as const,
    pos: [b.a, SORT.table.h + SORT.block / 2, b.b] as [number, number, number],
    yaw: b.yaw,
  })),
  ...KITCHEN.plateRack.slots.map((b, i) => ({
    id: `plate${i}`,
    kind: 'plate' as const,
    color: '#f8fafc',
    station: 'kitchen' as const,
    pos: [KITCHEN.plateRack.a, KITCHEN.counterH + 0.012 + KITCHEN.plate.r, b] as [number, number, number],
    yaw: 0,
  })),
  ...KITCHEN.cups.map((c, i) => ({
    id: `cup${i}`,
    kind: 'cup' as const,
    color: i ? '#0ea5e9' : '#e11d48',
    station: 'kitchen' as const,
    pos: [c.a, KITCHEN.counterH + KITCHEN.cup.h / 2, c.b] as [number, number, number],
    yaw: 0,
  })),
  ...WASHER.clothes.map((c, i) => ({
    id: `clothes${i}`,
    kind: 'clothes' as const,
    color: c.color,
    station: 'washer' as const,
    pos: [c.a, WASHER.stand.top + 0.035 + i * 0.012, c.b] as [number, number, number],
    yaw: i * 0.7,
  })),
]

/** Furniture boxes per station (station-local), for collisions with the base. */
export const FURNITURE: Record<StationId, Box[]> = {
  sort: [{ a0: SORT.table.a0, a1: SORT.table.a1, y0: 0, y1: SORT.table.h, b0: SORT.table.b0, b1: SORT.table.b1 }],
  kitchen: [
    ...KITCHEN.counters.map((c) => ({ ...c, y0: 0, y1: KITCHEN.counterH })),
    { ...KITCHEN.cabinet, y0: 0, y1: KITCHEN.cabinet.top },
  ],
  laundry: [{ a0: LAUNDRY.table.a0, a1: LAUNDRY.table.a1, y0: 0, y1: LAUNDRY.table.h, b0: LAUNDRY.table.b0, b1: LAUNDRY.table.b1 }],
  washer: [
    { ...WASHER.body, y0: 0, y1: WASHER.body.top },
    { ...WASHER.stand, y0: 0, y1: WASHER.stand.top },
  ],
}

/** World-space floor rectangles (x0, x1, z0, z1) of all furniture. */
export function furnitureRects() {
  const rects: { x0: number; x1: number; z0: number; z1: number }[] = []
  for (const id of Object.keys(FURNITURE) as StationId[]) {
    for (const bx of FURNITURE[id]) {
      const pts = [
        toWorld(id, bx.a0, 0, bx.b0),
        toWorld(id, bx.a1, 0, bx.b1),
      ]
      rects.push({
        x0: Math.min(pts[0].x, pts[1].x),
        x1: Math.max(pts[0].x, pts[1].x),
        z0: Math.min(pts[0].z, pts[1].z),
        z1: Math.max(pts[0].z, pts[1].z),
      })
    }
  }
  return rects
}
