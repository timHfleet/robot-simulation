/**
 * 3D models of the four work stations, their items and moving parts.
 * Each station is a group placed at its dock, so everything inside it uses the
 * station coordinates from lib/stations.ts (x = forward a, y = up, z = right b).
 */
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import {
  CLOTH_NU,
  CLOTH_NV,
  DOCKS,
  ITEMS,
  KITCHEN,
  LAUNDRY,
  SORT,
  WASHER,
  foldShirt,
  rackOrigin,
  shirtGrid,
  type ItemDef,
  type RackId,
  type StationId,
} from '../lib/stations'

export interface Mech {
  door: number
  lowerRack: number
  upperRack: number
  lid: number
  folds: number[]
  dishOn: boolean
  washOn: boolean
}
export const freshMech = (): Mech => ({ door: 0, lowerRack: 0, upperRack: 0, lid: 0, folds: [0, 0, 0], dishOn: false, washOn: false })

export interface Stations {
  groups: Record<StationId, THREE.Group>
  /** Things items can be put in: station groups, the dishwasher racks, the washer drum. */
  containers: Record<string, THREE.Object3D>
  items: Map<string, THREE.Object3D>
  /** Where each item is: 'home', 'held', 'bin:red', 'rack', 'drum', ... */
  where: Map<string, string>
  pickables: THREE.Object3D[]
  mech: Mech
  /** Push the mechanism state into the models (door, racks, lid, shirt, lights). */
  apply: (t: number, dt: number) => void
  reset: () => void
}

const COLORS: Record<string, string> = { red: '#dc2626', green: '#16a34a', blue: '#2563eb' }
const mat = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.1, ...extra })
function shadowed<T extends THREE.Object3D>(o: T): T {
  o.traverse((c) => {
    c.castShadow = true
    c.receiveShadow = true
  })
  return o
}
/** A box from (a0, y0, b0) to (a1, y1, b1) in its parent's coordinates. */
function slab(a0: number, a1: number, y0: number, y1: number, b0: number, b1: number, m: THREE.Material, round = 0) {
  const w = a1 - a0
  const h = y1 - y0
  const d = b1 - b0
  const geo = round > 0 ? new RoundedBoxGeometry(w, h, d, 2, Math.min(round, w / 2, h / 2, d / 2)) : new THREE.BoxGeometry(w, h, d)
  const mesh = new THREE.Mesh(geo, m)
  mesh.position.set((a0 + a1) / 2, (y0 + y1) / 2, (b0 + b1) / 2)
  return shadowed(mesh)
}

const M = {
  wood: mat('#b08158', { roughness: 0.75 }),
  woodDark: mat('#7c5a3c', { roughness: 0.8 }),
  counter: mat('#e5e7eb', { roughness: 0.35 }),
  cabinet: mat('#475569', { roughness: 0.55 }),
  steel: mat('#c8ced6', { metalness: 0.8, roughness: 0.3 }),
  steelDark: mat('#8b939e', { metalness: 0.8, roughness: 0.35 }),
  wire: mat('#d1d5db', { metalness: 0.6, roughness: 0.35 }),
  black: mat('#1f2328', { roughness: 0.5 }),
  white: mat('#f3f4f6', { roughness: 0.4 }),
  wicker: mat('#b7864b', { roughness: 0.9 }),
  glass: new THREE.MeshPhysicalMaterial({ color: '#9fb4c8', transparent: true, opacity: 0.35, roughness: 0.05, metalness: 0, transmission: 0 }),
  drum: mat('#aab2bd', { metalness: 0.9, roughness: 0.25, side: THREE.DoubleSide }),
}

/** A table: top slab and four legs. */
function table(a0: number, a1: number, b0: number, b1: number, h: number, top: THREE.Material, legs: THREE.Material) {
  const g = new THREE.Group()
  g.add(slab(a0, a1, h - 0.03, h, b0, b1, top, 0.01))
  for (const a of [a0 + 0.04, a1 - 0.04]) for (const b of [b0 + 0.04, b1 - 0.04]) g.add(slab(a - 0.02, a + 0.02, 0, h - 0.03, b - 0.02, b + 0.02, legs))
  return g
}

// ---------------------------------------------------------------------------
// Items
// ---------------------------------------------------------------------------
function makeItem(def: ItemDef): THREE.Object3D {
  if (def.kind === 'block') {
    const m = new THREE.Mesh(new RoundedBoxGeometry(SORT.block, SORT.block, SORT.block, 2, 0.004), mat(COLORS[def.color] ?? def.color, { roughness: 0.45 }))
    return shadowed(m)
  }
  if (def.kind === 'plate') {
    const g = new THREE.Group()
    const { r, t } = KITCHEN.plate
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 0.92, t, 40), mat('#f8fafc', { roughness: 0.25 }))
    disc.rotation.x = Math.PI / 2 // face along local z
    const rim = new THREE.Mesh(new THREE.TorusGeometry(r - 0.006, 0.004, 8, 40), mat('#38bdf8', { roughness: 0.3 }))
    g.add(disc, rim)
    return shadowed(g)
  }
  if (def.kind === 'cup') {
    // Standing mouth-down: the closed base is on top.
    const g = new THREE.Group()
    const { r, h } = KITCHEN.cup
    const m = mat(def.color, { roughness: 0.35, side: THREE.DoubleSide })
    const body = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.88, r, h, 28, 1, true), m)
    const base = new THREE.Mesh(new THREE.CircleGeometry(r * 0.88, 28), m)
    base.rotation.x = -Math.PI / 2
    base.position.y = h / 2
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.02, 0.005, 8, 20, Math.PI), m)
    handle.rotation.z = Math.PI / 2
    handle.position.set(0, 0, -r)
    handle.rotation.y = Math.PI / 2
    g.add(body, base, handle)
    return shadowed(g)
  }
  // Clothes: a soft, lumpy bundle.
  const geo = new THREE.IcosahedronGeometry(0.05, 3)
  const p = geo.getAttribute('position') as THREE.BufferAttribute
  const v = new THREE.Vector3()
  const seed = def.id.length * 7 + def.color.charCodeAt(2)
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i)
    const n = 1 + 0.14 * Math.sin(v.x * 60 + seed) * Math.cos(v.z * 50 + seed * 0.3) + 0.08 * Math.sin(v.y * 90 + seed)
    v.multiplyScalar(n)
    p.setXYZ(i, v.x, v.y, v.z)
  }
  geo.computeVertexNormals()
  const m = new THREE.Mesh(geo, mat(def.color, { roughness: 0.95 }))
  m.scale.set(1.15, 0.42, 0.9)
  return shadowed(m)
}

// ---------------------------------------------------------------------------
// Stations
// ---------------------------------------------------------------------------
function buildSort(g: THREE.Group, pick: THREE.Object3D[]) {
  const t = SORT.table
  const tb = table(t.a0, t.a1, t.b0, t.b1, t.h, M.wood, M.woodDark)
  g.add(tb)
  pick.push(tb)
  const s = SORT.binSize / 2
  for (const bin of SORT.bins) {
    const m = mat(COLORS[bin.color], { roughness: 0.55 })
    const y0 = t.h
    const y1 = t.h + SORT.binWall
    const w = 0.006
    const bg = new THREE.Group()
    bg.add(
      slab(bin.a - s, bin.a + s, y0, y0 + SORT.binFloor, bin.b - s, bin.b + s, m),
      slab(bin.a - s, bin.a + s, y0, y1, bin.b - s, bin.b - s + w, m),
      slab(bin.a - s, bin.a + s, y0, y1, bin.b + s - w, bin.b + s, m),
      slab(bin.a - s, bin.a - s + w, y0, y1, bin.b - s, bin.b + s, m),
      slab(bin.a + s - w, bin.a + s, y0, y1, bin.b - s, bin.b + s, m),
    )
    g.add(bg)
  }
}

interface KitchenParts {
  door: THREE.Group
  racks: Record<RackId, THREE.Group>
  led: THREE.MeshStandardMaterial
}
function buildKitchen(g: THREE.Group, pick: THREE.Object3D[]): KitchenParts {
  const K = KITCHEN
  for (const c of K.counters) {
    const base = slab(c.a0 + 0.02, c.a1, 0, K.counterH - 0.03, c.b0, c.b1, M.cabinet)
    const top = slab(c.a0, c.a1, K.counterH - 0.03, K.counterH, c.b0, c.b1, M.counter, 0.01)
    g.add(base, top)
    pick.push(top)
  }
  // Tall cabinet with the dishwasher cavity.
  const cb = K.cabinet
  const d = K.door
  const cavityTop = d.hingeY + d.height
  g.add(
    slab(cb.a0, cb.a1, 0, d.hingeY, cb.b0, cb.b1, M.cabinet),
    slab(cb.a0, cb.a1, cavityTop, cb.top, cb.b0, cb.b1, M.cabinet),
    slab(cb.a0, cb.a1, d.hingeY, cavityTop, cb.b0, cb.b0 + 0.03, M.cabinet),
    slab(cb.a0, cb.a1, d.hingeY, cavityTop, cb.b1 - 0.03, cb.b1, M.cabinet),
    slab(cb.a1 - 0.03, cb.a1, d.hingeY, cavityTop, cb.b0, cb.b1, M.cabinet),
  )
  // Stainless tub inside.
  const tub = mat('#b9c1cb', { metalness: 0.85, roughness: 0.3, side: THREE.BackSide })
  const tubBox = new THREE.Mesh(new THREE.BoxGeometry(cb.a1 - cb.a0 - 0.05, d.height - 0.02, cb.b1 - cb.b0 - 0.06), tub)
  tubBox.position.set((cb.a0 + cb.a1) / 2 + 0.01, d.hingeY + d.height / 2, 0)
  g.add(tubBox)
  // Control panel above the door: start button and a status light.
  const panel = slab(cb.a0 - 0.006, cb.a0, cavityTop + 0.01, cb.top - 0.04, cb.b0 + 0.04, cb.b1 - 0.04, M.black)
  const btn = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.012, 20), mat('#e5e7eb'))
  btn.rotation.z = Math.PI / 2
  btn.position.set(cb.a0 - 0.008, K.button.y, K.button.b)
  const led = new THREE.MeshStandardMaterial({ color: '#334155', emissive: '#000000' })
  const ledM = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8), led)
  ledM.position.set(cb.a0 - 0.007, K.button.y, K.button.b - 0.05)
  g.add(panel, btn, ledM)

  // Door: pivots at the bottom front edge, rotation about z (0 closed, pi/2 flat open).
  const door = new THREE.Group()
  door.position.set(d.hingeA, d.hingeY, 0)
  const outer = slab(-d.thick / 2, d.thick / 2, 0, d.height, -d.width / 2, d.width / 2, M.steel, 0.008)
  const inner = slab(d.thick / 2 - 0.002, d.thick / 2 + 0.004, 0.02, d.height - 0.02, -d.width / 2 + 0.02, d.width / 2 - 0.02, mat('#d1d5db', { metalness: 0.7 }))
  const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, d.width * 0.7, 14), M.black)
  bar.rotation.x = Math.PI / 2
  bar.position.set(K.handle.u, K.handle.s, 0)
  for (const z of [-d.width * 0.33, d.width * 0.33]) door.add(slab(-0.045, -d.thick / 2, K.handle.s - 0.006, K.handle.s + 0.006, z - 0.006, z + 0.006, M.black))
  door.add(outer, inner, shadowed(bar))
  g.add(door)

  // Wire racks; plates stand between tines in the lower one, cups sit in the upper one.
  const rack = (id: RackId) => {
    const r = K.racks[id]
    const rg = new THREE.Group()
    const w = r.width / 2
    const t = 0.006
    for (let i = 0; i <= 7; i++) rg.add(slab(0, r.depth, 0, t, -w + (i * r.width) / 7 - 0.002, -w + (i * r.width) / 7 + 0.002, M.wire))
    for (let i = 0; i <= 5; i++) rg.add(slab((i * r.depth) / 5 - 0.002, (i * r.depth) / 5 + 0.002, 0, t, -w, w, M.wire))
    for (const [a0, a1, b0, b1] of [
      [0, 0.006, -w, w],
      [r.depth - 0.006, r.depth, -w, w],
      [0, r.depth, -w, -w + 0.006],
      [0, r.depth, w - 0.006, w],
    ]) {
      rg.add(slab(a0, a1, 0, r.wall, b0, b1, M.wire))
    }
    // Handle on the front.
    const h = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.16, 10), M.black)
    h.rotation.x = Math.PI / 2
    h.position.set(-0.015, r.wall + 0.02, 0.05)
    rg.add(shadowed(h))
    rg.add(slab(-0.018, 0, r.wall, r.wall + 0.025, -0.01 + 0.05 - 0.08, -0.004 + 0.05 - 0.08, M.wire), slab(-0.018, 0, r.wall, r.wall + 0.025, 0.004 + 0.05 + 0.08, 0.01 + 0.05 + 0.08, M.wire))
    if (id === 'lower') {
      for (const b of K.plateSlots) for (const side of [-1, 1]) for (const a of [K.plateSlotA - 0.06, K.plateSlotA + 0.06]) {
        const tine = new THREE.Mesh(new THREE.CylinderGeometry(0.0025, 0.0025, 0.06, 6), M.wire)
        tine.position.set(a, 0.03, b + side * 0.012)
        rg.add(tine)
      }
    } else {
      for (const s of K.cupSlots) {
        const peg = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.03, 6), M.wire)
        peg.position.set(s.a, 0.015, s.b)
        rg.add(peg)
      }
    }
    g.add(rg)
    return rg
  }
  const racks = { lower: rack('lower'), upper: rack('upper') }

  // Drying rack for plates on the left counter.
  const pr = K.plateRack
  g.add(slab(pr.a - 0.07, pr.a + 0.07, K.counterH, K.counterH + 0.012, pr.b0, pr.b1, M.wire))
  for (const b of pr.slots) for (const side of [-1, 1]) for (const a of [pr.a - 0.05, pr.a + 0.05]) {
    const peg = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.07, 6), M.wire)
    peg.position.set(a, K.counterH + 0.045, b + side * 0.014)
    g.add(peg)
  }
  // A drying mat for the cups.
  g.add(slab(0.4, 0.54, K.counterH, K.counterH + 0.004, 0.31, 0.55, mat('#64748b', { roughness: 0.9 })))
  return { door, racks, led }
}

interface LaundryParts {
  cloth: THREE.Mesh
  flat: Float32Array
  verts: Float32Array
}
function buildLaundry(g: THREE.Group, pick: THREE.Object3D[]): LaundryParts {
  const t = LAUNDRY.table
  const tb = table(t.a0, t.a1, t.b0, t.b1, t.h, mat('#cbd5e1', { roughness: 0.5 }), M.steelDark)
  g.add(tb)
  pick.push(tb)
  const { verts: flat, mask } = shirtGrid()
  const geo = new THREE.BufferGeometry()
  const verts = new Float32Array(flat)
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3))
  // Colours: teal jersey with a darker hem, cuffs and collar.
  const colors = new Float32Array(flat.length)
  const base = new THREE.Color('#0d9488')
  const trim = new THREE.Color('#115e59')
  const { hemA, length, half } = LAUNDRY.shirt
  for (let i = 0; i <= CLOTH_NU; i++) {
    for (let j = 0; j <= CLOTH_NV; j++) {
      const k = (i * (CLOTH_NV + 1) + j) * 3
      const a = flat[k]
      const b = Math.abs(flat[k + 2])
      const c = a < hemA + 0.015 || b > LAUNDRY.shirt.sleeve - 0.02 || (a > hemA + length - 0.035 && b < 0.075) ? trim : base
      const shade = 0.92 + 0.08 * Math.sin(j * 1.7) * Math.sin(i * 1.3)
      colors[k] = c.r * shade
      colors[k + 1] = c.g * shade
      colors[k + 2] = c.b * shade
      void half
    }
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  const index: number[] = []
  for (let i = 0; i < CLOTH_NU; i++) {
    for (let j = 0; j < CLOTH_NV; j++) {
      if (!mask[i * CLOTH_NV + j]) continue
      const k = i * (CLOTH_NV + 1) + j
      index.push(k, k + 1, k + CLOTH_NV + 1, k + 1, k + CLOTH_NV + 2, k + CLOTH_NV + 1)
    }
  }
  geo.setIndex(index)
  geo.computeVertexNormals()
  const cloth = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, side: THREE.DoubleSide }))
  cloth.castShadow = true
  cloth.receiveShadow = true
  cloth.frustumCulled = false
  g.add(cloth)
  return { cloth, flat, verts }
}

interface WasherParts {
  lid: THREE.Group
  drum: THREE.Group
  led: THREE.MeshStandardMaterial
  display: THREE.MeshStandardMaterial
}
function buildWasher(g: THREE.Group, pick: THREE.Object3D[]): WasherParts {
  const W = WASHER
  const b = W.body
  const white = mat('#f8fafc', { roughness: 0.35 })
  // Body with a round opening in the top.
  g.add(slab(b.a0, b.a1, 0, b.top - 0.02, b.b0, b.b1, white, 0.02))
  const topShape = new THREE.Shape()
  topShape.moveTo(b.a0, b.b0)
  topShape.lineTo(b.a1, b.b0)
  topShape.lineTo(b.a1, b.b1)
  topShape.lineTo(b.a0, b.b1)
  topShape.closePath()
  const hole = new THREE.Path()
  hole.absarc(W.drum.a, 0, W.drum.r - 0.01, 0, Math.PI * 2, true)
  topShape.holes.push(hole)
  const topGeo = new THREE.ExtrudeGeometry(topShape, { depth: 0.02, bevelEnabled: false, curveSegments: 40 })
  topGeo.rotateX(Math.PI / 2) // shape x -> a, shape y -> b (extruded downward)
  const top = new THREE.Mesh(topGeo, white)
  top.position.y = b.top
  top.castShadow = top.receiveShadow = true
  g.add(top)
  pick.push(top)
  // Control strip at the front of the top.
  g.add(slab(b.a0 + 0.005, b.a0 + 0.07, b.top, b.top + 0.004, b.b0 + 0.03, b.b1 - 0.03, M.black))
  const btn = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.008, 20), mat('#e5e7eb'))
  btn.position.set(W.button.a, b.top + 0.006, W.button.b)
  const dial = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.022, 0.016, 24), M.steel)
  dial.position.set(W.button.a, b.top + 0.01, -0.05)
  const led = new THREE.MeshStandardMaterial({ color: '#334155', emissive: '#000000' })
  const ledM = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8), led)
  ledM.position.set(W.button.a, b.top + 0.006, W.button.b + 0.04)
  const display = new THREE.MeshStandardMaterial({ color: '#0f172a', emissive: '#000000' })
  const disp = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.003, 0.07), display)
  disp.position.set(W.button.a, b.top + 0.005, 0.07)
  g.add(shadowed(btn), shadowed(dial), ledM, disp)
  // Backsplash.
  g.add(slab(W.backsplash.a0, W.backsplash.a1, b.top, W.backsplash.top, b.b0, b.b1, white, 0.015))
  // Drum: open steel cylinder with perforation rings.
  const drum = new THREE.Group()
  drum.position.set(W.drum.a, W.drum.y0, 0)
  const wall = new THREE.Mesh(new THREE.CylinderGeometry(W.drum.r, W.drum.r, W.drum.y1 - W.drum.y0, 40, 1, true), M.drum)
  wall.position.y = (W.drum.y1 - W.drum.y0) / 2
  const floor = new THREE.Mesh(new THREE.CircleGeometry(W.drum.r, 40), M.drum)
  floor.rotation.x = -Math.PI / 2
  floor.position.y = 0.01
  const agitator = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.04, 0.16, 16), M.steelDark)
  agitator.position.y = 0.08
  drum.add(wall, floor, agitator)
  for (let i = 1; i < 4; i++) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(W.drum.r - 0.002, 0.002, 6, 40), M.steelDark)
    ring.rotation.x = Math.PI / 2
    ring.position.y = i * 0.09
    drum.add(ring)
  }
  g.add(drum)
  // Lid: pivots at its back edge; rotation about -z lifts the front edge.
  const lid = new THREE.Group()
  lid.position.set(W.lid.hingeA, W.lid.hingeY, 0)
  const lw = W.lid.width / 2
  const frame = new THREE.Group()
  frame.add(
    slab(-W.lid.depth, 0, 0, 0.02, -lw, -lw + 0.04, white),
    slab(-W.lid.depth, 0, 0, 0.02, lw - 0.04, lw, white),
    slab(-W.lid.depth, -W.lid.depth + 0.04, 0, 0.02, -lw, lw, white),
    slab(-0.04, 0, 0, 0.02, -lw, lw, white),
  )
  const glass = slab(-W.lid.depth + 0.04, -0.04, 0.006, 0.014, -lw + 0.04, lw - 0.04, M.glass)
  glass.castShadow = false
  const grip = slab(-W.lid.depth - 0.012, -W.lid.depth, 0, 0.02, -0.12, 0.12, M.black)
  lid.add(frame, glass, grip)
  g.add(lid)
  // Basket of clothes on a stand to the left.
  const s = W.stand
  const st = table(s.a0, s.a1, s.b0, s.b1, s.top, M.wood, M.woodDark)
  g.add(st)
  const bw = 0.008
  const bt = s.top + W.basketWall
  g.add(
    slab(s.a0 + 0.02, s.a1 - 0.02, s.top, s.top + bw, s.b0 + 0.02, s.b1 - 0.02, M.wicker),
    slab(s.a0 + 0.02, s.a0 + 0.02 + bw, s.top, bt, s.b0 + 0.02, s.b1 - 0.02, M.wicker),
    slab(s.a1 - 0.02 - bw, s.a1 - 0.02, s.top, bt, s.b0 + 0.02, s.b1 - 0.02, M.wicker),
    slab(s.a0 + 0.02, s.a1 - 0.02, s.top, bt, s.b0 + 0.02, s.b0 + 0.02 + bw, M.wicker),
    slab(s.a0 + 0.02, s.a1 - 0.02, s.top, bt, s.b1 - 0.02 - bw, s.b1 - 0.02, M.wicker),
  )
  return { lid, drum, led, display }
}

export function buildStations(scene: THREE.Scene): Stations {
  const groups = {} as Record<StationId, THREE.Group>
  for (const id of Object.keys(DOCKS) as StationId[]) {
    const g = new THREE.Group()
    g.position.set(DOCKS[id].x, 0, DOCKS[id].z)
    g.rotation.y = DOCKS[id].yaw
    scene.add(g)
    groups[id] = g
  }
  const pickables: THREE.Object3D[] = []
  buildSort(groups.sort, pickables)
  const kitchen = buildKitchen(groups.kitchen, pickables)
  const laundry = buildLaundry(groups.laundry, pickables)
  const washer = buildWasher(groups.washer, pickables)

  const containers: Record<string, THREE.Object3D> = {
    ...groups,
    'kitchen.lower': kitchen.racks.lower,
    'kitchen.upper': kitchen.racks.upper,
    'washer.drum': washer.drum,
  }
  const items = new Map<string, THREE.Object3D>()
  const where = new Map<string, string>()
  for (const def of ITEMS) {
    const o = makeItem(def)
    o.userData.itemId = def.id
    items.set(def.id, o)
  }
  const mech = freshMech()

  const reset = () => {
    for (const def of ITEMS) {
      const o = items.get(def.id)!
      groups[def.station].add(o)
      o.position.set(...def.pos)
      o.rotation.set(0, def.yaw, 0)
      o.userData.drop = null
      where.set(def.id, 'home')
    }
    Object.assign(mech, freshMech())
  }
  reset()

  let spin = 0
  let spinV = 0
  let lastFolds = ''
  const apply = (t: number, dt: number) => {
    kitchen.door.rotation.z = mech.door
    kitchen.racks.lower.position.copy(rackOrigin('lower', mech.lowerRack))
    kitchen.racks.upper.position.copy(rackOrigin('upper', mech.upperRack))
    kitchen.led.emissive.set(mech.dishOn ? '#22c55e' : '#000000')
    kitchen.led.emissiveIntensity = mech.dishOn ? 1.5 + 0.5 * Math.sin(t * 3) : 0
    washer.lid.rotation.z = -mech.lid
    washer.led.emissive.set(mech.washOn ? '#22c55e' : '#000000')
    washer.led.emissiveIntensity = mech.washOn ? 1.6 : 0
    washer.display.emissive.set(mech.washOn ? '#38bdf8' : '#000000')
    washer.display.emissiveIntensity = mech.washOn ? 0.8 + 0.2 * Math.sin(t * 2) : 0
    // The drum spins up when the washer runs; clothes inside turn with it.
    spinV += ((mech.washOn ? 7 : 0) - spinV) * Math.min(1, dt * 0.6)
    spin += spinV * dt
    washer.drum.rotation.y = spin
    const key = mech.folds.map((f) => f.toFixed(4)).join(',')
    if (key !== lastFolds) {
      lastFolds = key
      foldShirt(laundry.flat, mech.folds, laundry.verts)
      const pos = laundry.cloth.geometry.getAttribute('position') as THREE.BufferAttribute
      pos.needsUpdate = true
      laundry.cloth.geometry.computeVertexNormals()
    }
    // Items settling into place after being let go.
    for (const o of items.values()) {
      const d = o.userData.drop as null | { p0: THREE.Vector3; q0: THREE.Quaternion; p1: THREE.Vector3; q1: THREE.Quaternion; t: number }
      if (!d) continue
      d.t = Math.min(1, d.t + dt / 0.3)
      const e = d.t * d.t * (3 - 2 * d.t)
      o.position.lerpVectors(d.p0, d.p1, e)
      o.quaternion.slerpQuaternions(d.q0, d.q1, e)
      if (d.t >= 1) o.userData.drop = null
    }
  }
  return { groups, containers, items, where, pickables, mech, apply, reset }
}
