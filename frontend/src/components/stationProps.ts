/**
 * 3D models of the four work stations, their items and moving parts.
 * Each station is a group placed at its dock, so everything inside it uses the
 * station coordinates from lib/stations.ts (x = forward a, y = up, z = right b).
 */
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { knitBump, shirtTextures, stoneTexture, wickerTextures, woodTexture } from './textures'
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
  /** Moving parts, for the physics engine. */
  parts: {
    door: THREE.Object3D
    lid: THREE.Object3D
    drum: THREE.Object3D
    racks: Record<RackId, THREE.Object3D>
    cloth: { mesh: THREE.Mesh; flat: Float32Array; verts: Float32Array; mask: (u: number, v: number) => boolean }
  }
  /** Hand the shirt's shape over to the cloth simulation (fold angles then no longer move it). */
  useClothPhysics: () => void
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

const WICKER = wickerTextures()
const M = {
  wood: mat('#ffffff', { map: woodTexture([1, 1]), roughness: 0.55, metalness: 0 }),
  woodDark: mat('#ffffff', { map: woodTexture([1, 3], '#7a5434'), roughness: 0.7, metalness: 0 }),
  counter: mat('#ffffff', { map: stoneTexture([2, 1]), roughness: 0.28, metalness: 0 }),
  cabinet: mat('#475569', { roughness: 0.55 }),
  steel: mat('#c8ced6', { metalness: 0.8, roughness: 0.3 }),
  steelDark: mat('#8b939e', { metalness: 0.8, roughness: 0.35 }),
  wire: mat('#d1d5db', { metalness: 0.6, roughness: 0.35 }),
  black: mat('#1f2328', { roughness: 0.5 }),
  white: mat('#f3f4f6', { roughness: 0.4 }),
  wicker: mat('#ffffff', { map: WICKER.map, bumpMap: WICKER.bump, bumpScale: 2.5, roughness: 0.85, metalness: 0 }),
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
/** Glossy glazed ceramic. */
const ceramic = (color: string) =>
  new THREE.MeshPhysicalMaterial({ color, roughness: 0.18, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.08, side: THREE.DoubleSide })

/** A lathe profile from (radius, height) points, smoothed. */
const lathe = (pts: [number, number][], segs = 48) => new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), segs)

function makeItem(def: ItemDef): THREE.Object3D {
  if (def.kind === 'block') {
    // A painted wooden toy block: slightly rounded edges, satin paint over grain.
    const m = new THREE.Mesh(
      new RoundedBoxGeometry(SORT.block, SORT.block, SORT.block, 3, 0.005),
      new THREE.MeshPhysicalMaterial({ color: COLORS[def.color] ?? def.color, roughness: 0.42, clearcoat: 0.4, clearcoatRoughness: 0.4, bumpMap: woodTexture([1, 1]), bumpScale: 0.6 }),
    )
    return shadowed(m)
  }
  if (def.kind === 'plate') {
    // Dinner plate turned on a lathe: foot ring, flat well, sloped rim. Its axis is local z.
    const { r } = KITCHEN.plate
    const k = r / 0.1
    const prof: [number, number][] = [
      [0.001, 0.0035],
      [0.046, 0.0035],
      [0.05, 0.0],
      [0.056, 0.0],
      [0.06, 0.0035],
      [0.074, 0.006],
      [0.09, 0.0115],
      [0.0985, 0.0145],
      [0.1, 0.0158],
      [0.0985, 0.0172],
      [0.09, 0.0145],
      [0.074, 0.0092],
      [0.06, 0.0074],
      [0.001, 0.0074],
    ]
    const geo = lathe(prof.map(([x, y]) => [x * k, y - 0.0085] as [number, number]), 64)
    geo.rotateX(Math.PI / 2)
    const plate = new THREE.Mesh(geo, ceramic('#f7f5f0'))
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.092 * k, 0.0012, 6, 64), ceramic('#2f6fb0'))
    band.position.z = 0.0045
    const g = new THREE.Group()
    g.add(plate, band)
    return shadowed(g)
  }
  if (def.kind === 'cup') {
    // A mug standing mouth-down: lathe-turned walls with a rounded lip, base on top, and a handle.
    const { r, h } = KITCHEN.cup
    const prof: [number, number][] = [
      [0.001, -h / 2],
      [r * 0.86, -h / 2],
      [r * 0.92, -h / 2 + 0.003],
      [r, h / 2 - 0.004],
      [r * 0.995, h / 2 - 0.0005],
      [r * 0.95, h / 2],
      [r * 0.905, h / 2 - 0.002],
      [r * 0.87, -h / 2 + 0.008],
      [0.001, -h / 2 + 0.008],
    ]
    const geo = lathe(prof, 48)
    geo.rotateX(Math.PI) // mouth down
    const m = ceramic(def.color)
    const body = new THREE.Mesh(geo, m)
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.021, 0.0055, 12, 28, Math.PI * 1.1), m)
    handle.rotation.y = Math.PI / 2
    handle.rotation.x = Math.PI / 2 - 0.15
    handle.position.set(0, -0.002, -r - 0.012)
    const g = new THREE.Group()
    g.add(body, handle)
    return shadowed(g)
  }
  // Clothes: a folded garment, soft and slightly rumpled, with a knit texture.
  const seed = def.id.charCodeAt(def.id.length - 1)
  const geo = new RoundedBoxGeometry(0.13, 0.026, 0.1, 5, 0.011)
  const p = geo.getAttribute('position') as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i)
    const y = p.getY(i)
    const z = p.getZ(i)
    const bulge = 0.004 * Math.cos(x * 20) * Math.cos(z * 26) // softer, puffier middle
    const rumple = 0.0025 * Math.sin(x * 70 + seed) * Math.sin(z * 60 + seed * 0.7)
    p.setY(i, y + (y > 0 ? bulge + rumple : -0.3 * bulge))
    p.setX(i, x + 0.002 * Math.sin(z * 40 + seed))
  }
  geo.computeVertexNormals()
  const m = new THREE.Mesh(
    geo,
    new THREE.MeshPhysicalMaterial({ color: def.color, roughness: 0.95, sheen: 0.3, sheenRoughness: 0.8, sheenColor: new THREE.Color(def.color).lerp(new THREE.Color('#ffffff'), 0.2), envMapIntensity: 0.4, bumpMap: knitBump([3, 3]), bumpScale: 1.2 }),
  )
  // A visible fold line across the top.
  const crease = new THREE.Mesh(new THREE.BoxGeometry(0.128, 0.0015, 0.004), new THREE.MeshStandardMaterial({ color: new THREE.Color(def.color).multiplyScalar(0.7), roughness: 1 }))
  crease.position.set(0, 0.0135, 0.01)
  const g = new THREE.Group()
  g.add(m, crease)
  return shadowed(g)
}

/** A square plastic storage tub: tapered walls, a rolled rim and a floor. */
function tub(a: number, b: number, y: number, size: number, height: number, color: string) {
  const g = new THREE.Group()
  const plastic = new THREE.MeshPhysicalMaterial({ color, roughness: 0.35, clearcoat: 0.5, clearcoatRoughness: 0.3, side: THREE.DoubleSide })
  const top = (size / 2) * Math.SQRT2
  const bottom = top * 0.86
  const walls = new THREE.Mesh(new THREE.CylinderGeometry(top, bottom, height, 4, 1, true), plastic)
  walls.rotation.y = Math.PI / 4
  walls.position.y = height / 2
  const floor = new THREE.Mesh(new THREE.CylinderGeometry(bottom, bottom, 0.004, 4), plastic)
  floor.rotation.y = Math.PI / 4
  floor.position.y = 0.002
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(top * 1.05, top * 1.02, 0.008, 4, 1, true), plastic)
  rim.rotation.y = Math.PI / 4
  rim.position.y = height - 0.004
  g.add(walls, floor, rim)
  g.position.set(a, y, b)
  return shadowed(g)
}

// ---------------------------------------------------------------------------
// Stations
// ---------------------------------------------------------------------------
function buildSort(g: THREE.Group, pick: THREE.Object3D[]) {
  const t = SORT.table
  const tb = table(t.a0, t.a1, t.b0, t.b1, t.h, M.wood, M.woodDark)
  g.add(tb)
  pick.push(tb)
  for (const bin of SORT.bins) g.add(tub(bin.a, bin.b, t.h, SORT.binSize, SORT.binWall, COLORS[bin.color]))
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
  /** Is texture point (u, v) inside the T-shirt outline? */
  mask: (u: number, v: number) => boolean
}
function buildLaundry(g: THREE.Group, pick: THREE.Object3D[]): LaundryParts {
  const t = LAUNDRY.table
  const tb = table(t.a0, t.a1, t.b0, t.b1, t.h, mat('#ffffff', { map: woodTexture([1, 1], '#d9c3a3'), roughness: 0.5, metalness: 0 }), M.steelDark)
  g.add(tb)
  pick.push(tb)
  const { verts: flat, uv } = shirtGrid()
  const geo = new THREE.BufferGeometry()
  const verts = new Float32Array(flat)
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3))
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  const index: number[] = []
  for (let i = 0; i < CLOTH_NU; i++) {
    for (let j = 0; j < CLOTH_NV; j++) {
      const k = i * (CLOTH_NV + 1) + j
      index.push(k, k + 1, k + CLOTH_NV + 1, k + 1, k + CLOTH_NV + 2, k + CLOTH_NV + 1)
    }
  }
  geo.setIndex(index)
  geo.computeVertexNormals()
  // Heathered navy cotton jersey; the T-shirt outline is cut out of the grid by the alpha map.
  const tx = shirtTextures()
  const cloth = new THREE.Mesh(
    geo,
    new THREE.MeshPhysicalMaterial({
      map: tx.map,
      alphaMap: tx.alpha,
      alphaTest: 0.5,
      bumpMap: tx.bump,
      bumpScale: 0.9,
      roughness: 0.93,
      metalness: 0,
      sheen: 0.25,
      sheenRoughness: 0.8,
      sheenColor: new THREE.Color('#4f6788'),
      envMapIntensity: 0.4,
      side: THREE.DoubleSide,
    }),
  )
  // Shadows follow the shirt's outline, not the whole grid.
  cloth.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, alphaMap: tx.alpha, alphaTest: 0.5, side: THREE.DoubleSide })
  cloth.castShadow = true
  cloth.receiveShadow = true
  cloth.frustumCulled = false
  g.add(cloth)
  // The outline, read back from the transparency texture (u across the shirt, v hem to collar).
  const img = tx.alpha.image as HTMLCanvasElement
  const px = img.getContext('2d')!.getImageData(0, 0, img.width, img.height).data
  const mask = (u: number, v: number) => {
    const x = Math.min(img.width - 1, Math.max(0, Math.round(u * (img.width - 1))))
    const y = Math.min(img.height - 1, Math.max(0, Math.round((1 - v) * (img.height - 1))))
    return px[(y * img.width + x) * 4] > 127
  }
  return { cloth, flat, verts, mask }
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
  const bw = 0.01
  const bt = s.top + W.basketWall
  const rimMat = mat('#8a6232', { roughness: 0.8, metalness: 0 })
  g.add(
    slab(s.a0 + 0.02, s.a1 - 0.02, s.top, s.top + bw, s.b0 + 0.02, s.b1 - 0.02, M.wicker, 0.004),
    slab(s.a0 + 0.02, s.a0 + 0.02 + bw, s.top, bt, s.b0 + 0.02, s.b1 - 0.02, M.wicker, 0.004),
    slab(s.a1 - 0.02 - bw, s.a1 - 0.02, s.top, bt, s.b0 + 0.02, s.b1 - 0.02, M.wicker, 0.004),
    slab(s.a0 + 0.02, s.a1 - 0.02, s.top, bt, s.b0 + 0.02, s.b0 + 0.02 + bw, M.wicker, 0.004),
    slab(s.a0 + 0.02, s.a1 - 0.02, s.top, bt, s.b1 - 0.02 - bw, s.b1 - 0.02, M.wicker, 0.004),
    // Rolled rim around the top.
    slab(s.a0 + 0.015, s.a1 - 0.015, bt - 0.004, bt + 0.008, s.b0 + 0.015, s.b0 + 0.033, rimMat, 0.005),
    slab(s.a0 + 0.015, s.a1 - 0.015, bt - 0.004, bt + 0.008, s.b1 - 0.033, s.b1 - 0.015, rimMat, 0.005),
    slab(s.a0 + 0.015, s.a0 + 0.033, bt - 0.004, bt + 0.008, s.b0 + 0.015, s.b1 - 0.015, rimMat, 0.005),
    slab(s.a1 - 0.033, s.a1 - 0.015, bt - 0.004, bt + 0.008, s.b0 + 0.015, s.b1 - 0.015, rimMat, 0.005),
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
  let clothPhysics = false
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
    if (key !== lastFolds && !clothPhysics) {
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
  const parts = {
    door: kitchen.door,
    lid: washer.lid,
    drum: washer.drum,
    racks: kitchen.racks,
    cloth: { mesh: laundry.cloth, flat: laundry.flat, verts: laundry.verts, mask: laundry.mask },
  }
  const useClothPhysics = () => {
    clothPhysics = true
  }
  return { groups, containers, items, where, pickables, mech, apply, reset, parts, useClothPhysics }
}
