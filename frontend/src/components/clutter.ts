/**
 * Mess on the floor: fallen mugs and a coffee spill by the kitchen, towels dropped by the
 * laundry table, litter everywhere (paper balls, cans, bottles, wrappers, coffee cups, a
 * takeaway box), plus an empty pedal bin at the end of the kitchen counter and a laundry
 * hamper by the laundry table for the Clean up room task to fill.
 *
 * Each loose piece has an id, a kind (trash, towel or mug) and a simple collider, so the
 * physics engine makes it a dynamic body the hands can pick up. The bin and hamper are
 * hollow and fixed (and are furniture for the base's path and collision checks).
 */
import { BAYS, DINING, DINING_FOOTPRINTS, PLACES } from './dining'
import { BEDROOM_FOOTPRINTS } from './bedroom'
import { DOORWAYS, HOUSE_FOOTPRINTS, LIVING_FOOTPRINTS, START } from './house'
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { DOCKS, FURNITURE, KITCHEN, LAUNDRY, furnitureRects, stationQuat, toWorld, type StationId } from '../lib/stations'
import { createFloorCloth, type FloorCloth, type FloorTowel } from './floorCloth'
import { towelTextures, wickerTextures } from './textures'

export type PropShape =
  | { kind: 'ball'; r: number }
  /** `flat`: lies on its face (a plate): a true cylinder, not the faceted prism used for rolling things. */
  | { kind: 'cyl'; r: number; h: number; flat?: boolean }
  | { kind: 'box'; hx: number; hy: number; hz: number; y?: number }
  /** An open bowl standing on its foot (origin at the bottom): a base and a ring of sloping walls, so what's in it stays in. */
  | { kind: 'bowl'; r: number; h: number }

/** `decor`: the rest of what's on the dining table (napkins, crumbs, the fruit bowl and its fruit, the vase): loose, but no task's business. */
export type ClutterKind = 'trash' | 'towel' | 'mug' | 'dinnerplate' | 'cutlery' | 'glass' | 'decor'
export interface ClutterProp {
  id: string
  kind: ClutterKind
  obj: THREE.Object3D
  shape: PropShape
  /** kg per m³ of the collider */
  density: number
  friction: number
}
/** Something fixed standing on the floor: an upright cylinder or a box. */
export type ClutterStatic =
  | { kind: 'cyl'; x: number; z: number; r: number; h: number }
  | { kind: 'box'; x: number; z: number; w: number; d: number; h: number }
  /** An open-topped container: a floor and a ring of walls, so things dropped in stay in. */
  | { kind: 'tub'; x: number; z: number; r: number; h: number }
/** Where a container stands (world), its radius and rim height. */
export interface Container {
  x: number
  z: number
  r: number
  h: number
}
export interface Clutter {
  props: ClutterProp[]
  statics: ClutterStatic[]
  bin: Container
  hamper: Container
  /** The towels on the floor, as cloth (by prop id). */
  cloth: FloorCloth
  towels: Map<string, FloorTowel>
}
type Loose = Omit<ClutterProp, 'id' | 'kind'>

/** Deterministic random numbers, so the mess is the same on every load. */
function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}
const mat = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05, ...extra })
function shadowed<T extends THREE.Object3D>(o: T): T {
  o.traverse((c) => {
    c.castShadow = true
    c.receiveShadow = true
  })
  return o
}
/** Nudge every vertex a little (crumpled paper, crushed cans, scrunched cloth). */
function jitter(geo: THREE.BufferGeometry, amt: number, r: () => number, axes: [number, number, number] = [1, 1, 1]) {
  const p = geo.attributes.position
  // Vertices at the same spot (shared by neighbouring faces) move together, so the surface
  // crumples without tearing open.
  const moved = new Map<string, [number, number, number]>()
  for (let i = 0; i < p.count; i++) {
    const key = `${p.getX(i).toFixed(4)},${p.getY(i).toFixed(4)},${p.getZ(i).toFixed(4)}`
    let d = moved.get(key)
    if (!d) {
      d = [(r() - 0.5) * amt * axes[0], (r() - 0.5) * amt * axes[1], (r() - 0.5) * amt * axes[2]]
      moved.set(key, d)
    }
    p.setXYZ(i, p.getX(i) + d[0], p.getY(i) + d[1], p.getZ(i) + d[2])
  }
  geo.computeVertexNormals()
  return geo
}
function labelTexture(base: string, accent: string, text: string) {
  const c = document.createElement('canvas')
  c.width = 256
  c.height = 128
  const g = c.getContext('2d')!
  g.fillStyle = base
  g.fillRect(0, 0, 256, 128)
  g.fillStyle = accent
  g.beginPath()
  g.moveTo(0, 80)
  g.bezierCurveTo(64, 50, 128, 110, 256, 70)
  g.lineTo(256, 95)
  g.bezierCurveTo(128, 130, 64, 75, 0, 105)
  g.fill()
  g.fillStyle = 'rgba(255,255,255,0.9)'
  g.font = 'bold 34px sans-serif'
  g.fillText(text, 20, 48)
  g.fillText(text, 148, 48)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

// --- Pieces ------------------------------------------------------------------------------
const PAPER = [mat('#f4f1ea', { roughness: 0.95, flatShading: true }), mat('#fdf6c3', { roughness: 0.95, flatShading: true }), mat('#e6eef7', { roughness: 0.95, flatShading: true })]
const CAN_TEX = [labelTexture('#c8102e', '#ffffff', 'COLA'), labelTexture('#0f5132', '#a3e635', 'LIME'), labelTexture('#1e3a8a', '#93c5fd', 'FIZZ'), labelTexture('#f59e0b', '#7c2d12', 'ORNG')]
const RoundedBox = (w: number, h: number, d: number, r: number) => new RoundedBoxGeometry(w, h, d, 2, Math.min(r, h / 2 - 1e-4, d / 2 - 1e-4))
const ALU = mat('#d4d8de', { metalness: 0.9, roughness: 0.3 })

function paperBall(r: () => number): Loose {
  const rad = 0.028 + r() * 0.014
  const m = new THREE.Mesh(jitter(new THREE.IcosahedronGeometry(rad, 2), rad * 0.45, r), PAPER[Math.floor(r() * PAPER.length)])
  return { obj: shadowed(m), shape: { kind: 'ball', r: rad * 0.92 }, density: 60, friction: 0.7 }
}

function sodaCan(r: () => number, crushed = false): Loose {
  const g = new THREE.Group()
  const h = crushed ? 0.06 : 0.122
  const body = new THREE.CylinderGeometry(0.033, 0.033, h, 24, crushed ? 6 : 1)
  if (crushed) jitter(body, 0.018, r, [1, 0.3, 1])
  const label = mat('#ffffff', { map: CAN_TEX[Math.floor(r() * CAN_TEX.length)], metalness: 0.6, roughness: 0.35 })
  g.add(new THREE.Mesh(body, [label, ALU, ALU]))
  const tab = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.002, 0.02), ALU)
  tab.position.set(0, h / 2 + 0.001, 0.008)
  g.add(tab)
  return { obj: shadowed(g), shape: { kind: 'cyl', r: 0.033, h }, density: crushed ? 150 : 90, friction: 0.5 }
}

function bottle(r: () => number): Loose {
  const g = new THREE.Group()
  const pts = [
    [0, -0.1], [0.032, -0.1], [0.034, -0.095], [0.034, 0.02], [0.03, 0.045], [0.014, 0.075], [0.013, 0.09], [0, 0.09],
  ].map(([x, y]) => new THREE.Vector2(x, y))
  const tint = ['#bfe3f5', '#c7f2d0', '#e5e7eb'][Math.floor(r() * 3)]
  const plastic = new THREE.MeshPhysicalMaterial({ color: tint, roughness: 0.08, transparent: true, opacity: 0.55, clearcoat: 1 })
  g.add(new THREE.Mesh(new THREE.LatheGeometry(pts, 20), plastic))
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.018, 16), mat(['#2563eb', '#16a34a', '#f8fafc'][Math.floor(r() * 3)], { roughness: 0.4 }))
  cap.position.y = 0.098
  const label = new THREE.Mesh(new THREE.CylinderGeometry(0.0345, 0.0345, 0.05, 20, 1, true), mat('#ffffff', { map: CAN_TEX[Math.floor(r() * CAN_TEX.length)], side: THREE.DoubleSide }))
  label.position.y = -0.04
  g.add(cap, label)
  return { obj: shadowed(g), shape: { kind: 'cyl', r: 0.034, h: 0.2 }, density: 40, friction: 0.4 }
}

function wrapper(r: () => number): Loose {
  const w = 0.12 + r() * 0.06
  const d = 0.08 + r() * 0.04
  const geo = jitter(new THREE.PlaneGeometry(w, d, 6, 4), 0.02, r, [0.3, 0.3, 1])
  geo.rotateX(-Math.PI / 2)
  geo.translate(0, 0.008, 0)
  const colors: [string, string, string][] = [['#f97316', '#fde047', 'CHIPS'], ['#7c3aed', '#f0abfc', 'CNDY'], ['#dc2626', '#fef08a', 'SNAX'], ['#0ea5e9', '#e0f2fe', 'BAR']]
  const [c1, c2, txt] = colors[Math.floor(r() * colors.length)]
  const m = new THREE.Mesh(geo, mat('#ffffff', { map: labelTexture(c1, c2, txt), metalness: 0.5, roughness: 0.3, side: THREE.DoubleSide }))
  return { obj: shadowed(m), shape: { kind: 'box', hx: w / 2, hy: 0.006, hz: d / 2, y: 0.006 }, density: 80, friction: 0.9 }
}

function coffeeCup(): Loose {
  const g = new THREE.Group()
  const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.032, 0.13, 24), mat('#f5f5f4', { roughness: 0.8 }))
  const sleeve = new THREE.Mesh(new THREE.CylinderGeometry(0.0435, 0.038, 0.05, 24, 1, true), mat('#8b5a2b', { roughness: 0.95 }))
  sleeve.position.y = 0.01
  const lid = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.046, 0.012, 24), mat('#1f2937', { roughness: 0.5 }))
  lid.position.y = 0.071
  g.add(cup, sleeve, lid)
  return { obj: shadowed(g), shape: { kind: 'cyl', r: 0.04, h: 0.14 }, density: 50, friction: 0.6 }
}

function mug(color: string): Loose {
  const g = new THREE.Group()
  const ceramic = new THREE.MeshPhysicalMaterial({ color, roughness: 0.2, clearcoat: 0.8 })
  g.add(new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.038, 0.09, 28), ceramic))
  const inside = new THREE.Mesh(new THREE.CircleGeometry(0.035, 24), mat('#2a1a10', { roughness: 0.2 }))
  inside.rotation.x = -Math.PI / 2
  inside.position.y = 0.0451
  const handle = new THREE.Mesh(new THREE.TorusGeometry(0.024, 0.007, 10, 20, Math.PI * 1.2), ceramic)
  handle.rotation.z = -Math.PI * 0.6
  handle.position.set(0.042, 0, 0)
  g.add(inside, handle)
  return { obj: shadowed(g), shape: { kind: 'cyl', r: 0.04, h: 0.09 }, density: 700, friction: 0.5 }
}

// ---------------------------------------------------------------------------
// Tableware for the places laid on the dining table
// ---------------------------------------------------------------------------
const CHINA = new THREE.MeshPhysicalMaterial({ color: '#fbfaf7', roughness: 0.18, clearcoat: 0.7 })
const STEEL = mat('#d9dde2', { metalness: 1, roughness: 0.18 })

/** A dinner plate: a lathe-turned profile with a well and a raised rim (26 cm). */
function dinnerPlate(): Loose {
  const R = 0.13
  const pts = [
    [0, -0.009], [0.07, -0.009], [0.075, -0.006], [0.085, -0.004], [0.1, 0.002], [0.125, 0.007], [R, 0.009],
    [R, 0.011], [0.124, 0.0095], [0.1, 0.0045], [0.084, -0.0015], [0.07, -0.004], [0, -0.004],
  ].map(([x, y]) => new THREE.Vector2(x, y))
  const g = new THREE.Group()
  g.add(new THREE.Mesh(new THREE.LatheGeometry(pts, 48), CHINA))
  // A thin gold band near the rim.
  const band = new THREE.Mesh(new THREE.TorusGeometry(0.118, 0.0012, 6, 64), mat('#c9a35a', { metalness: 0.9, roughness: 0.3 }))
  band.rotation.x = Math.PI / 2
  band.position.y = 0.0062
  g.add(band)
  return { obj: shadowed(g), shape: { kind: 'cyl', r: R, h: 0.02, flat: true }, density: 900, friction: 0.5 }
}

/** What's left on a plate after dinner: a smear of sauce, a few peas, rice, a crust (seeded per plate). */
function leftovers(plate: THREE.Object3D, seed: number) {
  let s = seed
  const r = () => ((s = (s * 16807) % 2147483647) / 2147483647)
  const top = -0.0035
  const smear = new THREE.Mesh(new THREE.CircleGeometry(0.035, 24), new THREE.MeshStandardMaterial({ color: ['#8a3218', '#6b4a1e', '#9c5a18'][seed % 3], roughness: 0.35, transparent: true, opacity: 0.85 }))
  smear.rotation.x = -Math.PI / 2
  smear.scale.set(1.4 + r() * 0.6, 0.7 + r() * 0.4, 1)
  smear.rotation.z = r() * 3
  smear.position.set((r() - 0.5) * 0.06, top + 0.0007, (r() - 0.5) * 0.06)
  plate.add(smear)
  const pea = mat('#5f9c3a', { roughness: 0.5 })
  for (let i = 0; i < 2 + Math.floor(r() * 4); i++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.006, 10, 8), pea)
    m.position.set((r() - 0.5) * 0.12, top + 0.006, (r() - 0.5) * 0.12)
    plate.add(m)
  }
  const rice = mat('#f4f1e8', { roughness: 0.6 })
  for (let i = 0; i < 12; i++) {
    const m = new THREE.Mesh(new THREE.CapsuleGeometry(0.0015, 0.004, 2, 6), rice)
    m.rotation.set(Math.PI / 2, r() * 3, 0)
    m.position.set((r() - 0.5) * 0.1, top + 0.0016, (r() - 0.5) * 0.1)
    plate.add(m)
  }
  if (r() < 0.6) {
    const crust = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.012, 0.018), mat('#b07a3c', { roughness: 0.85 }))
    crust.rotation.y = r() * 3
    crust.position.set((r() - 0.5) * 0.08, top + 0.006, (r() - 0.5) * 0.08)
    plate.add(crust)
  }
}

/** A drinking glass (tumbler) with a heavy base, a little water in it. */
function tumbler(water = 0.04): Loose {
  const g = new THREE.Group()
  const pts = [
    [0, -0.05], [0.033, -0.05], [0.036, -0.045], [0.038, 0.05], [0.035, 0.05], [0.0335, -0.035], [0, -0.035],
  ].map(([x, y]) => new THREE.Vector2(x, y))
  const glass = new THREE.MeshPhysicalMaterial({ color: '#e8f1f5', roughness: 0.05, transparent: true, opacity: 0.35, clearcoat: 1, side: THREE.DoubleSide })
  g.add(new THREE.Mesh(new THREE.LatheGeometry(pts, 32), glass))
  if (water > 0.002) {
    const wm = new THREE.Mesh(new THREE.CylinderGeometry(0.033, 0.032, water, 28), new THREE.MeshPhysicalMaterial({ color: '#cfe6f0', roughness: 0.1, transparent: true, opacity: 0.45 }))
    wm.position.y = -0.035 + water / 2
    g.add(wm)
  }
  return { obj: g, shape: { kind: 'cyl', r: 0.038, h: 0.1 }, density: 600, friction: 0.5 }
}

/** A ceramic fruit bowl, origin under its foot. */
function fruitBowl(): Loose {
  const profile = [
    [0, 0], [0.05, 0], [0.09, 0.02], [0.12, 0.06], [0.125, 0.065], [0.118, 0.065], [0.085, 0.025], [0, 0.012],
  ].map(([a, b]) => new THREE.Vector2(a, b))
  const m = new THREE.Mesh(new THREE.LatheGeometry(profile, 40), new THREE.MeshPhysicalMaterial({ color: '#f1efe8', roughness: 0.25, clearcoat: 0.6, side: THREE.DoubleSide }))
  return { obj: shadowed(m), shape: { kind: 'bowl', r: 0.125, h: 0.065 }, density: 1600, friction: 0.6 }
}

/** A piece of fruit (centre origin). */
function fruitPiece(color: string, r: number): Loose {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), mat(color, { roughness: 0.35 }))
  return { obj: shadowed(m), shape: { kind: 'ball', r }, density: 850, friction: 0.7 }
}

/** A glass vase with a few flowers in it (centre origin, the flowers riding along). */
function vaseOfFlowers(): Loose {
  const H = 0.21
  const g = new THREE.Group()
  const vase = new THREE.Mesh(
    new THREE.LatheGeometry([[0, 0], [0.045, 0], [0.05, 0.05], [0.035, 0.16], [0.03, 0.2], [0.034, H], [0, H]].map(([a, b]) => new THREE.Vector2(a, b)), 30),
    new THREE.MeshPhysicalMaterial({ color: '#cfe3ea', roughness: 0.05, transparent: true, opacity: 0.55, clearcoat: 1 }),
  )
  vase.position.y = -H / 2
  g.add(vase)
  const stemM = mat('#3f6b35', { roughness: 0.7 })
  const petals = ['#f5f0e1', '#e8a2b0', '#f7d36b']
  for (let i = 0; i < 7; i++) {
    const a = (i / 7) * Math.PI * 2
    const lean = 0.18 + (i % 3) * 0.06
    const len = 0.3 + (i % 2) * 0.06
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, len, 5), stemM)
    stem.position.set(Math.cos(a) * lean * len * 0.5, 0.12 - H / 2 + len / 2, Math.sin(a) * lean * len * 0.5)
    stem.rotation.set(Math.sin(a) * lean, 0, -Math.cos(a) * lean)
    g.add(stem)
    const bloom = new THREE.Mesh(new THREE.IcosahedronGeometry(0.03, 1), mat(petals[i % 3], { roughness: 0.8, flatShading: true }))
    bloom.position.set(Math.cos(a) * lean * len, 0.12 - H / 2 + len, Math.sin(a) * lean * len)
    g.add(bloom)
  }
  // Glass with water in it, heavy at the bottom.
  return { obj: shadowed(g), shape: { kind: 'cyl', r: 0.047, h: H }, density: 700, friction: 0.5 }
}

/** A used napkin, crumpled (origin on the table under it). */
function napkin(seed: number): Loose {
  let s = seed
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647)
  // A sphere crumpled by a smooth noise of position (shared vertices move together, no cracks).
  const geo = new THREE.SphereGeometry(1, 18, 12)
  const pos = geo.attributes.position
  const ph = rnd() * 10
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i)
    const y = pos.getY(i)
    const z = pos.getZ(i)
    const k = 0.82 + 0.16 * Math.sin(5.3 * x + ph) * Math.cos(4.1 * y - ph) + 0.1 * Math.sin(7.7 * z + 2 * ph) * Math.sin(6.2 * x)
    pos.setXYZ(i, x * k, Math.max(y * k, -0.55), z * k)
  }
  geo.computeVertexNormals()
  const m = new THREE.Mesh(geo, mat('#efe9dc', { roughness: 1 }))
  m.scale.set(0.075, 0.035, 0.06)
  m.position.y = 0.019
  const g = new THREE.Group()
  g.add(m)
  return { obj: shadowed(g), shape: { kind: 'box', hx: 0.065, hy: 0.02, hz: 0.05, y: 0.02 }, density: 90, friction: 1 }
}

/** A crumb (origin on the table under it). */
function crumb(rnd: () => number): Loose {
  const w = 0.006 + rnd() * 0.006
  const d = 0.006 + rnd() * 0.005
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.004, d), mat('#c49a62', { roughness: 0.9 }))
  m.position.y = 0.002
  const g = new THREE.Group()
  g.add(m)
  return { obj: g, shape: { kind: 'box', hx: w / 2, hy: 0.002, hz: d / 2, y: 0.002 }, density: 400, friction: 0.9 }
}

/** Cutlery lies flat, its length along local x (the head at +x), origin on the table under its middle. */
function cutlery(kind: 'fork' | 'knife' | 'spoon'): Loose {
  const g = new THREE.Group()
  const L = kind === 'knife' ? 0.215 : 0.19
  const t = 0.004
  // Handle: a flat, slightly wider stem toward the diner.
  const handle = new THREE.Mesh(RoundedBox(L * 0.55, t, 0.018, 0.0018), STEEL)
  handle.position.set(-L / 2 + (L * 0.55) / 2, t / 2, 0)
  g.add(handle)
  const neck = new THREE.Mesh(new THREE.BoxGeometry(L * 0.12, t * 0.8, 0.008), STEEL)
  neck.position.set(-L / 2 + L * 0.61, t / 2, 0)
  g.add(neck)
  const headX = -L / 2 + L * 0.67
  const headLen = L - L * 0.67
  if (kind === 'knife') {
    const blade = new THREE.Mesh(RoundedBox(headLen, t * 0.6, 0.02, 0.001), STEEL)
    blade.position.set(headX + headLen / 2, t * 0.3, 0.002)
    g.add(blade)
  } else if (kind === 'fork') {
    const base = new THREE.Mesh(new THREE.BoxGeometry(0.02, t * 0.8, 0.024), STEEL)
    base.position.set(headX + 0.01, t / 2, 0)
    g.add(base)
    for (let k = 0; k < 4; k++) {
      const tine = new THREE.Mesh(new THREE.BoxGeometry(headLen - 0.02, t * 0.7, 0.0035), STEEL)
      tine.position.set(headX + 0.02 + (headLen - 0.02) / 2, t / 2, -0.009 + k * 0.006)
      g.add(tine)
    }
  } else {
    const bowl = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 10, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), STEEL)
    bowl.scale.set(headLen / 2, 0.008, 0.02)
    bowl.rotation.x = Math.PI
    bowl.position.set(headX + headLen / 2, 0.009, 0)
    g.add(bowl)
  }
  return { obj: shadowed(g), shape: { kind: 'box', hx: L / 2, hy: 0.004, hz: 0.012, y: 0.004 }, density: 1200, friction: 0.6 }
}

/**
 * A towel dropped on the floor. It is real cloth (see floorCloth.ts): this returns only a
 * marker, which follows the spot where a hand would grab the towel, and its material.
 */
function towel(r: () => number, color: string, band: string): Loose {
  const tex = towelTextures(color, band)
  const marker = new THREE.Object3D()
  marker.userData.towel = {
    material: mat('#ffffff', { map: tex.map, bumpMap: tex.bump, bumpScale: 1.5, roughness: 0.95, side: THREE.DoubleSide }),
    seed: Math.floor(r() * 1e9),
  }
  // The shape is only used for planning (how far it reaches); the cloth itself does the physics.
  return { obj: marker, shape: { kind: 'box', hx: 0.16, hy: 0.012, hz: 0.12, y: 0.012 }, density: 250, friction: 1.0 }
}

/** A small takeaway carton (it has to fit in the bin). */
function takeawayBox(): Loose {
  const g = new THREE.Group()
  const card = mat('#c9a26b', { roughness: 0.95 })
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.05, 0.1), [card, card, mat('#ffffff', { map: labelTexture('#c9a26b', '#b91c1c', 'TAKE'), roughness: 0.95 }), card, card, card])
  box.position.y = 0.025
  const stain = new THREE.Mesh(new THREE.CircleGeometry(0.022, 16), mat('#8a5a2b', { transparent: true, opacity: 0.45, roughness: 0.6 }))
  stain.rotation.x = -Math.PI / 2
  stain.position.set(0.03, 0.0506, -0.02)
  g.add(box, stain)
  return { obj: shadowed(g), shape: { kind: 'box', hx: 0.07, hy: 0.025, hz: 0.05, y: 0.025 }, density: 120, friction: 0.8 }
}

/** A flat spill: an irregular glossy puddle, just above the floor. */
function spill(color: string, size: number, r: () => number) {
  const s = new THREE.Shape()
  const n = 18
  for (let i = 0; i <= n; i++) {
    const t = (i / n) * Math.PI * 2
    const rad = size * (0.7 + r() * 0.5) * (1 + 0.35 * Math.sin(t * 2 + r()))
    const x = Math.cos(t) * rad
    const y = Math.sin(t) * rad * 0.7
    if (i === 0) s.moveTo(x, y)
    else s.lineTo(x, y)
  }
  const m = new THREE.Mesh(new THREE.ShapeGeometry(s), new THREE.MeshPhysicalMaterial({ color, roughness: 0.05, clearcoat: 1, transparent: true, opacity: 0.8, polygonOffset: true, polygonOffsetFactor: -2 }))
  m.rotation.x = -Math.PI / 2
  m.position.y = 0.0015
  m.receiveShadow = true
  return m
}

/** Stainless pedal bin with its lid propped open by an overflowing black bag. */
function trashCan(r: () => number) {
  const T = KITCHEN.trash
  const g = new THREE.Group()
  const steel = mat('#c3c8cf', { metalness: 0.9, roughness: 0.33 })
  const black = mat('#15181c', { roughness: 0.55 })
  const shell = new THREE.Mesh(new THREE.CylinderGeometry(T.r, T.r * 0.92, T.h - 0.03, 40, 1, true), steel)
  shell.position.y = 0.03 + (T.h - 0.03) / 2
  const foot = new THREE.Mesh(new THREE.CylinderGeometry(T.r * 0.93, T.r * 0.95, 0.03, 40), black)
  foot.position.y = 0.015
  // Pedal out the front (local -x faces the room, like the station's own frame).
  const pedal = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.014, 0.08), black)
  pedal.position.set(-T.r - 0.02, 0.03, 0)
  // An empty black liner inside, folded over the rim.
  const liner = new THREE.Mesh(new THREE.CylinderGeometry(T.r - 0.006, T.r * 0.9, T.h - 0.04, 32, 1, true), mat('#0b0d10', { roughness: 0.35, metalness: 0.1, side: THREE.BackSide }))
  liner.position.y = 0.04 + (T.h - 0.04) / 2
  const bottom = new THREE.Mesh(new THREE.CircleGeometry(T.r * 0.9, 32), mat('#0b0d10', { roughness: 0.4 }))
  bottom.rotation.x = -Math.PI / 2
  bottom.position.y = 0.045
  const fold = new THREE.Mesh(new THREE.TorusGeometry(T.r + 0.002, 0.008, 8, 40), mat('#0b0d10', { roughness: 0.35 }))
  fold.rotation.x = Math.PI / 2
  fold.position.y = T.h - 0.01
  // Lid hinged at the back, standing open.
  const lidG = new THREE.Group()
  lidG.position.set(T.r, T.h + 0.005, 0)
  const lid = new THREE.Mesh(new THREE.CylinderGeometry(T.r + 0.005, T.r + 0.005, 0.02, 40), steel)
  lid.position.set(-T.r, 0.01, 0)
  lidG.add(lid)
  lidG.rotation.z = -1.9 // tipped right back, clear of the opening
  g.add(shell, foot, pedal, liner, bottom, fold, lidG)
  const p = toWorld('kitchen', T.a, 0, T.b)
  g.position.copy(p)
  g.quaternion.copy(stationQuat('kitchen'))
  return { obj: shadowed(g), pos: p }
}

/** A woven laundry hamper with an open top, by the laundry table. */
function hamper() {
  const H = LAUNDRY.hamper
  const g = new THREE.Group()
  const weave = wickerTextures([6, 3])
  const wicker = mat('#ffffff', { map: weave.map, bumpMap: weave.bump, bumpScale: 2, roughness: 0.85, side: THREE.DoubleSide })
  const wall = new THREE.Mesh(new THREE.CylinderGeometry(H.r, H.r * 0.88, H.h, 36, 1, true), wicker)
  wall.position.y = H.h / 2
  const floor = new THREE.Mesh(new THREE.CircleGeometry(H.r * 0.88, 36), mat('#6b4f2a', { roughness: 0.9 }))
  floor.rotation.x = -Math.PI / 2
  floor.position.y = 0.012
  const rim = new THREE.Mesh(new THREE.TorusGeometry(H.r, 0.012, 10, 40), mat('#8a6a3c', { roughness: 0.8 }))
  rim.rotation.x = Math.PI / 2
  rim.position.y = H.h
  // Rope handles either side.
  for (const s of [-1, 1]) {
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.04, 0.007, 8, 16, Math.PI), mat('#d6c7a1', { roughness: 0.9 }))
    handle.position.set(0, H.h - 0.06, s * (H.r + 0.004))
    handle.rotation.y = Math.PI / 2
    g.add(handle)
  }
  g.add(wall, floor, rim)
  const p = toWorld('laundry', H.a, 0, H.b)
  g.position.copy(p)
  return { obj: shadowed(g), pos: p }
}

/** A loose piece as a sphere round where it is (for the cloth to drape over). */
export function ballOf(pr: ClutterProp, at: THREE.Vector3) {
  const sh = pr.shape
  const r = sh.kind === 'ball' ? sh.r : sh.kind === 'cyl' ? Math.max(sh.r, sh.h / 2) + 0.01 : sh.kind === 'bowl' ? sh.r : Math.max(sh.hx, sh.hz) * 0.8
  return { x: at.x, y: at.y, z: at.z, r }
}

export function buildClutter(scene: THREE.Scene, obstacles: { x: number; z: number; r: number }[]): Clutter {
  const r = rng(42)
  const props: ClutterProp[] = []
  const rects = [...furnitureRects(), ...DINING_FOOTPRINTS, ...BEDROOM_FOOTPRINTS, ...HOUSE_FOOTPRINTS, ...LIVING_FOOTPRINTS]
  const T = KITCHEN.trash

  const bin = trashCan(r)
  scene.add(bin.obj)
  const hp = hamper()
  scene.add(hp.obj)
  const H = LAUNDRY.hamper
  const statics: ClutterStatic[] = [
    { kind: 'tub', x: bin.pos.x, z: bin.pos.z, r: T.r, h: T.h },
    { kind: 'tub', x: hp.pos.x, z: hp.pos.z, r: H.r, h: H.h },
  ]
  const count: Record<ClutterKind, number> = { trash: 0, towel: 0, mug: 0, dinnerplate: 0, cutlery: 0, glass: 0, decor: 0 }

  /** Put a prop on the floor at (x, z), lying (axis level) or standing, turned by yaw. */
  const place = (loose: Loose, x: number, z: number, yaw: number, lying: boolean, kind: ClutterKind = 'trash') => {
    const pr: ClutterProp = { ...loose, kind, id: `${kind}${count[kind]++}` }
    const o = pr.obj
    o.name = pr.id
    const s = pr.shape
    if (s.kind === 'cyl' && lying) {
      // Turn to the heading first (Y), then tip over (X).
      o.quaternion.setFromEuler(new THREE.Euler(Math.PI / 2, yaw, 0, 'YXZ'))
      o.position.set(x, s.r + 0.001, z)
    } else {
      o.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
      const y = s.kind === 'ball' ? s.r + 0.001 : s.kind === 'cyl' ? s.h / 2 + 0.001 : 0.001
      o.position.set(x, y, z)
    }
    scene.add(o)
    props.push(pr)
  }
  const at = (st: 'kitchen' | 'laundry', a: number, b: number) => toWorld(st, a, 0, b)

  // Kitchen: two mugs knocked over in a coffee spill, one rolled away; a dropped towel.
  {
    const s = at('kitchen', 0.18, 0.95)
    const puddle = spill('#3a2212', 0.16, r)
    puddle.position.x = s.x
    puddle.position.z = s.z
    scene.add(puddle)
    // Kitchen coordinates (a out from the counter, which starts at a = 0.38), about 30 cm apart.
    const m0 = at('kitchen', 0.15, 0.88)
    place(mug('#b91c1c'), m0.x, m0.z, 0.6, true, 'mug')
    const m1 = at('kitchen', -0.1, 1.08)
    place(mug('#f8fafc'), m1.x, m1.z, 2.2, true, 'mug')
    const far = at('kitchen', 0.08, 1.36)
    place(mug('#1d4ed8'), far.x, far.z, 1.2, true, 'mug')
    // A dropped tea towel out on the kitchen floor (clear of where the robot works at the counter).
    const t = at('kitchen', -0.75, -0.3)
    place(towel(r, '#e2e8f0', '#94a3b8'), t.x, t.z, 0.4, false, 'towel')
    // Missed shots around the bin.
    // Well short of the bin, out on the floor and a good 30 cm apart: clear of where the robot
    // stands to drop things in (and turns), and of the walls, so it has room beside each.
    const misses: [number, number][] = [[-0.8, -2.15], [-1.1, -1.85], [-0.45, -2.25]]
    for (let i = 0; i < misses.length; i++) {
      const p = at('kitchen', misses[i][0], misses[i][1])
      place(paperBall(r), p.x, p.z, r() * 6, false)
    }
    const pz = at('kitchen', -0.5, -1.0)
    place(takeawayBox(), pz.x, pz.z, 0.3, false)
  }
  // Towels: two dropped in the laundry room past the dryer, one in the bedroom.
  {
    place(towel(r, '#a5b4fc', '#e0e7ff'), 4.0, -3.3, 1.1, false, 'towel')
    place(towel(r, '#fca5a5', '#fee2e2'), 4.5, -4.2, -0.5, false, 'towel')
    place(towel(r, '#fde68a', '#fffbeb'), -2.4, 2.0, 2.0, false, 'towel')
  }

  // After dinner for four: at each place a plate with leftovers and the knife, fork and spoon
  // laid across it (finished), and a half-drunk glass pushed about. All loose, like the rest.
  {
    const onTable = (loose: Loose, kind: ClutterKind, x: number, z: number, yaw: number, lift = 0) => {
      const pr: ClutterProp = { ...loose, kind, id: `${kind}${count[kind]++}` }
      const o = pr.obj
      o.name = pr.id
      o.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
      const sh = pr.shape
      o.position.set(x, DINING.h + lift + (sh.kind === 'cyl' ? sh.h / 2 : 0) + 0.001, z)
      scene.add(o)
      props.push(pr)
      return pr
    }
    const plateTop = 0.02
    PLACES.forEach((p, i) => {
      const rx = -p.fz
      const rz = p.fx
      const plate = dinnerPlate()
      leftovers(plate.obj, 7 + i * 13)
      onTable(plate, 'dinnerplate', p.x, p.z, i)
      // Knife, fork and spoon laid together on the left half of the plate, pointing away from
      // the diner ("finished"), a little askew.
      const yaw = Math.atan2(-p.fz, p.fx) + (i % 2 ? 0.12 : -0.1)
      const items: ['fork' | 'knife' | 'spoon', number][] = [['fork', -0.02], ['knife', -0.045], ['spoon', -0.07]]
      for (const [k, off] of items) onTable(cutlery(k), 'cutlery', p.x + rx * off, p.z + rz * off, yaw, plateTop + 0.002)
      // The glass: moved from its place up to the right, more or less drunk.
      const gx = p.x + p.fx * (0.1 + (i % 3) * 0.02) + rx * (0.13 + (i % 2 ? 0.05 : -0.03))
      const gz = p.z + p.fz * (0.1 + (i % 3) * 0.02) + rz * (0.13 + (i % 2 ? 0.05 : -0.03))
      onTable(tumbler([0.012, 0.0, 0.025, 0.006][i]), 'glass', gx, gz, 0)
    })
    // A crumpled napkin at each place, crumbs round the plates: loose, so they get pushed about.
    {
      let seed = 11
      const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
      PLACES.forEach((p, i) => {
        const rx = -p.fz
        const rz = p.fx
        onTable(napkin(31 + i * 7), 'decor', p.x - rx * 0.25 - p.fx * 0.04, p.z - rz * 0.25 - p.fz * 0.04, rnd() * Math.PI)
        for (let k = 0; k < 9; k++) {
          const a = rnd() * Math.PI * 2
          const d = 0.15 + rnd() * 0.08
          onTable(crumb(rnd), 'decor', p.x + Math.cos(a) * d, p.z + Math.sin(a) * d, rnd() * 3)
        }
      })
    }
    // The centrepiece: a bowl of fruit and a vase of flowers, standing on the runner.
    {
      const runnerTop = 0.003
      onTable(fruitBowl(), 'decor', DINING.x, DINING.z - 0.3, 0, runnerTop)
      const fruit: [string, number, number, number][] = [
        ['#c0392b', 0.04, -0.03, 0.02], ['#e67e22', 0.042, 0.04, -0.01], ['#27ae60', 0.038, 0.0, 0.05], ['#f1c40f', 0.035, -0.045, -0.045], ['#c0392b', 0.038, 0.05, 0.05],
      ]
      for (const [c, r, dx, dz] of fruit) onTable(fruitPiece(c, r), 'decor', DINING.x + dx, DINING.z - 0.3 + dz, 0, runnerTop + 0.03 + r * 0.8)
      onTable(vaseOfFlowers(), 'decor', DINING.x, DINING.z + 0.32, 0, runnerTop)
    }
  }

  // Litter all over the floor, kept out of furniture, obstacles, the docks and the robot's start.
  const docks = Object.values(DOCKS)
  const clear = (x: number, z: number, pad: number) => {
    if (Math.hypot(x - START.x, z - START.z) < 0.6) return false
    for (const q of DOORWAYS) if (x > q.x0 && x < q.x1 && z > q.z0 && z < q.z1) return false
    // Nor where the robot stands to clear the dining table, or works at the kitchen counter.
    for (const b of BAYS) if (Math.hypot(x - b.x, z - b.z) < 0.7) return false
    for (const k of [KITCHEN.dishPose, KITCHEN.stackPose]) {
      const p = toWorld('kitchen', k.a, 0, k.b)
      if (Math.hypot(x - p.x, z - p.z) < 0.75) return false
    }
    // Well out from the walls: the arms can't reach down beside a tall wall.
    for (const q of HOUSE_FOOTPRINTS) if (x > q.x0 - 0.75 && x < q.x1 + 0.75 && z > q.z0 - 0.75 && z < q.z1 + 0.75) return false
    // Somewhere beside it the base can stand (clear of furniture and walls by its turning
    // margin), so the robot can reach it.
    const standable = (cx: number, cz: number) => rects.every((q) => Math.hypot(Math.max(q.x0 - cx, 0, cx - q.x1), Math.max(q.z0 - cz, 0, cz - q.z1)) >= 0.62)
    let reachable = false
    for (let k = 0; k < 12 && !reachable; k++) reachable = standable(x + 0.55 * Math.cos((k * Math.PI) / 6), z + 0.55 * Math.sin((k * Math.PI) / 6))
    if (!reachable) return false
    for (const d of docks) if (Math.hypot(x - d.x, z - d.z) < 0.55) return false
    for (const q of rects) if (x > q.x0 - pad && x < q.x1 + pad && z > q.z0 - pad && z < q.z1 + pad) return false
    for (const o of obstacles) if (Math.hypot(x - o.x, z - o.z) < o.r + pad + 0.05) return false
    if (Math.hypot(x - bin.pos.x, z - bin.pos.z) < T.r + pad + 0.05) return false
    if (Math.hypot(x - hp.pos.x, z - hp.pos.z) < H.r + pad + 0.05) return false
    for (const p of props) if (Math.hypot(x - p.obj.position.x, z - p.obj.position.z) < 0.24) return false // a bottle is 20 cm long
    return true
  }
  const makers: (() => [Loose, boolean])[] = [
    () => [paperBall(r), false],
    () => [paperBall(r), false],
    () => [sodaCan(r), r() < 0.75],
    () => [sodaCan(r, true), true],
    () => [bottle(r), true],
    () => [wrapper(r), false],
    () => [wrapper(r), false],
    () => [coffeeCup(), true],
  ]
  let placed = 0
  // 10 pieces of rubbish in all: the 4 by the bin (3 missed shots and the takeaway box) and 6 about the room.
  const SCATTERED = 6
  for (let tries = 0; placed < SCATTERED && tries < 2000; tries++) {
    // Anywhere in the house.
    const x = -4.6 + r() * 9.2
    const z = -4.6 + r() * 9.2
    if (!clear(x, z, 0.15)) continue // room for the robot to reach it without touching the furniture
    const [pr, lying] = makers[Math.floor(r() * makers.length)]()
    place(pr, x, z, r() * Math.PI * 2, lying)
    placed++
  }
  // The towels as cloth: they drape on the floor and collide with the furniture, the bin and
  // the hamper (which have their own round walls, so their square furniture boxes are left out).
  const boxes = (Object.keys(FURNITURE) as StationId[]).flatMap((st) =>
    FURNITURE[st]
      .filter((b) => !(st === 'kitchen' && b.y1 === T.h) && !(st === 'laundry' && b.y1 === H.h))
      .map((b) => {
        const p = toWorld(st, b.a0, 0, b.b0)
        const q = toWorld(st, b.a1, 0, b.b1)
        return { x0: Math.min(p.x, q.x), x1: Math.max(p.x, q.x), z0: Math.min(p.z, q.z), z1: Math.max(p.z, q.z), y0: b.y0, y1: b.y1 }
      }),
  )
  const towelProps = props.filter((pr) => pr.kind === 'towel')
  const { length, half } = LAUNDRY.towel
  const cloth = createFloorCloth(
    towelProps.map((pr) => ({
      x: pr.obj.position.x,
      z: pr.obj.position.z,
      yaw: new THREE.Euler().setFromQuaternion(pr.obj.quaternion, 'YXZ').y,
      material: pr.obj.userData.towel.material,
      seed: pr.obj.userData.towel.seed,
    })),
    { w: 2 * half, l: length, nu: 19, nv: 13 },
    boxes,
    [
      { x: bin.pos.x, z: bin.pos.z, r: T.r, h: T.h, floor: 0.045 },
      { x: hp.pos.x, z: hp.pos.z, r: H.r, h: H.h, floor: 0.02 },
    ],
    props.filter((pr) => pr.kind !== 'towel').map((pr) => ballOf(pr, pr.obj.position)),
  )
  const towels = new Map<string, FloorTowel>()
  towelProps.forEach((pr, i) => {
    towels.set(pr.id, cloth.towels[i])
    scene.add(cloth.towels[i].mesh)
    pr.obj.position.copy(cloth.towels[i].grabPoint())
  })
  return {
    props,
    statics,
    bin: { x: bin.pos.x, z: bin.pos.z, r: T.r, h: T.h },
    hamper: { x: hp.pos.x, z: hp.pos.z, r: H.r, h: H.h },
    cloth,
    towels,
  }
}
