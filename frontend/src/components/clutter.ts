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
import * as THREE from 'three'
import { DOCKS, FURNITURE, KITCHEN, LAUNDRY, furnitureRects, stationQuat, toWorld, type StationId } from '../lib/stations'
import { createFloorCloth, type FloorCloth, type FloorTowel } from './floorCloth'
import { towelTextures, wickerTextures } from './textures'

export type PropShape =
  | { kind: 'ball'; r: number }
  | { kind: 'cyl'; r: number; h: number }
  | { kind: 'box'; hx: number; hy: number; hz: number; y?: number }

export type ClutterKind = 'trash' | 'towel' | 'mug'
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
  const r = sh.kind === 'ball' ? sh.r : sh.kind === 'cyl' ? Math.max(sh.r, sh.h / 2) + 0.01 : Math.max(sh.hx, sh.hz) * 0.8
  return { x: at.x, y: at.y, z: at.z, r }
}

export function buildClutter(scene: THREE.Scene, obstacles: { x: number; z: number; r: number }[]): Clutter {
  const r = rng(42)
  const props: ClutterProp[] = []
  const rects = furnitureRects()
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
  const count: Record<ClutterKind, number> = { trash: 0, towel: 0, mug: 0 }

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
    const t = at('kitchen', 0.16, -0.72)
    place(towel(r, '#e2e8f0', '#94a3b8'), t.x, t.z, 0.4, false, 'towel')
    // Missed shots around the bin.
    // In front of the bin and off past the end of the counter, a good 30 cm apart.
    const misses: [number, number][] = [[0.14, -2.16], [-0.12, -1.86], [0.3, -2.42]]
    for (let i = 0; i < misses.length; i++) {
      const p = at('kitchen', misses[i][0], misses[i][1])
      place(paperBall(r), p.x, p.z, r() * 6, false)
    }
    const pz = at('kitchen', -0.02, -1.3)
    place(takeawayBox(), pz.x, pz.z, 0.3, false)
  }
  // Laundry: towels dropped beside the table, one halfway across the room.
  {
    const a = at('laundry', 0.1, -1.05)
    place(towel(r, '#a5b4fc', '#e0e7ff'), a.x, a.z, 1.1, false, 'towel')
    const b = at('laundry', 0.55, -0.95)
    place(towel(r, '#fca5a5', '#fee2e2'), b.x, b.z, -0.5, false, 'towel')
    place(towel(r, '#fde68a', '#fffbeb'), 0.75, -0.55, 2.0, false, 'towel')
  }

  // Litter all over the floor, kept out of furniture, obstacles, the docks and the robot's start.
  const docks = Object.values(DOCKS)
  const clear = (x: number, z: number, pad: number) => {
    if (Math.hypot(x, z) < 0.55) return false
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
    const x = -1.55 + r() * 4.2
    const z = (r() - 0.5) * 5.2
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
