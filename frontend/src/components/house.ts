/**
 * The house: outer walls and partitions that split the floor into a kitchen (west), a laundry
 * room (north), a living room with the dining table (middle), and the bedroom (south), each
 * with its own floor. Every room opens through a pair of doors, swung back flat against the
 * wall so the opening is wide enough for the robot. Also the living room's furniture.
 *
 * Walls between the viewer's camera and what it looks at fade out (`fadeWalls`), so the rooms
 * stay visible from any angle.
 *
 * Everything is in world coordinates. `HOUSE_FOOTPRINTS` and `LIVING_FOOTPRINTS` list what
 * stands on the floor, for the robot's path planning and collision checks and for physics.
 */
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import type { Footprint } from './dining'
import { knitBump, paintTexture, woodTexture } from './textures'

/** Where the robot starts (living room). */
export const START = { x: 1.2, z: -1.0 }

const T = 0.1 // wall thickness
const H = 2.45 // wall height
const DOOR_W = 1.5 // the opening, frame to frame (room for the robot with its elbows out)
const DOOR_H = 2.05
const LEAF_W = 0.72
const LEAF_T = 0.04

/** Rooms (floor rectangles). */
export const ROOMS = {
  kitchen: { x0: -5.0, x1: -0.8, z0: -5.0, z1: 0.4 },
  laundry: { x0: -0.8, x1: 5.3, z0: -5.0, z1: -2.4 },
  living: { x0: -0.8, x1: 5.3, z0: -2.4, z1: 5.0 },
  bedroom: { x0: -5.0, x1: -0.8, z0: 0.4, z1: 5.0 },
}

/**
 * Wall lines. `axis: 'x'` runs along x at z = `at`; `'z'` runs along z at x = `at`. A door is
 * the span [c0, c0 + DOOR_W] along the wall; its leaves swing to side `swing` (+1 or -1 on
 * the wall's normal) and lie back against the wall.
 */
interface WallLine {
  name: string
  axis: 'x' | 'z'
  at: number
  from: number
  to: number
  door?: { c0: number; swing: 1 | -1 }
}
const WALLS: WallLine[] = [
  { name: 'north', axis: 'x', at: -5.0, from: -5.05, to: 5.35 },
  { name: 'west', axis: 'z', at: -5.0, from: -5.0, to: 5.0 },
  { name: 'east', axis: 'z', at: 5.3, from: -5.0, to: 5.0 },
  // The bedroom's back wall is its own, painted sage (bedroom.ts); this is the living room's.
  { name: 'south', axis: 'x', at: 5.05, from: -0.85, to: 5.35 },
  { name: 'kitchen', axis: 'z', at: -0.8, from: -4.95, to: 0.4, door: { c0: -2.0, swing: -1 } },
  { name: 'kitchenSouth', axis: 'x', at: 0.4, from: -4.95, to: -0.85 },
  { name: 'bedroom', axis: 'z', at: -0.8, from: 0.45, to: 5.0, door: { c0: 1.2, swing: -1 } },
  { name: 'laundry', axis: 'x', at: -2.4, from: -0.75, to: 5.25, door: { c0: 1.0, swing: -1 } },
]

/** A wall-aligned box: `u` along the wall, `n` across it (world rect). */
function rectOf(w: WallLine, u0: number, u1: number, n0: number, n1: number) {
  return w.axis === 'x'
    ? { x0: Math.min(u0, u1), x1: Math.max(u0, u1), z0: w.at + Math.min(n0, n1), z1: w.at + Math.max(n0, n1) }
    : { x0: w.at + Math.min(n0, n1), x1: w.at + Math.max(n0, n1), z0: Math.min(u0, u1), z1: Math.max(u0, u1) }
}
/** Solid stretches of a wall (the door left out). */
function spans(w: WallLine): [number, number][] {
  if (!w.door) return [[w.from, w.to]]
  return [
    [w.from, w.door.c0],
    [w.door.c0 + DOOR_W, w.to],
  ]
}
/** The two open leaves, as wall-aligned spans (along, across). */
function leaves(w: WallLine): { u0: number; u1: number; n0: number; n1: number; hinge: number; dir: number }[] {
  if (!w.door) return []
  const s = w.door.swing
  const n0 = s * (T / 2 + 0.008)
  const n1 = s * (T / 2 + 0.008 + LEAF_T)
  const a = w.door.c0 + 0.03
  const b = w.door.c0 + DOOR_W - 0.03
  return [
    { u0: a - LEAF_W, u1: a, n0, n1, hinge: a, dir: -1 },
    { u0: b, u1: b + LEAF_W, n0, n1, hinge: b, dir: 1 },
  ]
}

export const HOUSE_FOOTPRINTS: Footprint[] = WALLS.flatMap((w) => [
  ...spans(w).map(([u0, u1]) => ({ ...rectOf(w, u0, u1, -T / 2, T / 2), h: H })),
  ...leaves(w).map((l) => ({ ...rectOf(w, l.u0, l.u1, l.n0, l.n1), h: DOOR_H })),
])

/**
 * For the path planner only: the walls and open doors a little thicker. Unlike the furniture,
 * they're taller than the arms, and carrying things the elbows reach out past the base; this
 * keeps them clear (and routes through the middle of each doorway).
 */
export const PLAN_WALLS = HOUSE_FOOTPRINTS.map((f) => ({ x0: f.x0 - 0.17, x1: f.x1 + 0.17, z0: f.z0 - 0.17, z1: f.z1 + 0.17 }))

/** The floor in and either side of each doorway (kept clear of litter so the robot can pass). */
export const DOORWAYS = WALLS.filter((w) => w.door).map((w) => rectOf(w, w.door!.c0 - 0.3, w.door!.c0 + DOOR_W + 0.3, -1.2, 1.2))

// ---------------------------------------------------------------------------
// Living room furniture
// ---------------------------------------------------------------------------
// The lounge in the living room's north-east corner (TV against the laundry wall, the sofa
// facing it), leaving the rest of the room open round the dining table.
const TV = { x0: 3.4, x1: 4.6, z0: -2.35, z1: -1.95, h: 0.48 }
const SOFA = { x0: 3.3, x1: 4.7, z0: -0.25, z1: 0.59, seat: 0.43, back: 0.82, arm: 0.6 }
const COFFEE = { x0: 3.6, x1: 4.4, z0: -1.25, z1: -0.8, h: 0.42 }
const LAMP = { x: 4.95, z: 0.75, r: 0.16 }

export const LIVING_FOOTPRINTS: Footprint[] = [
  { x0: TV.x0, x1: TV.x1, z0: TV.z0, z1: TV.z1, h: 1.15 },
  { x0: SOFA.x0, x1: SOFA.x1, z0: SOFA.z0, z1: SOFA.z1, h: SOFA.back },
  { x0: COFFEE.x0, x1: COFFEE.x1, z0: COFFEE.z0, z1: COFFEE.z1, h: COFFEE.h },
  { x0: LAMP.x - LAMP.r, x1: LAMP.x + LAMP.r, z0: LAMP.z - LAMP.r, z1: LAMP.z + LAMP.r, h: 1.6 },
]

// ---------------------------------------------------------------------------
// Building it
// ---------------------------------------------------------------------------
const mat = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0, ...extra })
function shadowed<T extends THREE.Object3D>(o: T): T {
  o.traverse((c) => {
    c.castShadow = true
    c.receiveShadow = true
  })
  return o
}
function box(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, m: THREE.Material, r = 0) {
  const w = x1 - x0
  const h = y1 - y0
  const d = z1 - z0
  const rr = Math.min(r, w / 2 - 1e-4, h / 2 - 1e-4, d / 2 - 1e-4)
  const o = new THREE.Mesh(rr > 0.001 ? new RoundedBoxGeometry(w, h, d, 3, rr) : new THREE.BoxGeometry(w, h, d), m)
  o.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
  return shadowed(o)
}

function canvasTex(w: number, h: number, draw: (g: CanvasRenderingContext2D) => void, repeat: [number, number]) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  draw(c.getContext('2d')!)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.repeat.set(...repeat)
  t.anisotropy = 8
  return t
}
/** Square tiles with grout lines, a little variation from tile to tile. */
function tileTexture(base: [number, number, number], grout: string, n: number, repeat: [number, number], seed = 1) {
  let s = seed
  const r = () => ((s = (s * 16807) % 2147483647) / 2147483647)
  return canvasTex(
    512,
    512,
    (g) => {
      const cell = 512 / n
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          const v = (r() - 0.5) * 14
          g.fillStyle = `rgb(${base[0] + v},${base[1] + v},${base[2] + v})`
          g.fillRect(i * cell, j * cell, cell, cell)
          for (let k = 0; k < 40; k++) {
            g.fillStyle = r() < 0.5 ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)'
            g.fillRect(i * cell + r() * cell, j * cell + r() * cell, 2 + r() * 6, 2 + r() * 6)
          }
        }
      g.strokeStyle = grout
      g.lineWidth = 3
      for (let i = 0; i <= n; i++) {
        g.beginPath()
        g.moveTo(i * cell, 0)
        g.lineTo(i * cell, 512)
        g.moveTo(0, i * cell)
        g.lineTo(512, i * cell)
        g.stroke()
      }
    },
    repeat,
  )
}
/** Floorboards: long planks in staggered rows, each with its own tone and grain. */
function plankTexture(base: [number, number, number], repeat: [number, number], seed = 3) {
  let s = seed
  const r = () => ((s = (s * 16807) % 2147483647) / 2147483647)
  return canvasTex(
    1024,
    1024,
    (g) => {
      const rows = 8
      const rh = 1024 / rows
      for (let j = 0; j < rows; j++) {
        let x = -r() * 600
        while (x < 1024) {
          const len = 380 + r() * 420
          const v = (r() - 0.5) * 30
          g.fillStyle = `rgb(${base[0] + v},${base[1] + v * 0.8},${base[2] + v * 0.6})`
          g.fillRect(x, j * rh, len, rh)
          for (let k = 0; k < 14; k++) {
            const y = j * rh + 4 + r() * (rh - 8)
            g.strokeStyle = r() < 0.6 ? `rgba(80,48,20,${0.06 + r() * 0.1})` : `rgba(255,230,190,${0.05 + r() * 0.06})`
            g.lineWidth = 0.6 + r() * 1.4
            g.beginPath()
            g.moveTo(x, y)
            g.bezierCurveTo(x + len / 3, y + (r() - 0.5) * 6, x + (2 * len) / 3, y + (r() - 0.5) * 6, x + len, y + (r() - 0.5) * 4)
            g.stroke()
          }
          g.fillStyle = 'rgba(40,24,12,0.55)'
          g.fillRect(x, j * rh, 2, rh)
          x += len
        }
        g.fillStyle = 'rgba(40,24,12,0.5)'
        g.fillRect(0, j * rh, 1024, 2)
      }
    },
    repeat,
  )
}

/** Things that fade out when they're between the camera and what it looks at. */
const fadeables: { box: THREE.Box3; mats: THREE.MeshStandardMaterial[]; meshes: THREE.Mesh[]; o: number }[] = []

/** Make everything under `root` fade as one (its materials are cloned so it fades alone). */
export function addFadeable(root: THREE.Object3D) {
  const mats: THREE.MeshStandardMaterial[] = []
  const meshes: THREE.Mesh[] = []
  root.updateMatrixWorld(true)
  root.traverse((c) => {
    if (!(c instanceof THREE.Mesh)) return
    const m = (c.material as THREE.MeshStandardMaterial).clone()
    c.material = m
    mats.push(m)
    meshes.push(c)
  })
  // As tall as the sky: a line passing over the top still has the wall hiding the floor behind it.
  const b = new THREE.Box3().setFromObject(root).expandByScalar(0.02)
  b.max.y = 100
  fadeables.push({ box: b, mats, meshes, o: 1 })
}

const ray = new THREE.Ray()
const hit = new THREE.Vector3()
/**
 * Fade the walls the line from the camera to its target passes through (`on` false: show
 * them all, e.g. for the robot's own cameras). `dt` eases the change.
 */
export function fadeWalls(camera: THREE.Vector3, target: THREE.Vector3, on: boolean, dt: number) {
  const len = camera.distanceTo(target)
  ray.origin.copy(camera)
  ray.direction.copy(target).sub(camera).normalize()
  for (const f of fadeables) {
    let block = false
    if (on) {
      if (f.box.containsPoint(camera)) block = true
      else if (ray.intersectBox(f.box, hit) && hit.distanceTo(camera) < len - 0.05) block = true
    }
    const goal = block ? 0.12 : 1
    const o = dt > 0 ? f.o + (goal - f.o) * Math.min(1, dt * 8) : goal
    if (Math.abs(o - f.o) < 1e-4 && (o === 1 || o === goal)) continue
    f.o = Math.abs(o - goal) < 0.01 ? goal : o
    const solid = f.o > 0.99
    for (const m of f.mats) {
      m.transparent = !solid
      m.opacity = f.o
      m.depthWrite = solid
      m.needsUpdate = true
    }
    for (const m of f.meshes) m.castShadow = solid
  }
}

export function buildHouse(scene: THREE.Scene) {
  const root = new THREE.Group()
  root.name = 'house'

  // Floors: tiles in the kitchen and laundry room, oak boards in the living room, a paler
  // wood in the bedroom.
  const sz = (id: keyof typeof ROOMS) => [ROOMS[id].x1 - ROOMS[id].x0, ROOMS[id].z1 - ROOMS[id].z0]
  const floors: [keyof typeof ROOMS, THREE.Texture, number][] = [
    ['kitchen', tileTexture([206, 200, 190], '#9b948a', 4, [sz('kitchen')[0] / 1.2, sz('kitchen')[1] / 1.2]), 0.55],
    ['laundry', tileTexture([180, 186, 192], '#8a9097', 6, [sz('laundry')[0] / 1.2, sz('laundry')[1] / 1.2], 4), 0.6],
    ['living', plankTexture([150, 104, 66], [sz('living')[0] / 2.2, sz('living')[1] / 2.2]), 0.5],
    ['bedroom', plankTexture([196, 160, 120], [sz('bedroom')[0] / 2.2, sz('bedroom')[1] / 2.2], 9), 0.55],
  ]
  for (const [id, map, rough] of floors) {
    const r = ROOMS[id]
    const f = new THREE.Mesh(new THREE.PlaneGeometry(r.x1 - r.x0, r.z1 - r.z0), mat('#ffffff', { map, roughness: rough }))
    f.rotation.x = -Math.PI / 2
    f.position.set((r.x0 + r.x1) / 2, 0.003, (r.z0 + r.z1) / 2)
    f.receiveShadow = true
    f.name = `floor_${id}`
    root.add(f)
  }

  // Walls, each line its own group so it fades on its own.
  const paint = mat('#ffffff', { color: '#ece7de', map: paintTexture([4, 2]), roughness: 0.92 })
  const trim = mat('#f7f5f0', { roughness: 0.45 })
  const leafMat = mat('#f4f2ec', { roughness: 0.4 })
  const handleMat = mat('#2b2f33', { metalness: 0.7, roughness: 0.35 })
  for (const w of WALLS) {
    const g = new THREE.Group()
    g.name = `wall_${w.name}`
    const add = (u0: number, u1: number, n0: number, n1: number, y0: number, y1: number, m: THREE.Material, r = 0) => {
      const q = rectOf(w, u0, u1, n0, n1)
      g.add(box(q.x0, q.x1, y0, y1, q.z0, q.z1, m, r))
    }
    for (const [u0, u1] of spans(w)) {
      add(u0, u1, -T / 2, T / 2, 0, H, paint)
      // Skirting on both faces.
      for (const s of [-1, 1]) add(u0, u1, s * (T / 2), s * (T / 2 + 0.012), 0, 0.1, trim)
    }
    if (w.door) {
      const c0 = w.door.c0
      const c1 = c0 + DOOR_W
      // Header over the opening, the frame lining it, and an architrave round it on both faces.
      add(c0, c1, -T / 2, T / 2, DOOR_H, H, paint)
      add(c0, c0 + 0.03, -T / 2 - 0.005, T / 2 + 0.005, 0, DOOR_H, trim)
      add(c1 - 0.03, c1, -T / 2 - 0.005, T / 2 + 0.005, 0, DOOR_H, trim)
      add(c0, c1, -T / 2 - 0.005, T / 2 + 0.005, DOOR_H - 0.03, DOOR_H, trim)
      for (const s of [-1, 1]) {
        const n0 = s * (T / 2)
        const n1 = s * (T / 2 + 0.015)
        add(c0 - 0.07, c0, n0, n1, 0, DOOR_H + 0.07, trim, 0.004)
        add(c1, c1 + 0.07, n0, n1, 0, DOOR_H + 0.07, trim, 0.004)
        add(c0 - 0.07, c1 + 0.07, n0, n1, DOOR_H, DOOR_H + 0.07, trim, 0.004)
      }
      // The two leaves, swung back flat against the wall: a frame with two raised panels and a lever.
      for (const l of leaves(w)) {
        add(l.u0, l.u1, l.n0, l.n1, 0.01, DOOR_H - 0.04, leafMat, 0.006)
        const face = l.n1 + Math.sign(l.n1) * 0.004
        for (const [y0, y1] of [
          [0.18, 0.9],
          [1.08, DOOR_H - 0.2],
        ])
          add(l.u0 + 0.1, l.u1 - 0.1, l.n1, face, y0, y1, trim, 0.003)
        // Lever on the free edge (away from the hinge), on the face showing.
        const free = l.hinge + l.dir * (LEAF_W - 0.07)
        add(free - 0.012, free + 0.012, l.n1, l.n1 + Math.sign(l.n1) * 0.012, 0.98, 1.06, handleMat, 0.004)
        add(free - l.dir * 0.11, free + l.dir * 0.01, l.n1 + Math.sign(l.n1) * 0.012, l.n1 + Math.sign(l.n1) * 0.03, 1.01, 1.03, handleMat, 0.005)
      }
    }
    root.add(g)
  }

  buildLiving(root)
  scene.add(root)
  root.updateMatrixWorld(true)
  for (const g of root.children) if (g.name.startsWith('wall_')) addFadeable(g)
}

function buildLiving(root: THREE.Group) {
  const g = new THREE.Group()
  g.name = 'living'
  const walnut = mat('#ffffff', { map: woodTexture([1, 1], '#5a3d2a'), roughness: 0.45 })
  const black = mat('#1d2024', { roughness: 0.5, metalness: 0.3 })

  // Rug under the coffee table, reaching under the front of the sofa.
  const rugTex = canvasTex(
    512,
    512,
    (c) => {
      c.fillStyle = '#c9c2b4'
      c.fillRect(0, 0, 512, 512)
      for (let i = 0; i < 9000; i++) {
        c.fillStyle = Math.random() < 0.5 ? 'rgba(255,255,255,0.07)' : 'rgba(60,50,40,0.07)'
        c.fillRect(Math.random() * 512, Math.random() * 512, 2, 2)
      }
      c.strokeStyle = '#3e4a5a'
      c.lineWidth = 10
      c.strokeRect(26, 26, 460, 460)
      c.lineWidth = 3
      c.strokeRect(46, 46, 420, 420)
    },
    [1, 1],
  )
  const rug = new THREE.Mesh(new THREE.PlaneGeometry(2.1, 2.2), mat('#ffffff', { map: rugTex, roughness: 1 }))
  rug.rotation.x = -Math.PI / 2
  rug.position.set((COFFEE.x0 + COFFEE.x1) / 2, 0.004, (TV.z1 + SOFA.z0) / 2)
  rug.receiveShadow = true
  g.add(rug)

  // Low walnut media console against the laundry wall, a TV on its stand on top.
  g.add(box(TV.x0, TV.x1, 0.08, TV.h, TV.z0, TV.z1, walnut, 0.008))
  for (const x of [TV.x0 + 0.06, TV.x1 - 0.06]) for (const z of [TV.z0 + 0.06, TV.z1 - 0.06]) g.add(box(x - 0.015, x + 0.015, 0, 0.08, z - 0.015, z + 0.015, black, 0.004))
  // Slatted doors on the front.
  for (let i = 0; i < 22; i++) {
    const x = TV.x0 + 0.06 + ((TV.x1 - TV.x0 - 0.12) * i) / 21
    g.add(box(x - 0.008, x + 0.008, 0.12, TV.h - 0.05, TV.z1, TV.z1 + 0.008, walnut, 0.002))
  }
  const tvW = 1.05
  const tvH = 0.6
  const cx = (TV.x0 + TV.x1) / 2
  const cz = (TV.z0 + TV.z1) / 2 - 0.05
  g.add(box(cx - 0.12, cx + 0.12, TV.h, TV.h + 0.01, cz - 0.08, cz + 0.08, black, 0.003))
  g.add(box(cx - 0.02, cx + 0.02, TV.h, TV.h + 0.12, cz - 0.01, cz + 0.02, black))
  g.add(box(cx - tvW / 2, cx + tvW / 2, TV.h + 0.08, TV.h + 0.08 + tvH, cz - 0.015, cz + 0.015, black, 0.006))
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(tvW - 0.02, tvH - 0.02), new THREE.MeshStandardMaterial({ color: '#0b0f14', roughness: 0.15, metalness: 0.4 }))
  screen.position.set(cx, TV.h + 0.08 + tvH / 2, cz + 0.016)
  g.add(screen)
  // A speaker and a small plant on the console.
  g.add(box(TV.x0 + 0.08, TV.x0 + 0.2, TV.h, TV.h + 0.2, cz - 0.06, cz + 0.06, mat('#3a3f46', { roughness: 0.9 }), 0.02))
  {
    const pot = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.11, 24), mat('#e9e4da', { roughness: 0.4 }))
    pot.position.set(TV.x1 - 0.13, TV.h + 0.055, cz)
    g.add(shadowed(pot))
    const leaf = mat('#4c8a5c', { roughness: 0.6, side: THREE.DoubleSide })
    for (let i = 0; i < 7; i++) {
      const l = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), leaf)
      l.scale.set(0.3, 1.6, 0.9)
      const a = (i / 7) * Math.PI * 2
      l.position.set(TV.x1 - 0.13 + Math.cos(a) * 0.04, TV.h + 0.17, cz + Math.sin(a) * 0.04)
      l.rotation.set(Math.sin(a) * 0.5, -a, Math.cos(a) * 0.5)
      g.add(shadowed(l))
    }
  }

  // Sofa facing the TV: a plinth on short legs, seat and back cushions, rolled arms.
  const S = SOFA
  const fabric = mat('#4f6475', { roughness: 0.95, bumpMap: knitBump([8, 8]), bumpScale: 0.5 })
  const fabricDark = mat('#46596a', { roughness: 0.95, bumpMap: knitBump([8, 8]), bumpScale: 0.5 })
  for (const x of [S.x0 + 0.08, S.x1 - 0.08]) for (const z of [S.z0 + 0.08, S.z1 - 0.08]) g.add(box(x - 0.02, x + 0.02, 0, 0.09, z - 0.02, z + 0.02, walnut, 0.004))
  g.add(box(S.x0, S.x1, 0.09, 0.28, S.z0, S.z1, fabricDark, 0.03))
  g.add(box(S.x0, S.x1, 0.09, S.back, S.z1 - 0.2, S.z1, fabricDark, 0.05)) // back frame
  for (const x of [S.x0, S.x1 - 0.17]) g.add(box(x, x + 0.17, 0.09, S.arm, S.z0, S.z1, fabricDark, 0.06))
  const inner = S.x1 - S.x0 - 0.34
  for (let i = 0; i < 2; i++) {
    const x0 = S.x0 + 0.17 + (inner * i) / 2 + 0.005
    const x1 = S.x0 + 0.17 + (inner * (i + 1)) / 2 - 0.005
    g.add(box(x0, x1, 0.28, S.seat, S.z0 + 0.02, S.z1 - 0.18, fabric, 0.05)) // seat cushion
    const back = box(x0, x1, S.seat - 0.02, S.back - 0.02, S.z1 - 0.32, S.z1 - 0.17, fabric, 0.06)
    back.rotation.x = 0.12
    g.add(back)
  }
  // Two cushions and a folded throw.
  for (const [x, col, yaw] of [
    [S.x0 + 0.33, '#d8a25e', 0.25],
    [S.x1 - 0.33, '#e7e2d8', -0.2],
  ] as [number, string, number][]) {
    const c = box(x - 0.2, x + 0.2, S.seat, S.seat + 0.38, S.z1 - 0.32, S.z1 - 0.2, mat(col, { roughness: 1 }), 0.07)
    c.rotation.set(0.25, yaw, 0)
    g.add(c)
  }
  g.add(box(S.x1 - 0.17, S.x1, S.arm, S.arm + 0.04, S.z0 + 0.1, S.z0 + 0.5, mat('#9c5b45', { roughness: 1, bumpMap: knitBump([4, 4]), bumpScale: 0.8 }), 0.015))

  // Coffee table: walnut top on black steel legs, a tray with a candle, a stack of books.
  const C = COFFEE
  g.add(box(C.x0, C.x1, C.h - 0.035, C.h, C.z0, C.z1, walnut, 0.01))
  g.add(box(C.x0 + 0.05, C.x1 - 0.05, 0.12, 0.14, C.z0 + 0.05, C.z1 - 0.05, walnut, 0.005)) // shelf
  for (const x of [C.x0 + 0.04, C.x1 - 0.04]) for (const z of [C.z0 + 0.04, C.z1 - 0.04]) g.add(box(x - 0.012, x + 0.012, 0, C.h - 0.035, z - 0.012, z + 0.012, black, 0.003))
  g.add(box(C.x0 + 0.12, C.x0 + 0.42, C.h, C.h + 0.015, C.z0 + 0.1, C.z0 + 0.3, mat('#2b2b2b', { roughness: 0.4 }), 0.005))
  {
    const candle = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.08, 20), mat('#f3efe6', { roughness: 0.7 }))
    candle.position.set(C.x0 + 0.27, C.h + 0.055, C.z0 + 0.2)
    g.add(shadowed(candle))
  }
  ;['#3f5c7a', '#c9b27c', '#7a3e2e'].forEach((col, i) => {
    const b = box(-0.12, 0.12, 0, 0.025, -0.08, 0.08, mat(col, { roughness: 0.7 }), 0.003)
    b.position.set(C.x1 - 0.22, C.h + 0.0125 + i * 0.025, C.z1 - 0.2)
    b.rotation.y = i * 0.25 - 0.2
    g.add(b)
  })

  // Arc floor lamp in the corner by the sofa.
  {
    const base = new THREE.Mesh(new THREE.CylinderGeometry(LAMP.r, LAMP.r, 0.03, 32), mat('#d9dcdf', { roughness: 0.3, metalness: 0.2 }))
    base.position.set(LAMP.x, 0.015, LAMP.z)
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1.45, 12), black)
    pole.position.set(LAMP.x, 0.75, LAMP.z)
    const shade = new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.19, 0.24, 36, 1, true),
      new THREE.MeshStandardMaterial({ color: '#f1e7d4', emissive: '#ffd59a', emissiveIntensity: 0.5, side: THREE.DoubleSide, roughness: 0.9 }),
    )
    shade.position.set(LAMP.x, 1.5, LAMP.z)
    const bulb = new THREE.PointLight('#ffcf94', 1.4, 3.2, 1.6)
    bulb.position.set(LAMP.x, 1.42, LAMP.z)
    g.add(shadowed(base), shadowed(pole), shade, bulb)
  }

  // A framed print on the east wall, beside the lounge.
  {
    const pz = (TV.z1 + SOFA.z0) / 2
    const wx = ROOMS.living.x1 - T / 2 - 0.012
    const frame = box(wx - 0.03, wx, 1.25, 1.85, pz - 0.45, pz + 0.45, mat('#e7e2d8', { roughness: 0.5 }), 0.004)
    const art = canvasTex(
      360,
      240,
      (c) => {
        c.fillStyle = '#f2ede4'
        c.fillRect(0, 0, 360, 240)
        const shapes: [string, number, number, number][] = [
          ['#d8a25e', 120, 120, 70],
          ['#4f6475', 220, 140, 55],
          ['#9c5b45', 175, 85, 34],
        ]
        for (const [col, x, y, r] of shapes) {
          c.fillStyle = col
          c.beginPath()
          c.arc(x, y, r, 0, Math.PI * 2)
          c.fill()
        }
      },
      [1, 1],
    )
    const pic = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.5), mat('#ffffff', { map: art, roughness: 0.7 }))
    pic.rotation.y = -Math.PI / 2
    pic.position.set(wx - 0.032, 1.55, pz)
    // Hung on the wall, so it fades with it.
    ;(root.getObjectByName('wall_east') ?? g).add(frame, pic)
  }
  root.add(g)
}
