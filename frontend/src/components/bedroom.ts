/**
 * A bedroom along the +z side of the room: a painted back wall with a skirting board, a double
 * bed with a padded headboard, duvet, pillows and a throw, nightstands with lamps, a chest of
 * drawers with a mirror over it, a rug, and a framed print. The sorting table stands against
 * the same wall as a desk (it's a station, built in stationProps).
 *
 * Everything is in world coordinates. `BEDROOM_FOOTPRINTS` lists what stands on the floor, for
 * the robot's path planning and collision checks and for physics.
 */
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import type { Footprint } from './dining'
import { knitBump, paintTexture, woodTexture } from './textures'

/** The back wall's front face (z), and its ends and height. */
export const BEDROOM = { wallZ: 4.2, x0: -2.32, x1: 3.05, wallH: 2.45 }
const BED = { x0: 0.95, x1: 2.45, z0: 2.2, z1: 4.17, frameH: 0.32, mattressTop: 0.56 }
const STAND = { w: 0.45, d: 0.4, h: 0.56 }
const STANDS = [0.4, 2.55] // left edges (x) of the two nightstands
const DRESSER = { x0: -2.25, x1: -1.35, d: 0.46, h: 0.84 }

export const BEDROOM_FOOTPRINTS: Footprint[] = [
  { x0: BEDROOM.x0, x1: BEDROOM.x1, z0: BEDROOM.wallZ, z1: BEDROOM.wallZ + 0.06, h: BEDROOM.wallH },
  { x0: BED.x0, x1: BED.x1, z0: BED.z0, z1: BED.z1, h: BED.mattressTop + 0.06 },
  ...STANDS.map((x) => ({ x0: x, x1: x + STAND.w, z0: BEDROOM.wallZ - STAND.d, z1: BEDROOM.wallZ, h: STAND.h })),
  { x0: DRESSER.x0, x1: DRESSER.x1, z0: BEDROOM.wallZ - DRESSER.d, z1: BEDROOM.wallZ, h: DRESSER.h },
]

const mat = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0, ...extra })
function shadowed<T extends THREE.Object3D>(o: T): T {
  o.traverse((c) => {
    c.castShadow = true
    c.receiveShadow = true
  })
  return o
}
/** A rounded box spanning x0..x1, y0..y1, z0..z1. */
function box(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, m: THREE.Material, r = 0.006) {
  const w = x1 - x0
  const h = y1 - y0
  const d = z1 - z0
  const rr = Math.min(r, w / 2 - 1e-4, h / 2 - 1e-4, d / 2 - 1e-4)
  const o = new THREE.Mesh(rr > 0.001 ? new RoundedBoxGeometry(w, h, d, 3, rr) : new THREE.BoxGeometry(w, h, d), m)
  o.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
  return shadowed(o)
}

function rugTexture() {
  const c = document.createElement('canvas')
  c.width = 512
  c.height = 512
  const g = c.getContext('2d')!
  g.fillStyle = '#d8d0c2'
  g.fillRect(0, 0, 512, 512)
  // Soft diamond lattice and a plain border, like a flat-woven wool rug.
  g.strokeStyle = 'rgba(120,104,86,0.35)'
  g.lineWidth = 3
  for (let i = -512; i < 1024; i += 64) {
    g.beginPath()
    g.moveTo(i, 0)
    g.lineTo(i + 512, 512)
    g.moveTo(i, 512)
    g.lineTo(i + 512, 0)
    g.stroke()
  }
  g.strokeStyle = '#8c7a66'
  g.lineWidth = 18
  g.strokeRect(20, 20, 472, 472)
  // A little speckle so it doesn't look printed.
  for (let i = 0; i < 4000; i++) {
    g.fillStyle = `rgba(${Math.random() < 0.5 ? '255,255,255' : '60,50,40'},0.06)`
    g.fillRect(Math.random() * 512, Math.random() * 512, 2, 2)
  }
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 8
  return t
}

function printTexture() {
  const c = document.createElement('canvas')
  c.width = 384
  c.height = 256
  const g = c.getContext('2d')!
  const sky = g.createLinearGradient(0, 0, 0, 256)
  sky.addColorStop(0, '#f3d9b1')
  sky.addColorStop(1, '#f7efe2')
  g.fillStyle = sky
  g.fillRect(0, 0, 384, 256)
  // Layered hills at dusk.
  const hills: [string, number, number][] = [
    ['#c9a27e', 150, 30],
    ['#9a7b67', 175, 24],
    ['#5f5a63', 205, 18],
  ]
  for (const [col, base, amp] of hills) {
    g.fillStyle = col
    g.beginPath()
    g.moveTo(0, 256)
    for (let x = 0; x <= 384; x += 8) g.lineTo(x, base - amp * Math.sin(x / 55 + base) - (amp / 2) * Math.sin(x / 23))
    g.lineTo(384, 256)
    g.fill()
  }
  g.fillStyle = '#fbe7c6'
  g.beginPath()
  g.arc(270, 95, 22, 0, Math.PI * 2)
  g.fill()
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

/** A bedside lamp: turned ceramic base, linen drum shade that glows, and a warm light. */
function lamp(x: number, y: number, z: number) {
  const g = new THREE.Group()
  g.position.set(x, y, z)
  const base = new THREE.Mesh(
    new THREE.LatheGeometry(
      [
        [0.0, 0],
        [0.055, 0],
        [0.06, 0.02],
        [0.07, 0.09],
        [0.05, 0.17],
        [0.018, 0.2],
        [0.012, 0.25],
        [0, 0.25],
      ].map(([r, h]) => new THREE.Vector2(r, h)),
      32,
    ),
    mat('#5e7f7a', { roughness: 0.25 }),
  )
  const shade = new THREE.Mesh(
    new THREE.CylinderGeometry(0.085, 0.12, 0.17, 40, 1, true),
    new THREE.MeshStandardMaterial({ color: '#f5ead6', emissive: '#ffcf8a', emissiveIntensity: 0.55, roughness: 0.9, side: THREE.DoubleSide }),
  )
  shade.position.y = 0.3
  const bulb = new THREE.PointLight('#ffc98a', 0.9, 2.2, 1.6)
  bulb.position.y = 0.28
  g.add(shadowed(base), shade, bulb)
  return g
}

/** A pillow: a soft rounded box with its middle puffed up. */
function pillow(w: number, h: number, d: number, m: THREE.Material) {
  const geo = new RoundedBoxGeometry(w, h, d, 5, Math.min(h / 2 - 0.002, 0.05))
  const p = geo.attributes.position
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i) / (w / 2)
    const z = p.getZ(i) / (d / 2)
    const puff = (1 - x * x) * (1 - z * z)
    p.setY(i, p.getY(i) * (0.55 + 0.45 * puff))
  }
  geo.computeVertexNormals()
  return shadowed(new THREE.Mesh(geo, m))
}

export function buildBedroom(scene: THREE.Scene) {
  const root = new THREE.Group()
  root.name = 'bedroom'
  const W = BEDROOM
  const walnut = mat('#ffffff', { map: woodTexture([1, 1], '#6b4a33'), roughness: 0.5 })
  const walnutLong = mat('#ffffff', { map: woodTexture([2, 1], '#6b4a33'), roughness: 0.45 })
  const brass = mat('#c7a35a', { metalness: 0.85, roughness: 0.3 })

  // Back wall in a soft sage, with a skirting board.
  const paint = paintTexture([3, 1])
  const wall = box(W.x0, W.x1, 0, W.wallH, W.wallZ, W.wallZ + 0.06, mat('#ffffff', { color: '#b9c4b3', map: paint, roughness: 0.9 }), 0)
  wall.castShadow = false
  root.add(wall)
  root.add(box(W.x0, W.x1, 0, 0.1, W.wallZ - 0.015, W.wallZ, mat('#f1efe9', { roughness: 0.5 }), 0.003))

  // Rug under the bed, reaching out beyond its foot.
  const rug = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 2.3), mat('#ffffff', { map: rugTexture(), roughness: 1 }))
  rug.rotation.x = -Math.PI / 2
  rug.position.set((BED.x0 + BED.x1) / 2, 0.004, 3.05)
  rug.receiveShadow = true
  root.add(rug)

  // Bed: a walnut frame on short legs, a padded headboard against the wall.
  const B = BED
  const legH = 0.12
  for (const x of [B.x0 + 0.05, B.x1 - 0.05]) for (const z of [B.z0 + 0.05, B.z1 - 0.12]) root.add(box(x - 0.03, x + 0.03, 0, legH, z - 0.03, z + 0.03, walnut, 0.004))
  root.add(box(B.x0, B.x1, legH, B.frameH, B.z0, B.z1 - 0.07, walnutLong, 0.012))
  const upholstery = mat('#8a8f9c', { roughness: 0.95, bumpMap: knitBump([8, 4]), bumpScale: 0.6 })
  root.add(box(B.x0 - 0.04, B.x1 + 0.04, 0, 1.12, B.z1 - 0.08, B.z1, upholstery, 0.035))
  // Vertical channel tufting on the headboard.
  for (let i = 1; i < 6; i++) {
    const x = B.x0 - 0.04 + ((B.x1 - B.x0 + 0.08) * i) / 6
    root.add(box(x - 0.004, x + 0.004, 0.36, 1.06, B.z1 - 0.085, B.z1 - 0.078, mat('#767b88', { roughness: 1 }), 0.002))
  }
  // Mattress, then a duvet that hangs over the sides and the foot, folded back near the pillows.
  const mattressY0 = B.frameH
  root.add(box(B.x0 + 0.03, B.x1 - 0.03, mattressY0, B.mattressTop - 0.02, B.z0 + 0.03, B.z1 - 0.1, mat('#f4f1ea', { roughness: 0.9 }), 0.04))
  const duvetMat = mat('#e9e4da', { roughness: 0.95, bumpMap: knitBump([10, 10]), bumpScale: 0.3 })
  const duvetZ1 = 3.6
  root.add(box(B.x0 - 0.01, B.x1 + 0.01, B.mattressTop - 0.07, B.mattressTop + 0.035, B.z0 - 0.02, duvetZ1, duvetMat, 0.035))
  root.add(box(B.x0 - 0.005, B.x1 + 0.005, B.mattressTop + 0.01, B.mattressTop + 0.05, duvetZ1 - 0.22, duvetZ1 + 0.02, mat('#f7f4ee', { roughness: 0.9 }), 0.02))
  // Fitted sheet showing past the fold.
  root.add(box(B.x0 + 0.02, B.x1 - 0.02, B.mattressTop - 0.02, B.mattressTop + 0.0, duvetZ1, B.z1 - 0.11, mat('#fbfaf6', { roughness: 0.85 }), 0.01))
  // A throw across the foot.
  root.add(box(B.x0 - 0.02, B.x1 + 0.02, B.mattressTop + 0.035, B.mattressTop + 0.055, B.z0 + 0.1, B.z0 + 0.5, mat('#b7745a', { roughness: 1, bumpMap: knitBump([6, 2]), bumpScale: 0.8 }), 0.012))
  root.add(box(B.x0 - 0.04, B.x0 - 0.02, B.mattressTop - 0.18, B.mattressTop + 0.05, B.z0 + 0.1, B.z0 + 0.5, mat('#b7745a', { roughness: 1 }), 0.008))
  root.add(box(B.x1 + 0.02, B.x1 + 0.04, B.mattressTop - 0.18, B.mattressTop + 0.05, B.z0 + 0.1, B.z0 + 0.5, mat('#b7745a', { roughness: 1 }), 0.008))
  // Pillows: two sleeping pillows propped on the headboard, two smaller cushions in front.
  const pillowWhite = mat('#fbfaf7', { roughness: 0.9 })
  const pillowSage = mat('#9fb3a0', { roughness: 0.95 })
  const midX = (B.x0 + B.x1) / 2
  for (const s of [-1, 1]) {
    const p = pillow(0.62, 0.15, 0.4, pillowWhite)
    p.position.set(midX + s * 0.36, B.mattressTop + 0.1, B.z1 - 0.32)
    p.rotation.x = -0.45
    root.add(p)
    const c = pillow(0.42, 0.12, 0.3, pillowSage)
    c.position.set(midX + s * 0.24, B.mattressTop + 0.08, B.z1 - 0.56)
    c.rotation.set(-0.35, s * 0.12, 0)
    root.add(c)
  }

  // Nightstands with a drawer and an open shelf, a lamp on each.
  for (const x0 of STANDS) {
    const x1 = x0 + STAND.w
    const z0 = W.wallZ - STAND.d
    const z1 = W.wallZ - 0.005
    root.add(box(x0, x1, STAND.h - 0.03, STAND.h, z0, z1, walnutLong, 0.006)) // top
    root.add(box(x0, x0 + 0.025, 0.06, STAND.h - 0.03, z0, z1, walnut, 0.003))
    root.add(box(x1 - 0.025, x1, 0.06, STAND.h - 0.03, z0, z1, walnut, 0.003))
    root.add(box(x0, x1, 0.06, 0.085, z0, z1, walnut, 0.003)) // bottom shelf
    root.add(box(x0 + 0.02, x1 - 0.02, 0.085, STAND.h - 0.03, z1 - 0.02, z1, walnut, 0.002)) // back
    root.add(box(x0 + 0.03, x1 - 0.03, STAND.h - 0.17, STAND.h - 0.035, z0 - 0.005, z0 + 0.02, walnutLong, 0.004)) // drawer front
    root.add(box(x0 + 0.025, x1 - 0.025, STAND.h - 0.185, STAND.h - 0.17, z0, z1 - 0.02, walnut, 0.002)) // shelf under the drawer
    const knob = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 10), brass)
    knob.position.set((x0 + x1) / 2, STAND.h - 0.1, z0 - 0.015)
    root.add(shadowed(knob))
    for (const x of [x0 + 0.03, x1 - 0.03]) for (const z of [z0 + 0.03, z1 - 0.03]) root.add(box(x - 0.012, x + 0.012, 0, 0.06, z - 0.012, z + 0.012, walnut, 0.003))
    root.add(lamp(x0 + STAND.w / 2 + (x0 < midX ? -0.05 : 0.05), STAND.h, W.wallZ - 0.2))
    // A couple of books on the lower shelf.
    const cols = x0 < midX ? ['#3f5c7a', '#b45f45'] : ['#5b7b5e', '#c9b27c']
    cols.forEach((c, i) => root.add(box(x0 + 0.06, x1 - 0.1 + i * 0.03, 0.085 + i * 0.03, 0.115 + i * 0.03, z0 + 0.06 + i * 0.02, z1 - 0.08, mat(c, { roughness: 0.7 }), 0.003)))
  }

  // Chest of drawers, a round mirror above it, a plant and a little dish on top.
  const D = DRESSER
  const dz0 = W.wallZ - D.d
  const dz1 = W.wallZ - 0.005
  root.add(box(D.x0, D.x1, D.h - 0.03, D.h, dz0, dz1, walnutLong, 0.006))
  root.add(box(D.x0 + 0.01, D.x1 - 0.01, 0.08, D.h - 0.03, dz0 + 0.02, dz1, walnut, 0.004))
  const rows = 3
  const dh = (D.h - 0.03 - 0.08 - 0.02) / rows
  for (let i = 0; i < rows; i++) {
    const y0 = 0.09 + i * dh
    root.add(box(D.x0 + 0.02, D.x1 - 0.02, y0 + 0.004, y0 + dh - 0.004, dz0 - 0.002, dz0 + 0.022, walnutLong, 0.004))
    for (const s of [-1, 1]) {
      const pull = box(0, 0.09, 0, 0.012, 0, 0.012, brass, 0.004)
      pull.position.set((D.x0 + D.x1) / 2 + s * 0.2 - 0.045, y0 + dh / 2, dz0 - 0.012)
      root.add(pull)
    }
  }
  for (const x of [D.x0 + 0.04, D.x1 - 0.04]) for (const z of [dz0 + 0.05, dz1 - 0.04]) root.add(box(x - 0.015, x + 0.015, 0, 0.08, z - 0.015, z + 0.015, walnut, 0.004))
  {
    const mirror = new THREE.Group()
    const frame = new THREE.Mesh(new THREE.TorusGeometry(0.33, 0.018, 16, 64), brass)
    const glass = new THREE.Mesh(new THREE.CircleGeometry(0.33, 64), new THREE.MeshStandardMaterial({ color: '#c9d3dc', metalness: 1, roughness: 0.06 }))
    mirror.add(shadowed(frame), glass)
    mirror.rotation.y = Math.PI
    mirror.position.set((D.x0 + D.x1) / 2, D.h + 0.55, W.wallZ - 0.02)
    root.add(mirror)
  }
  {
    // Plant: a terracotta pot with a few leaves.
    const px = D.x0 + 0.17
    const pz = W.wallZ - 0.2
    const pot = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.06, 0.13, 28), mat('#b86b4b', { roughness: 0.85 }))
    pot.position.set(px, D.h + 0.065, pz)
    root.add(shadowed(pot))
    const leaf = mat('#3f7d4f', { roughness: 0.6, side: THREE.DoubleSide })
    for (let i = 0; i < 9; i++) {
      const l = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), leaf)
      l.scale.set(0.35, 1.5, 1)
      const a = (i / 9) * Math.PI * 2
      l.position.set(px + Math.cos(a) * 0.05, D.h + 0.2 + (i % 3) * 0.025, pz + Math.sin(a) * 0.05)
      l.rotation.set(Math.sin(a) * 0.5, -a, Math.cos(a) * 0.5)
      root.add(shadowed(l))
    }
    const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.04, 0.02, 28), mat('#e8e3d8', { roughness: 0.3 }))
    dish.position.set(D.x1 - 0.18, D.h + 0.01, W.wallZ - 0.22)
    root.add(shadowed(dish))
  }

  // A framed print over the bed.
  {
    const frame = box(midX - 0.5, midX + 0.5, 1.4, 2.0, W.wallZ - 0.03, W.wallZ, mat('#2b2b2b', { roughness: 0.5 }), 0.004)
    const art = new THREE.Mesh(new THREE.PlaneGeometry(0.88, 0.5), mat('#ffffff', { map: printTexture(), roughness: 0.7 }))
    art.rotation.y = Math.PI
    art.position.set(midX, 1.7, W.wallZ - 0.032)
    root.add(frame, art)
  }

  scene.add(root)
}
