/**
 * A dining area in the open part of the room: a solid oak table for six with an apron and
 * tapered legs, six upholstered chairs (one pulled out and turned, as if someone just got up),
 * a patterned rug, a pendant lamp that lights the table, and a fruit bowl and a vase of
 * flowers in the middle.
 *
 * Everything is in world coordinates. `DINING_FOOTPRINTS` lists what stands on the floor (as
 * boxes with a height), for the robot's path planning and collision checks and for physics.
 */
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { woodTexture } from './textures'

/** Table centre, its size (x across, z along the long side), and height. */
export const DINING = { x: 2.0, z: 2.2, w: 0.95, l: 1.8, h: 0.76 }

/**
 * Chairs: where each one stands and which way it faces (yaw: 0 faces +x). Dinner's over: the
 * four along the sides are pushed back to the corners and turned out (people have just got
 * up), which leaves room at each place for the robot; the two at the ends are tucked in.
 */
const CHAIR_W = 0.46
const CHAIR_D = 0.5
const SEAT_H = 0.46
const chairs: { x: number; z: number; yaw: number }[] = (() => {
  const { x, z, w, l } = DINING
  const side = w / 2 + CHAIR_D / 2 - 0.12
  const end = l / 2 + CHAIR_D / 2 - 0.1
  void side
  const out = w / 2 + 0.35
  return [
    { x: x - out, z: z - 1.3, yaw: 0.5 },
    { x: x - out - 0.03, z: z + 1.3, yaw: -0.45 },
    { x: x + out, z: z - 1.3, yaw: Math.PI - 0.5 },
    { x: x + out, z: z + 1.3, yaw: Math.PI + 0.55 },
    { x, z: z - end, yaw: -Math.PI / 2 }, // facing +z, toward the table
    { x, z: z + end, yaw: Math.PI / 2 },
  ]
})()

export interface Footprint {
  x0: number
  x1: number
  z0: number
  z1: number
  h: number
}

/** What stands on the floor, as world boxes (a turned chair gets the box round it). */
export const DINING_FOOTPRINTS: Footprint[] = [
  { x0: DINING.x - DINING.w / 2, x1: DINING.x + DINING.w / 2, z0: DINING.z - DINING.l / 2, z1: DINING.z + DINING.l / 2, h: DINING.h },
  ...chairs.map((c) => {
    const ext = (Math.abs(Math.cos(c.yaw)) * CHAIR_D + Math.abs(Math.sin(c.yaw)) * CHAIR_W) / 2
    const exz = (Math.abs(Math.sin(c.yaw)) * CHAIR_D + Math.abs(Math.cos(c.yaw)) * CHAIR_W) / 2
    return { x0: c.x - ext, x1: c.x + ext, z0: c.z - exz, z1: c.z + exz, h: 0.92 }
  }),
]

/**
 * The same for physics: each chair's box stops at the table's edge (a seat tucks under it, but
 * its box is as tall as the chair back, and would swallow whatever is laid at the table's edge).
 */
export const DINING_STATICS: Footprint[] = DINING_FOOTPRINTS.map((f, i) => {
  if (i === 0) return f
  const t = DINING_FOOTPRINTS[0]
  const c = { ...f }
  if (c.x1 > t.x0 && c.x0 < t.x1 && c.z1 > t.z0 && c.z0 < t.z1) {
    // Cut back along whichever side overlaps the least.
    const cuts: [keyof Footprint, number, number][] = [
      ['x1', t.x0, c.x1 - t.x0],
      ['x0', t.x1, t.x1 - c.x0],
      ['z1', t.z0, c.z1 - t.z0],
      ['z0', t.z1, t.z1 - c.z0],
    ]
    const [k, v] = cuts.filter((q) => q[2] > 0).sort((a, b) => a[2] - b[2])[0]
    c[k] = v
  }
  return c
})

/**
 * A place laid for each chair: where its plate goes (on the table, 17 cm in from the edge in
 * front of the chair) and which way the diner faces (fx, fz).
 */
export const PLACES: { x: number; z: number; fx: number; fz: number }[] = [
  // Dinner for four, two each side (the end seats weren't used).
  { x: DINING.x - (DINING.w / 2 - 0.17), z: DINING.z - 0.45, fx: 1, fz: 0 },
  { x: DINING.x - (DINING.w / 2 - 0.17), z: DINING.z + 0.45, fx: 1, fz: 0 },
  { x: DINING.x + (DINING.w / 2 - 0.17), z: DINING.z - 0.45, fx: -1, fz: 0 },
  { x: DINING.x + (DINING.w / 2 - 0.17), z: DINING.z + 0.45, fx: -1, fz: 0 },
]

/** Where the robot stands to work at each place: square to the table, its front at the edge. */
export const BAYS = PLACES.map((p) => ({
  x: DINING.x - p.fx * (DINING.w / 2 + 0.36),
  z: p.z,
  yaw: p.fx > 0 ? 0 : Math.PI,
}))

const mat = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0, ...extra })
function shadowed<T extends THREE.Object3D>(o: T): T {
  o.traverse((c) => {
    c.castShadow = true
    c.receiveShadow = true
  })
  return o
}
const rbox = (x: number, y: number, z: number, r: number, m: THREE.Material) => new THREE.Mesh(new RoundedBoxGeometry(x, y, z, 3, Math.min(r, x / 2, y / 2, z / 2)), m)

/** A leg tapering toward the floor (square section). */
function taperedLeg(top: number, bottom: number, h: number, m: THREE.Material) {
  const geo = new THREE.CylinderGeometry(top / Math.SQRT2, bottom / Math.SQRT2, h, 4, 1)
  geo.rotateY(Math.PI / 4)
  return new THREE.Mesh(geo, m)
}

function rugTexture() {
  const c = document.createElement('canvas')
  c.width = 512
  c.height = 384
  const g = c.getContext('2d')!
  g.fillStyle = '#7a3e2e'
  g.fillRect(0, 0, 512, 384)
  // Border bands.
  g.strokeStyle = '#d9c7a3'
  g.lineWidth = 14
  g.strokeRect(22, 22, 468, 340)
  g.strokeStyle = '#2f4858'
  g.lineWidth = 8
  g.strokeRect(46, 46, 420, 292)
  // A field of diamonds.
  for (let y = 80; y < 320; y += 48)
    for (let x = 84; x < 440; x += 56) {
      g.fillStyle = (x + y) % 3 ? '#c9a46a' : '#2f4858'
      g.beginPath()
      g.moveTo(x, y - 16)
      g.lineTo(x + 18, y)
      g.lineTo(x, y + 16)
      g.lineTo(x - 18, y)
      g.closePath()
      g.fill()
    }
  // Woven texture.
  for (let i = 0; i < 9000; i++) {
    g.fillStyle = Math.random() < 0.5 ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.06)'
    g.fillRect(Math.random() * 512, Math.random() * 384, 2, 1)
  }
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

function chair(oak: THREE.Material, fabric: THREE.Material) {
  const g = new THREE.Group()
  const hw = CHAIR_W / 2
  const hd = CHAIR_D / 2
  // Legs: back legs carry on up as the back posts, angled back a little.
  for (const sx of [-1, 1])
    for (const sz of [-1, 1]) {
      const leg = taperedLeg(0.036, 0.026, SEAT_H - 0.03, oak)
      leg.position.set(sx * (hd - 0.03), (SEAT_H - 0.03) / 2, sz * (hw - 0.03))
      g.add(leg)
    }
  for (const sz of [-1, 1]) {
    const post = taperedLeg(0.03, 0.036, 0.46, oak)
    post.position.set(-(hd - 0.03) - 0.02, SEAT_H + 0.22, sz * (hw - 0.03))
    post.rotation.z = 0.1
    g.add(post)
  }
  // Seat frame and a padded cushion.
  g.add(rbox(CHAIR_D - 0.02, 0.04, CHAIR_W - 0.02, 0.008, oak).translateY(SEAT_H - 0.02))
  const cushion = rbox(CHAIR_D - 0.06, 0.05, CHAIR_W - 0.06, 0.02, fabric)
  cushion.position.y = SEAT_H + 0.025
  g.add(cushion)
  // Stretchers between the legs.
  for (const sz of [-1, 1]) g.add(rbox(CHAIR_D - 0.06, 0.02, 0.02, 0.005, oak).translateY(0.14).translateZ(sz * (hw - 0.03)))
  // Back: a curved top rail and three slats.
  const rail = rbox(0.03, 0.09, CHAIR_W - 0.02, 0.012, oak)
  rail.position.set(-(hd - 0.03) - 0.06, SEAT_H + 0.41, 0)
  rail.rotation.z = 0.1
  g.add(rail)
  for (const dz of [-0.11, 0, 0.11]) {
    const slat = rbox(0.016, 0.28, 0.05, 0.006, oak)
    slat.position.set(-(hd - 0.03) - 0.045, SEAT_H + 0.22, dz)
    slat.rotation.z = 0.1
    g.add(slat)
  }
  return shadowed(g)
}

export function buildDining(scene: THREE.Scene) {
  const root = new THREE.Group()
  root.name = 'dining'
  const oak = mat('#ffffff', { map: woodTexture([1, 1], '#9a6b3f'), roughness: 0.45 })
  const oakLong = mat('#ffffff', { map: woodTexture([1, 2], '#9a6b3f'), roughness: 0.4 })
  const fabric = mat('#5b6b5a', { roughness: 0.95 })
  const { x, z, w, l, h } = DINING

  // Rug under the table, a little larger than the chairs' reach.
  // (The plane is turned a quarter below, so its first side runs along z.)
  const rug = new THREE.Mesh(new THREE.PlaneGeometry(l + 0.8, w + 0.95), mat('#ffffff', { map: rugTexture(), roughness: 1 }))
  rug.rotation.x = -Math.PI / 2
  rug.rotation.z = Math.PI / 2
  rug.position.set(x, 0.004, z)
  rug.receiveShadow = true
  root.add(rug)

  // Table: top with eased edges, an apron round under it, tapered legs set in from the corners.
  const table = new THREE.Group()
  const top = rbox(w, 0.04, l, 0.012, oakLong)
  top.position.y = h - 0.02
  table.add(top)
  const apronH = 0.09
  for (const sx of [-1, 1]) table.add(rbox(0.025, apronH, l - 0.16, 0.004, oak).translateX(sx * (w / 2 - 0.07)).translateY(h - 0.04 - apronH / 2))
  for (const sz of [-1, 1]) table.add(rbox(w - 0.16, apronH, 0.025, 0.004, oak).translateZ(sz * (l / 2 - 0.07)).translateY(h - 0.04 - apronH / 2))
  for (const sx of [-1, 1])
    for (const sz of [-1, 1]) {
      const leg = taperedLeg(0.075, 0.05, h - 0.04, oak)
      leg.position.set(sx * (w / 2 - 0.07), (h - 0.04) / 2, sz * (l / 2 - 0.07))
      table.add(leg)
    }
  table.position.set(x, 0, z)
  root.add(shadowed(table))

  // Chairs.
  for (const c of chairs) {
    const ch = chair(oak, fabric)
    ch.position.set(c.x, 0, c.z)
    ch.rotation.y = c.yaw
    root.add(ch)
  }

  // Centrepiece: a ceramic bowl of fruit and a glass vase of flowers on a linen runner.
  const runner = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.003, l * 0.8), mat('#e9e2d0', { roughness: 0.95 }))
  runner.position.set(x, h + 0.0015, z)
  runner.receiveShadow = true
  root.add(runner)
  const bowlProfile = [
    [0, 0], [0.05, 0], [0.09, 0.02], [0.12, 0.06], [0.125, 0.065], [0.118, 0.065], [0.085, 0.025], [0, 0.012],
  ].map(([a, b]) => new THREE.Vector2(a, b))
  const bowl = new THREE.Mesh(new THREE.LatheGeometry(bowlProfile, 40), new THREE.MeshPhysicalMaterial({ color: '#f1efe8', roughness: 0.25, clearcoat: 0.6, side: THREE.DoubleSide }))
  bowl.position.set(x, h, z - 0.3)
  root.add(shadowed(bowl))
  const fruit: [string, number, number, number][] = [
    ['#c0392b', 0.04, -0.03, 0.02], ['#e67e22', 0.042, 0.04, -0.01], ['#27ae60', 0.038, 0.0, 0.05], ['#f1c40f', 0.035, -0.045, -0.045], ['#c0392b', 0.038, 0.05, 0.05],
  ]
  for (const [c, r, dx, dz] of fruit) {
    const f = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), mat(c, { roughness: 0.35 }))
    f.position.set(x + dx, h + 0.03 + r * 0.8, z - 0.3 + dz)
    root.add(shadowed(f))
  }
  const vase = new THREE.Mesh(
    new THREE.LatheGeometry([[0, 0], [0.045, 0], [0.05, 0.05], [0.035, 0.16], [0.03, 0.2], [0.034, 0.21], [0, 0.21]].map(([a, b]) => new THREE.Vector2(a, b)), 30),
    new THREE.MeshPhysicalMaterial({ color: '#cfe3ea', roughness: 0.05, transparent: true, opacity: 0.55, clearcoat: 1 }),
  )
  vase.position.set(x, h, z + 0.32)
  root.add(vase)
  const stemM = mat('#3f6b35', { roughness: 0.7 })
  const petals = ['#f5f0e1', '#e8a2b0', '#f7d36b']
  for (let i = 0; i < 7; i++) {
    const a = (i / 7) * Math.PI * 2
    const lean = 0.18 + (i % 3) * 0.06
    const len = 0.3 + (i % 2) * 0.06
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, len, 5), stemM)
    stem.position.set(x + Math.cos(a) * lean * len * 0.5, h + 0.12 + len / 2, z + 0.32 + Math.sin(a) * lean * len * 0.5)
    stem.rotation.set(Math.sin(a) * lean, 0, -Math.cos(a) * lean)
    root.add(stem)
    const bloom = new THREE.Mesh(new THREE.IcosahedronGeometry(0.03, 1), mat(petals[i % 3], { roughness: 0.8, flatShading: true }))
    bloom.position.set(x + Math.cos(a) * lean * len, h + 0.12 + len, z + 0.32 + Math.sin(a) * lean * len)
    root.add(shadowed(bloom))
  }

  // Pendant lamp over the table: a cord from the ceiling, a dark metal shade, a warm bulb,
  // and a light that pools on the table.
  const lampY = h + 0.75
  const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 1.2, 6), mat('#111111'))
  cord.position.set(x, lampY + 0.6, z)
  const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.24, 0.2, 40, 1, true), mat('#1f2933', { metalness: 0.6, roughness: 0.4, side: THREE.DoubleSide }))
  shade.position.set(x, lampY, z)
  const inside = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.236, 0.198, 40, 1, true), mat('#f3e7c9', { side: THREE.BackSide, emissive: '#ffcf8a', emissiveIntensity: 0.4 }))
  inside.position.copy(shade.position)
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.04, 16, 12), new THREE.MeshStandardMaterial({ color: '#fff7e0', emissive: '#ffdca0', emissiveIntensity: 2.5 }))
  bulb.position.set(x, lampY - 0.06, z)
  const light = new THREE.SpotLight('#ffd9a0', 6, 3.5, 0.95, 0.6, 1.5)
  light.position.set(x, lampY - 0.05, z)
  light.target.position.set(x, 0, z)
  root.add(cord, shadowed(shade), inside, bulb, light, light.target)

  // After dinner: a crumpled napkin at each place, crumbs round the plates, a drip of sauce.
  {
    let seed = 11
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    const linen = mat('#efe9dc', { roughness: 1 })
    const crumbMat = mat('#c49a62', { roughness: 0.9 })
    for (const p of PLACES) {
      const rx = -p.fz
      const rz = p.fx
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
      const napkin = new THREE.Mesh(geo, linen)
      napkin.scale.set(0.075, 0.035, 0.06)
      napkin.rotation.y = rnd() * Math.PI
      napkin.position.set(p.x - rx * 0.25 - p.fx * 0.04, h + 0.025, p.z - rz * 0.25 - p.fz * 0.04)
      root.add(shadowed(napkin))
      for (let i = 0; i < 9; i++) {
        const a = rnd() * Math.PI * 2
        const d = 0.15 + rnd() * 0.08
        const c = new THREE.Mesh(new THREE.BoxGeometry(0.006 + rnd() * 0.006, 0.004, 0.006 + rnd() * 0.005), crumbMat)
        c.position.set(p.x + Math.cos(a) * d, h + 0.002, p.z + Math.sin(a) * d)
        c.rotation.y = rnd() * 3
        root.add(c)
      }
    }
    const drip = new THREE.Mesh(new THREE.CircleGeometry(0.022, 20), new THREE.MeshStandardMaterial({ color: '#7a2a14', roughness: 0.3, transparent: true, opacity: 0.8 }))
    drip.rotation.x = -Math.PI / 2
    drip.scale.set(1.4, 0.8, 1)
    drip.position.set(PLACES[2].x + 0.02, h + 0.001, PLACES[2].z + 0.17)
    root.add(drip)
  }

  scene.add(root)
  return root
}
