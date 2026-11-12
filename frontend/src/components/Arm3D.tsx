import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { Bin, Block, Detection, Fixture, Lengths, Q, Vec } from '../lib/types'
import { fk, type ClothGrid, type Vec3 } from '../lib/geom'
import { COLOR } from './ArmView'
import { shirtTextures, terryBump } from './fabric'

const FINGER = 0.05
const OPEN = 0.058
const CLOSED = 0.042
const BIN_DEPTH = 0.14
/** How far out the gripper can put a block down on the table (approach height included). */
export const tableReach = (lengths: Lengths) => Math.sqrt(Math.max(0, (lengths[0] + lengths[1]) ** 2 - 0.12 ** 2)) - 0.01

const CAM_POS = new THREE.Vector3(0.75, 0.62, 1.25)
const CAM_TARGET = new THREE.Vector3(0, 0.16, 0)

export type PickTarget = { kind: 'block'; id: string } | { kind: 'bin'; id: string } | { kind: 'table'; x: number; z: number }

interface Props {
  q: Q
  lengths: Lengths
  base: Vec
  grip: number
  /** Wrist roll (rad). */
  roll?: number
  blocks: Block[]
  bins: Bin[]
  target: Vec | null
  targetOk: boolean
  trail: Vec3[]
  showReachPlane: boolean
  interactive: boolean
  hint?: string
  onTarget?: (p: Vec) => void
  /** Pick & place mode: clicks select blocks, bins and table spots. */
  pickMode?: boolean
  selectedId?: string | null
  /** Queued table destinations, drawn as rings on the table. */
  markers?: { x: number; z: number }[]
  onPick?: (t: PickTarget) => void
  /** Dish holder, racks, laundry basket. */
  fixtures?: Fixture[]
  /** Cloth vertex positions plus its grid (towel or shirt), or null when there is none. */
  cloth?: { verts: Float32Array; grid: ClothGrid } | null
  /** Camera detections active right now (boxes + labels). */
  detections?: Detection[]
  /** A second arm (two-arm mode): its pose and where its base stands. */
  second?: { q: Q; grip: number; baseZ: number; roll?: number } | null
}

/** Everything the scene needs to keep between renders. */
interface Rig {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  waist: THREE.Group
  shoulder: THREE.Group
  elbow: THREE.Group
  wrist: THREE.Group
  links: THREE.Mesh[]
  wristLink: THREE.Mesh
  palm: THREE.Mesh
  fingers: [THREE.Mesh, THREE.Mesh]
  blocks: Map<string, THREE.Object3D>
  fixtures: Map<string, THREE.Group>
  cloth: THREE.Mesh
  detBoxes: THREE.LineSegments[]
  camCone: THREE.Mesh
  bins: Map<string, THREE.Group>
  target: THREE.Group
  targetMat: THREE.MeshBasicMaterial
  trail: THREE.Line
  reachPlane: THREE.Group
  reachRing: THREE.LineLoop
  table: THREE.Mesh
  selectRing: THREE.Mesh
  tableRing: THREE.LineLoop
  markers: THREE.Mesh[]
  markerMat: THREE.MeshBasicMaterial
  armA: ArmParts
  armB: ArmParts
}

const mat = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.25, ...extra })

function box(w: number, h: number, d: number, material: THREE.Material) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material)
  m.castShadow = true
  m.receiveShadow = true
  return m
}

/** A joint housing: a short cylinder whose axis is the joint's rotation axis (z). */
function housing(radius: number, width: number) {
  const g = new THREE.Group()
  const body = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, width, 32), mat('#1e293b', { metalness: 0.5 }))
  body.rotation.x = Math.PI / 2
  body.castShadow = true
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.55, radius * 0.55, width + 0.004, 24), mat('#22d3ee', { emissive: '#0e7490', emissiveIntensity: 0.6 }))
  cap.rotation.x = Math.PI / 2
  g.add(body, cap)
  return g
}

function buildRig(el: HTMLDivElement): Rig {
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  el.appendChild(renderer.domElement)
  renderer.domElement.style.display = 'block'
  renderer.domElement.style.width = '100%'
  renderer.domElement.style.height = '100%'

  const scene = new THREE.Scene()
  scene.background = new THREE.Color('#0b1120')
  scene.fog = new THREE.Fog('#0b1120', 2.6, 5)

  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20)
  camera.position.copy(CAM_POS)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.copy(CAM_TARGET)
  controls.enableDamping = true
  controls.maxPolarAngle = Math.PI * 0.49
  controls.minDistance = 0.5
  controls.maxDistance = 3.2
  controls.update()

  // Lights
  scene.add(new THREE.HemisphereLight('#dbeafe', '#0f172a', 1.1))
  const sun = new THREE.DirectionalLight('#ffffff', 2.2)
  sun.position.set(0.8, 1.8, 1.1)
  sun.castShadow = true
  sun.shadow.mapSize.set(2048, 2048)
  Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.1, far: 5 })
  sun.shadow.bias = -0.0005
  scene.add(sun)

  // Table and floor grid
  // Deep enough for a second robot across the table (two-arm mode).
  const table = box(1.7, 0.04, 1.3, mat('#334155', { metalness: 0.1, roughness: 0.8 }))
  table.position.set(0, -0.02, -0.2)
  table.castShadow = false
  scene.add(table)
  const grid = new THREE.GridHelper(1.6, 16, '#475569', '#3b4a61')
  grid.scale.z = 1.2 / 1.6
  grid.position.set(0, 0.001, -0.2)
  ;(grid.material as THREE.Material).transparent = true
  ;(grid.material as THREE.Material).opacity = 0.35
  scene.add(grid)

  const armA = buildArm(scene, '#f97316')
  const armB = buildArm(scene, '#38bdf8') // second arm: blue stripes so you can tell them apart
  armB.root.visible = false
  const { waist, shoulder, elbow, wrist, links, wristLink, palm, fingers } = armA

  // IK target marker
  const targetMat = new THREE.MeshBasicMaterial({ color: '#34d399' })
  const target = new THREE.Group()
  target.add(new THREE.Mesh(new THREE.SphereGeometry(0.008, 16, 12), targetMat))
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.022, 0.0025, 8, 40), targetMat)
  target.add(ring)
  target.visible = false
  scene.add(target)

  // Tool trail
  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#fbbf24', transparent: true, opacity: 0.8 }))
  trail.frustumCulled = false
  scene.add(trail)

  // The vertical plane you click on in Reach mode, with the reach circle on it.
  const reachPlane = new THREE.Group()
  const wall = new THREE.Mesh(
    new THREE.PlaneGeometry(1.6, 0.92),
    new THREE.MeshBasicMaterial({ color: '#22d3ee', transparent: true, opacity: 0.035, side: THREE.DoubleSide, depthWrite: false }),
  )
  wall.position.set(0, 0.46, 0)
  const wallGrid = new THREE.GridHelper(1.6, 16, '#155e75', '#155e75')
  wallGrid.rotation.x = Math.PI / 2
  wallGrid.scale.set(1, 1, 0.92 / 1.6)
  wallGrid.position.set(0, 0.46, 0)
  ;(wallGrid.material as THREE.Material).transparent = true
  ;(wallGrid.material as THREE.Material).opacity = 0.25
  reachPlane.add(wall, wallGrid)
  scene.add(reachPlane)
  const reachRing = new THREE.LineLoop(
    new THREE.BufferGeometry(),
    new THREE.LineDashedMaterial({ color: '#22d3ee', dashSize: 0.02, gapSize: 0.02, transparent: true, opacity: 0.5 }),
  )
  scene.add(reachRing)

  // Selection ring under the chosen block, and rings for queued drop spots.
  const selectRing = new THREE.Mesh(new THREE.RingGeometry(0.032, 0.042, 40), new THREE.MeshBasicMaterial({ color: '#fbbf24', side: THREE.DoubleSide }))
  selectRing.rotation.x = -Math.PI / 2
  selectRing.visible = false
  scene.add(selectRing)
  const tableRing = new THREE.LineLoop(
    new THREE.BufferGeometry(),
    new THREE.LineDashedMaterial({ color: '#fbbf24', dashSize: 0.025, gapSize: 0.02, transparent: true, opacity: 0.55 }),
  )
  scene.add(tableRing)
  const markerMat = new THREE.MeshBasicMaterial({ color: '#22d3ee', side: THREE.DoubleSide, transparent: true, opacity: 0.85 })

  // Cloth (towel or shirt): a grid mesh whose vertices are moved every frame.
  // Its geometry is (re)built when the cloth type changes.
  const cloth = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshStandardMaterial({ color: '#ffffff', vertexColors: true, side: THREE.DoubleSide, roughness: 0.95, metalness: 0 }),
  )
  cloth.castShadow = true
  cloth.receiveShadow = true
  cloth.frustumCulled = false
  cloth.visible = false
  scene.add(cloth)

  // Wrist camera's view cone, shown while it scans.
  const camCone = new THREE.Mesh(
    new THREE.ConeGeometry(1, 1, 32, 1, true),
    new THREE.MeshBasicMaterial({ color: '#22d3ee', transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }),
  )
  camCone.visible = false
  scene.add(camCone)
  // Detection boxes (a small pool, reused).
  const detBoxes = Array.from({ length: 3 }, () => {
    const l = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: '#22d3ee', transparent: true, opacity: 0.95, depthTest: false }),
    )
    l.renderOrder = 10
    l.visible = false
    scene.add(l)
    return l
  })

  return {
    renderer, scene, camera, controls, waist, shoulder, elbow, wrist,
    fixtures: new Map(), cloth, detBoxes, camCone,
    links, wristLink, palm, fingers, armA, armB,
    blocks: new Map(), bins: new Map(), target, targetMat, trail, reachPlane, reachRing,
    table, selectRing, markers: [], markerMat, tableRing,
  }
}


/** One robot: pedestal, waist, three links and the gripper, under a root group
 * that can be moved to where this robot stands. */
interface ArmParts {
  root: THREE.Group
  waist: THREE.Group
  shoulder: THREE.Group
  elbow: THREE.Group
  wrist: THREE.Group
  links: THREE.Mesh[]
  wristLink: THREE.Mesh
  palm: THREE.Mesh
  fingers: [THREE.Mesh, THREE.Mesh]
  /** Wrist roll: turns the gripper about its own axis. */
  rollG: THREE.Group
}

function buildArm(scene: THREE.Scene, accent: string): ArmParts {
  const root = new THREE.Group()
  scene.add(root)
  // Fixed pedestal
  const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.07, 0.1, 40), mat('#475569', { metalness: 0.5 }))
  pedestal.position.y = 0.05
  pedestal.castShadow = pedestal.receiveShadow = true
  root.add(pedestal)

  // Waist: everything above it turns about the vertical axis.
  const waist = new THREE.Group()
  root.add(waist)
  const turntable = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.016, 48), mat('#f472b6', { emissive: '#9d174d', emissiveIntensity: 0.35 }))
  turntable.position.y = 0.108
  turntable.castShadow = true
  waist.add(turntable)
  const tower = box(0.06, 0.03, 0.085, mat('#e2e8f0'))
  tower.position.set(0, 0.12, 0)
  waist.add(tower)

  const shoulder = new THREE.Group()
  waist.add(shoulder)
  const elbow = new THREE.Group()
  shoulder.add(elbow)
  const wrist = new THREE.Group()
  elbow.add(wrist)

  const linkMat = mat('#e2e8f0', { metalness: 0.35, roughness: 0.35 })
  const accentMat = mat(accent, { metalness: 0.2, roughness: 0.5 })
  // Unit-length boxes, scaled to the link length on every update.
  const l1 = box(1, 0.046, 0.052, linkMat)
  const l2 = box(1, 0.038, 0.044, linkMat)
  const stripe1 = box(1, 0.012, 0.054, accentMat)
  const stripe2 = box(1, 0.01, 0.046, accentMat)
  l1.add(stripe1)
  l2.add(stripe2)
  shoulder.add(l1, housing(0.036, 0.075))
  elbow.add(l2, housing(0.03, 0.064))
  const wristLink = box(1, 0.03, 0.036, linkMat)
  wrist.add(wristLink, housing(0.024, 0.05))

  const gripMat = mat('#cbd5e1', { metalness: 0.6, roughness: 0.3 })
  // Parallel gripper: the fingers close across the arm's plane (along local z),
  // so it can pinch a plate rim or the sides of a cup.
  const palm = box(0.014, 0.034, 1, gripMat)
  const f1 = box(FINGER, 0.026, 0.008, gripMat)
  const f2 = box(FINGER, 0.026, 0.008, gripMat)
  // Wrist roll joint: the palm and fingers turn about the gripper's axis.
  // A coloured collar with a notch makes the roll easy to see.
  const rollG = new THREE.Group()
  wrist.add(rollG)
  const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.021, 0.021, 0.014, 28), mat(accent, { metalness: 0.3, roughness: 0.4 }))
  collar.rotation.z = Math.PI / 2
  collar.position.x = -0.008
  collar.castShadow = true
  const notch = box(0.016, 0.008, 0.006, mat('#f8fafc'))
  notch.position.set(-0.008, 0.021, 0)
  rollG.add(collar, notch, palm, f1, f2)
  return { root, waist, shoulder, elbow, wrist, links: [l1, l2], wristLink, palm, fingers: [f1, f2], rollG }
}

/** Put one arm in pose q with the gripper opened to grip. */
function poseArm(arm: ArmParts, q: Q, lengths: Lengths, base: Vec, grip: number, roll = 0) {
  const [L1, L2, L3] = lengths
  arm.waist.rotation.y = -q[0] // three.js turns +x toward -z for positive angles
  arm.shoulder.position.set(0, base[1], 0)
  arm.shoulder.rotation.z = q[1]
  arm.links[0].scale.x = L1
  arm.links[0].position.x = L1 / 2
  arm.elbow.position.x = L1
  arm.elbow.rotation.z = q[2]
  arm.links[1].scale.x = L2
  arm.links[1].position.x = L2 / 2
  arm.wrist.position.x = L2
  arm.wrist.rotation.z = q[3]
  const wristLen = L3 - FINGER
  arm.wristLink.scale.x = wristLen
  arm.wristLink.position.x = wristLen / 2
  // grip 1 = open; 0 = closed on a 40 mm block; below 0 = closing on something thinner.
  const half = Math.max(0.0035, (CLOSED + (OPEN - CLOSED) * grip) / 2)
  arm.palm.scale.z = half * 2 + 0.02
  arm.rollG.position.x = wristLen
  arm.rollG.rotation.x = roll // about the gripper's own axis
  arm.palm.position.x = 0
  arm.fingers[0].position.set(FINGER / 2, 0, half)
  arm.fingers[1].position.set(FINGER / 2, 0, -half)
}

/** A block, plate or cup. Plates and cups live in a vertical plane given by
 * yaw; tilt turns them inside that plane (a cup upside down = pi). */
function makeObject(b: Block): THREE.Object3D {
  const color = COLOR[b.color] ?? b.color
  if (b.kind === 'plate') {
    const g = new THREE.Group()
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(b.size, b.size, 0.01, 48), mat(color, { roughness: 0.25, metalness: 0.05 }))
    disc.rotation.x = Math.PI / 2 // face along the plane normal (local z)
    disc.castShadow = disc.receiveShadow = true
    const rim = new THREE.Mesh(new THREE.TorusGeometry(b.size - 0.004, 0.0035, 8, 48), mat('#2563eb', { roughness: 0.4 }))
    rim.castShadow = true
    g.add(disc, rim)
    return g
  }
  if (b.kind === 'cup') {
    const g = new THREE.Group()
    const h = 0.075
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(b.size, b.size * 0.85, h, 32, 1, true),
      mat(color, { roughness: 0.35, side: THREE.DoubleSide }),
    )
    body.castShadow = true
    const bottom = new THREE.Mesh(new THREE.CircleGeometry(b.size * 0.85, 32), mat(color, { roughness: 0.35, side: THREE.DoubleSide }))
    bottom.rotation.x = -Math.PI / 2
    bottom.position.y = -h / 2
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.018, 0.004, 8, 24, Math.PI), mat(color, { roughness: 0.35 }))
    handle.rotation.z = -Math.PI / 2
    handle.position.set(-b.size + 0.002, 0.002, 0)
    handle.rotation.y = Math.PI
    g.add(body, bottom, handle)
    return g
  }
  const m = box(b.size, b.size, b.size, mat(color, { roughness: 0.5, metalness: 0.05 }))
  return m
}

function makeFixture(f: Fixture): THREE.Group {
  const g = new THREE.Group()
  g.userData.fixtureId = f.id
  const metal = mat('#94a3b8', { metalness: 0.7, roughness: 0.35 })
  const tine = (x: number, z: number, y0: number, h: number) => {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(0.0028, 0.0028, h, 8), metal)
    c.position.set(x, y0 + h / 2, z)
    c.castShadow = true
    g.add(c)
  }
  const slotFrame = (slot: { x: number; z: number; yaw: number }) => {
    const d = [Math.cos(slot.yaw), Math.sin(slot.yaw)]
    const n = [-Math.sin(slot.yaw), Math.cos(slot.yaw)]
    return { d, n }
  }
  if (f.kind === 'plate_holder') {
    const wood = mat('#a16207', { roughness: 0.8, metalness: 0 })
    for (const slot of f.slots ?? []) {
      const { d, n } = slotFrame(slot)
      const base = box(0.13, f.height, 0.04, wood)
      base.position.set(slot.x, f.height / 2, slot.z)
      base.rotation.y = -slot.yaw
      g.add(base)
      for (const side of [-1, 1]) {
        for (const along of [-0.04, 0.04]) {
          const peg = box(0.008, 0.035, 0.008, wood)
          peg.position.set(slot.x + d[0] * along + n[0] * side * 0.011, f.height + 0.0175, slot.z + d[1] * along + n[1] * side * 0.011)
          peg.rotation.y = -slot.yaw
          g.add(peg)
        }
      }
    }
    return g
  }
  if (f.kind === 'plate_rack' || f.kind === 'cup_rack') {
    const lift = f.kind === 'cup_rack' ? f.height : 0
    const floorY = f.kind === 'cup_rack' ? f.height : f.height / 2
    const floor = box(f.width, 0.006, f.depth, mat('#cbd5e1', { metalness: 0.6, roughness: 0.4, transparent: true, opacity: 0.85 }))
    floor.position.set(f.x, floorY, f.z)
    g.add(floor)
    // Wire grid on the floor
    for (let i = -3; i <= 3; i++) {
      const bar = box(f.width, 0.004, 0.003, metal)
      bar.position.set(f.x, floorY + 0.004, f.z + (i * f.depth) / 7)
      g.add(bar)
    }
    // Low rim
    const rimH = 0.035
    for (const [w, d, x, z] of [
      [f.width, 0.004, f.x, f.z - f.depth / 2],
      [f.width, 0.004, f.x, f.z + f.depth / 2],
      [0.004, f.depth, f.x - f.width / 2, f.z],
      [0.004, f.depth, f.x + f.width / 2, f.z],
    ]) {
      const r = box(w, rimH, d, metal)
      r.position.set(x, floorY + rimH / 2, z)
      g.add(r)
    }
    if (f.kind === 'cup_rack') {
      for (const sx of [-1, 1]) {
        for (const sz of [-1, 1]) tine(f.x + (sx * f.width) / 2, f.z + (sz * f.depth) / 2, 0, lift)
      }
      for (const slot of f.slots ?? []) tine(slot.x, slot.z, floorY, 0.02) // a peg under each cup
    } else {
      // Tines on both sides of each plate slot
      for (const slot of f.slots ?? []) {
        const { d, n } = slotFrame(slot)
        for (const side of [-1, 1]) {
          for (const along of [-0.05, 0, 0.05]) {
            tine(slot.x + d[0] * along + n[0] * side * 0.013, slot.z + d[1] * along + n[1] * side * 0.013, floorY, 0.055)
          }
        }
      }
    }
    return g
  }
  if (f.kind === 'basket') {
    const wicker = mat('#b45309', { roughness: 0.9, metalness: 0 })
    const t = 0.008
    const floor = box(f.width, t, f.depth, wicker)
    floor.position.set(f.x, t / 2, f.z)
    g.add(floor)
    for (const [w, d, x, z] of [
      [f.width, t, f.x, f.z - f.depth / 2],
      [t, f.depth, f.x - f.width / 2, f.z],
      [t, f.depth, f.x + f.width / 2, f.z],
    ]) {
      const wall = box(w, f.height, d, wicker)
      wall.position.set(x, f.height / 2, z)
      g.add(wall)
    }
    const front = box(f.width, f.height, t, mat('#b45309', { roughness: 0.9, transparent: true, opacity: 0.45 }))
    front.position.set(f.x, f.height / 2, f.z + f.depth / 2)
    g.add(front)
  }
  return g
}

function makeBin(b: Bin): THREE.Group {
  const g = new THREE.Group()
  const c = COLOR[b.color] ?? '#cbd5e1'
  const solid = mat(c, { roughness: 0.6, metalness: 0.05 })
  const glass = mat(c, { transparent: true, opacity: 0.35, roughness: 0.3 })
  const t = 0.006
  const floor = box(b.width, t, BIN_DEPTH, solid)
  floor.position.y = t / 2
  const left = box(t, b.wall, BIN_DEPTH, solid)
  left.position.set(-b.width / 2, b.wall / 2, 0)
  const right = left.clone()
  right.position.x = b.width / 2
  const back = box(b.width, b.wall, t, solid)
  back.position.set(0, b.wall / 2, -BIN_DEPTH / 2)
  const front = box(b.width, b.wall, t, glass)
  front.position.set(0, b.wall / 2, BIN_DEPTH / 2)
  front.castShadow = false
  g.add(floor, left, right, back, front)
  g.position.set(b.x, 0, b.z ?? 0)
  g.userData.binId = b.id
  return g
}

export default function Arm3D(props: Props) {
  const hostRef = useRef<HTMLDivElement>(null)
  const rigRef = useRef<Rig | null>(null)
  const propsRef = useRef(props)
  propsRef.current = props
  const labelRefs = useRef<(HTMLDivElement | null)[]>([])

  // Build the scene once.
  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const rig = buildRig(host)
    rigRef.current = rig

    const resize = () => {
      const w = host.clientWidth
      const h = host.clientHeight
      rig.renderer.setSize(w, h, false)
      rig.camera.aspect = w / Math.max(1, h)
      rig.camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(host)
    resize()

    // A click (not an orbit drag) either picks a block / bin / table spot
    // (Pick & place mode) or sets the IK target on the vertical work plane.
    let down: { x: number; y: number } | null = null
    const el = rig.renderer.domElement
    const rayFrom = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect()
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
      const ray = new THREE.Raycaster()
      ray.setFromCamera(ndc, rig.camera)
      return ray
    }
    const pickAt = (e: PointerEvent): PickTarget | null => {
      const ray = rayFrom(e)
      const targets: THREE.Object3D[] = [...rig.blocks.values(), ...rig.bins.values(), rig.table]
      for (const hit of ray.intersectObjects(targets, true)) {
        let o: THREE.Object3D | null = hit.object
        while (o) {
          if (o.userData.blockId) return { kind: 'block', id: o.userData.blockId }
          if (o.userData.binId) return { kind: 'bin', id: o.userData.binId }
          o = o.parent
        }
        if (hit.object === rig.table) return { kind: 'table', x: hit.point.x, z: hit.point.z }
      }
      return null
    }
    const onDown = (e: PointerEvent) => (down = { x: e.clientX, y: e.clientY })
    const onUp = (e: PointerEvent) => {
      const p = propsRef.current
      if (!down) return
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y)
      down = null
      if (moved > 5) return // that was an orbit drag
      if (p.pickMode && p.onPick) {
        const t = pickAt(e)
        if (t) p.onPick(t)
        return
      }
      if (!p.interactive || !p.onTarget) return
      const hit = new THREE.Vector3()
      if (rayFrom(e).ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), hit)) {
        p.onTarget([hit.x, hit.y])
      }
    }
    // Hand cursor over things you can click in Pick & place mode.
    const onMove = (e: PointerEvent) => {
      const p = propsRef.current
      if (!p.pickMode || down) return
      const t = pickAt(e)
      el.style.cursor = t && t.kind !== 'table' ? 'pointer' : t ? 'copy' : 'grab'
    }
    el.addEventListener('pointerdown', onDown)
    el.addEventListener('pointerup', onUp)
    el.addEventListener('pointermove', onMove)

    let raf = 0
    const loop = () => {
      rig.controls.update()
      // Keep the detection labels pinned above their boxes.
      const dets = propsRef.current.detections ?? []
      labelRefs.current.forEach((el, i) => {
        if (!el) return
        const d = dets[i]
        if (!d) {
          el.style.display = 'none'
          return
        }
        const v = new THREE.Vector3(d.center[0], d.center[1] + d.size[1] / 2 + 0.025, d.center[2]).project(rig.camera)
        el.style.display = 'block'
        el.style.left = `${((v.x + 1) / 2) * host.clientWidth}px`
        el.style.top = `${((1 - v.y) / 2) * host.clientHeight}px`
      })
      rig.renderer.render(rig.scene, rig.camera)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      el.removeEventListener('pointerdown', onDown)
      el.removeEventListener('pointerup', onUp)
      el.removeEventListener('pointermove', onMove)
      rig.controls.dispose()
      rig.scene.traverse((o) => {
        const m = o as THREE.Mesh
        m.geometry?.dispose?.()
        const material = m.material as THREE.Material | THREE.Material[] | undefined
        if (Array.isArray(material)) material.forEach((x) => x.dispose())
        else material?.dispose?.()
      })
      rig.renderer.dispose()
      el.remove()
      rigRef.current = null
    }
  }, [])

  // Push the current props into the scene on every render.
  useEffect(() => {
    const rig = rigRef.current
    if (!rig) return
    const { q, lengths, base, grip, blocks, bins, target, targetOk, trail, showReachPlane } = props
    const [L1, L2, L3] = lengths

    poseArm(rig.armA, q, lengths, base, grip, props.roll ?? 0)
    rig.armB.root.visible = !!props.second
    if (props.second) {
      rig.armB.root.position.z = props.second.baseZ
      poseArm(rig.armB, props.second.q, lengths, base, props.second.grip, props.second.roll ?? 0)
    }

    // Blocks: add new ones, drop removed ones, move all.
    const seen = new Set<string>()
    for (const b of blocks) {
      if (b.kind === 'towel' || b.kind === 'shirt') continue // drawn as cloth below
      seen.add(b.id)
      let m = rig.blocks.get(b.id)
      if (!m) {
        m = makeObject(b)
        m.traverse((o) => (o.userData.blockId = b.id))
        m.rotation.order = 'YZX' // turn within its own plane (tilt), then about the vertical (yaw)
        rig.blocks.set(b.id, m)
        rig.scene.add(m)
      }
      m.position.set(b.x, b.y, b.z ?? 0)
      m.rotation.y = -(b.yaw ?? 0) // a carried object turns with the arm and keeps that turn if dropped
      m.rotation.z = b.tilt ?? 0
      const selected = props.selectedId === b.id
      if (m instanceof THREE.Mesh) {
        const material = m.material as THREE.MeshStandardMaterial
        material.emissive.set(selected ? '#ffffff' : '#000000')
        material.emissiveIntensity = selected ? 0.35 : 0
      }
      if (selected) {
        rig.selectRing.position.set(b.x, Math.max(0.003, b.y - b.size / 2 + 0.003), b.z ?? 0)
      }
    }
    rig.selectRing.visible = !!props.selectedId && blocks.some((b) => b.id === props.selectedId)

    // Dashed ring on the table: how far out a block can be set down.
    rig.tableRing.visible = !!props.pickMode
    if (props.pickMode) {
      const r = tableReach(lengths)
      rig.tableRing.geometry.dispose()
      rig.tableRing.geometry = new THREE.BufferGeometry().setFromPoints(
        Array.from({ length: 120 }, (_, i) => {
          const a = (i / 120) * Math.PI * 2
          return new THREE.Vector3(r * Math.cos(a), 0.003, r * Math.sin(a))
        }).filter((v) => Math.abs(v.z) < 0.45),
      )
      rig.tableRing.computeLineDistances()
    }

    // Rings for queued table drop spots.
    const markers = props.markers ?? []
    while (rig.markers.length < markers.length) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.026, 0.034, 32), rig.markerMat)
      ring.rotation.x = -Math.PI / 2
      rig.scene.add(ring)
      rig.markers.push(ring)
    }
    rig.markers.forEach((ring, i) => {
      ring.visible = i < markers.length
      if (i < markers.length) ring.position.set(markers[i].x, 0.004, markers[i].z)
    })
    for (const [id, m] of rig.blocks) {
      if (!seen.has(id)) {
        rig.scene.remove(m)
        rig.blocks.delete(id)
      }
    }

    const binIds = new Set(bins.map((b) => b.id))
    for (const b of bins) {
      if (!rig.bins.has(b.id)) {
        const g = makeBin(b)
        rig.bins.set(b.id, g)
        rig.scene.add(g)
      }
    }
    for (const [id, g] of rig.bins) {
      if (!binIds.has(id)) {
        rig.scene.remove(g)
        rig.bins.delete(id)
      }
    }

    // Fixtures (dish holder, racks, basket)
    const fixtures = props.fixtures ?? []
    const fixtureIds = new Set(fixtures.map((x) => x.id))
    for (const fx of fixtures) {
      if (!rig.fixtures.has(fx.id)) {
        const g = makeFixture(fx)
        rig.fixtures.set(fx.id, g)
        rig.scene.add(g)
      }
    }
    for (const [id, g] of rig.fixtures) {
      if (!fixtureIds.has(id)) {
        rig.scene.remove(g)
        rig.fixtures.delete(id)
      }
    }

    // Cloth (towel or shirt)
    const cloth = props.cloth
    rig.cloth.visible = !!cloth
    if (cloth) {
      const { grid } = cloth
      if (rig.cloth.userData.key !== grid.key) {
        rig.cloth.userData.key = grid.key
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cloth.verts.length), 3))
        if (grid.colors.length) geo.setAttribute('color', new THREE.BufferAttribute(grid.colors, 3))
        // Texture coordinates: across the cloth (v) and along it (u, collar at the top).
        const uv = new Float32Array((grid.nu + 1) * (grid.nv + 1) * 2)
        for (let i = 0; i <= grid.nu; i++) {
          for (let j = 0; j <= grid.nv; j++) {
            const k = (i * (grid.nv + 1) + j) * 2
            uv[k] = j / grid.nv
            uv[k + 1] = 1 - i / grid.nu
          }
        }
        geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
        const index: number[] = []
        for (let i = 0; i < grid.nu; i++) {
          for (let j = 0; j < grid.nv; j++) {
            if (grid.mask && !grid.mask[i * grid.nv + j]) continue
            const k = i * (grid.nv + 1) + j
            index.push(k, k + 1, k + grid.nv + 1, k + 1, k + grid.nv + 2, k + grid.nv + 1)
          }
        }
        geo.setIndex(index)
        rig.cloth.geometry.dispose()
        rig.cloth.geometry = geo

        // Fabric material: cotton jersey for the shirt, terry cloth for the towel.
        const old = rig.cloth.material as THREE.MeshPhysicalMaterial
        ;[old.map, old.alphaMap, old.bumpMap].forEach((t) => t?.dispose())
        old.dispose()
        const tint = new THREE.Color('#ffffff').lerp(new THREE.Color(grid.color ?? '#ffffff'), 0.5)
        if (grid.texture === 'shirt') {
          const tex = shirtTextures(grid.color ?? '#e11d48')
          rig.cloth.material = new THREE.MeshPhysicalMaterial({
            map: tex.map,
            alphaMap: tex.alpha,
            alphaTest: 0.5,
            bumpMap: tex.bump,
            bumpScale: 0.6,
            side: THREE.DoubleSide,
            roughness: 0.92,
            metalness: 0,
            // A soft fabric sheen at grazing angles, kept subtle so the colour stays rich.
            sheen: 0.45,
            sheenRoughness: 0.6,
            sheenColor: new THREE.Color(grid.color ?? '#ffffff').lerp(new THREE.Color('#ffffff'), 0.25),
          })
          // Shadows follow the shirt's outline, not the whole grid.
          rig.cloth.customDepthMaterial = new THREE.MeshDepthMaterial({
            depthPacking: THREE.RGBADepthPacking,
            alphaMap: tex.alpha,
            alphaTest: 0.5,
            side: THREE.DoubleSide,
          })
        } else {
          rig.cloth.material = new THREE.MeshPhysicalMaterial({
            color: '#ffffff',
            vertexColors: true,
            bumpMap: terryBump(),
            bumpScale: 0.8,
            side: THREE.DoubleSide,
            roughness: 0.97,
            metalness: 0,
            sheen: 1,
            sheenRoughness: 0.8,
            sheenColor: tint,
          })
          rig.cloth.customDepthMaterial = undefined
        }
      }
      const pos = rig.cloth.geometry.getAttribute('position') as THREE.BufferAttribute
      ;(pos.array as Float32Array).set(cloth.verts)
      pos.needsUpdate = true
      rig.cloth.geometry.computeVertexNormals()
      rig.cloth.geometry.computeBoundingSphere()
    }

    // Camera detections: a box around what the camera thinks it sees.
    const dets = props.detections ?? []
    rig.detBoxes.forEach((boxLine, i) => {
      const d = dets[i]
      boxLine.visible = !!d
      if (!d) return
      boxLine.position.set(d.center[0], d.center[1], d.center[2])
      boxLine.scale.set(d.size[0] + 0.012, d.size[1] + 0.012, d.size[2] + 0.012)
      boxLine.rotation.y = -d.yaw
      ;(boxLine.material as THREE.LineBasicMaterial).color.set(d.phase === 'align' ? '#34d399' : '#22d3ee')
    })
    const scan = dets.find((d) => d.phase === 'scan')
    rig.camCone.visible = !!scan
    if (scan) {
      const fromB = scan.arm === 'B' && props.second
      const tip = fk(fromB ? props.second!.q : q, lengths, base).pts3[3]
      if (fromB) tip[2] += props.second!.baseZ
      const bottom = scan.center[1] + scan.size[1] / 2
      const h = Math.max(0.02, tip[1] - bottom)
      const r = Math.max(scan.size[0], scan.size[2]) * 0.75 + 0.02
      rig.camCone.scale.set(r, h, r)
      rig.camCone.position.set(tip[0], bottom + h / 2, tip[2])
    }

    rig.target.visible = !!target
    if (target) {
      rig.target.position.set(target[0], target[1], 0)
      rig.targetMat.color.set(targetOk ? '#34d399' : '#f87171')
    }

    rig.trail.geometry.dispose()
    rig.trail.geometry = new THREE.BufferGeometry().setFromPoints(trail.map(([x, y, z]) => new THREE.Vector3(x, y, z)))

    rig.reachPlane.visible = showReachPlane
    rig.reachRing.visible = showReachPlane
    const reach = L1 + L2 + L3
    const ringPts = Array.from({ length: 96 }, (_, i) => {
      const a = (i / 96) * Math.PI * 2
      return new THREE.Vector3(reach * Math.cos(a), base[1] + reach * Math.sin(a), 0)
    }).filter((v) => v.y >= 0)
    rig.reachRing.geometry.dispose()
    rig.reachRing.geometry = new THREE.BufferGeometry().setFromPoints(ringPts)
    rig.reachRing.computeLineDistances()
  })

  const resetView = () => {
    const rig = rigRef.current
    if (!rig) return
    rig.camera.position.copy(CAM_POS)
    rig.controls.target.copy(CAM_TARGET)
    rig.controls.update()
  }

  return (
    <div className="relative aspect-[16/10] w-full">
      <div ref={hostRef} className={`absolute inset-0 ${props.interactive && !props.pickMode ? 'cursor-crosshair' : 'cursor-grab'}`} />
      {(props.detections ?? []).map((d, i) => (
        <div
          key={i}
          ref={(el) => (labelRefs.current[i] = el)}
          className={`pointer-events-none absolute -translate-x-1/2 -translate-y-full whitespace-nowrap rounded px-1.5 py-0.5 font-mono text-[10px] ring-1 ${
            d.phase === 'align' ? 'bg-emerald-950/90 text-emerald-200 ring-emerald-400/50' : 'bg-cyan-950/90 text-cyan-200 ring-cyan-400/50'
          }`}
          style={{ display: 'none' }}
        >
          {d.phase === 'align' ? `${d.label} · locked` : `${d.label} ${Math.round(d.confidence * 100)}%`}
        </div>
      ))}
      {props.hint && (
        <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-slate-900/80 px-2.5 py-1.5 text-xs text-slate-300 ring-1 ring-slate-700">
          {props.hint}
        </div>
      )}
      <div className="absolute bottom-3 left-3 flex items-center gap-2 text-[11px] text-slate-400">
        <span className="rounded bg-slate-900/80 px-2 py-1 ring-1 ring-slate-700">Drag to orbit · scroll to zoom</span>
        <button type="button" onClick={resetView} className="pointer-events-auto rounded bg-slate-800 px-2 py-1 text-slate-300 ring-1 ring-slate-700 hover:bg-slate-700">
          Reset view
        </button>
      </div>
    </div>
  )
}
