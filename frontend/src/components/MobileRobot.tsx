import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Hand, Square } from 'lucide-react'
import clsx from 'clsx'
import { deg, fmt, rad } from '../lib/geom'
import { Button, Card, Segmented, Slider, Stat } from './ui'

// ---------------------------------------------------------------------------
// Geometry (metres). Robot frame: x forward, y up, z to the robot's right.
// ---------------------------------------------------------------------------
const BASE = { len: 0.62, w: 0.54, h: 0.24, y0: 0.06 } // y0 = underside of the shell
const BASE_TOP = BASE.y0 + BASE.h + 0.02
const WHEEL_R = 0.075
const WHEEL_Z = 0.21 // drive wheels, hidden under the skirt
const COLUMN = { x: -0.07, d: 0.11, w: 0.15, top: 1.34 }
const SHOULDER_Z = 0.235
const UPPER = 0.31
const FORE = 0.28
const LIFT_MIN = 0.78 // shoulder height range on the column
const LIFT_MAX = 1.18
const ROBOT_R = 0.42 // footprint radius used for collisions
const BOUNDS = 5.4
const ACCEL = 1.4 // m/s²
const TURN_ACCEL = 6 // rad/s²
const MAX_TURN = 1.6 // rad/s
const JOINT_SPEED = 2.2 // rad/s, how fast the arms move to a new pose

const CAM_OFFSET = new THREE.Vector3(1.5, 1.55, 2.5)
const CAM_LOOK = new THREE.Vector3(0, 0.8, 0)

const C = {
  white: '#eef1f5',
  gray: '#9aa3ae',
  dark: '#2a2f36',
  black: '#14171b',
  orange: '#f28c28',
}

type Obstacle =
  | { kind: 'crate'; x: number; z: number; w: number; d: number; h: number; color: string }
  | { kind: 'barrel'; x: number; z: number; r: number; h: number; color: string }

const OBSTACLES: Obstacle[] = [
  { kind: 'crate', x: 2.1, z: -1.2, w: 0.6, d: 0.6, h: 0.5, color: '#a16207' },
  { kind: 'crate', x: 2.6, z: -0.5, w: 0.45, d: 0.45, h: 0.35, color: '#b45309' },
  { kind: 'crate', x: -2.2, z: 1.8, w: 1.2, d: 0.5, h: 0.6, color: '#92400e' },
  { kind: 'crate', x: -1.4, z: -2.6, w: 0.5, d: 0.9, h: 0.45, color: '#a16207' },
  { kind: 'crate', x: 3.6, z: 2.4, w: 0.7, d: 0.7, h: 0.7, color: '#b45309' },
  { kind: 'barrel', x: 0.9, z: 2.2, r: 0.2, h: 0.55, color: '#2563eb' },
  { kind: 'barrel', x: 1.4, z: 2.5, r: 0.2, h: 0.55, color: '#dc2626' },
  { kind: 'barrel', x: -3.2, z: -0.8, r: 0.22, h: 0.6, color: '#16a34a' },
  { kind: 'barrel', x: 0.2, z: -3.3, r: 0.18, h: 0.5, color: '#eab308' },
]

/** True when a robot centred at (x, z) would overlap an obstacle or leave the floor. */
function blockedAt(x: number, z: number) {
  if (Math.abs(x) > BOUNDS - ROBOT_R || Math.abs(z) > BOUNDS - ROBOT_R) return true
  for (const o of OBSTACLES) {
    if (o.kind === 'crate') {
      const cx = Math.max(o.x - o.w / 2, Math.min(x, o.x + o.w / 2))
      const cz = Math.max(o.z - o.d / 2, Math.min(z, o.z + o.d / 2))
      if (Math.hypot(x - cx, z - cz) < ROBOT_R) return true
    } else if (Math.hypot(x - o.x, z - o.z) < ROBOT_R + o.r) return true
  }
  return false
}

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const approach = (v: number, target: number, step: number) => v + clamp(target - v, -step, step)

// ---------------------------------------------------------------------------
// Arm poses
// ---------------------------------------------------------------------------
interface ArmPose {
  /** Swing forward (+) / back about the shoulder. */
  pitch: number
  /** Out to the side. */
  spread: number
  /** Elbow bend, forward. */
  elbow: number
  /** Wrist bend. */
  wrist: number
  /** Wrist roll. */
  roll: number
  /** 0 closed .. 1 open */
  grip: number
}
type Side = 'L' | 'R'
type Arms = Record<Side, ArmPose>

const pose = (p: number, s: number, e: number, w = 0, r = 0, g = 0.6): ArmPose => ({
  pitch: rad(p),
  spread: rad(s),
  elbow: rad(e),
  wrist: rad(w),
  roll: rad(r),
  grip: g,
})
const both = (p: ArmPose): Arms => ({ L: { ...p }, R: { ...p } })

const PRESETS: { label: string; arms: Arms }[] = [
  { label: 'Ready', arms: both(pose(22, 12, 78, -10)) },
  { label: 'Rest', arms: both(pose(0, 6, 12, 0, 0, 0.2)) },
  { label: 'Reach', arms: both(pose(80, 8, 6, 0, 0, 1)) },
  { label: 'Carry', arms: both(pose(10, 4, 95, -5, 90, 0.4)) },
  { label: 'Arms up', arms: both(pose(165, 22, 8, 0, 0, 1)) },
]

const JOINTS: { key: keyof ArmPose; label: string; min: number; max: number; color: string }[] = [
  { key: 'pitch', label: 'Shoulder', min: -45, max: 180, color: '#f28c28' },
  { key: 'spread', label: 'Shoulder out', min: 0, max: 90, color: '#38bdf8' },
  { key: 'elbow', label: 'Elbow', min: 0, max: 145, color: '#a78bfa' },
  { key: 'wrist', label: 'Wrist bend', min: -90, max: 90, color: '#34d399' },
  { key: 'roll', label: 'Wrist roll', min: -180, max: 180, color: '#f472b6' },
]

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------
const mat = (color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.2, ...extra })

function shadowed<T extends THREE.Object3D>(o: T): T {
  o.traverse((c) => {
    c.castShadow = true
    c.receiveShadow = true
  })
  return o
}

const rbox = (x: number, y: number, z: number, r: number, m: THREE.Material) => shadowed(new THREE.Mesh(new RoundedBoxGeometry(x, y, z, 4, r), m))

/** A joint motor: a short cylinder along z with an orange ring. */
function motor(radius: number, width: number, shell: string = C.gray) {
  const g = new THREE.Group()
  const body = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, width, 28), mat(shell, { metalness: 0.35, roughness: 0.35 }))
  body.rotation.x = Math.PI / 2
  const ring = new THREE.Mesh(new THREE.CylinderGeometry(radius * 1.02, radius * 1.02, width * 0.22, 28), mat(C.orange, { roughness: 0.4 }))
  ring.rotation.x = Math.PI / 2
  g.add(body, ring)
  return shadowed(g)
}

interface ArmRig {
  s: number
  pitch: THREE.Group
  spread: THREE.Group
  elbow: THREE.Group
  wrist: THREE.Group
  roll: THREE.Group
  fingers: [THREE.Mesh, THREE.Mesh]
}

/** One arm hanging from the shoulder: white upper arm, grey forearm, black gripper. */
function buildArm(s: number): ArmRig {
  const white = mat(C.white, { roughness: 0.35 })
  const gray = mat(C.gray, { metalness: 0.35, roughness: 0.35 })
  const black = mat(C.black, { roughness: 0.5, metalness: 0.3 })
  const orange = mat(C.orange, { roughness: 0.4 })

  const pitch = new THREE.Group()
  pitch.position.set(0, 0, s * SHOULDER_Z)
  const cap = new THREE.Mesh(new THREE.SphereGeometry(0.068, 32, 20), white)
  cap.scale.set(1, 0.95, 0.8)
  pitch.add(shadowed(cap))

  const spread = new THREE.Group()
  pitch.add(spread)
  spread.add(motor(0.05, 0.09))
  const upper = rbox(0.085, UPPER - 0.04, 0.085, 0.03, white)
  upper.position.y = -UPPER / 2
  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.046, 0.02, 24), orange)
  band.position.y = -UPPER + 0.055
  spread.add(upper, shadowed(band))

  const elbow = new THREE.Group()
  elbow.position.y = -UPPER
  spread.add(elbow)
  elbow.add(motor(0.042, 0.085))
  const sleeve = rbox(0.075, 0.08, 0.075, 0.025, white)
  sleeve.position.y = -0.05
  const fore = rbox(0.064, FORE - 0.05, 0.064, 0.024, gray)
  fore.position.y = -FORE / 2 - 0.02
  elbow.add(sleeve, fore)

  const wrist = new THREE.Group()
  wrist.position.y = -FORE
  elbow.add(wrist)
  wrist.add(motor(0.032, 0.07, C.dark))
  const roll = new THREE.Group()
  wrist.add(roll)
  const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.034, 0.03, 24), orange)
  cuff.position.y = -0.03
  const palm = rbox(0.05, 0.06, 0.085, 0.012, black)
  palm.position.y = -0.07
  roll.add(shadowed(cuff), palm)
  const f1 = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.075, 0.012), black))
  const f2 = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.075, 0.012), black))
  for (const f of [f1, f2]) {
    const tip = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.018, 0.014), orange)
    tip.position.y = -0.03
    f.add(tip)
  }
  roll.add(f1, f2)
  return { s, pitch, spread, elbow, wrist, roll, fingers: [f1, f2] }
}

function poseArm(arm: ArmRig, p: ArmPose) {
  arm.pitch.rotation.z = p.pitch
  arm.spread.rotation.x = -arm.s * p.spread
  arm.elbow.rotation.z = p.elbow
  arm.wrist.rotation.z = p.wrist
  arm.roll.rotation.y = arm.s * p.roll
  const half = 0.008 + 0.028 * p.grip
  arm.fingers[0].position.set(0, -0.135, half)
  arm.fingers[1].position.set(0, -0.135, -half)
}

interface Head {
  pan: THREE.Group
  tilt: THREE.Group
  ringMat: THREE.MeshStandardMaterial
}

function buildHead(parent: THREE.Group): Head {
  const neck = new THREE.Group()
  neck.position.set(COLUMN.x + 0.02, COLUMN.top, 0)
  parent.add(neck)
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.034, 0.08, 20), mat(C.dark, { metalness: 0.5 }))
  stem.position.y = 0.04
  neck.add(shadowed(stem))

  const pan = new THREE.Group()
  pan.position.y = 0.085
  neck.add(pan)
  const tilt = new THREE.Group()
  pan.add(tilt)
  tilt.add(motor(0.028, 0.09, C.dark))

  const shell = rbox(0.14, 0.11, 0.17, 0.035, mat(C.dark, { metalness: 0.3, roughness: 0.4 }))
  shell.position.set(0.015, 0.065, 0)
  const cap = rbox(0.13, 0.03, 0.16, 0.012, mat(C.white))
  cap.position.set(0.01, 0.125, 0)
  tilt.add(shell, cap)

  // Main camera: a big lens with a status ring, plus a small depth sensor.
  const lensGroup = new THREE.Group()
  lensGroup.position.set(0.087, 0.065, 0)
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.045, 0.03, 32), mat(C.black, { metalness: 0.6 }))
  barrel.rotation.z = Math.PI / 2
  const ringMat = new THREE.MeshStandardMaterial({ color: '#67e8f9', emissive: '#22d3ee', emissiveIntensity: 1.4 })
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.036, 0.004, 10, 36), ringMat)
  ring.rotation.y = Math.PI / 2
  ring.position.x = 0.016
  const glass = new THREE.Mesh(new THREE.CircleGeometry(0.03, 32), new THREE.MeshStandardMaterial({ color: '#0b1a2e', metalness: 0.9, roughness: 0.05 }))
  glass.rotation.y = Math.PI / 2
  glass.position.x = 0.0155
  const glint = new THREE.Mesh(new THREE.CircleGeometry(0.008, 16), new THREE.MeshBasicMaterial({ color: '#93c5fd', transparent: true, opacity: 0.6 }))
  glint.rotation.y = Math.PI / 2
  glint.position.set(0.016, 0.011, 0.01)
  lensGroup.add(shadowed(barrel), ring, glass, glint)
  tilt.add(lensGroup)
  for (const z of [-0.062, 0.062]) {
    const eye = new THREE.Mesh(new THREE.CircleGeometry(0.008, 16), new THREE.MeshBasicMaterial({ color: '#1e3a5f' }))
    eye.rotation.y = Math.PI / 2
    eye.position.set(0.086, 0.065, z)
    tilt.add(eye)
  }
  return { pan, tilt, ringMat }
}

interface World {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  sun: THREE.DirectionalLight
  floor: THREE.Mesh
  robot: THREE.Group
  carriage: THREE.Group
  wheels: { spin: THREE.Group; s: number; phi: number }[]
  arms: Record<Side, ArmRig>
  head: Head
  goal: THREE.Group
  trail: THREE.Line
  brakeMats: THREE.MeshStandardMaterial[]
  stripMat: THREE.MeshStandardMaterial
}

function buildRobot() {
  const robot = new THREE.Group()
  const white = mat(C.white, { roughness: 0.35 })
  const dark = mat(C.dark, { metalness: 0.3, roughness: 0.5 })
  const black = mat(C.black, { roughness: 0.7 })

  // ---- Base: rounded shell on a dark skirt, wheels hidden underneath.
  const skirt = rbox(BASE.len + 0.03, 0.08, BASE.w + 0.03, 0.03, dark)
  skirt.position.y = 0.075
  const shell = rbox(BASE.len, BASE.h, BASE.w, 0.07, white)
  shell.position.y = BASE.y0 + BASE.h / 2 + 0.03
  const lid = rbox(BASE.len - 0.1, 0.03, BASE.w - 0.1, 0.012, mat('#c7ccd3', { roughness: 0.4 }))
  lid.position.y = BASE_TOP - 0.01
  const plinth = rbox(0.28, 0.05, 0.3, 0.015, dark) // where the column meets the base
  plinth.position.set(COLUMN.x, BASE_TOP + 0.02, 0)
  robot.add(skirt, shell, lid, plinth)

  // Front light strip and orange accents.
  const stripMat = new THREE.MeshStandardMaterial({ color: '#e0f2fe', emissive: '#bae6fd', emissiveIntensity: 1.3 })
  const strip = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.012, 0.24), stripMat)
  strip.position.set(BASE.len / 2 + 0.002, 0.2, 0)
  robot.add(strip)
  for (const s of [-1, 1]) {
    const accent = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.012, 0.01), mat(C.orange))
    accent.position.set(BASE.len / 2 - 0.12, 0.11, s * (BASE.w / 2 + 0.004))
    const vent = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.03, 0.006), black)
    vent.position.set(-0.08, 0.2, s * (BASE.w / 2 + 0.003))
    robot.add(accent, vent)
  }
  const brakeMats: THREE.MeshStandardMaterial[] = []
  for (const s of [-1, 1]) {
    const m = new THREE.MeshStandardMaterial({ color: '#fecaca', emissive: '#ef4444', emissiveIntensity: 0.5 })
    const tl = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.014, 0.08), m)
    tl.position.set(-BASE.len / 2 - 0.002, 0.2, s * 0.17)
    robot.add(tl)
    brakeMats.push(m)
  }

  // Drive wheels (two) and casters.
  const wheels: World['wheels'] = []
  for (const s of [-1, 1]) {
    const spin = new THREE.Group()
    spin.position.set(0, WHEEL_R, s * WHEEL_Z)
    const tire = new THREE.Mesh(new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, 0.045, 32), black)
    tire.rotation.x = Math.PI / 2
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.05, 6), mat(C.orange))
    hub.rotation.x = Math.PI / 2
    spin.add(shadowed(tire), hub)
    robot.add(spin)
    wheels.push({ spin, s, phi: 0 })
  }
  for (const x of [-0.22, 0.22]) {
    const caster = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), black)
    caster.position.set(x, 0.03, 0)
    robot.add(shadowed(caster))
  }

  // ---- Lift column: black front rail, white back panel.
  const colH = COLUMN.top - BASE_TOP
  const column = rbox(COLUMN.d, colH, COLUMN.w, 0.012, mat(C.black, { roughness: 0.55, metalness: 0.35 }))
  column.position.set(COLUMN.x, BASE_TOP + colH / 2, 0)
  const back = rbox(0.03, colH - 0.04, COLUMN.w + 0.01, 0.01, white)
  back.position.set(COLUMN.x - COLUMN.d / 2 - 0.005, BASE_TOP + colH / 2, 0)
  const rail = new THREE.Mesh(new THREE.BoxGeometry(0.004, colH - 0.1, 0.03), mat('#3f4650', { metalness: 0.7 }))
  rail.position.set(COLUMN.x + COLUMN.d / 2 + 0.002, BASE_TOP + colH / 2, 0)
  const topCap = rbox(COLUMN.d + 0.03, 0.03, COLUMN.w + 0.03, 0.01, white)
  topCap.position.set(COLUMN.x - 0.008, COLUMN.top, 0)
  robot.add(column, back, rail, topCap)

  // ---- Torso carriage: rides up and down the column, carries the shoulders.
  const carriage = new THREE.Group()
  robot.add(carriage)
  const torso = rbox(0.21, 0.19, 0.3, 0.05, white)
  torso.position.set(COLUMN.x + 0.03, 0.01, 0)
  const chest = rbox(0.02, 0.13, 0.17, 0.008, dark)
  chest.position.set(COLUMN.x + 0.14, 0.0, 0)
  const logo = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.012, 0.07), mat(C.orange))
  logo.position.set(COLUMN.x + 0.152, 0.035, 0)
  const clamp_ = rbox(0.16, 0.06, COLUMN.w + 0.05, 0.015, dark)
  clamp_.position.set(COLUMN.x - 0.02, -0.11, 0)
  carriage.add(torso, chest, logo, clamp_)
  for (const s of [-1, 1]) {
    const yoke = rbox(0.13, 0.13, 0.08, 0.035, white)
    yoke.position.set(0, 0.0, s * 0.165)
    carriage.add(yoke)
  }
  const arms = { L: buildArm(-1), R: buildArm(1) }
  carriage.add(arms.L.pitch, arms.R.pitch)

  const head = buildHead(robot)
  return { robot, carriage, wheels, arms, head, brakeMats, stripMat }
}

function buildWorld(el: HTMLDivElement): World {
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  el.appendChild(renderer.domElement)
  Object.assign(renderer.domElement.style, { display: 'block', width: '100%', height: '100%' })

  const scene = new THREE.Scene()
  scene.background = new THREE.Color('#0b1120')
  scene.fog = new THREE.Fog('#0b1120', 6, 13)

  const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 40)
  camera.position.copy(CAM_OFFSET)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.copy(CAM_LOOK)
  controls.enableDamping = true
  controls.maxPolarAngle = Math.PI * 0.48
  controls.minDistance = 0.9
  controls.maxDistance = 8
  controls.update()

  scene.add(new THREE.HemisphereLight('#e0ecff', '#111827', 1.15))
  const sun = new THREE.DirectionalLight('#ffffff', 2.3)
  sun.castShadow = true
  sun.shadow.mapSize.set(2048, 2048)
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 12 })
  sun.shadow.bias = -0.0006
  scene.add(sun, sun.target)
  const fill = new THREE.DirectionalLight('#cbd5e1', 0.6)
  fill.position.set(-2, 2, -1.5)
  scene.add(fill)

  // Floor, grid and a low wall around the edge.
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(BOUNDS * 2, BOUNDS * 2), mat('#1e293b', { metalness: 0.05, roughness: 0.9 }))
  floor.rotation.x = -Math.PI / 2
  floor.receiveShadow = true
  scene.add(floor)
  const grid = new THREE.GridHelper(BOUNDS * 2, 27, '#475569', '#334155')
  grid.position.y = 0.001
  ;(grid.material as THREE.Material).transparent = true
  ;(grid.material as THREE.Material).opacity = 0.45
  scene.add(grid)
  const wallMat = mat('#334155', { roughness: 0.8 })
  for (const [w, d, x, z] of [
    [BOUNDS * 2 + 0.1, 0.05, 0, -BOUNDS],
    [BOUNDS * 2 + 0.1, 0.05, 0, BOUNDS],
    [0.05, BOUNDS * 2, -BOUNDS, 0],
    [0.05, BOUNDS * 2, BOUNDS, 0],
  ]) {
    const wall = new THREE.Mesh(new THREE.BoxGeometry(w, 0.12, d), wallMat)
    wall.position.set(x, 0.06, z)
    scene.add(shadowed(wall))
  }

  for (const o of OBSTACLES) {
    if (o.kind === 'crate') {
      const crate = rbox(o.w, o.h, o.d, 0.015, mat(o.color, { roughness: 0.85, metalness: 0 }))
      crate.position.set(o.x, o.h / 2, o.z)
      scene.add(crate)
      const band = new THREE.Mesh(new THREE.BoxGeometry(o.w + 0.004, 0.04, o.d + 0.004), mat('#78350f', { roughness: 0.9 }))
      band.position.set(o.x, o.h * 0.7, o.z)
      scene.add(shadowed(band))
    } else {
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(o.r, o.r, o.h, 32), mat(o.color, { roughness: 0.5, metalness: 0.4 }))
      barrel.position.set(o.x, o.h / 2, o.z)
      scene.add(shadowed(barrel))
      for (const y of [0.25, 0.75]) {
        const hoop = new THREE.Mesh(new THREE.TorusGeometry(o.r + 0.002, 0.008, 8, 32), mat('#cbd5e1', { metalness: 0.8 }))
        hoop.rotation.x = Math.PI / 2
        hoop.position.set(o.x, o.h * y, o.z)
        scene.add(hoop)
      }
    }
  }

  const parts = buildRobot()
  scene.add(parts.robot)

  // Goal marker and trail
  const goal = new THREE.Group()
  const goalMat = new THREE.MeshBasicMaterial({ color: '#34d399', transparent: true, opacity: 0.9, side: THREE.DoubleSide })
  const goalRing = new THREE.Mesh(new THREE.RingGeometry(0.14, 0.18, 48), goalMat)
  goalRing.rotation.x = -Math.PI / 2
  goalRing.position.y = 0.004
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.4, 8), goalMat)
  pole.position.y = 0.2
  const flag = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.07, 16), goalMat)
  flag.position.y = 0.44
  flag.rotation.x = Math.PI
  goal.add(goalRing, pole, flag)
  goal.visible = false
  scene.add(goal)

  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#fbbf24', transparent: true, opacity: 0.7 }))
  trail.frustumCulled = false
  scene.add(trail)

  return { renderer, scene, camera, controls, sun, floor, goal, trail, ...parts }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
interface Settings {
  maxSpeed: number
  lift: number
  follow: boolean
  autoHead: boolean
  pan: number
  tilt: number
  arms: Arms
}

interface Telemetry {
  x: number
  z: number
  heading: number
  speed: number
  odo: number
  state: string
  lift: number
}

const KEYMAP: Record<string, string> = {
  w: 'fwd',
  arrowup: 'fwd',
  s: 'back',
  arrowdown: 'back',
  a: 'left',
  arrowleft: 'left',
  d: 'right',
  arrowright: 'right',
}

export default function MobileRobot() {
  const hostRef = useRef<HTMLDivElement>(null)
  const keys = useRef(new Set<string>())
  const resetRef = useRef<() => void>(() => {})
  const stopRef = useRef<() => void>(() => {})
  const waveRef = useRef(false)
  const [settings, setSettings] = useState<Settings>({
    maxSpeed: 0.7,
    lift: 1.12,
    follow: true,
    autoHead: true,
    pan: 0,
    tilt: 0,
    arms: PRESETS[0].arms,
  })
  const [armSel, setArmSel] = useState<'both' | Side>('both')
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const [tele, setTele] = useState<Telemetry>({ x: 0, z: 0, heading: 0, speed: 0, odo: 0, state: 'Idle', lift: settings.lift })
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v }))
  const setJoint = (key: keyof ArmPose, v: number) =>
    setSettings((s) => {
      const arms = { L: { ...s.arms.L }, R: { ...s.arms.R } }
      for (const side of armSel === 'both' ? (['L', 'R'] as Side[]) : [armSel]) arms[side][key] = v
      return { ...s, arms }
    })
  const shownArm = settings.arms[armSel === 'both' ? 'R' : armSel]

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const w = buildWorld(host)

    const cloneArms = (a: Arms): Arms => ({ L: { ...a.L }, R: { ...a.R } })
    const sim = {
      x: 0,
      z: 0,
      yaw: 0,
      v: 0,
      wz: 0,
      lift: settingsRef.current.lift,
      odo: 0,
      goal: null as { x: number; z: number } | null,
      blockedFor: 0,
      arrivedAt: -10,
      lastInput: 0,
      pan: 0,
      tilt: 0,
      arms: cloneArms(settingsRef.current.arms),
      waveUntil: -1,
      trail: [] as THREE.Vector3[],
    }
    const prevPos = new THREE.Vector3()
    resetRef.current = () => {
      Object.assign(sim, { x: 0, z: 0, yaw: 0, v: 0, wz: 0, odo: 0, goal: null, trail: [] })
      w.camera.position.copy(CAM_OFFSET)
      w.controls.target.copy(CAM_LOOK)
      prevPos.set(0, 0, 0)
    }
    stopRef.current = () => {
      sim.goal = null
    }

    const resize = () => {
      w.renderer.setSize(host.clientWidth, host.clientHeight, false)
      w.camera.aspect = host.clientWidth / Math.max(1, host.clientHeight)
      w.camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(host)
    resize()

    // Click the floor (not an orbit drag) to send the robot there.
    let down: { x: number; y: number } | null = null
    const el = w.renderer.domElement
    const onDown = (e: PointerEvent) => (down = { x: e.clientX, y: e.clientY })
    const onUp = (e: PointerEvent) => {
      if (!down) return
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y)
      down = null
      if (moved > 5) return
      const rect = el.getBoundingClientRect()
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
      const ray = new THREE.Raycaster()
      ray.setFromCamera(ndc, w.camera)
      const hit = ray.intersectObject(w.floor)[0]
      if (!hit) return
      sim.goal = {
        x: clamp(hit.point.x, -BOUNDS + ROBOT_R, BOUNDS - ROBOT_R),
        z: clamp(hit.point.z, -BOUNDS + ROBOT_R, BOUNDS - ROBOT_R),
      }
    }
    el.addEventListener('pointerdown', onDown)
    el.addEventListener('pointerup', onUp)

    const isTyping = (t: EventTarget | null) => t instanceof HTMLElement && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return
      const k = e.key.toLowerCase()
      if (k === ' ' && e.type === 'keydown') {
        sim.goal = null
        e.preventDefault()
        return
      }
      const action = KEYMAP[k]
      if (!action) return
      e.preventDefault()
      if (e.type === 'keydown') keys.current.add(action)
      else keys.current.delete(action)
    }
    const onBlur = () => keys.current.clear()
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKey)
    window.addEventListener('blur', onBlur)

    const clock = new THREE.Clock()
    let raf = 0
    let teleT = 0

    const loop = () => {
      // Fixed physics substeps so the robot moves in real time even at a low frame rate.
      const dt = Math.min(0.25, clock.getDelta())
      const steps = Math.max(1, Math.ceil(dt / 0.02))
      const h = dt / steps
      const t = clock.elapsedTime
      const st = settingsRef.current
      const k = keys.current

      let state = 'Idle'
      let vT = 0
      let vPrev = sim.v
      for (let step = 0; step < steps; step++) {
        const throttle = (k.has('fwd') ? 1 : 0) - (k.has('back') ? 1 : 0)
        const steer = (k.has('left') ? 1 : 0) - (k.has('right') ? 1 : 0)
        let wT = 0
        state = 'Idle'
        vT = 0
        if (throttle || steer) {
          sim.goal = null
          sim.lastInput = t
          vT = throttle * st.maxSpeed
          wT = steer * MAX_TURN
          state = 'Manual'
        } else if (sim.goal) {
          const dx = sim.goal.x - sim.x
          const dz = sim.goal.z - sim.z
          const dist = Math.hypot(dx, dz)
          if (dist < 0.06) {
            sim.goal = null
            sim.arrivedAt = t
          } else {
            const err = wrap(Math.atan2(-dz, dx) - sim.yaw)
            wT = clamp(3 * err, -MAX_TURN, MAX_TURN)
            vT = Math.abs(err) > 0.6 ? 0 : Math.min(st.maxSpeed, dist * 1.4) * Math.cos(err)
            state = 'Driving to goal'
          }
          sim.lastInput = t
        }

        vPrev = sim.v
        sim.v = approach(sim.v, vT, ACCEL * h)
        sim.wz = approach(sim.wz, wT, TURN_ACCEL * h)
        sim.yaw = wrap(sim.yaw + sim.wz * h)
        const nx = sim.x + sim.v * Math.cos(sim.yaw) * h
        const nz = sim.z - sim.v * Math.sin(sim.yaw) * h
        if (blockedAt(nx, nz) && !blockedAt(sim.x, sim.z)) {
          sim.v = 0
          sim.blockedFor += h
        } else {
          sim.odo += Math.hypot(nx - sim.x, nz - sim.z)
          sim.x = nx
          sim.z = nz
          if (Math.abs(sim.v) > 0.02) sim.blockedFor = Math.max(0, sim.blockedFor - h)
        }
        if (sim.blockedFor > 0.05) state = 'Blocked'
        if (sim.goal && sim.blockedFor > 2) {
          sim.goal = null // give up; let the driver take over
          sim.blockedFor = 0
        }
        if (state === 'Idle' && t - sim.arrivedAt < 1.5) state = 'Arrived'

        // Wheels: each side covers a different distance in a turn.
        for (const wh of w.wheels) wh.phi += (sim.v * h + wh.s * sim.wz * h * WHEEL_Z) / WHEEL_R
        sim.lift = approach(sim.lift, st.lift, 0.3 * h)
        // Arms move to their target pose at a limited joint speed.
        for (const side of ['L', 'R'] as Side[]) {
          const cur = sim.arms[side]
          const tgt = st.arms[side]
          for (const key of Object.keys(cur) as (keyof ArmPose)[]) {
            cur[key] = approach(cur[key], tgt[key], (key === 'grip' ? 1.5 : JOINT_SPEED) * h)
          }
        }
      }

      // ---- Pose the robot ------------------------------------------------
      w.robot.position.set(sim.x, 0, sim.z)
      w.robot.rotation.y = sim.yaw
      w.wheels.forEach((wh) => (wh.spin.rotation.z = -wh.phi))
      w.carriage.position.y = sim.lift
      if (waveRef.current) {
        waveRef.current = false
        sim.waveUntil = t + 3.5
      }
      poseArm(w.arms.L, sim.arms.L)
      if (t < sim.waveUntil) {
        // Right arm out to the side, forearm waving.
        const a = sim.arms.R
        poseArm(w.arms.R, { ...a, pitch: 0.35, spread: 1.35, elbow: 1.0 + 0.45 * Math.sin((t - sim.waveUntil) * 9), wrist: 0, grip: 1 })
      } else poseArm(w.arms.R, sim.arms.R)

      // Head: look where it's going, glance around when idle.
      let panT = st.pan
      let tiltT = st.tilt
      if (st.autoHead) {
        if (sim.goal) {
          panT = clamp(wrap(Math.atan2(-(sim.goal.z - sim.z), sim.goal.x - sim.x) - sim.yaw), -1.3, 1.3)
          tiltT = -0.25
        } else if (t - sim.lastInput > 3) {
          panT = 0.9 * Math.sin(t * 0.45) * Math.sin(t * 0.17)
          tiltT = -0.1 + 0.12 * Math.sin(t * 0.3)
        } else {
          panT = clamp(sim.wz * 0.5, -0.9, 0.9)
          tiltT = -0.12
        }
      }
      sim.pan += (panT - sim.pan) * Math.min(1, dt * 5)
      sim.tilt += (tiltT - sim.tilt) * Math.min(1, dt * 5)
      w.head.pan.rotation.y = sim.pan
      w.head.tilt.rotation.z = sim.tilt
      w.head.ringMat.emissive.set(state === 'Blocked' ? '#f87171' : state === 'Arrived' ? '#34d399' : '#22d3ee')
      w.stripMat.emissiveIntensity = 1.0 + 0.4 * Math.sin(t * 2)
      const braking = Math.abs(sim.v) > 0.02 && Math.abs(sim.v) <= Math.abs(vPrev) && Math.abs(vT) < Math.abs(sim.v)
      w.brakeMats.forEach((m) => (m.emissiveIntensity = braking ? 2.2 : 0.5))

      // Goal marker and trail
      w.goal.visible = !!sim.goal
      if (sim.goal) {
        w.goal.position.set(sim.goal.x, 0, sim.goal.z)
        w.goal.rotation.y = t * 1.5
      }
      const last = sim.trail[sim.trail.length - 1]
      if (!last || Math.hypot(last.x - sim.x, last.z - sim.z) > 0.04) {
        sim.trail.push(new THREE.Vector3(sim.x, 0.006, sim.z))
        if (sim.trail.length > 600) sim.trail.shift()
        w.trail.geometry.dispose()
        w.trail.geometry = new THREE.BufferGeometry().setFromPoints(sim.trail)
      }

      // Camera follows the robot (you can still orbit around it).
      const pos = new THREE.Vector3(sim.x, 0, sim.z)
      if (st.follow) {
        const delta = pos.clone().sub(prevPos)
        w.camera.position.add(delta)
        w.controls.target.add(delta)
      }
      prevPos.copy(pos)
      w.sun.position.set(sim.x + 1.5, 4, sim.z + 1.8)
      w.sun.target.position.copy(pos)
      w.controls.update()
      w.renderer.render(w.scene, w.camera)

      teleT += dt
      if (teleT > 0.1) {
        teleT = 0
        setTele({ x: sim.x, z: sim.z, heading: deg(sim.yaw), speed: sim.v, odo: sim.odo, state, lift: sim.lift })
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      el.removeEventListener('pointerdown', onDown)
      el.removeEventListener('pointerup', onUp)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKey)
      window.removeEventListener('blur', onBlur)
      w.controls.dispose()
      w.scene.traverse((o) => {
        const m = o as THREE.Mesh
        m.geometry?.dispose?.()
        const material = m.material as THREE.Material | THREE.Material[] | undefined
        if (Array.isArray(material)) material.forEach((x) => x.dispose())
        else material?.dispose?.()
      })
      w.renderer.dispose()
      el.remove()
    }
  }, [])

  const hold = (action: string) => ({
    onPointerDown: (e: RPointerEvent) => {
      ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
      keys.current.add(action)
    },
    onPointerUp: () => keys.current.delete(action),
    onPointerCancel: () => keys.current.delete(action),
    onLostPointerCapture: () => keys.current.delete(action),
  })
  const padBtn = 'grid h-11 place-items-center rounded-lg bg-slate-800 text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700 active:bg-cyan-500 active:text-slate-950 select-none touch-none'
  const stateTone = tele.state === 'Blocked' ? 'bad' : tele.state === 'Arrived' ? 'good' : undefined

  return (
    <main className="mx-auto grid max-w-[1400px] gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="min-w-0 space-y-4">
        <Card className="overflow-hidden" title="Mobile manipulator (3D)">
          <div className="-m-4">
            <div className="relative aspect-[16/10] w-full">
              <div ref={hostRef} className="absolute inset-0 cursor-crosshair" />
              <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-slate-900/80 px-2.5 py-1.5 text-xs text-slate-300 ring-1 ring-slate-700">
                WASD or arrow keys to drive · click the floor to send it there · space to stop
              </div>
              <div className="absolute bottom-3 left-3 flex items-center gap-2 text-[11px] text-slate-400">
                <span className="rounded bg-slate-900/80 px-2 py-1 ring-1 ring-slate-700">Drag to orbit · scroll to zoom</span>
                <button
                  type="button"
                  onClick={() => resetRef.current()}
                  className="rounded bg-slate-800 px-2 py-1 text-slate-300 ring-1 ring-slate-700 hover:bg-slate-700"
                >
                  Reset robot
                </button>
              </div>
            </div>
          </div>
        </Card>
        <Card title="Telemetry">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="State" value={tele.state} tone={stateTone} />
            <Stat label="Speed" value={`${fmt(tele.speed, 2)} m/s`} />
            <Stat label="Heading" value={`${Math.round(tele.heading)}°`} />
            <Stat label="Odometer" value={`${fmt(tele.odo, 2)} m`} />
            <Stat label="Position x" value={`${fmt(tele.x, 2)} m`} />
            <Stat label="Position z" value={`${fmt(tele.z, 2)} m`} />
            <Stat label="Shoulder height" value={`${Math.round(tele.lift * 100)} cm`} />
            <Stat label="Overall height" value={`${Math.round((COLUMN.top + 0.24) * 100)} cm`} />
          </div>
        </Card>
      </div>

      <aside className="space-y-4">
        <Card title="Drive">
          <div className="mx-auto grid w-44 grid-cols-3 gap-1.5">
            <span />
            <button type="button" className={padBtn} title="Forward (W)" {...hold('fwd')}>
              <ArrowUp className="h-5 w-5" />
            </button>
            <span />
            <button type="button" className={padBtn} title="Turn left (A)" {...hold('left')}>
              <ArrowLeft className="h-5 w-5" />
            </button>
            <button type="button" className={padBtn} title="Stop (space)" onClick={() => stopRef.current()}>
              <Square className="h-4 w-4" />
            </button>
            <button type="button" className={padBtn} title="Turn right (D)" {...hold('right')}>
              <ArrowRight className="h-5 w-5" />
            </button>
            <span />
            <button type="button" className={padBtn} title="Reverse (S)" {...hold('back')}>
              <ArrowDown className="h-5 w-5" />
            </button>
            <span />
          </div>
          <div className="mt-4 space-y-4">
            <Slider label="Top speed" value={settings.maxSpeed} min={0.2} max={1.4} step={0.05} unit=" m/s" onChange={(v) => set('maxSpeed', v)} />
            <div>
              <div className="mb-1 text-sm text-slate-300">Camera</div>
              <Segmented
                size="sm"
                value={settings.follow ? 'follow' : 'free'}
                onChange={(v) => set('follow', v === 'follow')}
                options={[
                  { value: 'follow', label: 'Follow robot' },
                  { value: 'free', label: 'Fixed' },
                ]}
              />
            </div>
          </div>
        </Card>

        <Card title="Lift">
          <Slider
            label="Shoulder height"
            value={Math.round(settings.lift * 100)}
            min={Math.round(LIFT_MIN * 100)}
            max={Math.round(LIFT_MAX * 100)}
            unit=" cm"
            onChange={(v) => set('lift', v / 100)}
          />
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            The torso rides up and down the column, so the arms can work at table height or reach a high shelf. The base drives on two hidden
            wheels (differential drive) with casters front and back.
          </p>
        </Card>

        <Card title="Arms">
          <div className="grid grid-cols-3 gap-2">
            {PRESETS.map((p) => (
              <Button key={p.label} onClick={() => set('arms', { L: { ...p.arms.L }, R: { ...p.arms.R } })}>
                {p.label}
              </Button>
            ))}
            <Button onClick={() => (waveRef.current = true)} title="Wave with the right arm">
              <Hand className="h-4 w-4" /> Wave
            </Button>
          </div>
          <div className="mt-4">
            <Segmented
              size="sm"
              value={armSel}
              onChange={setArmSel}
              options={[
                { value: 'both', label: 'Both' },
                { value: 'L', label: 'Left arm' },
                { value: 'R', label: 'Right arm' },
              ]}
            />
          </div>
          <div className="mt-4 space-y-4">
            {JOINTS.map((j) => (
              <Slider
                key={j.key}
                label={j.label}
                value={Math.round(deg(shownArm[j.key]))}
                min={j.min}
                max={j.max}
                unit="°"
                color={j.color}
                onChange={(v) => setJoint(j.key, rad(v))}
              />
            ))}
            <Slider label="Gripper" value={Math.round(shownArm.grip * 100)} min={0} max={100} unit="% open" onChange={(v) => setJoint('grip', v / 100)} />
          </div>
        </Card>

        <Card title="Head">
          <Segmented
            size="sm"
            value={settings.autoHead ? 'auto' : 'manual'}
            onChange={(v) => set('autoHead', v === 'auto')}
            options={[
              { value: 'auto', label: 'Look around' },
              { value: 'manual', label: 'Manual' },
            ]}
          />
          <div className={clsx('mt-4 space-y-4', settings.autoHead && 'opacity-60')}>
            <Slider label="Pan" value={Math.round(deg(settings.pan))} min={-90} max={90} unit="°" disabled={settings.autoHead} onChange={(v) => set('pan', rad(v))} />
            <Slider label="Tilt" value={Math.round(deg(settings.tilt))} min={-40} max={30} unit="°" disabled={settings.autoHead} onChange={(v) => set('tilt', rad(v))} />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            The camera ring glows cyan while driving, green on arrival and red when something is in the way.
          </p>
        </Card>
      </aside>
    </main>
  )
}
