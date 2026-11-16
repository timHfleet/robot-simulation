import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Boxes, Crosshair, Hand, Layers, RotateCcw, Shirt, SlidersHorizontal, Square, Trash2, UtensilsCrossed, WashingMachine } from 'lucide-react'
import clsx from 'clsx'
import { approach, clamp, deg, fmt, rad, servoStep, wrap } from '../lib/math'
import {
  ARM,
  DIR_VECTORS,
  DOF,
  FORE,
  UPPER,
  WAIST,
  WAIST_X,
  fk,
  gravityTorques,
  jointAxis,
  jointOffset,
  mirrorPoint,
  solveIK,
  torsoBase,
  waistFor,
  type DirGoal,
  type IKResult,
  type Side,
  type V3,
} from '../lib/kinematics'
import { Button, Card, Segmented, Slider, Stat } from './ui'
import { DOCKS, ITEMS, KITCHEN, LAUNDRY, STAGING, doorAngleFor, foldAngleFor, furnitureRects, lidAngleFor, toLocal, toWorld, type StationId } from '../lib/stations'
import { handPath, keepOutsAll, keepOutsFor, keepOutsRound, pathSteps, toRobotGoal, waistForMove, type HandGoal } from '../lib/motion'
import { TASKS, planTask, type Act, type Mechanism, type PlanCtx, type TaskId } from '../lib/tasks'
import { ballOf, buildClutter, type Clutter } from './clutter'
import { DINING, DINING_FOOTPRINTS, DINING_STATICS, buildDining } from './dining'
import { BEDROOM_FOOTPRINTS, buildBedroom } from './bedroom'
import { PLAN_WALLS, HOUSE_FOOTPRINTS, LIVING_FOOTPRINTS, START, buildHouse, fadeWalls } from './house'
import { FloorGrid, planPath, pathLength, type Disc, type Pt } from '../lib/pathPlan'
import { buildStations, type Mech, type Stations } from './stationProps'
import type { Physics, PhysicsStats } from './physicsWorld'
import { createCloth, type Feature } from './clothSim'

// ---------------------------------------------------------------------------
// Geometry (metres). Robot frame: x forward, y up, z to the robot's right.
// ---------------------------------------------------------------------------
const BASE = { len: 0.62, w: 0.54, h: 0.24, y0: 0.06 }
const BASE_TOP = BASE.y0 + BASE.h + 0.02
const WHEEL_R = 0.075
const WHEEL_Z = 0.21
const COLUMN = { x: -0.07, d: 0.11, w: 0.15, top: 1.34 }
const LIFT_MIN = 0.78
const LIFT_MAX = 1.18
const LIFT_V = 0.15 // m/s
const LIFT_A = 0.5 // m/s²
const ROBOT_R = 0.42
const BOUNDS = 5.4
const ACCEL = 1.4
const TURN_ACCEL = 6
const MAX_TURN = 1.6
const SIDES: Side[] = ['L', 'R']
/** Cameras view: the share of the panel's height the head camera takes (the wrist cameras share the rest). */
/** Cameras view rows, top to bottom: head camera, wrist cameras, base cameras. */
const CAM_ROWS = [0.44, 0.31, 0.25]
/** How quickly arm moves reach full speed along their path (fraction of full speed per second). */
const PATH_ACCEL = 2.5

const CAM_OFFSET = new THREE.Vector3(1.5, 1.55, 2.5)
const CAM_LOOK = new THREE.Vector3(0, 0.8, 0)

const C = {
  white: '#eef1f5',
  gray: '#9aa3ae',
  motor: '#2f353d',
  alu: '#c3c9d0',
  dark: '#2a2f36',
  black: '#14171b',
  accent: '#f8fafc', // trim on joints, gripper, wheels and base
}

type Obstacle =
  | { kind: 'crate'; x: number; z: number; w: number; d: number; h: number; color: string }
  | { kind: 'barrel'; x: number; z: number; r: number; h: number; color: string }

/** Loose crates and barrels (none in the house now; the type stays for the physics and the hoops). */
const OBSTACLES: Obstacle[] = []
/** Furniture the base must keep clear of: the stations', plus the dining table and chairs. */
const FURNITURE_RECTS = [...furnitureRects(), ...DINING_FOOTPRINTS, ...BEDROOM_FOOTPRINTS, ...HOUSE_FOOTPRINTS, ...LIVING_FOOTPRINTS]
/** How far the base's corners reach from its centre: the clearance it needs to turn on the spot. */
const TURN_R = 0.47 // 0.44 to the corners, plus slack for the 5 cm grid
/** The dishwasher door lying open (with the racks out over it): kitchen a -0.30..0.42, |b| < 0.29. */
const DOOR_OPEN_RECT = (() => {
  const p = toWorld('kitchen', KITCHEN.door.hingeA - KITCHEN.door.height - 0.02, 0, -0.29)
  const q = toWorld('kitchen', KITCHEN.door.hingeA, 0, 0.29)
  return { x0: Math.min(p.x, q.x), x1: Math.max(p.x, q.x), z0: Math.min(p.z, q.z), z1: Math.max(p.z, q.z) }
})()
/** Is the dishwasher door open (then its footprint is an obstacle for the base)? Set every frame. */
let doorOpen = false
const extraRects = () => (doorOpen ? [DOOR_OPEN_RECT] : [])
/** Where the base's centre may go: clear of furniture, crates, barrels and the walls by TURN_R. */
const FLOOR_GRID = new FloorGrid(
  BOUNDS,
  [
    ...FURNITURE_RECTS,
    ...PLAN_WALLS,
    ...OBSTACLES.flatMap((o) => (o.kind === 'crate' ? [{ x0: o.x - o.w / 2, x1: o.x + o.w / 2, z0: o.z - o.d / 2, z1: o.z + o.d / 2 }] : [])),
  ],
  OBSTACLES.flatMap((o) => (o.kind === 'barrel' ? [{ x: o.x, z: o.z, r: o.r }] : [])),
  TURN_R,
  0.45,
)
/** The same with the dishwasher door open. */
const FLOOR_GRID_DOOR = new FloorGrid(
  BOUNDS,
  [
    ...FURNITURE_RECTS,
    ...PLAN_WALLS,
    DOOR_OPEN_RECT,
    ...OBSTACLES.flatMap((o) => (o.kind === 'crate' ? [{ x0: o.x - o.w / 2, x1: o.x + o.w / 2, z0: o.z - o.d / 2, z1: o.z + o.d / 2 }] : [])),
  ],
  OBSTACLES.flatMap((o) => (o.kind === 'barrel' ? [{ x: o.x, z: o.z, r: o.r }] : [])),
  TURN_R,
  0.45,
)
const grid = () => (doorOpen ? FLOOR_GRID_DOOR : FLOOR_GRID)
/** The base's footprint: front and back from the centre, and half its width. */
const FOOT = { front: 0.33, back: 0.33, half: 0.29 }

/** True when the base, centred at (x, z) and facing yaw, would overlap furniture, an obstacle or a wall. */
function blockedAt(x: number, z: number, yaw: number) {
  if (Math.abs(x) > BOUNDS - 0.45 || Math.abs(z) > BOUNDS - 0.45) return true
  const c = Math.cos(yaw)
  const sn = Math.sin(yaw)
  const pts: [number, number][] = []
  for (let i = 0; i <= 6; i++) {
    const u = -FOOT.back + ((FOOT.front + FOOT.back) * i) / 6
    pts.push([u, -FOOT.half], [u, FOOT.half])
  }
  for (let i = 1; i < 6; i++) {
    const v = -FOOT.half + (2 * FOOT.half * i) / 6
    pts.push([FOOT.front, v], [-FOOT.back, v], [0, v])
  }
  for (const [u, v] of pts) {
    const px = x + c * u + sn * v
    const pz = z - sn * u + c * v
    for (const r of FURNITURE_RECTS) if (px > r.x0 && px < r.x1 && pz > r.z0 && pz < r.z1) return true
    for (const r of extraRects()) if (px > r.x0 && px < r.x1 && pz > r.z0 && pz < r.z1) return true
    for (const o of OBSTACLES) {
      if (o.kind === 'crate') {
        if (Math.abs(px - o.x) < o.w / 2 && Math.abs(pz - o.z) < o.d / 2) return true
      } else if (Math.hypot(px - o.x, pz - o.z) < o.r) return true
    }
  }
  return false
}

/**
 * Is the space the arms take up with the torso lowered (to pick something off the floor)
 * clear of furniture and obstacles? It spans the base and both elbows, and reaches forward
 * to where the working hand comes down. Nothing taller than the floor may be in it.
 */
function armsClearAt(x: number, z: number, yaw: number) {
  const c = Math.cos(yaw)
  const sn = Math.sin(yaw)
  for (let u = -0.2; u <= 0.48 + 1e-9; u += 0.04)
    for (let v = -0.58; v <= 0.58 + 1e-9; v += 0.04) {
      const px = x + c * u + sn * v
      const pz = z - sn * u + c * v
      // (The open dishwasher door lies at floor level: arms working above it don't touch it.)
      for (const r of FURNITURE_RECTS) if (px > r.x0 && px < r.x1 && pz > r.z0 && pz < r.z1) return false
      for (const o of OBSTACLES) {
        if (o.kind === 'crate') {
          if (Math.abs(px - o.x) < o.w / 2 && Math.abs(pz - o.z) < o.d / 2) return false
        } else if (Math.hypot(px - o.x, pz - o.z) < o.r) return false
      }
    }
  return true
}

/** `square`: roll straight on at the current heading (no steering onto the point): it arrives square, a little off to the side at worst. */
type Seg = { kind: 'turn'; yaw: number } | { kind: 'line'; x: number; z: number; vmax: number; square?: boolean }

/**
 * Base cameras: a wide-angle camera pod on each face of the base, tilted down at the floor.
 * Position and facing are in the robot's frame (x forward, z to the robot's right).
 */
type BaseCamId = 'front' | 'rear' | 'left' | 'right'
const BASE_CAMS: { id: BaseCamId; label: string; pos: V3; yaw: number }[] = [
  { id: 'front', label: 'Front', pos: [BASE.len / 2 + 0.004, 0.27, 0], yaw: 0 },
  { id: 'left', label: 'Left', pos: [0.12, 0.27, -(BASE.w / 2 + 0.004)], yaw: Math.PI / 2 },
  { id: 'right', label: 'Right', pos: [0.12, 0.27, BASE.w / 2 + 0.004], yaw: -Math.PI / 2 },
  { id: 'rear', label: 'Rear', pos: [-(BASE.len / 2 + 0.004), 0.27, 0], yaw: Math.PI },
]
/** How far the base cameras gauge, and the distances at which the base slows and warns. */
const CLEAR = { max: 1.5, slow: 0.6, warn: 0.6, danger: 0.25 }
type Clearance = Record<BaseCamId, number>
const FAR: Clearance = { front: CLEAR.max, rear: CLEAR.max, left: CLEAR.max, right: CLEAR.max }

/**
 * What a base camera measures: how far the whole base could slide in that camera's direction
 * (forward, back, or sideways) before it touched furniture, an obstacle or a wall.
 */
function clearanceAt(x: number, z: number, yaw: number, id: BaseCamId) {
  const c = Math.cos(yaw)
  const sn = Math.sin(yaw)
  const [dx, dz] = id === 'front' ? [c, -sn] : id === 'rear' ? [-c, sn] : id === 'right' ? [sn, c] : [-sn, -c]
  if (blockedAt(x, z, yaw)) return 0
  let lo = 0
  let hi = -1
  for (let d = 0.03; d <= CLEAR.max + 1e-9; d += 0.03) {
    if (blockedAt(x + dx * d, z + dz * d, yaw)) {
      hi = d
      break
    }
    lo = d
  }
  if (hi < 0) return CLEAR.max
  for (let k = 0; k < 4; k++) {
    const m = (lo + hi) / 2
    if (blockedAt(x + dx * m, z + dz * m, yaw)) hi = m
    else lo = m
  }
  return lo
}
/** Top speed allowed with `d` metres of room ahead: full speed when clear, easing to a crawl. */
const speedForClearance = (d: number) => (d >= CLEAR.slow ? Infinity : Math.max(0.05, (d - 0.02) * 0.9))

// ---------------------------------------------------------------------------
// Poses (degrees, in ARM order: shoulder pitch, roll, twist, elbow, forearm twist, wrist pitch, wrist yaw)
// ---------------------------------------------------------------------------
const Q = (...d: number[]) => d.map(rad)
/** Arms hanging relaxed at the sides (where they start). */
const HANG = Q(0, 6, 0, 12, 0, 0, 0)
/** The IK's preferred posture (elbows bent, hands forward): it biases solutions, it isn't a button. */
const REST = Q(22, 12, 0, 78, 0, -10, 0)
/** Pose buttons. (No forearms-out "ready"/"carry" or T-pose: they read as a mantis or a mannequin.) */
const PRESETS: { label: string; q: number[]; grip: number }[] = [
  { label: 'Rest', q: HANG, grip: 0.2 },
  { label: 'Reach', q: Q(80, 8, 0, 6, 0, 0, 0), grip: 1 },
  { label: 'Arms up', q: Q(165, 22, 0, 8, 0, 0, 0), grip: 1 },
]
const WAVE = Q(0, 80, -90, 90, 0, 0, 0)

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

const MAT = {
  motor: mat(C.motor, { metalness: 0.55, roughness: 0.35 }),
  fin: mat('#3c434c', { metalness: 0.6, roughness: 0.3 }),
  alu: mat(C.alu, { metalness: 0.85, roughness: 0.25 }),
  bolt: mat('#6b7280', { metalness: 0.9, roughness: 0.3 }),
  white: mat(C.white, { roughness: 0.35 }),
  gray: mat(C.gray, { metalness: 0.35, roughness: 0.35 }),
  black: mat(C.black, { roughness: 0.55, metalness: 0.3 }),
  rubber: mat('#0b0d10', { roughness: 0.9, metalness: 0 }),
  accent: mat(C.accent, { roughness: 0.35, metalness: 0.15 }),
  cable: mat('#111318', { roughness: 0.7 }),
  dark: mat(C.dark, { metalness: 0.3, roughness: 0.5 }),
}

/**
 * A rotary actuator: finned motor housing on the joint axis, an aluminium output
 * flange with bolts, a white end cap and a status LED. Built along y, then turned
 * so its axis matches the joint's.
 */
function actuator(r: number, len: number, axis: THREE.Vector3, flange = 1) {
  const g = new THREE.Group()
  const inner = new THREE.Group()
  g.add(inner)
  const housing = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 36), MAT.motor)
  inner.add(housing)
  for (let i = 0; i < 3; i++) {
    const fin = new THREE.Mesh(new THREE.CylinderGeometry(r * 1.05, r * 1.05, len * 0.07, 36), MAT.fin)
    fin.position.y = -flange * len * (0.05 + i * 0.13)
    inner.add(fin)
  }
  const band = new THREE.Mesh(new THREE.CylinderGeometry(r * 1.015, r * 1.015, len * 0.1, 36), MAT.accent)
  band.position.y = flange * len * 0.3
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.9, r * 0.96, len * 0.12, 36), MAT.white)
  cap.position.y = -flange * (len / 2 + len * 0.05)
  const fl = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.82, r * 0.82, len * 0.1, 36), MAT.alu)
  fl.position.y = flange * (len / 2 + len * 0.05)
  inner.add(band, cap, fl)
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2
    const bolt = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.07, r * 0.07, len * 0.04, 8), MAT.bolt)
    bolt.position.set(Math.cos(a) * r * 0.6, flange * (len / 2 + len * 0.11), Math.sin(a) * r * 0.6)
    inner.add(bolt)
  }
  const ledMat = new THREE.MeshStandardMaterial({ color: '#bbf7d0', emissive: '#22c55e', emissiveIntensity: 1.6 })
  const led = new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.004, r * 0.09), 10, 8), ledMat)
  led.position.set(0, -flange * len * 0.18, r * 1.0)
  inner.add(led)
  // Turn the y-built actuator onto the joint axis.
  inner.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.clone().normalize())
  shadowed(g)
  return { group: g, ledMat }
}

interface ArmRig {
  side: Side
  /** One group per joint; group i turns about joint i's axis. */
  joints: THREE.Group[]
  axes: THREE.Vector3[]
  leds: THREE.MeshStandardMaterial[]
  fingers: [THREE.Group, THREE.Group]
  tcp: THREE.Mesh
  /** Wrist camera on the palm, looking down past the fingers at what the hand is about to grip. */
  cam: THREE.PerspectiveCamera
}

/** Build one arm from the joint table, with an actuator at every joint. */
function buildArm(side: Side, carriage: THREE.Group): ArmRig {
  const s = side === 'R' ? 1 : -1
  const joints: THREE.Group[] = []
  const axes: THREE.Vector3[] = []
  let parent: THREE.Object3D = carriage
  for (let i = 0; i < DOF; i++) {
    const g = new THREE.Group()
    g.position.copy(jointOffset(side, i))
    parent.add(g)
    joints.push(g)
    axes.push(jointAxis(side, i))
    parent = g
  }
  const [g0, g1, g2, g3, g4, g5, g6] = joints
  const leds: THREE.MeshStandardMaterial[] = []
  // Each actuator is fixed to the link before its joint, centred on the joint.
  const mount = (link: THREE.Object3D, i: number, r: number, len: number, flange = 1) => {
    const a = actuator(r, len, jointAxis(side, i), flange)
    a.group.position.copy(jointOffset(side, i))
    link.add(a.group)
    leds[i] = a.ledMat
    return a
  }
  // J0 shoulder pitch: in the torso yoke, flange facing out.
  mount(carriage, 0, 0.052, 0.055, s)
  // J1 shoulder roll: hangs off the pitch flange, axis forward.
  mount(g0, 1, 0.045, 0.095, 1)
  const shoulderCap = new THREE.Mesh(new THREE.SphereGeometry(0.06, 32, 20, 0, Math.PI * 2, 0, Math.PI / 2), MAT.white)
  shoulderCap.position.set(0, 0.012, s * 0.065)
  shoulderCap.scale.set(1.05, 0.75, 0.95)
  g0.add(shadowed(shoulderCap))
  // J2 upper-arm twist: below the roll actuator, axis along the arm.
  const bracket = rbox(0.06, 0.04, 0.06, 0.012, MAT.white)
  bracket.position.y = -0.04
  g1.add(bracket)
  mount(g1, 2, 0.04, 0.05, -1)
  // Upper arm shell, elbow clevis and a cable running down the back.
  const elbowY = -(UPPER - 0.075)
  const upper = rbox(0.075, 0.145, 0.075, 0.026, MAT.white)
  upper.position.y = -0.1
  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.039, 0.039, 0.016, 28), MAT.accent)
  band.position.y = -0.165
  g2.add(upper, shadowed(band))
  for (const z of [-0.051, 0.051]) {
    const plate = rbox(0.075, 0.095, 0.012, 0.004, MAT.gray)
    plate.position.set(0, elbowY + 0.022, z)
    g2.add(plate)
  }
  const cable1 = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.17, 8), MAT.cable)
  cable1.position.set(-0.044, -0.11, 0)
  g2.add(shadowed(cable1))
  // J3 elbow: between the clevis plates.
  mount(g2, 3, 0.04, 0.085, s)
  // Forearm: sleeve, J4 twist actuator, grey shell, wrist clevis.
  const sleeve = rbox(0.068, 0.05, 0.068, 0.02, MAT.white)
  sleeve.position.y = -0.03
  g3.add(sleeve)
  mount(g3, 4, 0.033, 0.045, -1)
  const wristY = -(FORE - 0.07)
  const fore = rbox(0.062, 0.14, 0.062, 0.022, MAT.gray)
  fore.position.y = -0.095
  g4.add(fore)
  for (const z of [-0.039, 0.039]) {
    const plate = rbox(0.05, 0.07, 0.01, 0.003, MAT.alu)
    plate.position.set(0, wristY + 0.018, z)
    g4.add(plate)
  }
  const cable2 = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.15, 8), MAT.cable)
  cable2.position.set(-0.036, -0.1, 0)
  g4.add(shadowed(cable2))
  // J5 wrist pitch, J6 wrist yaw.
  mount(g4, 5, 0.026, 0.066, s)
  const knuckle = rbox(0.04, 0.03, 0.04, 0.01, MAT.black)
  knuckle.position.y = -0.02
  g5.add(knuckle)
  mount(g5, 6, 0.021, 0.056, 1)

  // Gripper: cuff, palm and two fingers on pivots.
  const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.033, 0.02, 28), MAT.accent)
  cuff.position.y = -0.03
  const palm = rbox(0.05, 0.05, 0.09, 0.012, MAT.black)
  palm.position.y = -0.06
  g6.add(shadowed(cuff), palm)
  const fingers: THREE.Group[] = []
  for (const fz of [0.024, -0.024]) {
    const pivot = new THREE.Group()
    pivot.position.set(0, -0.085, fz)
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.054, 12), MAT.alu)
    pin.rotation.z = Math.PI / 2
    const link = rbox(0.024, 0.062, 0.012, 0.004, MAT.black)
    link.position.y = -0.031
    const pad = rbox(0.026, 0.02, 0.006, 0.002, MAT.rubber)
    pad.position.set(0, -0.052, -Math.sign(fz) * 0.008)
    const tip = rbox(0.026, 0.012, 0.014, 0.003, MAT.accent)
    tip.position.y = -0.064
    pivot.add(shadowed(pin), link, pad, tip)
    g6.add(pivot)
    fingers.push(pivot)
  }
  // Tool centre point marker (shown in IK mode).
  const tcp = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 8), new THREE.MeshBasicMaterial({ color: '#fbbf24', depthTest: false }))
  tcp.position.set(0, -0.14, 0)
  tcp.renderOrder = 5
  tcp.visible = false
  g6.add(tcp)
  // Wrist camera: a small lens on the back of the palm, aimed down between the fingertips
  // (fingers close left-right in its picture).
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.01, 16), MAT.black)
  lens.rotation.z = Math.PI / 2
  lens.position.set(0.032, -0.08, 0)
  g6.add(lens)
  const cam = new THREE.PerspectiveCamera(72, 4 / 3, 0.005, 4)
  const eye = new THREE.Vector3(0.046, -0.082, 0)
  cam.position.copy(eye)
  cam.quaternion.setFromRotationMatrix(new THREE.Matrix4().lookAt(eye, new THREE.Vector3(0, -0.165, 0), new THREE.Vector3(1, 0, 0)))
  g6.add(cam)
  return { side, joints, axes, leds, fingers: fingers as [THREE.Group, THREE.Group], tcp, cam }
}

function poseArm(arm: ArmRig, q: number[], grip: number) {
  arm.joints.forEach((g, i) => g.quaternion.setFromAxisAngle(arm.axes[i], q[i]))
  const a = -0.3 + grip * 0.86
  arm.fingers.forEach((f) => (f.rotation.x = -Math.sign(f.position.z) * a))
}

interface Head {
  pan: THREE.Group
  tilt: THREE.Group
  ringMat: THREE.MeshStandardMaterial
  cam: THREE.PerspectiveCamera
}

function buildHead(parent: THREE.Group): Head {
  const neck = new THREE.Group()
  neck.position.set(COLUMN.x + 0.02, COLUMN.top, 0)
  parent.add(neck)
  neck.add(actuator(0.034, 0.05, new THREE.Vector3(0, 1, 0), 1).group)
  neck.children[0].position.y = 0.03
  const pan = new THREE.Group()
  pan.position.y = 0.07
  neck.add(pan)
  const yoke = rbox(0.05, 0.03, 0.12, 0.01, MAT.dark)
  yoke.position.y = 0.0
  pan.add(yoke)
  const tiltAct = actuator(0.024, 0.04, new THREE.Vector3(0, 0, 1), 1).group
  tiltAct.position.set(0, 0.035, 0.07)
  pan.add(tiltAct)
  const tilt = new THREE.Group()
  tilt.position.y = 0.035
  pan.add(tilt)

  const shell = rbox(0.14, 0.11, 0.17, 0.035, mat(C.dark, { metalness: 0.3, roughness: 0.4 }))
  shell.position.set(0.015, 0.045, 0)
  const cap = rbox(0.13, 0.03, 0.16, 0.012, MAT.white)
  cap.position.set(0.01, 0.105, 0)
  tilt.add(shell, cap)
  const lensGroup = new THREE.Group()
  lensGroup.position.set(0.087, 0.045, 0)
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
    eye.position.set(0.086, 0.045, z)
    tilt.add(eye)
  }
  // The head camera, behind the lens, looking where the head looks.
  const cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.02, 20)
  cam.position.set(0.1, 0.045, 0)
  cam.rotation.y = -Math.PI / 2
  tilt.add(cam)
  return { pan, tilt, ringMat, cam }
}

function buildRobot() {
  const robot = new THREE.Group()
  const dark = mat(C.dark, { metalness: 0.3, roughness: 0.5 })

  // Base: rounded shell on a dark skirt, drive wheels hidden underneath.
  const skirt = rbox(BASE.len + 0.03, 0.08, BASE.w + 0.03, 0.03, dark)
  skirt.position.y = 0.075
  const shell = rbox(BASE.len, BASE.h, BASE.w, 0.07, MAT.white)
  shell.position.y = BASE.y0 + BASE.h / 2 + 0.03
  const lid = rbox(BASE.len - 0.1, 0.03, BASE.w - 0.1, 0.012, mat('#c7ccd3', { roughness: 0.4 }))
  lid.position.y = BASE_TOP - 0.01
  const plinth = rbox(0.28, 0.05, 0.3, 0.015, dark)
  plinth.position.set(COLUMN.x, BASE_TOP + 0.02, 0)
  robot.add(skirt, shell, lid, plinth)
  const stripMat = new THREE.MeshStandardMaterial({ color: '#e0f2fe', emissive: '#bae6fd', emissiveIntensity: 1.3 })
  const strip = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.012, 0.24), stripMat)
  strip.position.set(BASE.len / 2 + 0.002, 0.2, 0)
  robot.add(strip)
  for (const s of [-1, 1]) {
    const accent = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.012, 0.01), MAT.accent)
    accent.position.set(BASE.len / 2 - 0.12, 0.11, s * (BASE.w / 2 + 0.004))
    const vent = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.03, 0.006), MAT.black)
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
  const wheels: { spin: THREE.Group; s: number; phi: number }[] = []
  for (const s of [-1, 1]) {
    const spin = new THREE.Group()
    spin.position.set(0, WHEEL_R, s * WHEEL_Z)
    const tire = new THREE.Mesh(new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, 0.045, 32), MAT.rubber)
    tire.rotation.x = Math.PI / 2
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.05, 6), MAT.accent)
    hub.rotation.x = Math.PI / 2
    spin.add(shadowed(tire), hub)
    robot.add(spin)
    wheels.push({ spin, s, phi: 0 })
  }
  for (const x of [-0.22, 0.22]) {
    const caster = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), MAT.black)
    caster.position.set(x, 0.03, 0)
    robot.add(shadowed(caster))
  }

  // Waist: a turntable on the base. Everything above it (column, torso, arms,
  // head) turns about a vertical axis through the column.
  const waist = new THREE.Group()
  waist.position.set(WAIST_X, 0, 0)
  robot.add(waist)
  const upper = new THREE.Group()
  upper.position.x = -WAIST_X
  waist.add(upper)
  const bearing = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.012, 56), MAT.alu)
  bearing.position.set(WAIST_X, BASE_TOP + 0.05, 0)
  robot.add(shadowed(bearing))
  const waistMotor = actuator(0.12, 0.045, new THREE.Vector3(0, 1, 0), 1)
  waistMotor.group.position.set(0, BASE_TOP + 0.085, 0)
  waist.add(waistMotor.group)
  const mark = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.012, 0.012), MAT.accent)
  mark.position.set(0.125, BASE_TOP + 0.085, 0)
  waist.add(mark)

  // Lift column: black body, linear rails and a lead screw on the front, white back panel.
  const colH = COLUMN.top - BASE_TOP
  const column = rbox(COLUMN.d, colH, COLUMN.w, 0.012, mat(C.black, { roughness: 0.55, metalness: 0.35 }))
  column.position.set(COLUMN.x, BASE_TOP + colH / 2, 0)
  const back = rbox(0.03, colH - 0.04, COLUMN.w + 0.01, 0.01, MAT.white)
  back.position.set(COLUMN.x - COLUMN.d / 2 - 0.005, BASE_TOP + colH / 2, 0)
  for (const z of [-0.045, 0.045]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.008, colH - 0.08, 0.014), MAT.alu)
    rail.position.set(COLUMN.x + COLUMN.d / 2 + 0.004, BASE_TOP + colH / 2, z)
    upper.add(shadowed(rail))
  }
  const screw = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, colH - 0.08, 10), MAT.bolt)
  screw.position.set(COLUMN.x + COLUMN.d / 2 + 0.01, BASE_TOP + colH / 2, 0)
  const topCap = rbox(COLUMN.d + 0.03, 0.03, COLUMN.w + 0.03, 0.01, MAT.white)
  topCap.position.set(COLUMN.x - 0.008, COLUMN.top, 0)
  upper.add(column, back, shadowed(screw), topCap)

  // Torso carriage rides the rails and carries both shoulders.
  const carriage = new THREE.Group()
  upper.add(carriage)
  const torso = rbox(0.21, 0.19, 0.3, 0.05, MAT.white)
  torso.position.set(COLUMN.x + 0.03, 0.01, 0)
  const chest = rbox(0.02, 0.13, 0.17, 0.008, dark)
  chest.position.set(COLUMN.x + 0.14, 0.0, 0)
  const logo = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.012, 0.07), MAT.accent)
  logo.position.set(COLUMN.x + 0.152, 0.035, 0)
  const clampBlock = rbox(0.16, 0.06, COLUMN.w + 0.05, 0.015, dark)
  clampBlock.position.set(COLUMN.x - 0.02, -0.11, 0)
  carriage.add(torso, chest, logo, clampBlock)
  for (const s of [-1, 1]) {
    const yoke = rbox(0.12, 0.12, 0.07, 0.03, MAT.white)
    yoke.position.set(0, 0, s * 0.135)
    carriage.add(yoke)
  }
  const arms = { L: buildArm('L', carriage), R: buildArm('R', carriage) }
  const head = buildHead(upper)

  // Base cameras: a dark pod with a wide lens and a status ring on each face of the base,
  // looking out and down at the floor to gauge how close the nearest obstacle is.
  const lensMat = new THREE.MeshStandardMaterial({ color: '#0b1220', metalness: 0.9, roughness: 0.08 })
  const baseCams = BASE_CAMS.map((bc) => {
    const pod = new THREE.Group()
    pod.name = `baseCam_${bc.id}`
    pod.position.set(...bc.pos)
    pod.rotation.y = bc.yaw
    const body = rbox(0.022, 0.034, 0.062, 0.008, MAT.black)
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.0105, 0.012, 0.01, 24), lensMat)
    lens.rotation.z = Math.PI / 2
    lens.position.x = 0.014
    const ringMat = new THREE.MeshStandardMaterial({ color: '#cffafe', emissive: '#22d3ee', emissiveIntensity: 1.2 })
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.0125, 0.0018, 8, 28), ringMat)
    ring.rotation.y = Math.PI / 2
    ring.position.x = 0.0125
    const cam = new THREE.PerspectiveCamera(100, 16 / 10, 0.02, 6)
    cam.position.x = 0.02
    cam.rotation.order = 'YXZ'
    cam.rotation.y = -Math.PI / 2 // look along the pod's +x
    cam.rotation.x = -rad(28) // and down at the floor
    pod.add(body, lens, ring, cam)
    robot.add(pod)
    return { id: bc.id, cam, ringMat }
  })

  // Proximity zones on the floor around the base, one per camera: hidden while the way is clear,
  // amber when something is within 60 cm, red within 25 cm.
  const zones = BASE_CAMS.map((bc) => {
    const r0 = bc.id === 'front' || bc.id === 'rear' ? FOOT.front + 0.04 : FOOT.half + 0.04
    const span = bc.id === 'front' || bc.id === 'rear' ? rad(64) : rad(80)
    const m = new THREE.MeshBasicMaterial({ color: '#f59e0b', transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide })
    const mesh = new THREE.Mesh(new THREE.RingGeometry(r0, r0 + CLEAR.warn, 40, 1, -span / 2, span), m)
    // Ring angle 0 is local +x; the ring's +y becomes -z after laying it flat.
    mesh.rotation.x = -Math.PI / 2
    mesh.rotation.z = bc.yaw
    mesh.position.y = 0.006
    mesh.visible = false
    mesh.renderOrder = 2
    robot.add(mesh)
    return { id: bc.id, mesh }
  })
  return { robot, waist, waistLed: waistMotor.ledMat, carriage, wheels, arms, head, brakeMats, stripMat, baseCams, zones }
}

interface World {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  gizmo: TransformControls
  sun: THREE.DirectionalLight
  floor: THREE.Mesh
  pickables: THREE.Object3D[]
  robot: THREE.Group
  waist: THREE.Group
  waistLed: THREE.MeshStandardMaterial
  carriage: THREE.Group
  wheels: { spin: THREE.Group; s: number; phi: number }[]
  arms: Record<Side, ArmRig>
  head: Head
  stations: Stations
  clutter: Clutter
  targets: Record<Side, THREE.Group>
  reachLines: Record<Side, THREE.Line>
  goal: THREE.Group
  trail: THREE.Line
  brakeMats: THREE.MeshStandardMaterial[]
  stripMat: THREE.MeshStandardMaterial
  baseCams: { id: BaseCamId; cam: THREE.PerspectiveCamera; ringMat: THREE.MeshStandardMaterial }[]
  zones: { id: BaseCamId; mesh: THREE.Mesh }[]
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
  scene.fog = new THREE.Fog('#0b1120', 10, 24)

  const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 40)
  camera.position.set(START.x, 0, START.z).add(CAM_OFFSET)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.set(START.x, 0, START.z).add(CAM_LOOK)
  controls.enableDamping = true
  controls.maxPolarAngle = Math.PI * 0.48
  controls.minDistance = 0.6
  controls.maxDistance = 8
  controls.update()

  // Soft studio reflections so steel, glazed ceramics and plastics look like the real thing.
  const pmrem = new THREE.PMREMGenerator(renderer)
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
  scene.environmentIntensity = 0.45
  pmrem.dispose()
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.05
  scene.add(new THREE.HemisphereLight('#e0ecff', '#111827', 0.75))
  const sun = new THREE.DirectionalLight('#ffffff', 2.3)
  sun.castShadow = true
  sun.shadow.mapSize.set(2048, 2048)
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 12 })
  sun.shadow.bias = -0.0006
  scene.add(sun, sun.target)
  const fill = new THREE.DirectionalLight('#cbd5e1', 0.6)
  fill.position.set(-2, 2, -1.5)
  scene.add(fill)

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

  const pickables: THREE.Object3D[] = [floor]
  for (const o of OBSTACLES) {
    if (o.kind === 'crate') {
      const crate = rbox(o.w, o.h, o.d, 0.015, mat(o.color, { roughness: 0.85, metalness: 0 }))
      crate.position.set(o.x, o.h / 2, o.z)
      scene.add(crate)
      pickables.push(crate)
      const band = new THREE.Mesh(new THREE.BoxGeometry(o.w + 0.004, 0.04, o.d + 0.004), mat('#78350f', { roughness: 0.9 }))
      band.position.set(o.x, o.h * 0.7, o.z)
      scene.add(shadowed(band))
    } else {
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(o.r, o.r, o.h, 32), mat(o.color, { roughness: 0.5, metalness: 0.4 }))
      barrel.position.set(o.x, o.h / 2, o.z)
      scene.add(shadowed(barrel))
      pickables.push(barrel)
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
  const stations = buildStations(scene)
  pickables.push(...stations.pickables)
  const clutter = buildClutter(
    scene,
    OBSTACLES.map((o) => ({ x: o.x, z: o.z, r: o.kind === 'crate' ? Math.hypot(o.w, o.d) / 2 : o.r })),
  )
  buildHouse(scene)
  buildDining(scene)
  buildBedroom(scene)
  // The walls, the dining set and the living room and bedroom furniture are solid for loose things too (as their footprints).
  for (const f of [...HOUSE_FOOTPRINTS, ...DINING_STATICS, ...BEDROOM_FOOTPRINTS, ...LIVING_FOOTPRINTS]) clutter.statics.push({ kind: 'box', x: (f.x0 + f.x1) / 2, z: (f.z0 + f.z1) / 2, w: f.x1 - f.x0, d: f.z1 - f.z0, h: f.h })
  // Crates and barrels are solid for the litter too.
  for (const o of OBSTACLES)
    clutter.statics.push(o.kind === 'crate' ? { kind: 'box', x: o.x, z: o.z, w: o.w, d: o.d, h: o.h } : { kind: 'cyl', x: o.x, z: o.z, r: o.r, h: o.h })

  // IK hand targets: children of the robot, so they ride along when it drives.
  const mkTarget = (color: string) => {
    const g = new THREE.Group()
    const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthTest: false })
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.018, 20, 14), m)
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.035, 0.003, 8, 40), m)
    ring.rotation.x = Math.PI / 2
    ball.renderOrder = ring.renderOrder = 6
    g.add(ball, ring)
    g.visible = false
    parts.robot.add(g)
    return g
  }
  const targets = { L: mkTarget('#38bdf8'), R: mkTarget('#34d399') }
  const mkLine = () => {
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineDashedMaterial({ color: '#f87171', dashSize: 0.015, gapSize: 0.01, depthTest: false }))
    l.renderOrder = 6
    l.frustumCulled = false
    l.visible = false
    parts.robot.add(l)
    return l
  }
  const reachLines = { L: mkLine(), R: mkLine() }

  const gizmo = new TransformControls(camera, renderer.domElement)
  gizmo.setSize(0.65)
  gizmo.addEventListener('dragging-changed', (e) => (controls.enabled = !(e as unknown as { value: boolean }).value))
  scene.add(gizmo.getHelper())

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

  return { renderer, scene, camera, controls, gizmo, sun, floor, pickables, goal, trail, targets, reachLines, stations, clutter, ...parts }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
type ArmMode = 'joint' | 'ik'
type ArmSel = 'both' | Side

interface Settings {
  maxSpeed: number
  lift: number
  follow: boolean
  autoHead: boolean
  /** What the 3D panel shows: the scene, or the robot's own cameras. */
  view: 'scene' | 'cameras'
  pan: number
  tilt: number
  /** Joint-mode targets. */
  arms: Record<Side, number[]>
  grip: Record<Side, number>
  armMode: ArmMode
  armSel: ArmSel
  /** IK hand targets in the robot frame (m). With "both", the left one mirrors the right. */
  targets: Record<Side, V3>
  dirGoal: DirGoal
  payload: number
  /** Simulation speed while running tasks. */
  speed: number
  /** Waist angle set by the slider (joint mode). */
  waist: number
}

interface ArmTele {
  q: number[]
  v: number[]
  tau: number[]
  tcp: V3
  dir: V3
}

interface Telemetry {
  x: number
  z: number
  heading: number
  speed: number
  odo: number
  state: string
  lift: number
  waist: number
  arms: Record<Side, ArmTele>
  ik: Record<Side, IKResult | null>
  task: { id: TaskId; label: string; step: number; total: number; text: string } | null
  log: string[]
  physics: PhysicsStats | null
  clear: Clearance
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

const activeSides = (sel: ArmSel): Side[] => (sel === 'both' ? SIDES : [sel])
const effTargets = (st: Settings): Record<Side, V3> => ({
  R: st.targets.R,
  L: st.armSel === 'both' ? mirrorPoint('L', st.targets.R) : st.targets.L,
})
const emptyArm = (): ArmTele => ({ q: new Array(DOF).fill(0), v: new Array(DOF).fill(0), tau: new Array(DOF).fill(0), tcp: [0, 0, 0], dir: [0, -1, 0] })

export default function Humanoid() {
  const hostRef = useRef<HTMLDivElement>(null)
  const keys = useRef(new Set<string>())
  const resetRef = useRef<() => void>(() => {})
  const stopRef = useRef<() => void>(() => {})
  const syncRef = useRef<() => Record<Side, number[]>>(() => ({ L: REST, R: REST }))
  const waistRef = useRef<() => number>(() => 0)
  const waveRef = useRef(false)
  const taskRef = useRef<{ start: (id: TaskId) => void; stop: () => void; resetScene: () => void }>({ start: () => {}, stop: () => {}, resetScene: () => {} })
  const [settings, setSettings] = useState<Settings>({
    maxSpeed: 0.7,
    lift: 1.12,
    follow: true,
    autoHead: true,
    view: 'scene',
    pan: 0,
    tilt: 0,
    // Idle: arms hanging straight down at the sides.
    arms: { L: [...HANG], R: [...HANG] },
    grip: { L: 0.6, R: 0.6 },
    armMode: 'joint',
    armSel: 'both',
    targets: { R: [0.45, 0.92, 0.27], L: [0.45, 0.92, -0.27] },
    dirGoal: 'free',
    payload: 0,
    speed: 2,
    waist: 0,
  })
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const [infoTab, setInfoTab] = useState<'joints' | 'telemetry'>('joints')
  const [tele, setTele] = useState<Telemetry>({
    x: 0,
    z: 0,
    heading: 0,
    speed: 0,
    odo: 0,
    state: 'Idle',
    lift: settings.lift,
    waist: 0,
    arms: { L: emptyArm(), R: emptyArm() },
    ik: { L: null, R: null },
    task: null,
    physics: null,
    log: [],
    clear: FAR,
  })
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v }))
  const shownSide: Side = settings.armSel === 'both' ? 'R' : settings.armSel

  const setJoint = (i: number, v: number) =>
    setSettings((s) => {
      const arms = { L: [...s.arms.L], R: [...s.arms.R] }
      for (const side of activeSides(s.armSel)) arms[side][i] = v
      return { ...s, arms }
    })
  const setGrip = (v: number) =>
    setSettings((s) => {
      const grip = { ...s.grip }
      for (const side of activeSides(s.armSel)) grip[side] = v
      return { ...s, grip }
    })
  const setTarget = (axis: 0 | 1 | 2, v: number) =>
    setSettings((s) => {
      const side = s.armSel === 'L' ? 'L' : 'R'
      const t = [...s.targets[side]] as V3
      t[axis] = v
      return { ...s, targets: { ...s.targets, [side]: t } }
    })
  const setMode = (m: ArmMode) =>
    setSettings((s) => {
      // Going back to joint control: start from wherever IK left the arms.
      if (m === 'joint' && s.armMode === 'ik') return { ...s, armMode: m, arms: syncRef.current(), waist: waistRef.current() }
      return { ...s, armMode: m }
    })

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const w = buildWorld(host)
    const st0 = settingsRef.current
    /** Rigid-body physics for items, the robot's links and the stations' moving parts (loads asynchronously). */
    let phys: Physics | null = null
    let physFailed = false
    let disposed = false
    /** Build the physics world from the scene as it stands (items stay where they are). */
    // Loaded on demand: the engine is a large WebAssembly module, so the scene shows first.
    const startPhysics = () =>
      import('./physicsWorld')
        .then(({ createPhysics }) => createPhysics(w.scene, w.stations, { robot: w.robot, waist: w.waist, carriage: w.carriage, arms: w.arms }, w.clutter))
        .then((p) => {
          if (disposed) return p.dispose()
          phys = p
          if (import.meta.env.DEV) (window as unknown as { __physics: Physics }).__physics = p
        })
    startPhysics()
    // The towel is simulated cloth: the hands pinch it and it drapes and folds under gravity.
    const cloth = createCloth(w.stations.parts.cloth)
    const clothFrame = () => cloth.tune.maxSteps / 120

    // ---- Towels: a basket of them on the table. The top one is simulated cloth; the ones under
    // it, and the folded ones on the stack, are still meshes (copies of a cloth shape).
    const TOWEL_COLORS = ['#3f8fb0', '#c9734a', '#7aa36a', '#d9b44a', '#9a7fc2']
    const clothMesh = w.stations.parts.cloth.mesh
    const laundryG = w.stations.groups.laundry
    const T0 = LAUNDRY.table.h
    const tw = { left: 0, basket: [] as THREE.Mesh[], stack: [] as THREE.Mesh[], layer: 0.03 }
    const towelMesh = (verts: Float32Array, color: string) => {
      const g = clothMesh.geometry.clone()
      g.setAttribute('position', new THREE.BufferAttribute(verts, 3))
      g.computeVertexNormals()
      const m = new THREE.Mesh(g, (clothMesh.material as THREE.MeshPhysicalMaterial).clone())
      ;(m.material as THREE.MeshPhysicalMaterial).color.set(color)
      m.castShadow = m.receiveShadow = true
      laundryG.add(m)
      return m
    }
    const dropMesh = (m: THREE.Mesh) => {
      laundryG.remove(m)
      m.geometry.dispose()
      ;(m.material as THREE.Material).dispose()
    }
    const setClothColor = (c: string) => (clothMesh.material as THREE.MeshPhysicalMaterial).color.set(c)
    const fillBasket = () => {
      for (const m of [...tw.basket, ...tw.stack]) dropMesh(m)
      tw.basket = []
      tw.stack = []
      cloth.clearStatic()
      tw.left = TOWEL_COLORS.length
      // One heap in the bottom of the basket, copied (turned) for the towels under the top one.
      cloth.setBasketLift(0)
      cloth.reset()
      const shape = cloth.snapshot()
      let lo = Infinity
      let hi = -Infinity
      for (let i = 1; i < shape.length; i += 3) {
        lo = Math.min(lo, shape[i])
        hi = Math.max(hi, shape[i])
      }
      tw.layer = Math.max(0.02, (hi - lo) * 0.5)
      const B = LAUNDRY.basket
      // Copies are turned about the heap's own middle, so they stay inside the basket.
      let ca = 0
      let cb = 0
      for (let k = 0; k < shape.length; k += 3) {
        ca += shape[k]
        cb += shape[k + 2]
      }
      ca /= shape.length / 3
      cb /= shape.length / 3
      for (let i = 0; i < tw.left - 1; i++) {
        const v = new Float32Array(shape)
        const ang = [Math.PI, 0.5, -0.5, Math.PI + 0.4][i % 4]
        const c = Math.cos(ang)
        const sn = Math.sin(ang)
        for (let k = 0; k < v.length; k += 3) {
          const x = v[k] - ca
          const z = v[k + 2] - cb
          v[k] = ca + x * c - z * sn
          v[k + 2] = cb + x * sn + z * c
          v[k + 1] += i * tw.layer
          // Keep every copy inside the basket walls (turning a heap can swing part of it out).
          v[k] = Math.min(B.a1 - B.wall - 0.006, Math.max(B.a0 + B.wall + 0.006, v[k]))
          v[k + 2] = Math.min(B.b1 - B.wall - 0.006, Math.max(B.b0 + B.wall + 0.006, v[k + 2]))
        }
        // Bottom of the basket first; the last towel in this list sits just under the top one.
        tw.basket.push(towelMesh(v, TOWEL_COLORS[tw.left - 1 - i]))
      }
      const lift = (tw.left - 1) * tw.layer
      cloth.lift(lift)
      cloth.setBasketLift(lift)
      setClothColor(TOWEL_COLORS[0])
      clothMesh.visible = true
    }
    /** The folded towel the hand holds goes onto the stack; the next towel in the basket becomes the cloth. */
    const stackTowel = () => {
      // The towel has been let go over the stack and has settled there under physics: keep it
      // exactly as it landed (a still copy the next towels can land on), and ready the next one.
      const v = cloth.snapshot()
      cloth.addStatic(v)
      const done = TOWEL_COLORS.length - tw.left
      tw.stack.push(towelMesh(v, TOWEL_COLORS[done]))
      tw.left--
      if (tw.left > 0) {
        const m = tw.basket.pop()
        if (m) dropMesh(m)
        const lift = (tw.left - 1) * tw.layer
        cloth.setBasketLift(lift)
        cloth.reset()
        setClothColor(TOWEL_COLORS[done + 1])
        say(`${tw.left} towel${tw.left > 1 ? 's' : ''} left in the basket`)
      } else {
        clothMesh.visible = false
        say('The basket is empty')
      }
    }
    fillBasket()
    w.stations.useClothPhysics()
    if (import.meta.env.DEV) Object.assign(window, { __world: w, __cloth: cloth })
    /** Items let go of, to check where they ended up once they settle. */
    let landing: { id: string; where: string; at: number; container: string; pos: [number, number, number] }[] = []
    /** Towels (cloth) each hand has pinched. */
    const clothHeld: Partial<Record<Side, string>> = {}
    /** Dishes put in a rack this task, with the slot they were meant for, to check again at the end. */
    let racked: { id: string; container: string; pos: [number, number, number] }[] = []

    const sim = {
      x: START.x,
      z: START.z,
      yaw: 0,
      v: 0,
      wz: 0,
      lift: st0.lift,
      liftV: 0,
      waist: 0,
      waistV: 0,
      waistT: 0,
      odo: 0,
      goal: null as { x: number; z: number } | null,
      blockedFor: 0,
      arrivedAt: -10,
      lastInput: 0,
      pan: 0,
      tilt: 0,
      panV: 0,
      tiltV: 0,
      q: { L: [...st0.arms.L], R: [...st0.arms.R] } as Record<Side, number[]>,
      qv: { L: new Array(DOF).fill(0), R: new Array(DOF).fill(0) } as Record<Side, number[]>,
      qT: { L: [...st0.arms.L], R: [...st0.arms.R] } as Record<Side, number[]>,
      grip: { ...st0.grip } as Record<Side, number>,
      ik: { L: null, R: null } as Record<Side, IKResult | null>,
      ikKey: '',
      waveUntil: -1,
      tau: { L: new Array(DOF).fill(0), R: new Array(DOF).fill(0) } as Record<Side, number[]>,
      trail: [] as THREE.Vector3[],
      // Task runner
      time: 0,
      path: [] as Seg[],
      blockedT: 0,
      clear: { ...FAR } as Clearance,
      clearT: 0,
      task: null as null | { id: TaskId; label: string; acts: Act[]; i: number; started: boolean; text: string },
      log: [] as string[],
      gripT: { ...st0.grip } as Record<Side, number>,
      liftT: null as number | null,
      slow: { L: 1, R: 1 } as Record<Side, number>,
      binds: {} as Partial<Record<Side, { mech: Mechanism; a0: number; out0: number }>>,
      waitUntil: 0,
      anims: [] as { key: 'door' | 'lowerRack' | 'upperRack' | 'lid'; from: number; to: number; t: number; dur: number }[],
      look: null as THREE.Vector3 | null,
      /** The move being executed: joint targets for each small step of the hand's path. */
      /**
       * The move being executed, as a trajectory: joint angles at each small step of the hand's
       * path, how long each step takes at full joint speed, and how far along it the arm is.
       */
      mq: null as null | {
        K: number
        qs: Partial<Record<Side, number[][]>>
        ws: number[]
        q0: Record<Side, number[]>
        w0: number
        /** Duration of each step at full speed (s), and their running total. */
        dur: number[]
        cum: number[]
        /** Progress along the move, in seconds-at-full-speed. */
        s: number
        via: boolean
      },
      /** Speed along the current path, 0 to 1 of full joint speed (carried through via moves). */
      pathSpeed: 0,
      /** Per-joint speed scale so all joints of an arm start and finish a move together. */
      vscale: { L: new Array(DOF).fill(1), R: new Array(DOF).fill(1) } as Record<Side, number[]>,
    }
    const prevPos = new THREE.Vector3(START.x, 0, START.z)
    resetRef.current = () => {
      if (sim.task) endTask('Stopped')
      Object.assign(sim, { x: START.x, z: START.z, yaw: 0, v: 0, wz: 0, odo: 0, goal: null, trail: [], path: [] })
      w.camera.position.set(START.x, 0, START.z).add(CAM_OFFSET)
      w.controls.target.set(START.x, 0, START.z).add(CAM_LOOK)
      prevPos.set(START.x, 0, START.z)
    }
    stopRef.current = () => {
      sim.goal = null
    }
    syncRef.current = () => ({ L: [...sim.qT.L], R: [...sim.qT.R] })
    waistRef.current = () => sim.waistT

    // ---------------------------------------------------------------- task runner
    const S = w.stations
    const clutterOf = (id: string) => w.clutter.props.find((pr) => pr.id === id)
    /** The 3D object for a station item or a piece of floor clutter. */
    const objOf = (id: string) => S.items.get(id) ?? clutterOf(id)?.obj
    const itemName = (id: string) => {
      const pr = clutterOf(id)
      if (pr) return pr.kind === 'trash' ? 'piece of rubbish' : pr.kind === 'dinnerplate' ? 'plate' : pr.kind
      const def = ITEMS.find((d) => d.id === id)
      if (!def) return id
      return def.kind === 'block' ? `${def.color} block` : def.kind === 'cup' ? 'mug' : def.kind
    }
    /**
     * Is a dish really where the task put it? Upright (plates on edge, mugs standing) and
     * near its slot in the rack (3 cm; 5 cm for a mug on the flat wires). Returns what went wrong, or null.
     */
    const rackProblem = (id: string, container: string, pos: [number, number, number]): string | null => {
      const o = objOf(id)
      const rack = S.containers[container]
      if (!o || !rack) return null
      o.updateMatrixWorld()
      rack.updateMatrixWorld()
      const local = o.getWorldPosition(new THREE.Vector3()).applyMatrix4(rack.matrixWorld.clone().invert())
      const e = o.matrixWorld.elements
      const kind = ITEMS.find((d) => d.id === id)?.kind ?? (clutterOf(id)?.kind === 'mug' ? 'cup' : undefined)
      // Plates stand with their axis (local z) level; mugs with their axis (local y) vertical.
      const tilt = kind === 'plate' ? Math.asin(Math.min(1, Math.abs(e[9]))) : Math.acos(Math.min(1, Math.abs(e[5])))
      if (tilt > THREE.MathUtils.degToRad(25)) return `tipped over in the rack (${Math.round(THREE.MathUtils.radToDeg(tilt))}° off upright)`
      const off = Math.hypot(local.x - pos[0], local.z - pos[2])
      // Plates and cups go in slots (channels, posts); a mug from the floor stands on the flat
      // wires, where a few centimetres either way is still a good place for it.
      const tol = clutterOf(id)?.kind === 'mug' ? 0.05 : 0.03
      if (off > tol) return `landed ${Math.round(off * 100)} cm from its slot`
      return null
    }
    /** At the end of a task, check every dish again: racks sliding in can still knock one over. */
    const reportRack = () => {
      if (!racked.length) return
      const bad: string[] = []
      for (const r of racked) {
        if (S.where.get(r.id) === 'floor') continue
        const why = rackProblem(r.id, r.container, r.pos)
        if (why) {
          S.where.set(r.id, 'misplaced')
          bad.push(`${itemName(r.id)} ${why}`)
        } else S.where.set(r.id, 'rack')
      }
      const ok = racked.filter((r) => S.where.get(r.id) === 'rack').length
      say(bad.length ? `Loaded ${ok} of ${racked.length} upright. Check: ${bad.join('; ')}` : `All ${racked.length} dishes loaded upright in their slots`)
      racked = []
    }
    /** The towel's left or right end as it lies (its furthest point along b), at its middle depth. */
    const clothEdge = (sign: number) => {
      const P = cloth.particles()
      let best = -1
      let cx = 0
      for (let i = 0; i < P.length; i += 3) cx += P[i]
      cx /= P.length / 3
      for (let i = 0; i < P.length; i += 3) {
        if (Math.abs(P[i] - cx) > 0.04) continue
        if (best < 0 || sign * P[i + 2] > sign * P[best + 2]) best = i
      }
      return new THREE.Vector3(P[best], cloth.top().y, P[best + 2])
    }
    const say = (text: string) => {
      sim.log = [...sim.log.slice(-7), text]
    }
    const planCtx = (): PlanCtx => ({
      item: (id) => {
        const o = objOf(id)!
        o.updateWorldMatrix(true, false)
        return { pos: o.getWorldPosition(new THREE.Vector3()), quat: o.getWorldQuaternion(new THREE.Quaternion()), where: S.where.get(id) ?? (clutterOf(id) ? 'floor' : 'home') }
      },
      ids: (kind) => {
        const own = ITEMS.filter((d) => d.kind === kind).map((d) => d.id)
        return own.length ? own : w.clutter.props.filter((pr) => pr.kind === kind).map((pr) => pr.id)
      },
      shape: (id) => clutterOf(id)?.shape ?? null,
      kindOf: (id) => clutterOf(id)?.kind ?? null,
      held: () => ({ ...(phys ? phys.stats().held : {}), ...clothHeld }),
      tcp: (side) => tcpWorld(side),
      closeAxis: (side) => new THREE.Vector3(0, 0, 1).applyQuaternion(w.arms[side].tcp.getWorldQuaternion(new THREE.Quaternion())),
      robot: () => ({ x: sim.x, z: sim.z, yaw: sim.yaw }),
      free: (x, z, yaw) => !blockedAt(x, z, yaw),
      workClear: (x, z, yaw) => armsClearAt(x, z, yaw),
      driveLength: (x, z, yaw, except) => gotoLength(x, z, yaw, except),
      bin: w.clutter.bin,
      hamper: w.clutter.hamper,
      cloth: (f) => {
        const p =
          f === 'top'
            ? cloth.top()
            : f === 'centre'
              ? cloth.centroid()
              : f === 'edgeL' || f === 'edgeR'
                ? clothEdge(f === 'edgeL' ? -1 : 1)
                : f === 'stack'
                ? new THREE.Vector3(LAUNDRY.stack.a, cloth.staticHeight(LAUNDRY.stack.a, LAUNDRY.stack.b), LAUNDRY.stack.b)
                : cloth.feature(f as Feature)
        return toWorld('laundry', p.x, p.y, p.z)
      },
    })
    const stagingOf = (st: StationId) => {
      const d = DOCKS[st]
      return { x: d.x - Math.cos(d.yaw) * STAGING, z: d.z + Math.sin(d.yaw) * STAGING }
    }
    /** Loose things on the floor the base should steer round (not held, not put away), as discs. */
    const floorLitter = (except: string[] = []): Disc[] => {
      if (!phys) return []
      const out: Disc[] = []
      for (const pr of w.clutter.props) {
        if (except.includes(pr.id)) continue
        const where = S.where.get(pr.id) ?? 'floor'
        if (where !== 'floor') continue
        const tw = w.clutter.towels.get(pr.id)
        if (tw) {
          // A towel covers a patch of floor round its middle.
          const c = tw.centroid()
          out.push({ x: c.x, z: c.z, r: Math.max(0.1, tw.spread() * 0.85) })
          continue
        }
        const p = phys.state(pr.id).pos
        if (p.y > 0.25) continue
        const sh = pr.shape
        const r = sh.kind === 'ball' ? sh.r : sh.kind === 'cyl' ? Math.max(sh.r, sh.h / 2) : Math.hypot(sh.hx, sh.hz)
        out.push({ x: p.x, z: p.z, r })
      }
      return out
    }
    /**
     * Parked where there's no room to turn (next to furniture)? Back straight out (or roll
     * forward) along the current heading until the base can turn freely. Returns the move and
     * where it ends, or null if there's no need (or no way).
     */
    const escapeSeg = (soft: Disc[] = floorLitter()): { seg: Seg; at: Pt } | null => {
      // Room to turn here: clear of furniture, and of litter (the corners sweep round).
      const turnable = (px: number, pz: number) => !grid().isHard(px, pz) && soft.every((d) => Math.hypot(d.x - px, d.z - pz) >= 0.46 + d.r)
      if (turnable(sim.x, sim.z)) return null
      const fx = Math.cos(sim.yaw)
      const fz = -Math.sin(sim.yaw)
      let fallback: { seg: Seg; at: Pt } | null = null
      for (const dir of [-1, 1]) {
        let dirty = false
        for (let d = 0.05; d <= 1.2; d += 0.05) {
          const x = sim.x + dir * fx * d
          const z = sim.z + dir * fz * d
          if (blockedAt(x, z, sim.yaw)) break
          if (litterUnder(x, z, sim.yaw, soft)) dirty = true
          if (turnable(x, z)) {
            const out = { seg: { kind: 'line' as const, x, z, vmax: 0.25 }, at: { x, z } }
            if (!dirty) return out // the way with no litter in it
            fallback ??= out
            break
          }
        }
      }
      return fallback
    }
    /** Would the base, standing here, be on top of any of these loose things? */
    const litterUnder = (x: number, z: number, yaw: number, soft: Disc[], margin = 0.05) => {
      const c = Math.cos(yaw)
      const sn = Math.sin(yaw)
      for (const d of soft) {
        const dx = d.x - x
        const dz = d.z - z
        const u = dx * c - dz * sn // ahead
        const v = dx * sn + dz * c // to the right
        if (u > -FOOT.back - d.r - margin && u < FOOT.front + d.r + margin && Math.abs(v) < FOOT.half + d.r + margin) return true
      }
      return false
    }
    /** Turn-and-drive segments along planned waypoints. */
    const segsAlong = (from: Pt, pts: Pt[], vmax = 0.7): Seg[] => {
      const segs: Seg[] = []
      let c = from
      for (const p of pts) {
        segs.push({ kind: 'turn', yaw: Math.atan2(-(p.z - c.z), p.x - c.x) }, { kind: 'line', x: p.x, z: p.z, vmax })
        c = p
      }
      return segs
    }
    /**
     * Drive to (x, z) and end up facing `yaw`, steering round furniture and litter.
     * Spots right next to furniture (a bin, a counter) are inside the planner's turning margin,
     * so it plans to a point backed off along the final heading, then rolls straight in.
     * Returns null if there's no way there.
     */
    const planGoto = (x: number, z: number, yaw: number, except: string[] = [], strict = true): Seg[] | null => {
      // The spot must hold with a little heading error too (driving straight in, the base steers a touch).
      if (blockedAt(x, z, yaw) || blockedAt(x, z, yaw + 0.05) || blockedAt(x, z, yaw - 0.05)) return null
      const soft = floorLitter() // the route steers round everything, the fetched piece too
      const others = floorLitter(except) // only the final spot may be right beside the fetched piece
      const fx = Math.cos(yaw)
      const fz = -Math.sin(yaw)
      // Where to finish turning: backed off along the final heading until the base can turn freely
      // there (clear of furniture, and of litter so the sweeping corners touch nothing), then it
      // rolls straight in.
      const turnable = (px: number, pz: number) => !grid().isHard(px, pz) && soft.every((d) => Math.hypot(d.x - px, d.z - pz) >= 0.46 + d.r)
      let back = 0
      while (back <= 1.2 && !turnable(x - fx * back, z - fz * back)) back += 0.05
      if (back > 1.2) {
        if (strict) return null
        back = 0
        while (back <= 0.8 && grid().isHard(x - fx * back, z - fz * back)) back += 0.05
        if (back > 0.8) return null
      }
      for (let k = 0; k <= 12; k++) if (blockedAt(x - fx * back * (k / 12), z - fz * back * (k / 12), yaw)) return null
      const pre = { x: x - fx * back, z: z - fz * back }
      // Rolling in must not run over anything but the piece being fetched (which it stops short of).
      if (strict) for (let k = 0; k <= 12; k++) if (litterUnder(pre.x + (x - pre.x) * (k / 12), pre.z + (z - pre.z) * (k / 12), yaw, others)) return null
      const esc = escapeSeg(soft)
      const start = esc ? esc.at : { x: sim.x, z: sim.z }
      let pts: Pt[] | null = []
      if (Math.hypot(pre.x - start.x, pre.z - start.z) >= 0.03) {
        // The route itself always tries to touch nothing first.
        pts = planPath(grid(), start, pre, soft, 0.46, true) ?? (strict ? null : planPath(grid(), start, pre, soft))
      }
      if (!pts) return null
      const segs = [...(esc ? [esc.seg] : []), ...segsAlong(start, pts)]
      segs.push({ kind: 'turn', yaw })
      if (back > 0.001) segs.push({ kind: 'line', x, z, vmax: 0.2, square: true })
      return segs
    }
    /**
     * Length of the drive planGoto would take (Infinity if none). Ending up on top of litter,
     * or rolling over it on the last straight bit, counts as 5 m extra, so a clean spot wins.
     */
    const gotoLength = (x: number, z: number, yaw: number, except: string[] = []) => {
      const clean = planGoto(x, z, yaw, except, true)
      const segs = clean ?? planGoto(x, z, yaw, except, false)
      if (!segs) return Infinity
      const soft = floorLitter(except)
      const last = segs[segs.length - 1]
      const from = last.kind === 'line' && segs.length > 1 ? (segs[segs.length - 3]?.kind === 'line' ? (segs[segs.length - 3] as { x: number; z: number }) : { x: sim.x, z: sim.z }) : { x, z }
      let dirty = false
      for (let k = 0; k <= 8 && !dirty; k++) dirty = litterUnder(from.x + ((x - from.x) * k) / 8, from.z + ((z - from.z) * k) / 8, yaw, soft)
      // Turning to the final heading sweeps the corners round: nothing within reach of them either.
      if (!dirty) dirty = soft.some((d) => Math.hypot(d.x - from.x, d.z - from.z) < 0.46 + d.r)
      // A drive that has to brush past litter is a last resort.
      const penalty = clean ? 0 : 20 + (dirty ? 5 : 0)
      const pts = segs.flatMap((sg) => (sg.kind === 'line' ? [{ x: sg.x, z: sg.z }] : []))
      // Each stop to turn costs time too: count it as a little extra distance.
      return pathLength({ x: sim.x, z: sim.z }, pts) + 0.1 * segs.filter((sg) => sg.kind === 'turn').length + penalty
    }
    /** Path to a dock: back out of the current dock if parked at one, turn, drive to the staging point, turn, roll in. */
    const pathTo = (to: StationId): Seg[] => {
      const segs: Seg[] = []
      let cx = sim.x
      let cz = sim.z
      for (const id of Object.keys(DOCKS) as StationId[]) {
        const d = DOCKS[id]
        if (Math.hypot(sim.x - d.x, sim.z - d.z) < 0.15 && Math.abs(wrap(sim.yaw - d.yaw)) < 0.25) {
          if (id === to) return [{ kind: 'turn', yaw: d.yaw }, { kind: 'line', x: d.x, z: d.z, vmax: 0.25 }]
          const sg = stagingOf(id)
          segs.push({ kind: 'line', x: sg.x, z: sg.z, vmax: 0.3 })
          cx = sg.x
          cz = sg.z
        }
      }
      const d = DOCKS[to]
      const sg = stagingOf(to)
      if (!segs.length) {
        const esc = escapeSeg()
        if (esc) {
          segs.push(esc.seg)
          cx = esc.at.x
          cz = esc.at.z
        }
      }
      if (Math.hypot(sg.x - cx, sg.z - cz) > 0.03) {
        // Round furniture and litter to the staging point (straight there if the planner can't find a way).
        const lit = floorLitter()
        const pts = planPath(grid(), { x: cx, z: cz }, sg, lit, 0.46, true) ?? planPath(grid(), { x: cx, z: cz }, sg, lit)
        if (pts) segs.push(...segsAlong({ x: cx, z: cz }, pts, 0.8))
        else segs.push({ kind: 'turn', yaw: Math.atan2(-(sg.z - cz), sg.x - cx) }, { kind: 'line', x: sg.x, z: sg.z, vmax: 0.8 })
      }
      segs.push({ kind: 'turn', yaw: d.yaw }, { kind: 'line', x: d.x, z: d.z, vmax: 0.25 })
      return segs
    }
    const endTask = (msg: string | null) => {
      sim.task = null
      sim.path = []
      sim.binds = {}
      sim.slow = { L: 1, R: 1 }
      sim.mq = null
      sim.look = null
      const arms = { L: [...sim.qT.L], R: [...sim.qT.R] }
      const grip = { ...sim.gripT }
      const lift = sim.liftT ?? settingsRef.current.lift
      sim.liftT = null
      // Hand control back to the panels exactly where the task left the robot.
      const waist = sim.waistT
      settingsRef.current = { ...settingsRef.current, armMode: 'joint', arms, grip, lift, waist }
      setSettings((s) => ({ ...s, armMode: 'joint', arms, grip, lift, waist }))
      if (msg) say(msg)
    }
    taskRef.current = {
      start: (id) => {
        if (sim.task) return
        racked = []
        w.scene.updateMatrixWorld(true)
        const acts = planTask(id, planCtx(), { towelsLeft: tw.left })
        const meta = TASKS.find((x) => x.id === id)!
        if (!acts.length) {
          say(`${meta.label}: nothing left to do. Press Reset scene to set it up again.`)
          return
        }
        sim.goal = null
        sim.gripT = { ...sim.grip }
        sim.liftT = sim.lift
        sim.task = { id, label: meta.label, acts, i: 0, started: false, text: '' }
        settingsRef.current = { ...settingsRef.current, armMode: 'joint' }
        setSettings((s) => ({ ...s, armMode: 'joint' }))
      },
      stop: () => sim.task && endTask('Stopped'),
      resetScene: () => {
        if (sim.task) endTask(null)
        S.reset()
        for (const pr of w.clutter.props) S.where.delete(pr.id) // back on the floor
        for (const tw of w.clutter.towels.values()) tw.reset()
        for (const side of SIDES) delete clothHeld[side]
        phys?.reset()
        fillBasket()
        landing = []
        racked = []
        say('Scene reset')
      },
    }
    /** Robot-frame point to the frame of the torso turned by the waist. */
    const torsoFrame = (l: THREE.Vector3) => l.clone().sub(new THREE.Vector3(WAIST_X, 0, 0)).applyAxisAngle(new THREE.Vector3(0, 1, 0), -sim.waist).add(new THREE.Vector3(WAIST_X, 0, 0))
    const robotFrame = (p: THREE.Vector3) => p.clone().sub(new THREE.Vector3(sim.x, 0, sim.z)).applyAxisAngle(new THREE.Vector3(0, 1, 0), -sim.yaw)
    /** Set an arm's joint targets, scaling each joint's speed so they all arrive together. */
    const setQT = (side: Side, q: number[]) => {
      const t = q.map((v, i) => Math.abs(v - sim.q[side][i]) / ARM[i].vmax)
      const tm = Math.max(...t)
      sim.vscale[side] = t.map((x) => (tm > 1e-6 ? Math.max(0.04, x / tm) : 1))
      sim.qT[side] = [...q]
    }
    const mechStation = (m: Mechanism): StationId => (m.kind === 'lid' ? 'washer' : m.kind === 'fold' ? 'laundry' : 'kitchen')
    const tcpWorld = (side: Side) => w.arms[side].tcp.getWorldPosition(new THREE.Vector3())

    const begin = (a: Act) => {
      if (a.k !== 'move') sim.pathSpeed = 0
      const T = sim.task!
      switch (a.k) {
        case 'say':
          say(a.text)
          T.text = a.text
          break
        case 'drive':
          sim.path = pathTo(a.to)
          sim.blockedT = 0
          break
        case 'goto': {
          const segs = planGoto(a.x, a.z, a.yaw, a.except, true) ?? planGoto(a.x, a.z, a.yaw, a.except, false)
          if (segs) sim.path = segs
          else say(`No clear way to ${a.what ?? 'there'}`)
          sim.blockedT = 0
          break
        }
        case 'nudge':
          sim.path = [{ kind: 'line', x: sim.x + Math.cos(sim.yaw) * a.d, z: sim.z - Math.sin(sim.yaw) * a.d, vmax: 0.25 }]
          sim.blockedT = 0
          break
        case 'lift':
          sim.liftT = a.h
          break
        case 'move': {
          const lift = sim.liftT ?? sim.lift
          const robot = { x: sim.x, z: sim.z, yaw: sim.yaw }
          const w0 = sim.waistT
          // The waist turns so the working shoulder(s) are a comfortable distance from their targets.
          let w1 = waistForMove(a.arms, robot, lift, w0)
          // Beside the dishwasher the left counter is close on the left: the waist only turns the
          // right shoulder forward there (which swings the left one back, away from the counter).
          const dp = toWorld('kitchen', KITCHEN.dishPose.a, 0, KITCHEN.dishPose.b)
          if (Math.hypot(sim.x - dp.x, sim.z - dp.z) < 0.2) w1 = clamp(w1, -0.05, 1.4)
          const station = TASKS.find((x) => x.id === sim.task!.id)!.station
          // Tables, tubs, the basket and machines the arm must stay above, where the robot stands now.
          // Cleaning up, the robot works all round the room at any angle: keep clear of every
          // station's furniture and of the bin and hamper, wherever they are relative to it.
          const roaming = sim.task!.id === 'cleanup' || sim.task!.id === 'table'
          const keepOut = roaming ? keepOutsAll(robot, [w.clutter.bin, w.clutter.hamper]) : keepOutsFor(station, robot)
          // Clearing the table: the arm keeps above the dining table (the hand reaches down past its edge).
          if (sim.task!.id === 'table') keepOut.push(...keepOutsRound(robot, [], [{ rect: DINING_FOOTPRINTS[0], y: DINING.h }]))
          // The robot's own base: reaching down to the floor, the elbow and forearm stay above it.
          keepOut.push({ x0: -0.34, x1: 0.34, z0: -0.31, z1: 0.31, y: 0.36, hand: false })
          // The dishwasher door lying open: the whole arm, hand too, stays above it.
          if (S.mech.door > 1) keepOut.push(...keepOutsRound(robot, [], [{ rect: DOOR_OPEN_RECT, y: KITCHEN.door.hingeY + KITCHEN.door.thick }]))
          const goals: Partial<Record<Side, HandGoal>> = {}
          for (const side of SIDES) {
            const g = a.arms[side]
            if (g && !Array.isArray(g)) goals[side] = toRobotGoal(g, robot, lift, w1)
          }
          const K = Math.max(1, ...SIDES.filter((side) => goals[side]).map((side) => pathSteps(side, sim.qT[side], lift, w0, w1, goals[side]!)))
          const qs: Partial<Record<Side, number[][]>> = {}
          for (const side of SIDES) {
            const g = a.arms[side]
            if (!g) continue
            sim.slow[side] = a.slow ? 0.35 : 1
            if (Array.isArray(g)) {
              const q0 = sim.qT[side]
              qs[side] = Array.from({ length: K }, (_, k) => q0.map((v, i) => v + ((g[i] - v) * (k + 1)) / K))
              continue
            }
            // The hand travels in a straight line in small steps, each solved from the last,
            // keeping the elbow, wrist and palm above the work surface.
            const r = handPath(side, sim.qT[side], lift, w0, w1, goals[side]!, K, { rest: REST, keepOut })
            qs[side] = r.qs
            sim.ik[side] = r.res
            if (import.meta.env.DEV && !r.res.reached)
              console.debug(`[ik] ${side} misses by ${(r.res.posErr * 1000).toFixed(0)} mm at step ${sim.task?.i}: target ${toLocal(station, (g as { pos: THREE.Vector3 }).pos).toArray().map((v) => v.toFixed(3))}`)
            if (g.frame !== 'torso') sim.look = g.pos.clone()
          }
          const ws = Array.from({ length: K }, (_, k) => w0 + ((w1 - w0) * (k + 1)) / K)
          // How long each step takes with the slowest joint at its top speed.
          const q0 = { L: [...sim.q.L], R: [...sim.q.R] }
          const dur: number[] = []
          const cum: number[] = []
          let total = 0
          for (let k = 0; k < K; k++) {
            let d = Math.abs(ws[k] - (k ? ws[k - 1] : sim.waist)) / WAIST.vmax
            for (const side of SIDES) {
              const path = qs[side]
              if (!path) continue
              const prev = k ? path[k - 1] : q0[side]
              for (let i = 0; i < DOF; i++) d = Math.max(d, Math.abs(path[k][i] - prev[i]) / (ARM[i].vmax * sim.slow[side]))
            }
            d = Math.max(d, 1e-3)
            dur.push(d)
            total += d
            cum.push(total)
          }
          sim.mq = { K, qs, ws, q0, w0: sim.waist, dur, cum, s: 0, via: !!a.via }
          break
        }
        case 'grip':
          sim.gripT[a.side] = a.g
          break
        case 'grab': {
          const towel = w.clutter.towels.get(a.item)
          if (towel) {
            // Cloth: pinch whatever is under the fingertips.
            if (towel.pinch(a.side, tcpWorld(a.side))) {
              S.where.set(a.item, 'held')
              clothHeld[a.side] = a.item
            } else say('Missed the towel: no cloth between the fingers')
            break
          }
          if (phys) {
            // The grasp holds only if the item is really between the fingers.
            if (phys.grab(a.side, a.item)) S.where.set(a.item, 'held')
            else say(`Missed the ${itemName(a.item)}: it wasn't between the fingers`)
            break
          }
          const o = S.items.get(a.item)!
          o.userData.drop = null
          w.arms[a.side].joints[6].attach(o)
          S.where.set(a.item, 'held')
          break
        }
        case 'release': {
          const towel = w.clutter.towels.get(a.item)
          if (towel) {
            if (clothHeld[a.side] === a.item) {
              towel.release(a.side)
              delete clothHeld[a.side]
              S.where.set(a.item, a.where)
              landing.push({ id: a.item, where: a.where, at: sim.time + 2.5, container: a.container, pos: a.pos })
            }
            break
          }
          if (phys) {
            // Let go: the item drops and lands wherever physics puts it.
            if (phys.release(a.side) === a.item) {
              S.where.set(a.item, a.where)
              landing.push({ id: a.item, where: a.where, at: sim.time + 1.5, container: a.container, pos: a.pos })
              if (a.where === 'rack') racked.push({ id: a.item, container: a.container, pos: a.pos })
            }
            break
          }
          const o = S.items.get(a.item)!
          S.containers[a.container].attach(o)
          o.userData.drop = {
            p0: o.position.clone(),
            q0: o.quaternion.clone(),
            p1: new THREE.Vector3(...a.pos),
            q1: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), a.yaw),
            t: 0,
          }
          S.where.set(a.item, a.where)
          break
        }
        case 'glue':
          // Cutlery still lying where it was becomes part of its plate (a plate keeps its own status).
          if (phys?.glue(a.item, a.onto) && (S.where.get(a.item) ?? 'floor') === 'floor') S.where.set(a.item, `on:${a.onto}`)
          break
        case 'unglue':
          phys?.unglue(a.item)
          break
        case 'bind': {
          if (a.mech.kind === 'fold') {
            // Pinch the cloth under every closed hand that is down at it (both hands for the bottom fold).
            for (const side of SIDES) if (sim.gripT[side] < 0.1) cloth.pinch(side, toLocal('laundry', tcpWorld(side)))
          }
          const local = toLocal(mechStation(a.mech), tcpWorld(a.side))
          const out0 = a.mech.kind === 'rack' ? (a.mech.rack === 'lower' ? S.mech.lowerRack : S.mech.upperRack) : 0
          sim.binds[a.side] = { mech: a.mech, a0: local.x, out0 }
          break
        }
        case 'unbind':
          if (sim.binds[a.side]?.mech.kind === 'fold') for (const side of SIDES) cloth.release(side)
          delete sim.binds[a.side]
          break
        case 'animate':
          sim.anims.push({ key: a.key, from: S.mech[a.key], to: a.to, t: 0, dur: a.dur })
          break
        case 'set':
          S.mech[a.key] = a.value
          break
        case 'clothGrab': {
          const tcp = toLocal('laundry', tcpWorld(a.side))
          if (import.meta.env.DEV)
            console.debug(`[clothGrab] ${a.side} ${a.feature}: hand ${tcp.toArray().map((v) => v.toFixed(3))}, point ${cloth.feature(a.feature as Feature).toArray().map((v) => v.toFixed(3))}, q ${sim.q[a.side].map((v) => Math.round(deg(v)))}`)
          const caught =
            a.feature === 'all'
              ? cloth.grabAll(a.side, tcp)
              : a.feature === 'nearest'
                ? cloth.grabAt(a.side, tcp, a.radius)
                : cloth.grabFeature(a.side, a.feature as Feature, tcp, a.stack, a.radius)
          if (!caught) say(`Missed the towel: the ${a.side === 'L' ? 'left' : 'right'} hand couldn't reach it`)
          break
        }
        case 'clothRelease':
          cloth.release(a.side)
          break
        case 'clothSweep':
          cloth.sweep(a.side, a.on)
          break
        case 'clothPress':
          cloth.press(a.side, a.on)
          break
        case 'wait':
          sim.waitUntil = sim.time + a.t
          break
        case 'plan': {
          // Plan the next steps from the scene as it is now, and run them next.
          const T = sim.task!
          T.acts.splice(T.i + 1, 0, ...a.make())
          break
        }
        case 'clothFolded':
          S.mech.folds = [Math.PI, Math.PI, Math.PI]
          break
        case 'towelStack':
          stackTowel()
          break
      }
    }
    const finished = (a: Act) => {
      switch (a.k) {
        case 'drive':
        case 'goto':
        case 'nudge':
          return sim.path.length === 0
        case 'lift':
          return Math.abs(sim.lift - a.h) < 0.002 && Math.abs(sim.liftV) < 0.01
        case 'move': {
          const mq = sim.mq
          if (!mq) return true
          if (mq.s < mq.cum[mq.K - 1] - 1e-9) return false
          sim.mq = null
          return true
        }
        case 'grip':
          return Math.abs(sim.grip[a.side] - a.g) < 0.01
        case 'animate':
          return !sim.anims.some((x) => x.key === a.key)
        case 'wait':
          return sim.time >= sim.waitUntil
        default:
          return true
      }
    }
    const runTask = () => {
      for (let guard = 0; guard < 25 && sim.task; guard++) {
        const T = sim.task
        const a = T.acts[T.i]
        if (!a) {
          endTask(null)
          reportRack()
          return
        }
        if (!T.started) {
          T.started = true
          begin(a)
        }
        if (!finished(a)) return
        T.i++
        T.started = false
      }
    }
    /** Doors, racks and the lid follow a hand that holds them. */
    const updateBindings = (mech: Mech) => {
      for (const side of SIDES) {
        const b = sim.binds[side]
        if (!b) continue
        const local = toLocal(mechStation(b.mech), tcpWorld(side))
        const m = b.mech
        if (m.kind === 'door') {
          let phi = clamp(doorAngleFor(m.u, m.s, local.x, local.y), 0, Math.PI / 2)
          if (m.pushOnly) phi = Math.min(mech.door, phi)
          mech.door = phi
        } else if (m.kind === 'rack') {
          const v = clamp(b.out0 + (b.a0 - local.x), 0, KITCHEN.racks[m.rack].travel)
          if (m.rack === 'lower') mech.lowerRack = v
          else mech.upperRack = v
        } else if (m.kind === 'lid') {
          mech.lid = lidAngleFor(local.x, local.y)
        } else {
          mech.folds[m.fold] = foldAngleFor(m.fold, local)
        }
      }
    }

    const resize = () => {
      w.renderer.setSize(host.clientWidth, host.clientHeight, false)
      w.camera.aspect = host.clientWidth / Math.max(1, host.clientHeight)
      w.camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(host)
    resize()

    // Dragging the gizmo moves the IK target of the active hand.
    w.gizmo.addEventListener('objectChange', () => {
      const obj = w.gizmo.object
      if (!obj) return
      const side: Side = obj === w.targets.L ? 'L' : 'R'
      const p = obj.position
      setSettings((s) => ({ ...s, targets: { ...s.targets, [side]: [p.x, p.y, p.z] as V3 } }))
    })

    // A click (not an orbit drag): in IK mode it puts the hand target on what you
    // clicked; otherwise it sends the robot to that spot on the floor.
    let down: { x: number; y: number } | null = null
    const el = w.renderer.domElement
    const onDown = (e: PointerEvent) => (down = { x: e.clientX, y: e.clientY })
    const onUp = (e: PointerEvent) => {
      if (!down) return
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y)
      down = null
      if (moved > 5 || w.gizmo.dragging || w.gizmo.axis || sim.task) return
      const rect = el.getBoundingClientRect()
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
      const ray = new THREE.Raycaster()
      ray.setFromCamera(ndc, w.camera)
      const st = settingsRef.current
      if (st.armMode === 'ik') {
        const hit = ray.intersectObjects(w.pickables, true)[0]
        if (!hit) return
        const local = w.robot.worldToLocal(hit.point.clone().add(new THREE.Vector3(0, 0.05, 0)))
        const side: Side = st.armSel === 'L' ? 'L' : 'R'
        setSettings((s) => ({ ...s, targets: { ...s.targets, [side]: [local.x, local.y, local.z] as V3 } }))
        return
      }
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
    const insetSize = new THREE.Vector2()
    let raf = 0
    let teleT = 0
    const baseM = new THREE.Matrix4()

    let ffDt = 0
    const loop = () => {
      // While the cloth is moving, simulated time advances at most 16 cloth steps (1/120 s each)
      // per frame: on a slow machine or at high task speed the whole scene slows down rather than
      // the cloth taking coarse steps (which would let folded layers pass through each other).
      const dt = ffDt || Math.min(cloth.sleeping() ? 0.5 : clothFrame(), Math.min(0.25, clock.getDelta()) * (sim.task ? settingsRef.current.speed : 1))
      const steps = Math.max(1, Math.ceil(dt / 0.01))
      const h = dt / steps
      const t = clock.elapsedTime
      const st = settingsRef.current
      const k = keys.current

      // ---- Arm targets: joint sliders, or inverse kinematics ---------------
      if (!sim.task)
        for (const side of SIDES)
          if ((st.armMode === 'joint' || !activeSides(st.armSel).includes(side)) && st.arms[side].some((v, i) => v !== sim.qT[side][i])) setQT(side, st.arms[side])
      if (st.armMode === 'ik' && !sim.task) {
        const tg = effTargets(st)
        sim.waistT = waistFor(activeSides(st.armSel).map((side) => ({ side, p: new THREE.Vector3(...tg[side]) })))
        const key = JSON.stringify([tg, st.dirGoal, st.armSel, st.lift, sim.waistT])
        if (key !== sim.ikKey) {
          sim.ikKey = key
          baseM.copy(torsoBase(st.lift, sim.waistT))
          const dir = st.dirGoal === 'free' ? null : DIR_VECTORS[st.dirGoal]
          for (const side of activeSides(st.armSel)) {
            const res = solveIK(side, sim.qT[side], baseM, new THREE.Vector3(...tg[side]), { dir, rest: REST })
            sim.ik[side] = res
            setQT(side, res.q)
          }
        }
      } else sim.ikKey = ''
      if (!sim.task && st.armMode !== 'ik') sim.waistT = st.waist
      if (waveRef.current) {
        waveRef.current = false
        if (!sim.task) sim.waveUntil = t + 5
      }
      const waving = t < sim.waveUntil && !sim.task
      const waveQ = [...WAVE]
      waveQ[3] = WAVE[3] + 0.4 * Math.sin((t - sim.waveUntil) * 4)

      // Base cameras gauge the room on every side (20 Hz).
      sim.clearT += dt
      if (sim.clearT >= 0.05) {
        sim.clearT = 0
        for (const bc of BASE_CAMS) sim.clear[bc.id] = clearanceAt(sim.x, sim.z, sim.yaw, bc.id)
        for (const zn of w.zones) {
          const d = sim.clear[zn.id]
          zn.mesh.visible = d < CLEAR.warn
          const m = zn.mesh.material as THREE.MeshBasicMaterial
          m.color.set(d < CLEAR.danger ? '#ef4444' : '#f59e0b')
          m.opacity = d < CLEAR.danger ? 0.5 : 0.3
        }
        for (const bc of w.baseCams) {
          const d = sim.clear[bc.id]
          bc.ringMat.emissive.set(d < CLEAR.danger ? '#ef4444' : d < CLEAR.warn ? '#f59e0b' : '#22d3ee')
        }
      }

      let state = 'Idle'
      let vT = 0
      let vPrev = sim.v
      let slowed = false
      for (let step = 0; step < steps; step++) {
        const throttle = sim.task ? 0 : (k.has('fwd') ? 1 : 0) - (k.has('back') ? 1 : 0)
        const steer = sim.task ? 0 : (k.has('left') ? 1 : 0) - (k.has('right') ? 1 : 0)
        let wT = 0
        state = sim.task ? 'Working' : 'Idle'
        vT = 0
        if (sim.path.length) {
          const seg = sim.path[0]
          state = 'Driving'
          if (seg.kind === 'turn') {
            const err = wrap(seg.yaw - sim.yaw)
            wT = clamp(3 * err, -MAX_TURN * 0.8, MAX_TURN * 0.8)
            if (Math.abs(err) < 0.004 && Math.abs(sim.wz) < 0.06) {
              sim.yaw = seg.yaw
              sim.wz = 0
              wT = 0
              sim.path.shift()
            }
          } else {
            const dx = seg.x - sim.x
            const dz = seg.z - sim.z
            const along = dx * Math.cos(sim.yaw) - dz * Math.sin(sim.yaw)
            const dist = Math.hypot(dx, dz)
            const cap = Math.min(seg.vmax, st.maxSpeed)
            vT = clamp(along * 1.6, -cap, cap)
            if (along > 0.1 && !seg.square) wT = clamp(2.5 * wrap(Math.atan2(-dz, dx) - sim.yaw), -1, 1)
            // Arrived, or as near as the base can get (within 3 cm, with furniture stopping it).
            if (((dist < 0.006 || Math.abs(along) < 0.004) && Math.abs(sim.v) < 0.03) || (dist < 0.03 && sim.blockedFor > 0.2)) {
              sim.blockedFor = 0
              sim.v = 0
              vT = 0
              sim.path.shift()
            }
          }
        } else if (throttle || steer) {
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
        // Ease off before contact: cap the speed by the room the base cameras see ahead.
        if (vT !== 0) {
          const room = vT > 0 ? sim.clear.front : sim.clear.rear
          // Driving by hand, it stops 3 cm short; scripted docking may creep the last bit.
          const cap = room < 0.03 && !sim.task ? 0 : speedForClearance(room)
          if (Math.abs(vT) > cap) {
            vT = Math.sign(vT) * cap
            slowed = true
          }
        }
        vPrev = sim.v
        sim.v = approach(sim.v, vT, ACCEL * h)
        sim.wz = approach(sim.wz, wT, TURN_ACCEL * h)
        const ny = wrap(sim.yaw + sim.wz * h)
        if (blockedAt(sim.x, sim.z, ny) && !blockedAt(sim.x, sim.z, sim.yaw)) {
          sim.wz = 0
          // A turn that can't happen counts as blocked too (so a task can't wait on it forever).
          if (wT !== 0) sim.blockedFor += h
        } else sim.yaw = ny
        const nx = sim.x + sim.v * Math.cos(sim.yaw) * h
        const nz = sim.z - sim.v * Math.sin(sim.yaw) * h
        if (blockedAt(nx, nz, sim.yaw) && !blockedAt(sim.x, sim.z, sim.yaw)) {
          sim.v = 0
          sim.blockedFor += h
        } else {
          sim.odo += Math.hypot(nx - sim.x, nz - sim.z)
          sim.x = nx
          sim.z = nz
          if (Math.abs(sim.v) > 0.02) sim.blockedFor = Math.max(0, sim.blockedFor - h)
        }
        if (slowed && state !== 'Idle' && state !== 'Working') state = Math.abs(vT) < 1e-6 ? 'Stopped: obstacle' : 'Slowing: obstacle'
        if (sim.blockedFor > 0.05) state = 'Blocked'
        if (sim.goal && sim.blockedFor > 2) {
          sim.goal = null
          sim.blockedFor = 0
        }
        if (sim.task && sim.path.length && sim.blockedFor > 2.5) {
          sim.blockedFor = 0
          endTask("Stopped: the robot's path is blocked")
        }
        if (state === 'Idle' && t - sim.arrivedAt < 1.5) state = 'Arrived'
        for (const wh of w.wheels) wh.phi += (sim.v * h + wh.s * sim.wz * h * WHEEL_Z) / WHEEL_R

        // Motors: every joint follows its target with a trapezoidal velocity profile.
        ;[sim.lift, sim.liftV] = servoStep(sim.lift, sim.liftV, sim.liftT ?? st.lift, LIFT_V, LIFT_A, h)
        const mq = sim.mq
        if (mq) {
          // Follow the path smoothly: speed up at the start, cruise, and slow down only where the
          // motion really ends (a via move hands its speed on to the next move). Every joint stays
          // within its top speed, and nothing surges and brakes at each small step of the path.
          const total = mq.cum[mq.K - 1]
          const left = total - mq.s
          const cap = mq.via ? 1 : Math.sqrt(2 * PATH_ACCEL * left)
          sim.pathSpeed = Math.max(0.03, Math.min(1, cap, sim.pathSpeed + PATH_ACCEL * h))
          mq.s = Math.min(total, mq.s + sim.pathSpeed * h)
          let k = 0
          while (k < mq.K - 1 && mq.cum[k] < mq.s) k++
          const f = Math.min(1, Math.max(0, 1 - (mq.cum[k] - mq.s) / mq.dur[k]))
          for (const side of SIDES) {
            const path = mq.qs[side]
            if (!path) continue
            const prev = k ? path[k - 1] : mq.q0[side]
            for (let i = 0; i < DOF; i++) {
              const q = prev[i] + (path[k][i] - prev[i]) * f
              sim.qv[side][i] = (q - sim.q[side][i]) / h
              sim.q[side][i] = q
              sim.qT[side][i] = q
            }
          }
          const wPrev = k ? mq.ws[k - 1] : mq.w0
          const wq = wPrev + (mq.ws[k] - wPrev) * f
          sim.waistV = (wq - sim.waist) / h
          sim.waist = wq
          sim.waistT = wq
          if (mq.s >= total && !mq.via) sim.pathSpeed = 0
        } else {
          ;[sim.waist, sim.waistV] = servoStep(sim.waist, sim.waistV, sim.waistT, WAIST.vmax, WAIST.amax, h)
        }
        for (const side of SIDES) {
          const target = side === 'R' && waving ? waveQ : sim.qT[side]
          if (mq?.qs[side]) {
            sim.grip[side] = approach(sim.grip[side], sim.task ? sim.gripT[side] : st.grip[side], 1.5 * h)
            continue
          }
          for (let i = 0; i < DOF; i++) {
            ;[sim.q[side][i], sim.qv[side][i]] = servoStep(sim.q[side][i], sim.qv[side][i], target[i], ARM[i].vmax * sim.slow[side] * (waving && side === 'R' ? 1 : sim.vscale[side][i]), ARM[i].amax * (waving && side === 'R' ? 1 : sim.vscale[side][i]), h)
          }
          sim.grip[side] = approach(sim.grip[side], sim.task ? sim.gripT[side] : side === 'R' && waving ? 1 : st.grip[side], 1.5 * h)
        }
      }

      // ---- Pose the robot ------------------------------------------------
      w.robot.position.set(sim.x, 0, sim.z)
      w.robot.rotation.y = sim.yaw
      w.wheels.forEach((wh) => (wh.spin.rotation.z = -wh.phi))
      w.carriage.position.y = sim.lift
      w.waist.rotation.y = sim.waist
      for (const side of SIDES) poseArm(w.arms[side], sim.q[side], sim.grip[side])
      w.robot.updateMatrixWorld(true)
      sim.time += dt
      updateBindings(S.mech)
      for (const an of sim.anims) {
        an.t = Math.min(1, an.t + dt / an.dur)
        // A door falls open like gravity; a rack rolls smoothly, easing in and out (no jolt to tip a plate).
        const isRack = an.key === 'lowerRack' || an.key === 'upperRack'
        const e = isRack ? an.t * an.t * (3 - 2 * an.t) : an.to > an.from ? an.t * an.t : 1 - (1 - an.t) * (1 - an.t)
        S.mech[an.key] = an.from + (an.to - an.from) * e
      }
      sim.anims = sim.anims.filter((an) => an.t < 1)
      S.apply(sim.time, dt)
      cloth.step(dt, { L: toLocal('laundry', tcpWorld('L')), R: toLocal('laundry', tcpWorld('R')) })
      doorOpen = S.mech.door > 0.15
      // The towels on the floor: cloth too. They meet the robot's base as a solid box.
      w.clutter.cloth.step(
        dt,
        { L: tcpWorld('L'), R: tcpWorld('R') },
        { x: sim.x, z: sim.z, yaw: sim.yaw, half: [0.31, 0.27], top: 0.32 },
        w.clutter.props.filter((pr) => pr.kind !== 'towel' && (S.where.get(pr.id) ?? 'floor') === 'floor').map((pr) => ballOf(pr, pr.obj.position)),
      )
      for (const [id, tw] of w.clutter.towels) clutterOf(id)!.obj.position.copy(tw.grabPoint())
      if (phys) {
        try {
          phys.step(dt)
        } catch (err) {
          // Safety net: a failed WebAssembly engine can't be reused, so carry on without physics
          // (items then attach to the hand and settle into place, as before physics was added).
          console.error('Physics engine failed; continuing without it', err)
          phys = null
          physFailed = true
          for (const [id, where] of S.where) if (where === 'held') S.where.set(id, 'home')
          say('Physics stopped after an engine error; reload the page to restart it')
        }
      }
      if (phys) {
        // Where did let-go items end up? One that fell off or bounced out goes back on the to-do list.
        for (const l of landing) {
          if (sim.time < l.at) continue
          const p = w.clutter.towels.get(l.id)?.centroid() ?? phys.state(l.id).pos
          if (l.where === 'bin' || l.where === 'hamper') {
            // Dropped into the bin or hamper: it lands low, but inside.
            const c = w.clutter[l.where]
            if (Math.hypot(p.x - c.x, p.z - c.z) > c.r || p.y > c.h) {
              S.where.set(l.id, 'floor')
              say(`The ${itemName(l.id)} missed the ${l.where}: it'll be picked up again`)
            }
          } else if (p.y < 0.2) {
            S.where.set(l.id, 'floor')
            say(`The ${itemName(l.id)} fell on the floor`)
          } else if (l.where === 'rack') {
            const why = rackProblem(l.id, l.container, l.pos)
            if (why) {
              S.where.set(l.id, 'misplaced')
              say(`The ${itemName(l.id)} ${why}`)
            }
          }
        }
        landing = landing.filter((l) => sim.time < l.at)
      }
      runTask()

      // Head
      let panT = st.pan
      let tiltT = st.tilt
      if (st.autoHead) {
        const ikSide: Side = st.armSel === 'L' ? 'L' : 'R'
        if (sim.task && sim.look && !sim.path.length) {
          const l = torsoFrame(robotFrame(sim.look))
          panT = clamp(Math.atan2(-l.z, l.x - COLUMN.x), -1.3, 1.3)
          tiltT = clamp(Math.atan2(l.y - (COLUMN.top + 0.1), Math.hypot(l.x - COLUMN.x, l.z)), -0.8, 0.5)
        } else if (sim.task) {
          panT = clamp(sim.wz * 0.5, -0.9, 0.9)
          tiltT = -0.15
        } else if (st.armMode === 'ik') {
          const l = torsoFrame(new THREE.Vector3(...effTargets(st)[ikSide]))
          panT = clamp(Math.atan2(-l.z, l.x - COLUMN.x), -1.3, 1.3)
          tiltT = clamp(Math.atan2(l.y - (COLUMN.top + 0.1), Math.hypot(l.x - COLUMN.x, l.z)), -0.7, 0.5)
        } else if (sim.goal) {
          panT = clamp(wrap(Math.atan2(-(sim.goal.z - sim.z), sim.goal.x - sim.x) - sim.yaw - sim.waist), -1.3, 1.3)
          tiltT = -0.25
        } else if (t - sim.lastInput > 3) {
          panT = 0.9 * Math.sin(t * 0.45) * Math.sin(t * 0.17)
          tiltT = -0.1 + 0.12 * Math.sin(t * 0.3)
        } else {
          panT = clamp(sim.wz * 0.5, -0.9, 0.9)
          tiltT = -0.12
        }
      }
      ;[sim.pan, sim.panV] = servoStep(sim.pan, sim.panV, panT, 3, 10, dt)
      ;[sim.tilt, sim.tiltV] = servoStep(sim.tilt, sim.tiltV, tiltT, 3, 10, dt)
      w.head.pan.rotation.y = sim.pan
      w.head.tilt.rotation.z = sim.tilt
      w.head.ringMat.emissive.set(state === 'Blocked' ? '#f87171' : state === 'Arrived' ? '#34d399' : '#22d3ee')
      w.stripMat.emissiveIntensity = 1.0 + 0.4 * Math.sin(t * 2)
      const braking = Math.abs(sim.v) > 0.02 && Math.abs(sim.v) <= Math.abs(vPrev) && Math.abs(vT) < Math.abs(sim.v)
      w.brakeMats.forEach((m) => (m.emissiveIntensity = braking ? 2.2 : 0.5))

      // IK targets, gizmo and "can't reach" lines
      const ikOn = st.armMode === 'ik' && !sim.task
      const tg = effTargets(st)
      for (const side of SIDES) {
        const on = ikOn && activeSides(st.armSel).includes(side)
        const tObj = w.targets[side]
        tObj.visible = on
        if (on && !(w.gizmo.dragging && w.gizmo.object === tObj)) tObj.position.set(...tg[side])
        w.arms[side].tcp.visible = on
        const res = sim.ik[side]
        const line = w.reachLines[side]
        line.visible = on && !!res && !res.reached
        if (line.visible) {
          baseM.copy(torsoBase(st.lift, sim.waistT))
          const tip = fk(side, res!.q, baseM).tcp
          ;(line.geometry as THREE.BufferGeometry).setFromPoints([tip, new THREE.Vector3(...tg[side])])
          line.computeLineDistances()
        }
      }
      const gizmoSide: Side = st.armSel === 'L' ? 'L' : 'R'
      if (ikOn) {
        if (w.gizmo.object !== w.targets[gizmoSide]) w.gizmo.attach(w.targets[gizmoSide])
      } else if (w.gizmo.object) w.gizmo.detach()

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
      // Walls in the way of the view fade out (not for the robot's own cameras).
      fadeWalls(w.camera.position, w.controls.target, st.view !== 'cameras', ffDt ? 0 : dt)
      if (st.view === 'cameras') {
        // The robot's cameras: the head camera across the top, the two wrist cameras below it,
        // and the four base cameras along the bottom.
        const r = w.renderer
        const size = r.getSize(insetSize)
        const headH = Math.round(size.y * CAM_ROWS[0])
        const baseH = Math.round(size.y * CAM_ROWS[2])
        const wristH = size.y - headH - baseH - 4
        const halfW = Math.floor(size.x / 2) - 1
        const quarter = (size.x - 6) / 4
        const views: [THREE.PerspectiveCamera, number, number, number, number][] = [
          [w.head.cam, 0, size.y - headH, size.x, headH],
          [w.arms.L.cam, 0, baseH + 2, halfW, wristH],
          [w.arms.R.cam, size.x - halfW, baseH + 2, halfW, wristH],
          ...w.baseCams.map((bc, i): [THREE.PerspectiveCamera, number, number, number, number] => [
            bc.cam,
            Math.round(i * (quarter + 2)),
            0,
            Math.round(quarter),
            baseH,
          ]),
        ]
        r.setScissorTest(true)
        views.forEach(([cam, x, y, vw, vh], i) => {
          cam.aspect = vw / Math.max(1, vh)
          cam.updateProjectionMatrix()
          r.setViewport(x, y, vw, vh)
          r.setScissor(x, y, vw, vh)
          r.shadowMap.autoUpdate = i === 0
          r.render(w.scene, cam)
        })
        r.shadowMap.autoUpdate = true
        r.setScissorTest(false)
        r.setViewport(0, 0, size.x, size.y)
      } else if (!ffDt) w.renderer.render(w.scene, w.camera)

      // ---- Telemetry and joint loads (10 Hz) ------------------------------
      teleT += dt
      if (teleT > 0.1) {
        teleT = 0
        baseM.copy(torsoBase(sim.lift, sim.waist))
        w.waistLed.emissive.set(Math.abs(sim.waistV) > 0.05 ? '#38bdf8' : '#22c55e')
        const arms = {} as Record<Side, ArmTele>
        for (const side of SIDES) {
          const tau = gravityTorques(side, sim.q[side], baseM, st.payload)
          sim.tau[side] = tau
          // Actuator LEDs: green < 50 % of rated torque, amber < 85 %, red above.
          tau.forEach((v, i) => {
            const load = Math.abs(v) / ARM[i].rated
            w.arms[side].leds[i]?.emissive.set(load < 0.5 ? '#22c55e' : load < 0.85 ? '#f59e0b' : '#ef4444')
          })
          const f = fk(side, sim.q[side], baseM)
          arms[side] = {
            q: [...sim.q[side]],
            v: [...sim.qv[side]],
            tau,
            tcp: [f.tcp.x, f.tcp.y, f.tcp.z],
            dir: [f.dir.x, f.dir.y, f.dir.z],
          }
        }
        const T = sim.task
        setTele({
          x: sim.x,
          z: sim.z,
          heading: deg(sim.yaw),
          speed: sim.v,
          odo: sim.odo,
          state,
          lift: sim.lift,
          waist: sim.waist,
          arms,
          ik: { ...sim.ik },
          task: T ? { id: T.id, label: T.label, step: Math.min(T.i + 1, T.acts.length), total: T.acts.length, text: T.text } : null,
          log: sim.log,
          physics: phys ? phys.stats() : physFailed ? { awake: -1, contacts: 0, held: {} } : null,
          clear: { ...sim.clear },
        })
      }
      if (!ffDt) raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    if (import.meta.env.DEV) {
      // Testing aid: advance `seconds` of simulated time in steps of `dt`, without drawing.
      Object.assign(window, {
        __sim: sim,
        __ctx: () => planCtx(),
        __ff: (seconds: number, dt = 0.02) => {
          for (let t = 0; t < seconds; t += dt) {
            ffDt = dt
            loop()
          }
          ffDt = 0
        },
      })
    }

    return () => {
      disposed = true
      phys?.dispose()
      cancelAnimationFrame(raf)
      ro.disconnect()
      el.removeEventListener('pointerdown', onDown)
      el.removeEventListener('pointerup', onUp)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKey)
      window.removeEventListener('blur', onBlur)
      w.gizmo.detach()
      w.gizmo.dispose()
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
  const busy = tele.task ? 'pointer-events-none opacity-50' : undefined
  const TASK_ICONS: Record<TaskId, typeof Boxes> = { sort: Boxes, dishes: UtensilsCrossed, fold: Shirt, wash: WashingMachine, cleanup: Trash2, table: Layers }
  const ikMode = settings.armMode === 'ik'
  const ikSide: Side = settings.armSel === 'L' ? 'L' : 'R'
  const ik = tele.ik[ikSide]
  const target = settings.targets[ikSide]
  const arm = tele.arms[shownSide]
  // Joint sliders show the commanded angles (joint mode) or what IK chose (IK mode).
  const sliderQ = ikMode ? (tele.ik[shownSide]?.q ?? arm.q) : settings.arms[shownSide]

  return (
    <main className="mx-auto grid max-w-[1400px] gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[minmax(0,1fr)_380px]">
      <div className="min-w-0 space-y-4">
        <Card
          className="overflow-hidden"
          title="Humanoid (3D)"
          right={
            <Segmented
              size="sm"
              value={settings.view}
              onChange={(v) => set('view', v)}
              options={[
                { value: 'scene', label: 'Scene' },
                { value: 'cameras', label: 'Cameras' },
              ]}
            />
          }
        >
          <div className="-m-4">
            <div className="relative aspect-[16/10] w-full">
              <div ref={hostRef} className="absolute inset-0 cursor-crosshair" />
              <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-slate-900/80 px-2.5 py-1.5 text-xs text-slate-300 ring-1 ring-slate-700">
                {tele.task
                  ? `${tele.task.label}: ${tele.task.text}`
                  : ikMode
                  ? 'Drag the arrows to move the hand target · click a crate, barrel or the floor to reach for it'
                  : 'WASD or arrow keys to drive · click the floor to send it there · space to stop'}
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
              {settings.view === 'cameras' && (
                <div className="pointer-events-none absolute inset-0 text-[11px] text-slate-200">
                  <div className="absolute inset-x-0 top-0 border-b-2 border-slate-950" style={{ height: `${CAM_ROWS[0] * 100}%` }}>
                    <span className="absolute right-2 top-2 rounded bg-slate-900/80 px-2 py-0.5">Head camera</span>
                  </div>
                  {(['L', 'R'] as Side[]).map((side, i) => (
                    <div
                      key={side}
                      className={clsx('absolute w-1/2 border-b-2 border-slate-950', i ? 'right-0' : 'left-0 border-r-2')}
                      style={{ top: `${CAM_ROWS[0] * 100}%`, height: `${CAM_ROWS[1] * 100}%` }}
                    >
                      <span className="absolute left-2 top-2 rounded bg-slate-900/80 px-2 py-0.5">{side === 'L' ? 'Left' : 'Right'} gripper camera</span>
                      <span className="absolute left-1/2 top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full ring-1 ring-cyan-300/80" />
                    </div>
                  ))}
                  {BASE_CAMS.map((bc, i) => {
                    const d = tele.clear[bc.id]
                    const tone = d < CLEAR.danger ? 'bg-red-500/90 text-white' : d < CLEAR.warn ? 'bg-amber-400/90 text-slate-950' : 'bg-slate-900/80'
                    return (
                      <div
                        key={bc.id}
                        className={clsx('absolute bottom-0 w-1/4', i < 3 && 'border-r-2 border-slate-950')}
                        style={{ left: `${i * 25}%`, height: `${CAM_ROWS[2] * 100}%` }}
                      >
                        <span className="absolute left-1.5 top-1.5 rounded bg-slate-900/80 px-1.5 py-0.5">{bc.label} base cam</span>
                        <span className={clsx('absolute bottom-1.5 right-1.5 rounded px-1.5 py-0.5 font-mono', tone)}>
                          {d >= CLEAR.max ? 'clear > 1.5 m' : `${Math.round(d * 100)} cm`}
                        </span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </Card>

        {/* Joints and telemetry share one card, a tab each. */}
        <Card
          title={infoTab === 'joints' ? `Joints · ${shownSide === 'R' ? 'right' : 'left'} arm` : 'Telemetry'}
          right={
            <div className="flex items-center gap-3">
              {infoTab === 'joints' && (
                <span className="hidden font-mono text-[11px] text-slate-400 sm:inline">
                  hand at ({fmt(arm.tcp[0] * 100, 0)}, {fmt(arm.tcp[1] * 100, 0)}, {fmt(arm.tcp[2] * 100, 0)}) cm
                </span>
              )}
              <Segmented
                size="sm"
                value={infoTab}
                onChange={setInfoTab}
                options={[
                  { value: 'joints', label: 'Joints' },
                  { value: 'telemetry', label: 'Telemetry' },
                ]}
              />
            </div>
          }
        >
          {infoTab === 'joints' ? (
            <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-sm">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
                  <th className="pb-2 font-medium">Joint</th>
                  <th className="pb-2 font-medium">Angle within limits</th>
                  <th className="pb-2 text-right font-medium">Angle</th>
                  <th className="pb-2 text-right font-medium">Speed</th>
                  <th className="pb-2 pl-3 font-medium">Holding torque</th>
                </tr>
              </thead>
              <tbody className="font-mono text-xs">
                {ARM.map((j, i) => {
                  const a = deg(arm.q[i])
                  const pos = (a - j.min) / (j.max - j.min)
                  const load = Math.abs(arm.tau[i]) / j.rated
                  return (
                    <tr key={j.key} className="border-t border-slate-800/80">
                      <td className="py-1.5 pr-2 font-sans text-slate-300">
                        <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: j.color }} />
                        {j.label}
                      </td>
                      <td className="py-1.5 pr-3">
                        <div className="relative h-1.5 w-full min-w-[90px] rounded-full bg-slate-800">
                          <div className="absolute top-1/2 h-3 w-1 -translate-x-1/2 -translate-y-1/2 rounded-sm" style={{ left: `${clamp(pos, 0, 1) * 100}%`, background: j.color }} />
                        </div>
                        <div className="mt-0.5 flex justify-between text-[10px] text-slate-600">
                          <span>{j.min}°</span>
                          <span>{j.max}°</span>
                        </div>
                      </td>
                      <td className="py-1.5 text-right text-slate-100">{fmt(a, 0)}°</td>
                      <td className="py-1.5 text-right text-slate-400">{fmt(deg(arm.v[i]), 0)}°/s</td>
                      <td className="py-1.5 pl-3">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 rounded-full bg-slate-800">
                            <div
                              className={clsx('h-full rounded-full', load < 0.5 ? 'bg-emerald-400' : load < 0.85 ? 'bg-amber-400' : 'bg-rose-500')}
                              style={{ width: `${clamp(load, 0, 1) * 100}%` }}
                            />
                          </div>
                          <span className="w-24 text-slate-300">
                            {fmt(arm.tau[i], 1)} <span className="text-slate-500">/ {j.rated} N·m</span>
                          </span>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            Holding torque is what each motor must push to hold the arm still against gravity: for every link further down the arm,
            its weight times its lever arm about that joint's axis. The LED on each actuator turns amber above half its rated torque and red
            near the limit. Each joint moves with a trapezoidal velocity profile: it accelerates, cruises at its top speed, then brakes to stop on target.
          </p>
            </>
          ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="State" value={tele.state} tone={stateTone} />
            <Stat label="Speed" value={`${fmt(tele.speed, 2)} m/s`} />
            <Stat label="Heading" value={`${Math.round(tele.heading)}°`} />
            <Stat label="Odometer" value={`${fmt(tele.odo, 2)} m`} />
            <Stat
              label="Nearest obstacle"
              value={(() => {
                const [id, d] = (Object.entries(tele.clear) as [BaseCamId, number][]).reduce((a, b) => (b[1] < a[1] ? b : a))
                return d >= CLEAR.max ? 'Clear' : `${Math.round(d * 100)} cm ${id}`
              })()}
              tone={Math.min(...Object.values(tele.clear)) < CLEAR.danger ? 'bad' : undefined}
            />
            <Stat label="Position x" value={`${fmt(tele.x, 2)} m`} />
            <Stat label="Position z" value={`${fmt(tele.z, 2)} m`} />
            <Stat label="Shoulder height" value={`${Math.round(tele.lift * 100)} cm`} />
            <Stat label="Waist" value={`${Math.round(deg(tele.waist))}°`} />
            <Stat label="Arm reach" value={`${Math.round((UPPER + FORE + 0.14) * 100)} cm`} />
            <Stat label="Physics" value={!tele.physics ? 'Loading…' : tele.physics.awake < 0 ? 'Stopped' : 'Rapier, 120 Hz'} />
            <Stat label="Items moving" value={tele.physics && tele.physics.awake >= 0 ? String(tele.physics.awake) : '–'} />
            <Stat label="Item contacts" value={tele.physics && tele.physics.awake >= 0 ? String(tele.physics.contacts) : '–'} />
          </div>
          )}
        </Card>
      </div>

      <aside className="space-y-4">
        <Card
          title="Tasks"
          right={tele.task ? <span className="text-[11px] font-medium text-cyan-300">Running</span> : <span className="text-[11px] text-slate-500">4 stations</span>}
        >
          <div className="grid grid-cols-2 gap-2">
            {TASKS.map((task) => {
              const Icon = TASK_ICONS[task.id]
              return (
                <Button key={task.id} onClick={() => taskRef.current.start(task.id)} disabled={!!tele.task} title={task.blurb}>
                  <Icon className="h-4 w-4" /> {task.label}
                </Button>
              )
            })}
          </div>
          {tele.task && (
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between gap-2 text-sm">
                <span className="font-medium text-slate-100">{tele.task.label}</span>
                <span className="font-mono text-xs text-slate-400">
                  step {tele.task.step} / {tele.task.total}
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-800">
                <div className="h-full rounded-full bg-cyan-400 transition-[width]" style={{ width: `${(tele.task.step / tele.task.total) * 100}%` }} />
              </div>
              <p className="text-xs text-slate-300">{tele.task.text}</p>
              <Button onClick={() => taskRef.current.stop()}>
                <Square className="h-3.5 w-3.5" /> Stop
              </Button>
            </div>
          )}
          <div className="mt-4 flex items-end gap-2">
            <div className="flex-1">
              <div className="mb-1 text-xs text-slate-400">Task speed</div>
              <Segmented
                size="sm"
                value={settings.speed}
                onChange={(v) => set('speed', v)}
                options={[1, 2, 4, 8].map((v) => ({ value: v, label: `${v}×` }))}
              />
            </div>
            <Button onClick={() => taskRef.current.resetScene()} title="Put every item back and close everything">
              <RotateCcw className="h-4 w-4" /> Reset scene
            </Button>
          </div>
          {tele.log.length > 0 && (
            <ol className="mt-4 space-y-1 border-t border-slate-800 pt-3 text-xs text-slate-400">
              {tele.log.map((line, i) => (
                <li key={i} className={clsx(i === tele.log.length - 1 && 'text-slate-200')}>
                  {line}
                </li>
              ))}
            </ol>
          )}
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            The robot drives to each station, docks, and plans every grasp with inverse kinematics (hand position plus which way the
            fingers point and close). Items, the towel and the doors, racks and lid move under physics as the hands act on them.
          </p>
        </Card>

        <Card title="Arms" className={busy}>
          <Segmented
            size="sm"
            value={settings.armMode}
            onChange={setMode}
            options={[
              { value: 'joint', label: <span className="inline-flex items-center gap-1"><SlidersHorizontal className="h-3.5 w-3.5" />Joints (FK)</span> },
              { value: 'ik', label: <span className="inline-flex items-center gap-1"><Crosshair className="h-3.5 w-3.5" />Reach (IK)</span> },
            ]}
          />
          <div className="mt-3">
            <Segmented
              size="sm"
              value={settings.armSel}
              onChange={(v) => set('armSel', v)}
              options={[
                { value: 'both', label: ikMode ? 'Both (mirror)' : 'Both' },
                { value: 'L', label: 'Left arm' },
                { value: 'R', label: 'Right arm' },
              ]}
            />
          </div>

          {!ikMode && (
            <div className="mt-4 grid grid-cols-3 gap-2">
              {PRESETS.map((p) => (
                <Button
                  key={p.label}
                  onClick={() =>
                    setSettings((s) => {
                      const arms = { L: [...s.arms.L], R: [...s.arms.R] }
                      const grip = { ...s.grip }
                      for (const side of activeSides(s.armSel)) {
                        arms[side] = [...p.q]
                        grip[side] = p.grip
                      }
                      return { ...s, arms, grip }
                    })
                  }
                >
                  {p.label}
                </Button>
              ))}
              <Button onClick={() => (waveRef.current = true)} title="Wave with the right arm">
                <Hand className="h-4 w-4" /> Wave
              </Button>
            </div>
          )}

          {ikMode && (
            <div className="mt-4 space-y-4">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Target" value={ik ? (ik.reached ? 'Reached' : 'Out of reach') : '–'} tone={ik ? (ik.reached ? 'good' : 'bad') : undefined} />
                <Stat label="Position error" value={ik ? `${fmt(ik.posErr * 1000, 1)} mm` : '–'} />
                <Stat label="Iterations" value={ik ? `${ik.iters}` : '–'} />
                <Stat label="Solve time" value={ik ? `${fmt(ik.ms, 1)} ms` : '–'} />
              </div>
              <Slider label="Target forward (x)" value={Math.round(target[0] * 100)} min={-20} max={100} unit=" cm" onChange={(v) => setTarget(0, v / 100)} />
              <Slider label="Target height (y)" value={Math.round(target[1] * 100)} min={20} max={190} unit=" cm" onChange={(v) => setTarget(1, v / 100)} />
              <Slider label="Target sideways (z, + = right)" value={Math.round(target[2] * 100)} min={-90} max={90} unit=" cm" onChange={(v) => setTarget(2, v / 100)} />
              <div>
                <div className="mb-1 text-sm text-slate-300">Fingers point</div>
                <Segmented
                  size="sm"
                  value={settings.dirGoal}
                  onChange={(v) => set('dirGoal', v)}
                  options={[
                    { value: 'free', label: 'Any way' },
                    { value: 'forward', label: 'Forward' },
                    { value: 'down', label: 'Down' },
                  ]}
                />
              </div>
              <p className="text-xs leading-relaxed text-slate-400">
                Damped least squares on the 7-joint arm: each step moves the joints by Jᵀ(JJᵀ + λ²I)⁻¹·error, where the Jacobian J says how
                each joint moves the hand. Seven joints is one more than a hand pose needs, so the spare freedom keeps the elbow close to a natural
                pose. Out-of-reach targets get the closest the arm can do (red dashed line).
              </p>
            </div>
          )}

          <div className="mt-4 space-y-3">
            {ARM.map((j, i) => (
              <Slider
                key={j.key}
                label={j.label}
                value={Math.round(deg(sliderQ[i]))}
                min={j.min}
                max={j.max}
                unit="°"
                color={j.color}
                disabled={ikMode}
                onChange={(v) => setJoint(i, rad(v))}
              />
            ))}
            <Slider label="Gripper" value={Math.round(settings.grip[shownSide] * 100)} min={0} max={100} unit="% open" onChange={(v) => setGrip(v / 100)} />
            <Slider label="Payload in hand" value={settings.payload} min={0} max={5} step={0.5} unit=" kg" onChange={(v) => set('payload', v)} />
          </div>
        </Card>

        <Card title="Drive" className={busy}>
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

        <Card title="Torso" className={busy}>
          <Slider
            label="Shoulder height"
            value={Math.round(settings.lift * 100)}
            min={Math.round(LIFT_MIN * 100)}
            max={Math.round(LIFT_MAX * 100)}
            unit=" cm"
            onChange={(v) => set('lift', v / 100)}
          />
          <div className="mt-4">
            <Slider
              label={ikMode ? 'Waist (turns toward the target)' : 'Waist'}
              value={Math.round(deg(ikMode ? tele.waist : settings.waist))}
              min={WAIST.min}
              max={WAIST.max}
              unit="°"
              color="#22d3ee"
              disabled={ikMode}
              onChange={(v) => set('waist', rad(v))}
            />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            The column, torso, arms and head turn together on a waist turntable at the top of the base. In Reach mode and during tasks the
            waist turns by itself so the working shoulder is a comfortable distance from the target. A lead screw drives the torso up and
            down the column (15 cm/s).
          </p>
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
            In Reach mode the head watches the hand target. The camera ring glows cyan while driving, green on arrival and red when blocked.
          </p>
        </Card>
      </aside>
    </main>
  )
}
