import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Crosshair, Hand, SlidersHorizontal, Square } from 'lucide-react'
import clsx from 'clsx'
import { approach, clamp, deg, fmt, rad, servoStep, wrap } from '../lib/math'
import {
  ARM,
  DIR_VECTORS,
  DOF,
  FORE,
  UPPER,
  fk,
  gravityTorques,
  jointAxis,
  jointOffset,
  mirrorPoint,
  solveIK,
  type DirGoal,
  type IKResult,
  type Side,
  type V3,
} from '../lib/kinematics'
import { Button, Card, Segmented, Slider, Stat } from './ui'

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

const OBSTACLES: Obstacle[] = [
  { kind: 'crate', x: 2.1, z: -1.2, w: 0.6, d: 0.6, h: 0.5, color: '#a16207' },
  { kind: 'crate', x: 2.6, z: -0.5, w: 0.45, d: 0.45, h: 0.35, color: '#b45309' },
  { kind: 'crate', x: -2.2, z: 1.8, w: 1.2, d: 0.5, h: 0.6, color: '#92400e' },
  { kind: 'crate', x: -1.4, z: -2.6, w: 0.5, d: 0.9, h: 0.45, color: '#a16207' },
  { kind: 'crate', x: 3.6, z: 2.4, w: 0.7, d: 0.7, h: 0.7, color: '#b45309' },
  { kind: 'crate', x: 1.3, z: 0.9, w: 0.5, d: 0.5, h: 0.75, color: '#a16207' },
  { kind: 'barrel', x: 0.9, z: 2.2, r: 0.2, h: 0.55, color: '#2563eb' },
  { kind: 'barrel', x: 1.4, z: 2.5, r: 0.2, h: 0.55, color: '#dc2626' },
  { kind: 'barrel', x: -3.2, z: -0.8, r: 0.22, h: 0.6, color: '#16a34a' },
  { kind: 'barrel', x: 0.2, z: -3.3, r: 0.18, h: 0.5, color: '#eab308' },
]

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

// ---------------------------------------------------------------------------
// Poses (degrees, in ARM order: shoulder pitch, roll, twist, elbow, forearm twist, wrist pitch, wrist yaw)
// ---------------------------------------------------------------------------
const Q = (...d: number[]) => d.map(rad)
const PRESETS: { label: string; q: number[]; grip: number }[] = [
  { label: 'Ready', q: Q(22, 12, 0, 78, 0, -10, 0), grip: 0.6 },
  { label: 'Rest', q: Q(0, 6, 0, 12, 0, 0, 0), grip: 0.2 },
  { label: 'Reach', q: Q(80, 8, 0, 6, 0, 0, 0), grip: 1 },
  { label: 'Carry', q: Q(10, 4, 0, 95, 90, -5, 0), grip: 0.4 },
  { label: 'Arms up', q: Q(165, 22, 0, 8, 0, 0, 0), grip: 1 },
  { label: 'T-pose', q: Q(0, 90, 0, 0, 0, 0, 0), grip: 1 },
]
const REST = PRESETS[0].q
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
  return { side, joints, axes, leds, fingers: fingers as [THREE.Group, THREE.Group], tcp }
}

function poseArm(arm: ArmRig, q: number[], grip: number) {
  arm.joints.forEach((g, i) => g.quaternion.setFromAxisAngle(arm.axes[i], q[i]))
  const a = 0.06 + grip * 0.5
  arm.fingers.forEach((f) => (f.rotation.x = -Math.sign(f.position.z) * a))
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
  return { pan, tilt, ringMat }
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

  // Lift column: black body, linear rails and a lead screw on the front, white back panel.
  const colH = COLUMN.top - BASE_TOP
  const column = rbox(COLUMN.d, colH, COLUMN.w, 0.012, mat(C.black, { roughness: 0.55, metalness: 0.35 }))
  column.position.set(COLUMN.x, BASE_TOP + colH / 2, 0)
  const back = rbox(0.03, colH - 0.04, COLUMN.w + 0.01, 0.01, MAT.white)
  back.position.set(COLUMN.x - COLUMN.d / 2 - 0.005, BASE_TOP + colH / 2, 0)
  for (const z of [-0.045, 0.045]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.008, colH - 0.08, 0.014), MAT.alu)
    rail.position.set(COLUMN.x + COLUMN.d / 2 + 0.004, BASE_TOP + colH / 2, z)
    robot.add(shadowed(rail))
  }
  const screw = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, colH - 0.08, 10), MAT.bolt)
  screw.position.set(COLUMN.x + COLUMN.d / 2 + 0.01, BASE_TOP + colH / 2, 0)
  const topCap = rbox(COLUMN.d + 0.03, 0.03, COLUMN.w + 0.03, 0.01, MAT.white)
  topCap.position.set(COLUMN.x - 0.008, COLUMN.top, 0)
  robot.add(column, back, shadowed(screw), topCap)

  // Torso carriage rides the rails and carries both shoulders.
  const carriage = new THREE.Group()
  robot.add(carriage)
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
  const head = buildHead(robot)
  return { robot, carriage, wheels, arms, head, brakeMats, stripMat }
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
  carriage: THREE.Group
  wheels: { spin: THREE.Group; s: number; phi: number }[]
  arms: Record<Side, ArmRig>
  head: Head
  targets: Record<Side, THREE.Group>
  reachLines: Record<Side, THREE.Line>
  goal: THREE.Group
  trail: THREE.Line
  brakeMats: THREE.MeshStandardMaterial[]
  stripMat: THREE.MeshStandardMaterial
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
  controls.minDistance = 0.6
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

  return { renderer, scene, camera, controls, gizmo, sun, floor, pickables, goal, trail, targets, reachLines, ...parts }
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
  arms: Record<Side, ArmTele>
  ik: Record<Side, IKResult | null>
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
  const waveRef = useRef(false)
  const [settings, setSettings] = useState<Settings>({
    maxSpeed: 0.7,
    lift: 1.12,
    follow: true,
    autoHead: true,
    pan: 0,
    tilt: 0,
    arms: { L: [...REST], R: [...REST] },
    grip: { L: 0.6, R: 0.6 },
    armMode: 'joint',
    armSel: 'both',
    targets: { R: [0.45, 0.92, 0.27], L: [0.45, 0.92, -0.27] },
    dirGoal: 'free',
    payload: 0,
  })
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const [tele, setTele] = useState<Telemetry>({
    x: 0,
    z: 0,
    heading: 0,
    speed: 0,
    odo: 0,
    state: 'Idle',
    lift: settings.lift,
    arms: { L: emptyArm(), R: emptyArm() },
    ik: { L: null, R: null },
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
      if (m === 'joint' && s.armMode === 'ik') return { ...s, armMode: m, arms: syncRef.current() }
      return { ...s, armMode: m }
    })

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const w = buildWorld(host)
    const st0 = settingsRef.current

    const sim = {
      x: 0,
      z: 0,
      yaw: 0,
      v: 0,
      wz: 0,
      lift: st0.lift,
      liftV: 0,
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
    syncRef.current = () => ({ L: [...sim.qT.L], R: [...sim.qT.R] })

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
      if (moved > 5 || w.gizmo.dragging || w.gizmo.axis) return
      const rect = el.getBoundingClientRect()
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
      const ray = new THREE.Raycaster()
      ray.setFromCamera(ndc, w.camera)
      const st = settingsRef.current
      if (st.armMode === 'ik') {
        const hit = ray.intersectObjects(w.pickables, false)[0]
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
    let raf = 0
    let teleT = 0
    const baseM = new THREE.Matrix4()

    const loop = () => {
      const dt = Math.min(0.25, clock.getDelta())
      const steps = Math.max(1, Math.ceil(dt / 0.01))
      const h = dt / steps
      const t = clock.elapsedTime
      const st = settingsRef.current
      const k = keys.current

      // ---- Arm targets: joint sliders, or inverse kinematics ---------------
      for (const side of SIDES) if (st.armMode === 'joint' || !activeSides(st.armSel).includes(side)) sim.qT[side] = [...st.arms[side]]
      if (st.armMode === 'ik') {
        const tg = effTargets(st)
        const key = JSON.stringify([tg, st.dirGoal, st.armSel, st.lift])
        if (key !== sim.ikKey) {
          sim.ikKey = key
          baseM.makeTranslation(0, st.lift, 0)
          const dir = st.dirGoal === 'free' ? null : DIR_VECTORS[st.dirGoal]
          for (const side of activeSides(st.armSel)) {
            const res = solveIK(side, sim.qT[side], baseM, new THREE.Vector3(...tg[side]), { dir, rest: REST })
            sim.ik[side] = res
            sim.qT[side] = res.q
          }
        }
      } else sim.ikKey = ''
      if (waveRef.current) {
        waveRef.current = false
        sim.waveUntil = t + 5
      }
      const waving = t < sim.waveUntil
      const waveQ = [...WAVE]
      waveQ[3] = WAVE[3] + 0.4 * Math.sin((t - sim.waveUntil) * 4)

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
          sim.goal = null
          sim.blockedFor = 0
        }
        if (state === 'Idle' && t - sim.arrivedAt < 1.5) state = 'Arrived'
        for (const wh of w.wheels) wh.phi += (sim.v * h + wh.s * sim.wz * h * WHEEL_Z) / WHEEL_R

        // Motors: every joint follows its target with a trapezoidal velocity profile.
        ;[sim.lift, sim.liftV] = servoStep(sim.lift, sim.liftV, st.lift, LIFT_V, LIFT_A, h)
        for (const side of SIDES) {
          const target = side === 'R' && waving ? waveQ : sim.qT[side]
          for (let i = 0; i < DOF; i++) {
            ;[sim.q[side][i], sim.qv[side][i]] = servoStep(sim.q[side][i], sim.qv[side][i], target[i], ARM[i].vmax, ARM[i].amax, h)
          }
          sim.grip[side] = approach(sim.grip[side], side === 'R' && waving ? 1 : st.grip[side], 1.5 * h)
        }
      }

      // ---- Pose the robot ------------------------------------------------
      w.robot.position.set(sim.x, 0, sim.z)
      w.robot.rotation.y = sim.yaw
      w.wheels.forEach((wh) => (wh.spin.rotation.z = -wh.phi))
      w.carriage.position.y = sim.lift
      for (const side of SIDES) poseArm(w.arms[side], sim.q[side], sim.grip[side])

      // Head
      let panT = st.pan
      let tiltT = st.tilt
      if (st.autoHead) {
        const ikSide: Side = st.armSel === 'L' ? 'L' : 'R'
        if (st.armMode === 'ik') {
          const tg = effTargets(st)[ikSide]
          panT = clamp(Math.atan2(-tg[2], tg[0] - COLUMN.x), -1.3, 1.3)
          tiltT = clamp(Math.atan2(tg[1] - (COLUMN.top + 0.1), Math.hypot(tg[0] - COLUMN.x, tg[2])), -0.7, 0.5)
        } else if (sim.goal) {
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
      ;[sim.pan, sim.panV] = servoStep(sim.pan, sim.panV, panT, 3, 10, dt)
      ;[sim.tilt, sim.tiltV] = servoStep(sim.tilt, sim.tiltV, tiltT, 3, 10, dt)
      w.head.pan.rotation.y = sim.pan
      w.head.tilt.rotation.z = sim.tilt
      w.head.ringMat.emissive.set(state === 'Blocked' ? '#f87171' : state === 'Arrived' ? '#34d399' : '#22d3ee')
      w.stripMat.emissiveIntensity = 1.0 + 0.4 * Math.sin(t * 2)
      const braking = Math.abs(sim.v) > 0.02 && Math.abs(sim.v) <= Math.abs(vPrev) && Math.abs(vT) < Math.abs(sim.v)
      w.brakeMats.forEach((m) => (m.emissiveIntensity = braking ? 2.2 : 0.5))

      // IK targets, gizmo and "can't reach" lines
      const ikOn = st.armMode === 'ik'
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
          baseM.makeTranslation(0, st.lift, 0)
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
      w.renderer.render(w.scene, w.camera)

      // ---- Telemetry and joint loads (10 Hz) ------------------------------
      teleT += dt
      if (teleT > 0.1) {
        teleT = 0
        baseM.makeTranslation(0, sim.lift, 0)
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
        setTele({ x: sim.x, z: sim.z, heading: deg(sim.yaw), speed: sim.v, odo: sim.odo, state, lift: sim.lift, arms, ik: { ...sim.ik } })
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
        <Card className="overflow-hidden" title="Humanoid (3D)">
          <div className="-m-4">
            <div className="relative aspect-[16/10] w-full">
              <div ref={hostRef} className="absolute inset-0 cursor-crosshair" />
              <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-slate-900/80 px-2.5 py-1.5 text-xs text-slate-300 ring-1 ring-slate-700">
                {ikMode
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
            </div>
          </div>
        </Card>

        <Card
          title={`Joints · ${shownSide === 'R' ? 'right' : 'left'} arm`}
          right={
            <span className="font-mono text-[11px] text-slate-400">
              hand at ({fmt(arm.tcp[0] * 100, 0)}, {fmt(arm.tcp[1] * 100, 0)}, {fmt(arm.tcp[2] * 100, 0)}) cm
            </span>
          }
        >
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
            <Stat label="Arm reach" value={`${Math.round((UPPER + FORE + 0.14) * 100)} cm`} />
          </div>
        </Card>
      </div>

      <aside className="space-y-4">
        <Card title="Arms">
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
            A lead screw drives the torso carriage along two rails on the column (15 cm/s top speed). The base drives on two hidden wheels
            (differential drive) with casters front and back.
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
