/**
 * Task planners. Each turns the current state of a station into a list of steps
 * (drive, lift, arm moves with IK targets, gripper, grab/release, mechanisms),
 * which the runner in Humanoid.tsx then executes one by one.
 */
import * as THREE from 'three'
import type { ClutterKind, Container, PropShape } from '../components/clutter'
import { BAYS, PLACES } from '../components/dining'
import { fk, torsoBase, type Side } from './kinematics'
import { rad } from './math'
import {
  DOCKS,
  KITCHEN,
  toLocal,
  LAUNDRY,
  LIFTS,
  SORT,
  UP,
  WASHER,
  dirWorld,
  doorPoint,
  lidEdge,
  rackOrigin,
  toWorld,
  type RackId,
  type StationId,
} from './stations'

export type TaskId = 'sort' | 'dishes' | 'fold' | 'wash' | 'cleanup' | 'table'
export const TASKS: { id: TaskId; label: string; station: StationId; blurb: string }[] = [
  { id: 'sort', label: 'Sort blocks', station: 'sort', blurb: 'Pick each block and drop it in the bin of its colour, turning at the waist to reach across.' },
  { id: 'dishes', label: 'Load dishwasher', station: 'kitchen', blurb: 'Open the door, pull out the racks, load plates and cups, close up and start it.' },
  { id: 'fold', label: 'Fold towel', station: 'laundry', blurb: 'Take a towel from the basket, spread it flat by two corners, fold it in half twice and put it on the stack.' },
  { id: 'wash', label: 'Wash clothes', station: 'washer', blurb: 'Open the lid, move the clothes from the basket into the drum, close it and press start.' },
  { id: 'table', label: 'Clear the table', station: 'kitchen', blurb: 'Collect the plates into one stack (the cutlery rides on them), carry it to the kitchen and set it down on the counter by the dishwasher.' },
  { id: 'cleanup', label: 'Clean up room', station: 'kitchen', blurb: 'Pick the rubbish off the floor into the bin, towels into the hamper and mugs into the dishwasher, steering round everything.' },
]

/** A hand target in world coordinates: where the fingertips go and how the hand is turned. */
export interface Target {
  pos: THREE.Vector3
  /** Direction the fingers point. */
  approach: THREE.Vector3
  /** Direction the fingers close along (optional: free if omitted). */
  close?: THREE.Vector3
  /** 'world' (default), or 'torso': relative to the shoulders, turning with the waist and riding with the lift. */
  frame?: 'world' | 'torso'
  /** Don't turn the waist for this target. */
  keepWaist?: boolean
  /**
   * Joint angles the arm should settle into on the way (the elbow down at the side, not
   * flared out): the arm's spare freedom drifts toward them while the hand moves.
   */
  posture?: number[]
  /** Close along `close` exactly, not either way round: a hand holding something must not roll over. */
  exactClose?: boolean
}
export type ArmGoal = Target | number[]

export type Mechanism =
  | { kind: 'door'; u: number; s: number; pushOnly?: boolean }
  | { kind: 'rack'; rack: RackId }
  | { kind: 'lid' }
  | { kind: 'fold'; fold: number }

export type Act =
  | { k: 'say'; text: string }
  | { k: 'drive'; to: StationId }
  /** Drive to a spot on the floor and face `yaw`, steering round furniture and litter (`except`: ids not to avoid). */
  | { k: 'goto'; x: number; z: number; yaw: number; except?: string[]; what?: string }
  | { k: 'nudge'; d: number }
  | { k: 'lift'; h: number }
  | { k: 'move'; arms: Partial<Record<Side, ArmGoal>>; via?: boolean; slow?: boolean }
  | { k: 'grip'; side: Side; g: number }
  | { k: 'grab'; side: Side; item: string }
  | { k: 'release'; side: Side; item: string; container: string; pos: [number, number, number]; yaw: number; where: string }
  | { k: 'bind'; side: Side; mech: Mechanism }
  | { k: 'unbind'; side: Side }
  | { k: 'animate'; key: 'door' | 'lowerRack' | 'upperRack' | 'lid'; to: number; dur: number }
  | { k: 'set'; key: 'dishOn' | 'washOn'; value: boolean }
  /** Pinch a named point of the shirt under the hand (`stack`: through all the layers there). */
  | { k: 'clothGrab'; side: Side; feature: string; stack?: boolean; radius?: number }
  | { k: 'clothRelease'; side: Side }
  | { k: 'clothSweep'; side: Side; on: boolean }
  | { k: 'clothPress'; side: Side; on: boolean }
  | { k: 'wait'; t: number }
  /** Fix an item to another where it is now (a plate onto the stack, cutlery onto its plate), or loose it again. */
  | { k: 'glue'; item: string; onto: string }
  | { k: 'unglue'; item: string }
  /** Plan the next steps now, from what the scene looks like at this point in the task. */
  | { k: 'plan'; make: () => Act[] }
  | { k: 'clothFolded' }
  /** Put the towel the hand holds onto the stack of folded towels, and ready the next one. */
  | { k: 'towelStack' }

export interface PlanCtx {
  /** Item world pose and where it is ('home', 'bin', 'rack', 'drum', ...). */
  item: (id: string) => { pos: THREE.Vector3; quat: THREE.Quaternion; where: string }
  ids: (kind: string) => string[]
  /** World position of a named point on the towel ('top' = its highest point, 'centre' = its middle). */
  cloth: (feature: string) => THREE.Vector3
  /** Floor clutter: collider shape and kind of a piece (null for station items). */
  shape: (id: string) => PropShape | null
  kindOf: (id: string) => ClutterKind | null
  /** Where the robot is now. */
  robot: () => { x: number; z: number; yaw: number }
  /** Could the base stand here without touching furniture? */
  free: (x: number, z: number, yaw: number) => boolean
  /** Is the space the arms use when the torso is lowered (round and in front of the base) clear of furniture? */
  workClear: (x: number, z: number, yaw: number) => boolean
  /** Length of the drive there round furniture and litter (Infinity if there's no way). */
  driveLength: (x: number, z: number, yaw: number, except: string[]) => number
  /** What each hand is holding right now. */
  held: () => Partial<Record<Side, string>>
  /** Where a hand's fingertips are now. */
  tcp: (side: Side) => THREE.Vector3
  /** The direction a hand's fingers close along now (world). */
  closeAxis: (side: Side) => THREE.Vector3
  /** The direction a hand's fingers point now (world). */
  handDir: (side: Side) => THREE.Vector3
  bin: Container
  hamper: Container
}

/** The axis a hand's fingers close along, from its tool frame. */
const closeOf = (tool: THREE.Matrix4) => new THREE.Vector3(0, 0, 1).transformDirection(tool)

/** Arms hanging beside the body (used before driving off and at the end). */
export const REST_Q = [0, 8, 0, 15, 0, 0, 0].map(rad)
/**
 * Ready pose between items, like a person's: upper arms down at the sides, forearms
 * forward and the hands carrying on from them, fingers forward and a little down,
 * above table height but short of the table edge.
 */
export const TUCK_Q = [-10, 8, 0, 130, 0, -65, 0].map(rad)
/** Where the hand is when the arm hangs (REST_Q), relative to the shoulders, as a hand target. */
export const hang = (side: Side): Target => {
  const f = fk(side, REST_Q, torsoBase(0, 0))
  return { frame: 'torso', pos: f.tcp, approach: f.dir, close: closeOf(f.tool), keepWaist: true, posture: REST_Q }
}
/** The same ready pose as a hand target, so getting there is a smooth straight-line move. */
export const tuck = (side: Side): Target => {
  const f = fk(side, TUCK_Q, torsoBase(0, 0))
  // The hand's roll is fixed too (palms facing in), so the arm always comes back to this
  // same untwisted posture, wherever it was working.
  return { frame: 'torso', pos: f.tcp, approach: f.dir, close: closeOf(f.tool), keepWaist: true, posture: TUCK_Q }
}

/** Gripper setting whose fingertips just touch an object this wide. */
export function gripFor(width: number) {
  const a = Math.asin(THREE.MathUtils.clamp((width / 2 + 0.003 - 0.024) / 0.062, -1, 1))
  return THREE.MathUtils.clamp((a + 0.3) / 0.86, 0, 1)
}

const DOWN = new THREE.Vector3(0, -1, 0)
const T = (pos: THREE.Vector3, approach: THREE.Vector3, close?: THREE.Vector3): Target => ({ pos: pos.clone(), approach: approach.clone(), close: close?.clone() })
const up = (p: THREE.Vector3, h: number) => p.clone().addScaledVector(UP, h)
const other = (s: Side): Side => (s === 'L' ? 'R' : 'L')

const tuckBoth = (): Act => ({ k: 'move', arms: { L: TUCK_Q, R: TUCK_Q } })
const tuckHands = (): Act => ({ k: 'move', arms: { L: tuck('L'), R: tuck('R') } })

function arrive(st: StationId, text: string): Act[] {
  return [{ k: 'say', text }, tuckBoth(), { k: 'drive', to: st }, { k: 'lift', h: LIFTS[st] }]
}

/** Grasp axis for a block: whichever pair of faces lines up best with `ref`. */
function blockCloseAxis(q: THREE.Quaternion, ref: THREE.Vector3) {
  const cands = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1)].map((v) => v.applyQuaternion(q))
  const best = cands.reduce((a, b) => (Math.abs(b.dot(ref)) > Math.abs(a.dot(ref)) ? b : a))
  return best.dot(ref) < 0 ? best.negate() : best
}

/**
 * "Down, leaning toward the target": pointing straight down puts the wrist 17 cm
 * above the fingertips, which costs reach, so the farther a target is from the
 * shoulder the more the hand tilts toward it (like a person reaching across a table).
 * `shift` is how far the robot stands behind its dock (negative = backed up).
 */
/** Where the robot stands while a plan is built, if not at its dock (station coordinates). */
let standAt: { a: number; b: number } | null = null

function downish(st: StationId, side: Side, p: THREE.Vector3, maxTilt = 55, shift = 0, forwardOnly = false) {
  const l = toLocal(st, p)
  const at = standAt ?? { a: shift, b: 0 }
  const h = new THREE.Vector3(l.x - at.a, 0, l.z - at.b - (side === 'L' ? -0.265 : 0.265))
  const d = h.length()
  // A plate held by its rim must stay in its own plane (a-y): a sideways lean would tilt it
  // out of the narrow rack channel. Lean only forward/back then.
  if (forwardOnly) h.z = 0
  const a = rad(maxTilt) * THREE.MathUtils.clamp((d - 0.36) / 0.24, 0, 1)
  const local = new THREE.Vector3(0, -Math.cos(a), 0).addScaledVector(h.normalize(), Math.sin(a))
  return dirWorld(st, local.x, local.y, local.z)
}

/** Pick up an item from above (hand leaning toward it if far) and lift it clear. */
function pickFromAbove(st: StationId, side: Side, id: string, tcp: THREE.Vector3, close: THREE.Vector3 | undefined, width: number, clear = 0.12, lift = 0.15, maxTilt = 55, forwardOnly = false): Act[] {
  const at = (p: THREE.Vector3) => T(p, downish(st, side, p, maxTilt, 0, forwardOnly), close)
  return [
    { k: 'grip', side, g: Math.min(1, gripFor(width) + 0.35) },
    { k: 'move', arms: { [side]: at(up(tcp, clear)) } },
    { k: 'move', arms: { [side]: at(tcp) }, slow: true },
    { k: 'grip', side, g: gripFor(width) },
    { k: 'grab', side, item: id },
    { k: 'move', arms: { [side]: at(up(tcp, lift)) }, via: true },
  ]
}

/** Lower a held item onto a spot from above and let go. */
function placeFromAbove(
  st: StationId,
  side: Side,
  id: string,
  tcp: THREE.Vector3,
  close: THREE.Vector3 | undefined,
  open: number,
  rel: { container: string; pos: [number, number, number]; yaw: number; where: string },
  clear = 0.15,
  maxTilt = 55,
  /** How high above its resting place the item is let go (it drops the rest of the way). */
  release = 0.012,
  forwardOnly = false,
): Act[] {
  const at = (p: THREE.Vector3) => T(p, downish(st, side, p, maxTilt, 0, forwardOnly), close)
  return [
    { k: 'move', arms: { [side]: at(up(tcp, clear)) } },
    { k: 'move', arms: { [side]: at(up(tcp, release)) }, slow: true },
    { k: 'grip', side, g: open },
    { k: 'release', side, item: id, ...rel },
    { k: 'move', arms: { [side]: at(up(tcp, clear)) }, via: true },
  ]
}

// ---------------------------------------------------------------------------
// Sort blocks into bins by colour
// ---------------------------------------------------------------------------
function planSort(ctx: PlanCtx): Act[] {
  const st: StationId = 'sort'
  const blocks = ctx.ids('block').filter((id) => ctx.item(id).where === 'home')
  if (!blocks.length) return []
  const acts = arrive(st, 'Sorting blocks by colour')
  const right = dirWorld(st, 0, 0, 1)
  const filled: Record<string, number> = {}
  for (const id of ctx.ids('block')) {
    const w = ctx.item(id).where
    if (w.startsWith('bin:')) filled[w.slice(4)] = (filled[w.slice(4)] ?? 0) + 1
  }
  const floorY = SORT.table.h + SORT.binFloor
  for (const id of blocks) {
    const color = SORT.blocks[Number(id.replace('block', ''))].color
    const bin = SORT.bins.find((b) => b.color === color)!
    const n = filled[color] ?? 0
    filled[color] = n + 1
    const [da, db] = SORT.slots[n % SORT.slots.length]
    const { pos, quat } = ctx.item(id)
    const local = toLocal(st, pos)
    // The hand on the block's side picks it (a precise grasp), then turns at the waist
    // to drop it in its bin, even across the body (letting go needs no particular wrist angle).
    const placer: Side = local.z < 0 ? 'L' : 'R'
    const close = blockCloseAxis(quat, right)
    const slotLocal: [number, number, number] = [bin.a + da, floorY + SORT.block / 2, bin.b + db]
    const slot = toWorld(st, ...slotLocal)
    acts.push({ k: 'say', text: `${color[0].toUpperCase() + color.slice(1)} block` })
    acts.push({ k: 'move', arms: { [other(placer)]: tuck(other(placer)) } }, ...pickFromAbove(st, placer, id, pos, close, SORT.block))
    // Let go with the fingertips just above the tub rim: the block drops in, and the hand
    // never dips into a tub where it could catch the rim.
    const rimClear = SORT.binWall + 0.004 + 0.03 - SORT.block / 2
    acts.push(...placeFromAbove(st, placer, id, slot, undefined, gripFor(SORT.block) + 0.3, { container: st, pos: slotLocal, yaw: 0, where: `bin:${color}` }, rimClear + 0.09, 55, rimClear))
  }
  acts.push(tuckHands(), { k: 'say', text: 'All blocks sorted' })
  return acts
}

// ---------------------------------------------------------------------------
// Load the dishwasher
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Dishwasher door and racks (shared by Load dishwasher and Clean up room)
// ---------------------------------------------------------------------------
/**
 * The dishwasher is built in under the worktop. Its door folds down and reaches out over the
 * floor in front, so the robot works it from beside the door (KITCHEN.dishPose), turning at
 * the waist; the right arm handles the door, racks and cups (it's on the dishwasher side).
 */
const kitchenFrame = () => {
  const st: StationId = 'kitchen'
  return { st, fwd: dirWorld(st, 1, 0, 0), right: dirWorld(st, 0, 0, 1), W: (a: number, y: number, b: number) => toWorld(st, a, y, b) }
}

/** An idle arm folded up close to the chest: above the shoulders and barely forward, clear of any counter edge. */
const fold = (side: Side): Target => ({
  frame: 'torso',
  pos: new THREE.Vector3(0.18, 0.04, side === 'L' ? -0.2 : 0.2),
  approach: new THREE.Vector3(0.3, -0.95, 0).normalize(),
  keepWaist: true,
})

/** Drive to the spot beside the dishwasher door (and face the counter). */
function dishGoto(): Act {
  const p = KITCHEN.dishPose
  const w = toWorld('kitchen', p.a, 0, p.b)
  return { k: 'goto', x: w.x, z: w.z, yaw: DOCKS.kitchen.yaw, what: 'the dishwasher' }
}

/**
 * Torso height for working at fingertip height y: the shoulders about level with the wrist,
 * which is 0.17 m above the fingertips when they point down, level with them when they point ahead.
 */
const liftFor = (y: number, down = true) => THREE.MathUtils.clamp(y + (down ? 0.2 : 0.06), 0.55, 1.12)

/** Open the door: hook the handle, pull it down a little, then let it drop open. */
function doorOpenActs(): Act[] {
  const K = KITCHEN
  const { fwd, W } = kitchenFrame()
  const handleAt = (phi: number) => {
    const [a, y] = doorPoint(K.handle.u, K.handle.s, phi)
    return W(a, y, 0.04)
  }
  const acts: Act[] = [
    { k: 'say', text: 'Opening the door' },
    { k: 'lift', h: liftFor(handleAt(0).y, false) },
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'move', arms: { R: T(handleAt(0).addScaledVector(fwd, -0.08), fwd, UP) } },
    { k: 'move', arms: { R: T(handleAt(0), fwd, UP) }, slow: true },
    { k: 'grip', side: 'R', g: gripFor(0.02) },
    { k: 'bind', side: 'R', mech: { kind: 'door', u: K.handle.u, s: K.handle.s } },
  ]
  const openDeg = Math.round((K.openByHand * 180) / Math.PI)
  for (let d = 10; d <= openDeg; d += 10) acts.push({ k: 'move', arms: { R: T(handleAt(rad(d)), fwd, UP) }, via: d < openDeg, slow: true })
  acts.push(
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'unbind', side: 'R' },
    { k: 'move', arms: { R: T(handleAt(K.openByHand).addScaledVector(UP, 0.06).addScaledVector(fwd, -0.06), fwd, UP) } },
    { k: 'animate', key: 'door', to: Math.PI / 2, dur: 1.1 },
    { k: 'move', arms: { R: tuck('R') } },
  )
  return acts
}

/**
 * Pull a rack out (or push it back in) by its front handle. Pushed all the way in, the handle
 * is under the worktop's edge: the hand pushes it most of the way and a last shove sends it home.
 */
function rackMoveActs(rack: RackId, from: number, to: number): Act[] {
  const K = KITCHEN
  const { fwd, W } = kitchenFrame()
  const handle = (out: number) => {
    const p = rackOrigin(rack, out)
    return W(p.x - 0.015, p.y + K.racks[rack].wall + 0.02, 0.05)
  }
  const handEnd = Math.max(to, 0.14) // pushing in, the hand (fingertips too) stops short of the cabinet; the rack runs on
  const h0 = handle(from) // the hand takes the handle where it really is
  const h1 = handle(handEnd)
  // The low rack's handle is near the floor: the hand takes it from above and in front (leaning
  // 35° down), a grip the wrist can hold that low; the upper rack's straight on.
  const lean = rack === 'lower' ? rad(35) : 0
  const ap = fwd.clone().multiplyScalar(Math.cos(lean)).addScaledVector(UP, -Math.sin(lean))
  const cl = UP.clone().multiplyScalar(Math.cos(lean)).addScaledVector(fwd, Math.sin(lean))
  const acts: Act[] = [
    { k: 'say', text: `${to > from ? 'Pulling out' : 'Pushing in'} the ${rack} rack` },
    { k: 'lift', h: liftFor(h0.y + 0.1, false) },
    { k: 'grip', side: 'R', g: 0.7 },
    // Come in from out in front of the rack, not across it (dishes stand near its front edge):
    // well above the plate tops first, then down in front of the rack.
    { k: 'move', arms: { R: T(h0.clone().addScaledVector(fwd, -0.25).addScaledVector(UP, 0.3), ap, cl) } },
    { k: 'move', arms: { R: T(h0.clone().addScaledVector(fwd, -0.2).addScaledVector(UP, 0.12), ap, cl) } },
    { k: 'move', arms: { R: T(h0.clone().addScaledVector(fwd, -0.08).addScaledVector(UP, 0.02), ap, cl) } },
    { k: 'move', arms: { R: T(h0, ap, cl) }, slow: true },
    { k: 'grip', side: 'R', g: gripFor(0.012) },
    { k: 'bind', side: 'R', mech: { kind: 'rack', rack } },
    { k: 'move', arms: { R: T(h1, ap, cl) }, slow: true },
    { k: 'grip', side: 'R', g: 0.7 },
    { k: 'unbind', side: 'R' },
  ]
  // The rack runs on to its stop (all the way out, or home), wherever the hand let go of it.
  // Gently: plates standing on edge fall over if the rack jerks.
  acts.push({ k: 'animate', key: rack === 'lower' ? 'lowerRack' : 'upperRack', to, dur: 0.8 })
  // Straight back off the handle at its height first, then up: the fingers stay level, clear of the open door below.
  acts.push(
    { k: 'move', arms: { R: T(h1.clone().addScaledVector(fwd, -0.1), ap, cl) } },
    { k: 'move', arms: { R: T(h1.clone().addScaledVector(fwd, -0.14).addScaledVector(UP, 0.14), ap, cl) } },
    { k: 'move', arms: { R: tuck('R') } },
  )
  return acts
}

/**
 * Close the door from beside it: take its free edge where it lies open, lift it up through its
 * arc, then push it shut with the fingertips (and press start).
 */
function doorCloseActs(start: boolean): Act[] {
  const K = KITCHEN
  const { fwd, W } = kitchenFrame()
  const edgeAt = (phi: number) => {
    const [a, y] = doorPoint(K.edge.u, K.edge.s, phi)
    return W(a, y, 0.04)
  }
  const pushAt = (phi: number) => {
    const [a, y] = doorPoint(K.push.u, K.push.s, phi)
    return W(a, y, 0.04)
  }
  const liftDeg = Math.round((K.liftTo * 180) / Math.PI)
  const acts: Act[] = [
    { k: 'say', text: 'Closing the door: lifting it by its edge' },
    { k: 'move', arms: { R: tuck('R') } },
    { k: 'lift', h: 0.62 },
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'move', arms: { R: T(edgeAt(Math.PI / 2).addScaledVector(UP, 0.08), DOWN, fwd) } },
    { k: 'move', arms: { R: T(edgeAt(Math.PI / 2), DOWN, fwd) }, slow: true },
    { k: 'grip', side: 'R', g: gripFor(K.door.thick) },
    { k: 'bind', side: 'R', mech: { kind: 'door', u: K.edge.u, s: K.edge.s } },
  ]
  // Up through the arc; the fingers turn from pointing down to pointing at the cabinet.
  for (let d = 80; d >= liftDeg; d -= 10) {
    const t = (90 - d) / (90 - liftDeg)
    const dir = DOWN.clone().lerp(fwd, t).normalize()
    acts.push({ k: 'lift', h: 0.62 + 0.3 * t }, { k: 'move', arms: { R: T(edgeAt(rad(d)), dir, fwd.clone().cross(UP).cross(dir).normalize()) }, via: d > liftDeg, slow: true })
  }
  acts.push(
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'unbind', side: 'R' },
    { k: 'move', arms: { R: T(edgeAt(K.liftTo).addScaledVector(fwd, -0.1), fwd) } },
    { k: 'say', text: 'Pushing the door shut' },
    { k: 'grip', side: 'R', g: 0 },
    { k: 'move', arms: { R: T(pushAt(K.liftTo).addScaledVector(fwd, -0.05), fwd) } },
    { k: 'bind', side: 'R', mech: { kind: 'door', u: K.push.u, s: K.push.s, pushOnly: true } },
  )
  for (let d = liftDeg - 10; d >= 0; d -= 10) acts.push({ k: 'move', arms: { R: T(pushAt(rad(d)), fwd) }, via: d > 0, slow: true })
  acts.push(
    { k: 'unbind', side: 'R' },
    { k: 'animate', key: 'door', to: 0, dur: 0.15 },
    // Straight back from the door before tucking (not diagonally past the counter's corner).
    { k: 'move', arms: { R: T(pushAt(0).addScaledVector(fwd, -0.16), fwd) } },
  )
  if (!start) {
    acts.push({ k: 'move', arms: { R: tuck('R') } })
    return acts
  }
  const btn = W(K.button.a, K.button.y, K.button.b)
  acts.push(
    { k: 'move', arms: { R: T(btn.clone().addScaledVector(fwd, -0.08), fwd) } },
    { k: 'move', arms: { R: T(btn.clone().addScaledVector(fwd, 0.004), fwd) }, slow: true },
    { k: 'set', key: 'dishOn', value: true },
    { k: 'move', arms: { R: T(btn.clone().addScaledVector(fwd, -0.16), fwd) } },
    { k: 'move', arms: { R: tuck('R') } },
    tuckHands(),
  )
  return acts
}

/** A point clear in front of the counters, above the rack, to bring a held dish to before the torso goes down. */
const outFront = (b: number, y: number) => kitchenFrame().W(0.18, y, b)

function planDishes(ctx: PlanCtx): Act[] {
  const st: StationId = 'kitchen'
  const plates = ctx.ids('plate').filter((id) => ctx.item(id).where === 'home')
  const cups = ctx.ids('cup').filter((id) => ctx.item(id).where === 'home')
  if (!plates.length && !cups.length) return []
  const K = KITCHEN
  const right = dirWorld(st, 0, 0, 1)
  const W = (a: number, y: number, b: number) => toWorld(st, a, y, b)
  // Everything happens from beside the dishwasher; the hand directions are worked out from there.
  standAt = { a: K.dishPose.a, b: K.dishPose.b }
  // The left arm isn't needed: it waits folded up against the chest, clear of the counter whatever the torso height.
  const acts: Act[] = [{ k: 'say', text: 'Loading the dishwasher' }, tuckBoth(), { k: 'lift', h: 0.95 }, dishGoto(), { k: 'move', arms: { L: fold('L') } }, ...doorOpenActs()]
  const out = K.racks.lower.travel
  if (plates.length) {
    acts.push(...rackMoveActs('lower', 0, out))
    plates.forEach((id, i) => {
      const { pos } = ctx.item(id)
      const rim = up(pos, K.plate.r - 0.018)
      const o = rackOrigin('lower', out)
      // Farthest slot first: the arm then never reaches over a plate already in the rack.
      const slots = [...K.plateSlots].sort((x, y) => y - x)
      const slotLocal: [number, number, number] = [K.plateSlotA, 0.01 + K.plate.r, slots[i % slots.length]]
      const slotRim = up(W(o.x + slotLocal[0], o.y + slotLocal[1], slotLocal[2]), K.plate.r - 0.018)
      const slotCentre = W(o.x + slotLocal[0], o.y + slotLocal[1], slotLocal[2])
      acts.push(
        // The right hand: it's the one on the dishwasher side, in reach of both the drying rack and the slots.
        { k: 'say', text: `Plate ${i + 1}: right hand pinches the rim` },
        { k: 'lift', h: liftFor(rim.y) },
        ...pickFromAbove(st, 'R', id, rim, right, K.plate.t, 0.1, 0.16, 30, true),
        // Out in front of the counter before the torso goes down to the rack.
        { k: 'move', arms: { R: T(outFront(slotLocal[2], rim.y + 0.16), downish(st, 'R', outFront(slotLocal[2], rim.y), 30, 0, true), right) } },
        { k: 'lift', h: liftFor(slotRim.y) },
        // Put the plate (not the hand) on the slot: allow for where it really sits in the fingers.
        {
          k: 'plan',
          make: () => {
            const tcp = slotCentre.clone().add(ctx.tcp('R').sub(ctx.item(id).pos))
            standAt = { a: K.dishPose.a, b: K.dishPose.b }
            const place = placeFromAbove(st, 'R', id, tcp, right, 0.45, { container: 'kitchen.lower', pos: slotLocal, yaw: 0, where: 'rack' }, 0.14, 30, 0.012, true)
            standAt = null
            return place
          },
        },
        // Up out of the rack with the torso before the hand turns back to the ready pose,
        // so the fingers never swing through the dishes.
        { k: 'lift', h: 0.95 },
        { k: 'move', arms: { R: tuck('R') } },
      )
    })
    acts.push(...rackMoveActs('lower', out, 0))
  }
  if (cups.length) {
    acts.push(...rackMoveActs('upper', 0, out))
    cups.forEach((id, i) => {
      const { pos } = ctx.item(id)
      const o = rackOrigin('upper', out)
      // Farthest slot first, for the same reason as the plates.
      const cupSlots = [...K.cupSlots].sort((x, y) => y.b - x.b)
      const sl = cupSlots[i % cupSlots.length]
      const slotLocal: [number, number, number] = [sl.a, K.cup.h / 2 + 0.004, sl.b]
      const slot = W(o.x + slotLocal[0], o.y + slotLocal[1], slotLocal[2])
      acts.push(
        { k: 'say', text: `Cup ${i + 1}: right hand grips it from above` },
        { k: 'lift', h: liftFor(pos.y) },
        ...pickFromAbove(st, 'R', id, pos, undefined, 2 * K.cup.r, 0.1, 0.16, 30),
        { k: 'move', arms: { R: T(outFront(sl.b, pos.y + 0.16), DOWN) } },
        { k: 'lift', h: liftFor(slot.y) },
        // Straight down onto the slot (leaning costs reach out here), allowing for where the cup sits in the hand.
        {
          k: 'plan',
          make: () => {
            const tcp = slot.clone().add(ctx.tcp('R').sub(ctx.item(id).pos))
            standAt = { a: K.dishPose.a, b: K.dishPose.b }
            const place = placeFromAbove(st, 'R', id, tcp, undefined, Math.min(1, gripFor(2 * K.cup.r) + 0.25), { container: 'kitchen.upper', pos: slotLocal, yaw: 0, where: 'rack' }, 0.15, 0)
            standAt = null
            return place
          },
        },
        { k: 'move', arms: { R: tuck('R') } },
      )
    })
    acts.push(...rackMoveActs('upper', out, 0))
  }
  acts.push(...doorCloseActs(true), { k: 'say', text: 'Dishwasher loaded and running' })
  standAt = null
  return acts
}

// ---------------------------------------------------------------------------
// Fold the shirt
// ---------------------------------------------------------------------------
function planFold(ctx: PlanCtx, towelsLeft: number): Act[] {
  if (towelsLeft <= 0) return []
  const st: StationId = 'laundry'
  const T0 = LAUNDRY.table.h
  const pinchY = T0 + 0.014
  const half = LAUNDRY.towel.half - 0.01
  const L = (p: THREE.Vector3) => toLocal(st, p)
  const W = (a: number, y: number, b: number) => toWorld(st, a, y, b)
  const at = (side: Side, p: THREE.Vector3) => T(p, downish(st, side, p, 55))
  const loc = (f: string) => {
    const p = L(ctx.cloth(f))
    p.y = pinchY
    return p
  }
  const Wv = (p: THREE.Vector3) => W(p.x, p.y, p.z)
  const acts = arrive(st, 'Folding a towel from the basket')
  const steps = 10

  // 0. Take the top towel out of the basket and drop it on the table in front of the robot.
  acts.push({
    k: 'plan',
    make: () => {
      const p = ctx.cloth('top').addScaledVector(UP, 0.004)
      const hold = (q: THREE.Vector3) => ({ ...at('L', q), keepWaist: true })
      return [
        { k: 'say', text: 'Taking a towel from the basket' },
        { k: 'grip', side: 'L', g: 0.5 },
        { k: 'move', arms: { L: at('L', up(p, 0.1)) } },
        { k: 'move', arms: { L: at('L', p) }, slow: true },
        { k: 'grip', side: 'L', g: 0 },
        { k: 'clothGrab', side: 'L', feature: 'nearest' },
        // Straight up out of the basket, across over the table, then down and forward so it
        // lands in a loose heap in front of the robot.
        { k: 'move', arms: { L: hold(W(0.52, T0 + 0.38, -0.5)) }, slow: true },
        { k: 'move', arms: { L: hold(W(0.5, T0 + 0.36, -0.04)) }, slow: true },
        { k: 'move', arms: { L: hold(W(0.6, T0 + 0.05, 0.02)) }, slow: true },
        { k: 'grip', side: 'L', g: 0.5 },
        { k: 'clothRelease', side: 'L' },
        { k: 'move', arms: { L: at('L', W(0.58, T0 + 0.15, 0.0)) } },
        { k: 'move', arms: { L: tuck('L') } },
        { k: 'wait', t: 0.8 },
      ]
    },
  })

  /** The towel's own axes as it lies now: u from the near edge to the far edge, v across (toward +b). */
  const axes = () => {
    const far = loc('topInL').add(loc('topInR')).multiplyScalar(0.5)
    const near = loc('botInL').add(loc('botInR')).multiplyScalar(0.5)
    const u = far.clone().sub(near).setY(0).normalize()
    const v = new THREE.Vector3(-u.z, 0, u.x)
    if (v.z < 0) v.negate()
    return { u, v, centre: far.clone().add(near).multiplyScalar(0.5), far, near }
  }
  /**
   * The towel's axes once it has been folded in half (its near and far edges then lie on each
   * other, so they no longer give a direction): v along the strip from end to end, u across it.
   */
  const stripAxes = () => {
    const a = loc('endL')
    const b = loc('endR')
    const v = b.clone().sub(a).setY(0).normalize()
    if (v.z < 0) v.negate()
    const u = new THREE.Vector3(v.z, 0, -v.x)
    if (u.x < 0) u.negate()
    return { u, v }
  }
  /**
   * Hand position carrying point p over the line through p0 along `axis`, at angle th (0 to
   * pi), along a slightly shortened arc so the flap is carried with a little slack.
   */
  const over = (p: THREE.Vector3, p0: THREE.Vector3, axis: THREE.Vector3, th: number, height = 1) => {
    const r = p.clone().sub(p0).setY(0)
    const along = axis.clone().multiplyScalar(r.dot(axis))
    const w = r.clone().sub(along)
    const d = w.length()
    const R = th > 0 ? Math.max(0.02, d - 0.005) : d
    const q = p0.clone().add(along).addScaledVector(w, (Math.cos(th) * R) / Math.max(d, 1e-6))
    q.y = pinchY + R * Math.sin(th) * height
    return W(q.x, q.y, q.z)
  }
  /**
   * One fold with both hands: grip points pL (left hand) and pR (right hand), pull the edge
   * between them 1 cm taut (if `taut`), carry both over the line through p0 along `axis`,
   * let go and lift clear.
   */
  const fold = (label: string, pL: THREE.Vector3, fL: string, pR: THREE.Vector3, fR: string, p0: THREE.Vector3, axis: THREE.Vector3, height: number, taut: boolean): Act[] => {
    const dir = pR.clone().sub(pL).setY(0).normalize()
    const tL = taut ? pL.clone().addScaledVector(dir, -0.01) : pL
    const tR = taut ? pR.clone().addScaledVector(dir, 0.01) : pR
    const list: Act[] = [
      { k: 'say', text: label },
      { k: 'grip', side: 'L', g: 0.5 },
      { k: 'grip', side: 'R', g: 0.5 },
      { k: 'move', arms: { L: at('L', up(Wv(pL), 0.08)), R: at('R', up(Wv(pR), 0.08)) } },
      { k: 'move', arms: { L: at('L', Wv(pL)), R: at('R', Wv(pR)) }, slow: true },
      { k: 'grip', side: 'L', g: 0 },
      { k: 'grip', side: 'R', g: 0 },
      { k: 'clothGrab', side: 'L', feature: fL, stack: true },
      { k: 'clothGrab', side: 'R', feature: fR, stack: true },
    ]
    if (taut) list.push({ k: 'move', arms: { L: at('L', Wv(tL)), R: at('R', Wv(tR)) }, slow: true })
    for (let i = 1; i <= steps; i++) {
      const th = (Math.PI * i) / steps
      list.push({ k: 'move', arms: { L: at('L', over(tL, p0, axis, th, height)), R: at('R', over(tR, p0, axis, th, height)) }, via: i < steps, slow: true })
    }
    list.push(
      { k: 'grip', side: 'L', g: 0.5 },
      { k: 'grip', side: 'R', g: 0.5 },
      { k: 'clothRelease', side: 'L' },
      { k: 'clothRelease', side: 'R' },
      { k: 'move', arms: { L: at('L', up(over(tL, p0, axis, Math.PI, height), 0.1)), R: at('R', up(over(tR, p0, axis, Math.PI, height), 0.1)) }, via: true },
      tuckHands(),
      { k: 'wait', t: 0.4 },
    )
    return list
  }

  /**
   * Press the folded towel flat, as the robot in the reference video does between folds: both
   * closed grippers come down on it side by side at each spot along it, push the layers flat,
   * and lift off. `spots` are station-local points along the towel's middle.
   */
  const press = (spots: THREE.Vector3[], label: string): Act[] => {
    const { v } = stripAxes()
    const h = T0 + 0.02
    const list: Act[] = [{ k: 'say', text: label }, { k: 'grip', side: 'L', g: 0 }, { k: 'grip', side: 'R', g: 0 }]
    for (const c of spots) {
      const pL = c.clone().addScaledVector(v, -0.05)
      const pR = c.clone().addScaledVector(v, 0.05)
      const P = (p: THREE.Vector3, y: number) => W(p.x, y, p.z)
      list.push(
        { k: 'move', arms: { L: at('L', P(pL, h + 0.05)), R: at('R', P(pR, h + 0.05)) } },
        { k: 'clothPress', side: 'L', on: true },
        { k: 'clothPress', side: 'R', on: true },
        { k: 'move', arms: { L: at('L', P(pL, h)), R: at('R', P(pR, h)) }, slow: true },
        { k: 'wait', t: 0.3 },
        { k: 'move', arms: { L: at('L', P(pL, h + 0.05)), R: at('R', P(pR, h + 0.05)) }, slow: true },
        { k: 'clothPress', side: 'L', on: false },
        { k: 'clothPress', side: 'R', on: false },
      )
    }
    list.push(tuckHands(), { k: 'wait', t: 0.3 })
    return list
  }

  /**
   * Second fold: the left hand takes the left end of the strip (through both layers) and lays it
   * on the right end, along a low arc, with the waist turning to give the reach across.
   */
  const halfAgain = (): Act[] => {
    const { u, v } = stripAxes()
    const lefter = loc('endL').dot(v) <= loc('endR').dot(v)
    const fL = lefter ? 'endL' : 'endR'
    const fR = lefter ? 'endR' : 'endL'
    const pL = loc(fL).addScaledVector(v, 0.012)
    const line = pL.clone().add(loc(fR)).multiplyScalar(0.5)
    const list: Act[] = [
      { k: 'say', text: 'Folding it in half again' },
      { k: 'move', arms: { R: tuck('R') } },
      { k: 'grip', side: 'L', g: 0.5 },
      { k: 'move', arms: { L: at('L', up(Wv(pL), 0.08)) } },
      { k: 'move', arms: { L: at('L', Wv(pL)) }, slow: true },
      { k: 'grip', side: 'L', g: 0 },
      { k: 'clothGrab', side: 'L', feature: fL, stack: true, radius: 0.06 },
    ]
    for (let i = 1; i <= steps; i++) list.push({ k: 'move', arms: { L: at('L', over(pL, line, u, (Math.PI * i) / steps, 0.55)) }, via: i < steps, slow: true })
    list.push(
      { k: 'grip', side: 'L', g: 0.5 },
      { k: 'clothRelease', side: 'L' },
      { k: 'move', arms: { L: at('L', up(over(pL, line, u, Math.PI, 0.55), 0.1)) }, via: true },
      tuckHands(),
      { k: 'wait', t: 0.4 },
    )
    return list
  }

  /** Both ends of the folded strip in to the middle at once, each hand gripping its end's middle. */
  const gate = (): Act[] => {
    const { u, v } = stripAxes()
    const lefter = loc('endL').dot(v) <= loc('endR').dot(v)
    const fL = lefter ? 'endL' : 'endR'
    const fR = lefter ? 'endR' : 'endL'
    const pL = loc(fL).addScaledVector(v, 0.01)
    const pR = loc(fR).addScaledVector(v, -0.01)
    const mid = pL.clone().add(pR).multiplyScalar(0.5)
    // Each end is folded over the line halfway between it and the middle, so the ends meet.
    const lineL = pL.clone().add(mid).multiplyScalar(0.5)
    const lineR = pR.clone().add(mid).multiplyScalar(0.5)
    const list: Act[] = [
      { k: 'say', text: 'Both hands fold the ends in to the middle' },
      { k: 'grip', side: 'L', g: 0.5 },
      { k: 'grip', side: 'R', g: 0.5 },
      { k: 'move', arms: { L: at('L', up(Wv(pL), 0.08)), R: at('R', up(Wv(pR), 0.08)) } },
      { k: 'move', arms: { L: at('L', Wv(pL)), R: at('R', Wv(pR)) }, slow: true },
      { k: 'grip', side: 'L', g: 0 },
      { k: 'grip', side: 'R', g: 0 },
      { k: 'clothGrab', side: 'L', feature: fL, stack: true, radius: 0.05 },
      { k: 'clothGrab', side: 'R', feature: fR, stack: true, radius: 0.05 },
    ]
    for (let i = 1; i <= steps; i++) {
      const th = (Math.PI * i) / steps
      list.push({ k: 'move', arms: { L: at('L', over(pL, lineL, u, th)), R: at('R', over(pR, lineR, u, th)) }, via: i < steps, slow: true })
    }
    list.push(
      { k: 'grip', side: 'L', g: 0.5 },
      { k: 'grip', side: 'R', g: 0.5 },
      { k: 'clothRelease', side: 'L' },
      { k: 'clothRelease', side: 'R' },
      { k: 'move', arms: { L: at('L', up(over(pL, lineL, u, Math.PI), 0.1)), R: at('R', up(over(pR, lineR, u, Math.PI), 0.1)) }, via: true },
      tuckHands(),
      { k: 'wait', t: 0.4 },
    )
    return list
  }

  /** Smooth it flat: both hands press down in the middle and slide apart, then lengthways. */
  const smooth = (): Act[] => {
    const { u, v, centre } = axes()
    const y = T0 + 0.01
    const pt = (du: number, dv: number) => {
      const p = centre.clone().addScaledVector(u, du).addScaledVector(v, dv)
      return W(p.x, y, p.z)
    }
    const pass = (from: [number, number, number, number], to: [number, number, number, number]): Act[] => [
      { k: 'move', arms: { L: at('L', up(pt(from[0], from[1]), 0.06)), R: at('R', up(pt(from[2], from[3]), 0.06)) } },
      { k: 'move', arms: { L: at('L', pt(from[0], from[1])), R: at('R', pt(from[2], from[3])) }, slow: true },
      { k: 'clothSweep', side: 'L', on: true },
      { k: 'clothSweep', side: 'R', on: true },
      { k: 'move', arms: { L: at('L', pt(to[0], to[1])), R: at('R', pt(to[2], to[3])) }, slow: true },
      { k: 'clothSweep', side: 'L', on: false },
      { k: 'clothSweep', side: 'R', on: false },
      { k: 'move', arms: { L: at('L', up(pt(to[0], to[1]), 0.06)), R: at('R', up(pt(to[2], to[3]), 0.06)) } },
    ]
    return [
      { k: 'say', text: 'Smoothing it flat with both hands' },
      { k: 'grip', side: 'L', g: 0.15 },
      { k: 'grip', side: 'R', g: 0.15 },
      ...pass([0.06, -0.04, 0.06, 0.04], [0.06, -0.2, 0.06, 0.2]),
      ...pass([-0.06, -0.04, -0.06, 0.04], [-0.06, -0.2, -0.06, 0.2]),
      { k: 'move', arms: { L: tuck('L'), R: tuck('R') } },
      { k: 'wait', t: 0.4 },
    ]
  }

  /**
   * 2. Fold in half, then both ends in to the middle (a gate fold), each step planned from
   * where the towel actually lies at that moment.
   */
  const folds = (): Act[] => [
    {
      // In half: both hands take the near corners and lay the near edge on the far edge.
      k: 'plan',
      make: () => {
        const { v, centre } = axes()
        const [fl, fr] = loc('botL').dot(v) <= loc('botR').dot(v) ? ['botL', 'botR'] : ['botR', 'botL']
        return fold('Both hands fold it in half', loc(fl), fl, loc(fr), fr, centre, v, 0.8, true)
      },
    },
    // Press the strip flat along its length before the next fold.
    {
      k: 'plan',
      make: () => {
        const { v } = stripAxes()
        const mid = loc('endL').add(loc('endR')).multiplyScalar(0.5)
        return press([-0.12, 0, 0.12].map((d) => mid.clone().addScaledVector(v, d)), 'Pressing the fold flat')
      },
    },
    // In half again: the left end of the strip over onto the right end.
    { k: 'plan', make: halfAgain },
    // Press the finished towel flat.
    {
      k: 'plan',
      make: () => {
        const { u } = stripAxes()
        const mid = loc('endL').add(loc('endR')).multiplyScalar(0.5)
        return press([0].map((d) => mid.clone().addScaledVector(u, d)), 'Pressing the towel flat')
      },
    },
    { k: 'clothFolded' },
    { k: 'say', text: 'Towel folded' },
    {
      // Carry the folded towel to the stack with both hands, one at each end (through all its
      // layers), and lay it down gently on top of the pile before letting go.
      k: 'plan',
      make: () => {
        // Pinch points: 2 cm in from each end, through every layer.
        const eL = L(ctx.cloth('edgeL'))
        const eR = L(ctx.cloth('edgeR'))
        const top = ctx.cloth('top').y
        const half = Math.max(0.04, (eR.z - eL.z) / 2 - 0.02)
        const c = eL.clone().add(eR).multiplyScalar(0.5)
        const grip = (d: number) => W(c.x, top - 0.004, c.z + d)
        const S = LAUNDRY.stack
        const lift = 0.06
        const list: Act[] = [
          { k: 'say', text: 'Both hands pinch it by the ends and carry it to the stack' },
          { k: 'grip', side: 'L', g: 0.5 },
          { k: 'grip', side: 'R', g: 0.5 },
          { k: 'move', arms: { L: at('L', up(grip(-half), 0.07)), R: at('R', up(grip(half), 0.07)) } },
          { k: 'move', arms: { L: at('L', grip(-half)), R: at('R', grip(half)) }, slow: true },
          { k: 'grip', side: 'L', g: 0 },
          { k: 'grip', side: 'R', g: 0 },
          // Each hand pinches its end through every layer there.
          { k: 'clothGrab', side: 'L', feature: 'nearest', radius: 0.035 },
          { k: 'clothGrab', side: 'R', feature: 'nearest', radius: 0.035 },
          { k: 'move', arms: { L: at('L', up(grip(-half), lift)), R: at('R', up(grip(half), lift)) }, slow: true },
          {
            k: 'plan',
            make: () => {
              const pile = ctx.cloth('stack').y
              const P = (d: number, h: number) => W(S.a, h, S.b + d)
              return [
                { k: 'move', arms: { L: at('L', P(-half, pile + lift)), R: at('R', P(half, pile + lift)) }, slow: true },
                // Down gently until it rests on the pile, then let go and lift the hands clear.
                { k: 'move', arms: { L: at('L', P(-half, pile + 0.03)), R: at('R', P(half, pile + 0.03)) }, slow: true },
                { k: 'move', arms: { L: at('L', P(-half, pile + 0.018)), R: at('R', P(half, pile + 0.018)) }, slow: true },
                { k: 'grip', side: 'L', g: 0.5 },
                { k: 'grip', side: 'R', g: 0.5 },
                { k: 'clothRelease', side: 'L' },
                { k: 'clothRelease', side: 'R' },
                { k: 'move', arms: { L: at('L', P(-half, pile + 0.1)), R: at('R', P(half, pile + 0.1)) }, slow: true },
                { k: 'wait', t: 1 },
                { k: 'towelStack' },
                tuckHands(),
              ]
            },
          },
        ]
        return list
      },
    },
  ]

  // 1. Spread it: pick it up by one corner and let it hang, catch the corner at the other end
  //    of that long edge, stretch the edge out, fling it, and lay it down flat.
  acts.push({
    k: 'plan',
    make: () => {
      const [fl, fr] = L(ctx.cloth('topL')).z <= L(ctx.cloth('topR')).z ? ['topL', 'topR'] : ['topR', 'topL']
      // While one hand holds the towel, the other reaches without turning the waist.
      const grab = (side: Side, f: string, holding = false): Act[] => {
        const p = ctx.cloth(f).addScaledVector(UP, 0.006)
        const go = (q: THREE.Vector3) => ({ ...at(side, q), keepWaist: holding })
        return [
          { k: 'grip', side, g: 0.5 },
          { k: 'move', arms: { [side]: go(up(p, 0.09)) } },
          {
            // Look again just before closing: cloth shifts.
            k: 'plan',
            make: () => [
              { k: 'move', arms: { [side]: go(ctx.cloth(f).addScaledVector(UP, 0.006)) }, slow: true },
              { k: 'grip', side, g: 0 },
              { k: 'clothGrab', side, feature: f },
            ],
          },
        ]
      }
      const both = (a: number, h: number, w = half) => ({ L: at('L', W(a, T0 + h, -w)), R: at('R', W(a, T0 + h, w)) })
      const list: Act[] = [
        { k: 'say', text: 'Picking it up by one corner and letting it hang' },
        { k: 'move', arms: { L: TUCK_Q, R: TUCK_Q } },
        ...grab('L', fl, true),
        // Lift that corner up close in front of the body: the towel hangs from it and untwists,
        // and its next corner ends up within easy reach of the other hand.
        { k: 'move', arms: { L: { ...at('L', W(0.42, T0 + 0.3, -0.12)), keepWaist: true } }, slow: true },
        { k: 'wait', t: 1.2 },
        { k: 'say', text: 'Catching the next corner' },
        { k: 'plan', make: () => grab('R', fr, true) },
        // Both corners held: torso up, so the towel can hang clear of the table.
        { k: 'lift', h: 1.18 },
        { k: 'say', text: 'Stretching the edge out and flinging it flat' },
        { k: 'move', arms: both(0.45, 0.4), slow: true },
        { k: 'wait', t: 0.6 },
        { k: 'move', arms: both(0.6, 0.45) },
        { k: 'move', arms: both(0.4, 0.42) },
        { k: 'wait', t: 0.4 },
        { k: 'lift', h: LIFTS[st] },
      ]
      // Lay it down: forward as fast as down, so it lies out flat behind the hands.
      const n = 10
      const [a0, h0, a1, h1] = [0.4, 0.42 - (1.18 - LIFTS[st]), 0.66, 0.016]
      for (let i = 1; i <= n; i++) {
        const t = i / n
        list.push({ k: 'move', arms: both(a0 + (a1 - a0) * t, h0 + (h1 - h0) * t), via: i < n, slow: true })
      }
      list.push(
        { k: 'grip', side: 'L', g: 0.5 },
        { k: 'grip', side: 'R', g: 0.5 },
        { k: 'clothRelease', side: 'L' },
        { k: 'clothRelease', side: 'R' },
        { k: 'move', arms: both(0.64, 0.1) },
        { k: 'move', arms: { L: tuck('L'), R: tuck('R') } },
        { k: 'wait', t: 0.8 },
        { k: 'plan', make: smooth },
        { k: 'plan', make: folds },
      )
      return list
    },
  })
  return acts
}

// ---------------------------------------------------------------------------
// Wash clothes
// ---------------------------------------------------------------------------
function planWash(ctx: PlanCtx): Act[] {
  const st: StationId = 'washer'
  // Take the pile from the top down, so nothing has to be pulled out from under the rest.
  const clothes = ctx
    .ids('clothes')
    .filter((id) => ctx.item(id).where === 'home')
    .sort((x, y) => ctx.item(y).pos.y - ctx.item(x).pos.y)
  if (!clothes.length) return []
  const fwd = dirWorld(st, 1, 0, 0)
  const right = dirWorld(st, 0, 0, 1)
  const W = (a: number, y: number, b: number) => toWorld(st, a, y, b)
  // The right hand holds the lid's front edge this far to the right of centre.
  const lb = 0.1
  const edge = (phi: number) => {
    const [a, y] = lidEdge(phi, 0.008)
    return W(a, y, lb)
  }
  const acts = arrive(st, 'Doing the laundry')
  const lidMove = (from: number, to: number, text: string): Act[] => {
    const list: Act[] = [
      { k: 'say', text },
      { k: 'grip', side: 'R', g: 0.6 },
      { k: 'move', arms: { R: T(edge(from).addScaledVector(fwd, -0.08), fwd, UP) } },
      { k: 'move', arms: { R: T(edge(from), fwd, UP) }, slow: true },
      { k: 'grip', side: 'R', g: gripFor(0.02) },
      { k: 'bind', side: 'R', mech: { kind: 'lid' } },
    ]
    const n = 7
    for (let i = 1; i <= n; i++) list.push({ k: 'move', arms: { R: T(edge(from + ((to - from) * i) / n), fwd, UP) }, via: i < n, slow: true })
    list.push(
      { k: 'grip', side: 'R', g: 0.6 },
      { k: 'unbind', side: 'R' },
      { k: 'move', arms: { R: T(edge(to).addScaledVector(fwd, -0.1), fwd, UP) }, via: true },
      { k: 'move', arms: { R: tuck('R') } },
    )
    return list
  }
  acts.push(...lidMove(0, WASHER.lid.open, 'Lifting the lid'))
  const dropY = WASHER.drum.y1 + 0.14
  clothes.forEach((id, i) => {
    const { pos } = ctx.item(id)
    const rest = WASHER.drumRest[i % WASHER.drumRest.length]
    const drop = W(WASHER.drum.a - 0.07, dropY, -0.06 + rest[2] * 0.3)
    const at = (p: THREE.Vector3) => T(p, downish(st, 'L', p, 55))
    acts.push(
      { k: 'say', text: `Clothes ${i + 1} of ${clothes.length} into the drum` },
      ...pickFromAbove(st, 'L', id, up(pos, 0.008), undefined, 0.02, 0.12, 0.2),
      { k: 'move', arms: { L: at(drop) } },
      { k: 'grip', side: 'L', g: 0.55 },
      { k: 'release', side: 'L', item: id, container: 'washer.drum', pos: rest, yaw: i * 1.3, where: 'drum' },
      { k: 'move', arms: { L: at(up(drop, 0.06)) }, via: true },
    )
  })
  acts.push({ k: 'move', arms: { L: tuck('L') } }, ...lidMove(WASHER.lid.open, 0, 'Closing the lid'))
  const btn = W(WASHER.button.a, WASHER.button.y + 0.012, WASHER.button.b)
  acts.push(
    { k: 'say', text: 'Pressing start' },
    { k: 'grip', side: 'R', g: 0 },
    { k: 'move', arms: { R: T(up(btn, 0.08), DOWN) } },
    { k: 'move', arms: { R: T(up(btn, -0.004), DOWN) }, slow: true },
    { k: 'set', key: 'washOn', value: true },
    { k: 'move', arms: { R: T(up(btn, 0.1), DOWN) }, via: true },
    tuckHands(),
    { k: 'say', text: 'Washing machine running' },
  )
  return acts
}

// ---------------------------------------------------------------------------
// Clean up the room: rubbish in the bin, towels in the hamper, mugs in the dishwasher
// ---------------------------------------------------------------------------
/** Torso height for picking things off the floor, and for carrying. */
const FLOOR_LIFT = 0.52
const CARRY_LIFT = 1.05 // arms ride above every work surface (the washer's top is 0.88 m)
/** How far the base's front edge is from its centre. */
const BASE_FRONT = 0.33

/** Holding pose for a hand that carries something: the ready pose, held in front at the waist. */
const carry = tuck
/**
 * While the torso drops to pick something off the floor, a hand that holds something
 * comes up in front of its shoulder, so what it holds (a towel hangs 30 cm) clears the floor.
 */
const carryUp = (side: Side): Target => ({
  frame: 'torso',
  pos: new THREE.Vector3(0.26, -0.02, side === 'L' ? -0.26 : 0.26),
  approach: new THREE.Vector3(0.2, -0.98, 0).normalize(),
  keepWaist: true,
})

/** How far a piece reaches along the robot's forward direction from its centre, and its grip. */
function graspOf(shape: PropShape, kind: ClutterKind) {
  if (kind === 'towel') return { extent: Math.max(shape.kind === 'box' ? shape.hx : 0.15, 0.13), width: 0.008, y: 0.012 }
  switch (shape.kind) {
    case 'ball':
      return { extent: shape.r, width: 2 * shape.r, y: shape.r }
    case 'cyl':
      return { extent: Math.max(shape.r, shape.h / 2), width: 2 * shape.r, y: 0 }
    case 'box':
      // Flat things (wrappers) are pinched; boxes gripped across their short side.
      return shape.hy < 0.01
        ? { extent: Math.max(shape.hx, shape.hz), width: 0.006, y: (shape.y ?? 0) + 0.002 }
        : { extent: Math.max(shape.hx, shape.hz), width: 2 * Math.min(shape.hx, shape.hz), y: (shape.y ?? 0) }
  }
}

/** World-space direction of a body's local axis. */
const axisOf = (q: THREE.Quaternion, x: number, y: number, z: number) => new THREE.Vector3(x, y, z).applyQuaternion(q)

function planCleanup(ctx: PlanCtx): Act[] {
  const tries = new Map<string, number>()
  const done = { bin: 0, hamper: 0, rack: 0 }
  let dishwasherOpen = false
  let mugSlot = 0
  const K = KITCHEN

  const floorIds = (kind: ClutterKind) => ctx.ids(kind).filter((id) => ctx.item(id).where === 'floor' && (tries.get(id) ?? 0) < 2)

  /** Robot pose (centre, heading) that puts a piece in front of the given hand, ready to pick. */
  const approachFor = (id: string, side: Side) => {
    const { pos } = ctx.item(id)
    const shape = ctx.shape(id)!
    const kind = ctx.kindOf(id)!
    const g = graspOf(shape, kind)
    const r = ctx.robot()
    // Any heading works: the fingers turn to close across the piece wherever the robot stands.
    // Nearest-facing first, then round the clock in 45° steps.
    const headings: number[] = []
    const yaw0 = Math.atan2(-(pos.z - r.z), pos.x - r.x)
    for (let k = 0; k < 8; k++) headings.push(yaw0 + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (Math.PI / 4))
    // And square to the room (the furniture is), which fits best beside a counter or table.
    for (let k = 0; k < 8; k++) headings.push((k * Math.PI) / 4)
    // Keep the base's front edge clear of the whole piece, however it's turned (and a mug's handle).
    const D = BASE_FRONT + g.extent + (kind === 'mug' ? 0.06 : 0.04)
    const lateral = side === 'L' ? -0.2 : 0.2
    let best: { x: number; z: number; yaw: number; len: number } | null = null
    for (const yaw of headings) {
      const f = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw))
      const rt = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw))
      const x = pos.x - f.x * D - rt.x * lateral
      const z = pos.z - f.z * D - rt.z * lateral
      // Lowered to pick up, the arms and elbows spread round the base: nothing may be there.
      if (!ctx.free(x, z, yaw) || !ctx.workClear(x, z, yaw)) continue
      const len = ctx.driveLength(x, z, yaw, [id])
      if (len < (best?.len ?? Infinity)) best = { x, z, yaw, len }
    }
    return best
  }

  /** Drive up to a piece, then (planned on arrival) pick it up with `side`. */
  const fetch = (id: string, side: Side, upright = false): Act[] => {
    const ap = approachFor(id, side)
    tries.set(id, (tries.get(id) ?? 0) + 1)
    if (!ap) return [{ k: 'say', text: `Can't get to the ${ctx.kindOf(id) === 'trash' ? 'rubbish' : ctx.kindOf(id)} there: skipping it` }]
    return [
      { k: 'move', arms: { L: carry('L'), R: carry('R') } },
      { k: 'lift', h: CARRY_LIFT },
      { k: 'goto', x: ap.x, z: ap.z, yaw: ap.yaw, except: [id], what: `the ${ctx.kindOf(id)}` },
      { k: 'plan', make: () => pickUp(id, side, upright) },
    ]
  }

  /** Reach down and pick a piece off the floor, from wherever the robot ended up. */
  const pickUp = (id: string, side: Side, upright: boolean): Act[] => {
    const { pos, quat } = ctx.item(id)
    const shape = ctx.shape(id)!
    const kind = ctx.kindOf(id)!
    const g = graspOf(shape, kind)
    const r = ctx.robot()
    const f = new THREE.Vector3(Math.cos(r.yaw), 0, -Math.sin(r.yaw))
    const rt = new THREE.Vector3(Math.sin(r.yaw), 0, Math.cos(r.yaw))
    // Too far from the piece (the drive failed, or something pushed it)? Try again later.
    const local = pos.clone().sub(new THREE.Vector3(r.x, 0, r.z))
    const ahead = local.dot(f)
    if (ahead < BASE_FRONT || ahead > 0.75 || Math.abs(local.dot(rt)) > 0.4) return [{ k: 'say', text: `Not close enough to the ${kind}: trying again later` }]
    // Close across a lying can (fingers along the robot's right); boxes across their short side.
    let close = rt.clone()
    const lying = shape.kind === 'cyl' && Math.abs(axisOf(quat, 0, 1, 0).y) < 0.5
    if (lying) {
      // Across the can's axis, whichever way it points.
      const a = axisOf(quat, 0, 1, 0)
      a.y = 0
      close = new THREE.Vector3(-a.z, 0, a.x).normalize()
      if (close.dot(rt) < 0) close.negate()
    } else if (shape.kind === 'box' && g.width > 0.02) {
      const ax = axisOf(quat, shape.hx < shape.hz ? 1 : 0, 0, shape.hx < shape.hz ? 0 : 1)
      ax.y = 0
      close = ax.normalize()
    }
    // Lean the fingers forward a little (it saves reach); mugs straight down so they can be stood up later.
    const tilt = upright ? 0 : rad(28)
    const approach = new THREE.Vector3(0, -Math.cos(tilt), 0).addScaledVector(f, Math.sin(tilt)).normalize()
    close.addScaledVector(approach, -close.dot(approach)).normalize()
    // Cans at their middle; a towel where it's highest near its middle (where fingers meet it).
    const at = new THREE.Vector3(pos.x, shape.kind === 'cyl' || kind === 'towel' ? Math.max(pos.y, 0.012) : g.y, pos.z)
    const other: Side = side === 'L' ? 'R' : 'L'
    return [
      { k: 'move', arms: { [other]: ctx.held()[other] ? carryUp(other) : carry(other) } },
      { k: 'lift', h: FLOOR_LIFT },
      { k: 'grip', side, g: Math.min(1, gripFor(Math.max(g.width, 0.02)) + 0.35) },
      { k: 'move', arms: { [side]: T(at.clone().addScaledVector(approach, -0.12), approach, close) } },
      { k: 'move', arms: { [side]: T(at, approach, close) }, slow: true },
      { k: 'grip', side, g: gripFor(g.width) },
      { k: 'grab', side, item: id },
      // Up and back toward the robot, away from any furniture beyond the piece.
      { k: 'move', arms: { [side]: T(up(at, 0.22).addScaledVector(f, -0.12), approach, close) }, via: true },
      { k: 'lift', h: CARRY_LIFT },
      { k: 'move', arms: { [side]: carry(side) } },
    ]
  }

  /** Drive to a container and let go of whatever each hand holds over it. */
  const dropInto = (box: 'bin' | 'hamper'): Act[] => {
    const held = ctx.held()
    const sides = SIDES.filter((s) => held[s])
    if (!sides.length) return []
    const c = ctx[box]
    const r = ctx.robot()
    const D = c.r + BASE_FRONT + 0.07
    let best: { x: number; z: number; yaw: number; len: number } | null = null
    const yaw0 = Math.atan2(-(c.z - r.z), c.x - r.x)
    for (let k = 0; k < 16; k++) {
      const yaw = yaw0 + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (Math.PI / 8)
      const x = c.x - Math.cos(yaw) * D
      const z = c.z + Math.sin(yaw) * D
      if (!ctx.free(x, z, yaw)) continue
      const len = ctx.driveLength(x, z, yaw, [])
      if (len < (best?.len ?? Infinity)) best = { x, z, yaw, len }
    }
    if (!best) return [{ k: 'say', text: `Can't reach the ${box}` }]
    const acts: Act[] = [
      { k: 'say', text: box === 'bin' ? 'Into the bin' : 'Into the hamper' },
      { k: 'lift', h: CARRY_LIFT },
      { k: 'goto', x: best.x, z: best.z, yaw: best.yaw, what: `the ${box}` },
    ]
    // A towel hangs ~30 cm below the fingers: let go high enough that it clears the rim.
    const centre = new THREE.Vector3(c.x, c.h + (box === 'hamper' ? 0.42 : 0.16), c.z)
    // Fingers leaning 30° toward the container: the wrist stays nearer the robot, well within reach.
    const f = new THREE.Vector3(Math.cos(best.yaw), 0, -Math.sin(best.yaw))
    const lean = DOWN.clone().multiplyScalar(Math.cos(rad(30))).addScaledVector(f, Math.sin(rad(30)))
    for (const s of sides) {
      const id = held[s]!
      acts.push(
        { k: 'move', arms: { [s]: T(centre, lean) } },
        { k: 'grip', side: s, g: 1 },
        { k: 'release', side: s, item: id, container: 'kitchen', pos: [0, 0, 0], yaw: 0, where: box },
        { k: 'move', arms: { [s]: carry(s) } },
      )
    }
    return acts
  }

  /** Stand a mug the hand holds (lying, gripped from above) upright in a back-row slot of the upper rack. */
  const rackMug = (side: Side): Act[] => {
    const id = ctx.held()[side]
    if (!id) return []
    const { fwd, right, W } = kitchenFrame()
    // Farthest slot first: the arm never reaches past a mug already standing in the rack.
    const order = [...K.mugSlots].sort((x, y) => y.b - x.b)
    const sl = order[mugSlot % order.length]
    mugSlot++
    const o = rackOrigin('upper', K.racks.upper.travel)
    const slotLocal: [number, number, number] = [sl.a, K.cup.h / 2 + 0.004, sl.b]
    const slot = W(o.x + sl.a, o.y + K.cup.h / 2 + 0.015, sl.b) // let go just above the slot
    // Fingers forward and closing along `right`: the mug's axis (along the hand) turns vertical.
    // Put the mug (not the hand) over the slot: allow for where it really sits in the fingers,
    // which varies, picked up off the floor. Off-centre, it lands on a post and topples.
    const at = () => slot.clone().add(ctx.tcp(side).sub(ctx.item(id).pos))
    return [
      { k: 'lift', h: liftFor(slot.y, false) },
      {
        k: 'plan',
        make: (): Act[] => {
          const hand = at()
          // In from the front well above the posts round the slot, over it, then straight down
          // (coming in low, the mug's base catches the front posts and twists in the fingers).
          return [
            { k: 'move', arms: { [side]: T(hand.clone().addScaledVector(fwd, -0.14).addScaledVector(UP, 0.1), fwd, right) } },
            { k: 'move', arms: { [side]: T(hand.clone().addScaledVector(UP, 0.09), fwd, right) }, slow: true },
            { k: 'move', arms: { [side]: T(hand, fwd, right) }, slow: true },
          ]
        },
      },
      // Open just enough to let go (wide open, the fingers knock the mug next to it).
      { k: 'grip', side, g: Math.min(1, gripFor(2 * K.cup.r) + 0.12) },
      { k: 'release', side, item: id, container: 'kitchen.upper', pos: slotLocal, yaw: 0, where: 'rack' },
      // Let it settle, then straight back out of the slot (not up through the mug), slowly so the
      // fingers don't drag it over, then up.
      { k: 'wait', t: 0.4 },
      {
        k: 'plan',
        make: (): Act[] => {
          const hand = ctx.tcp(side)
          return [
            { k: 'move', arms: { [side]: T(hand.clone().addScaledVector(fwd, -0.12), fwd, right) }, slow: true },
            { k: 'move', arms: { [side]: T(hand.clone().addScaledVector(fwd, -0.16).addScaledVector(UP, 0.08), fwd, right) } },
          ]
        },
      },
      { k: 'move', arms: { [side]: tuck(side) } },
    ]
  }

  /** The next leg of the job, planned from the room as it is now. */
  const next = (): Act[] => {
    const r = ctx.robot()
    const nearest = (ids: string[], n: number) =>
      ids
        .map((id) => ({ id, d: ctx.item(id).pos.distanceTo(new THREE.Vector3(r.x, 0, r.z)) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, n)
        .map((x) => x.id)
    const loop: Act = { k: 'plan', make: next }
    // 1. Rubbish, two pieces a trip (one in each hand), into the bin.
    const trash = floorIds('trash')
    if (trash.length) {
      const [a] = nearest(trash, 1)
      return [
        ...fetch(a, 'L'),
        {
          k: 'plan',
          make: () => {
            const rest = floorIds('trash').filter((id) => id !== a)
            return rest.length ? fetch(nearest(rest, 1)[0], 'R') : []
          },
        },
        { k: 'plan', make: () => dropInto('bin') },
        loop,
      ]
    }
    // 2. Towels into the hamper.
    const towels = floorIds('towel')
    if (towels.length) {
      const [a] = nearest(towels, 1)
      return [
        { k: 'say', text: 'Towels go in the hamper' },
        ...fetch(a, 'L'),
        {
          k: 'plan',
          make: () => {
            const rest = floorIds('towel').filter((id) => id !== a)
            return rest.length ? fetch(nearest(rest, 1)[0], 'R') : []
          },
        },
        { k: 'plan', make: () => dropInto('hamper') },
        loop,
      ]
    }
    // 3. Mugs into the dishwasher: open it once, then bring them two at a time.
    const mugs = floorIds('mug')
    if (mugs.length) {
      const acts: Act[] = []
      if (!dishwasherOpen) {
        dishwasherOpen = true
        acts.push(
          { k: 'say', text: 'Mugs go in the dishwasher: opening it first' },
          { k: 'move', arms: { L: TUCK_Q, R: TUCK_Q } },
          { k: 'lift', h: CARRY_LIFT },
          dishGoto(),
          ...doorOpenActs(),
          ...rackMoveActs('upper', 0, K.racks.upper.travel),
          { k: 'move', arms: { L: carry('L'), R: carry('R') } },
          { k: 'lift', h: CARRY_LIFT },
        )
      }
      // One at a time, in the right hand: it's the arm on the dishwasher side.
      const [a] = nearest(mugs, 1)
      acts.push(
        ...fetch(a, 'R', true),
        { k: 'lift', h: CARRY_LIFT },
        dishGoto(),
        { k: 'plan', make: () => rackMug('R') },
        loop,
      )
      return acts
    }
    // 4. All done: close the dishwasher if it was opened, and report.
    const acts: Act[] = []
    if (dishwasherOpen) {
      dishwasherOpen = false
      acts.push(
        { k: 'move', arms: { L: TUCK_Q, R: TUCK_Q } },
        { k: 'lift', h: CARRY_LIFT },
        dishGoto(),
        ...rackMoveActs('upper', K.racks.upper.travel, 0),
        ...doorCloseActs(false),
      )
    }
    const left = ['trash', 'towel', 'mug'].flatMap((k) => ctx.ids(k).filter((id) => ctx.item(id).where === 'floor')).length
    done.bin = ctx.ids('trash').filter((id) => ctx.item(id).where === 'bin').length
    done.hamper = ctx.ids('towel').filter((id) => ctx.item(id).where === 'hamper').length
    // In the rack, upright or not: the final rack check says which (if any) need standing up.
    done.rack = ctx.ids('mug').filter((id) => ['rack', 'misplaced'].includes(ctx.item(id).where)).length
    acts.push(
      { k: 'move', arms: { L: TUCK_Q, R: TUCK_Q } },
      { k: 'lift', h: CARRY_LIFT },
      {
        k: 'say',
        text: `Room cleaned up: ${done.bin} pieces of rubbish in the bin, ${done.hamper} towels in the hamper, ${done.rack} mugs in the dishwasher${left ? ` (${left} left that it couldn't reach)` : ''}`,
      },
    )
    return acts
  }

  return [{ k: 'say', text: 'Cleaning up the room' }, { k: 'move', arms: { L: TUCK_Q, R: TUCK_Q } }, { k: 'lift', h: CARRY_LIFT }, { k: 'plan', make: next }]
}

// ---------------------------------------------------------------------------
// Clear the table: stack the plates (the cutlery rides on them) and carry the stack to the counter
// ---------------------------------------------------------------------------
/** Plate (dinner) radius and how far in from its edge the fingers take the rim. */
const DPLATE_R = 0.13
const RIM_IN = 0.125
/**
 * How a hand holds a plate by its rim, like a person: the fingers come in from outside the
 * plate tilted this far from straight down, the thumb on top and the fingers under the rim.
 */
const RIM_TILT = rad(45)
/** The right arm carrying a plate: elbow down at the side, forearm forward, wrist turned in. */
const PLATE_CARRY_Q = [22, 13, 2, 123, 146, 135, 10].map(rad)
/** Torso height carrying plates: the hand rides low by the waist, so the torso goes up to keep the plate above tables and counters. */
const PLATE_LIFT = 1.18
/** Which way round the right hand closes on a plate's rim (relative to rimGrip's axis) to suit PLATE_CARRY_Q. */
const PLATE_SIGN = -1
/** Fingers' direction and closing axis for a rim grip, `out` pointing from the plate's centre to the hand. */
const rimGrip = (out: THREE.Vector3) => ({
  approach: DOWN.clone().multiplyScalar(Math.cos(RIM_TILT)).addScaledVector(out, -Math.sin(RIM_TILT)),
  close: out.clone().multiplyScalar(Math.cos(RIM_TILT)).addScaledVector(UP, -Math.sin(RIM_TILT)),
})
/** A stacked plate sits this far above the one under it (plate thickness and the cutlery between). */
const STACK_STEP = 0.028

function planTable(ctx: PlanCtx): Act[] {
  const K = KITCHEN
  const onTable = (id: string) => ctx.item(id).where === 'floor' && ctx.item(id).pos.y > 0.5
  const plates = ctx.ids('dinnerplate').filter(onTable)
  if (!plates.length) return []
  // Which plate is at which place (nearest), and what cutlery lies on a plate.
  const plateAt = (i: number) => {
    const p = PLACES[i]
    return plates.map((id) => ({ id, d: Math.hypot(ctx.item(id).pos.x - p.x, ctx.item(id).pos.z - p.z) })).sort((a, b) => a.d - b.d)[0]
  }
  const cutleryOn = (plate: string) => {
    const c = ctx.item(plate).pos
    return ctx
      .ids('cutlery')
      .filter((id) => ctx.item(id).where === 'floor')
      .filter((id) => {
        const p = ctx.item(id).pos
        return Math.hypot(p.x - c.x, p.z - c.z) < DPLATE_R + 0.02 && p.y > c.y && p.y < c.y + 0.03
      })
  }
  // The stack is built at the place nearest the kitchen (the first), the others brought to it.
  const base = 0
  const basePlate = plateAt(base)
  const stack: string[] = [basePlate.id]
  const order = [1, 3, 2].filter((i) => plateAt(i) && plateAt(i).d < 0.15 && plateAt(i).id !== basePlate.id)
  const bayGoto = (i: number): Act => ({ k: 'goto', x: BAYS[i].x, z: BAYS[i].z, yaw: BAYS[i].yaw, what: 'the table' })
  const frame = () => {
    const r = ctx.robot()
    return { f: new THREE.Vector3(Math.cos(r.yaw), 0, -Math.sin(r.yaw)), rt: new THREE.Vector3(Math.sin(r.yaw), 0, Math.cos(r.yaw)) }
  }
  /**
   * Carrying a plate (or the stack) in front by its rim, the way a person does: the hand angled
   * in on the rim and the plate held level, so nothing on it slides off.
   */
  const carryFlat = (side: Side, sign: number): Target => {
    const g = rimGrip(new THREE.Vector3(0, 0, side === 'L' ? -1 : 1))
    g.close.multiplyScalar(sign)
    // Close in: driving up to a counter, what's carried stays in front of its edge.
    return {
      frame: 'torso',
      // The hand under its shoulder with the elbow down at the side, the plate across in front.
      pos: new THREE.Vector3(0.28, -0.2, side === 'L' ? -0.24 : 0.24),
      approach: g.approach,
      close: g.close,
      exactClose: true,
      keepWaist: true,
      posture: side === 'R' ? PLATE_CARRY_Q : undefined,
    }
  }
  /** Which way round the right hand closes relative to `c` (+1 or -1): moves holding something keep it that way. */
  const signOf = (c: THREE.Vector3) => (ctx.closeAxis('R').dot(c) < 0 ? -1 : 1)
  /** A hand target holding something: exactly this way round. */
  const H = (p: THREE.Vector3, a: THREE.Vector3, c: THREE.Vector3): Target => ({ ...T(p, a, c), exactClose: true })
  /** Which way (level) from the held plate's centre to the hand: the plate is set back level with the hand that side of it. */
  const outFrom = (id: string) => ctx.tcp('R').sub(ctx.item(id).pos).setY(0).normalize()
  /** Take a plate lying flat by its rim on the robot's right, at an angle like a person's hand. */
  const takeRim = (id: string): Act[] => {
    const { f, rt } = frame()
    const c = ctx.item(id).pos
    const P = c.clone().addScaledVector(rt, RIM_IN)
    P.y = c.y + 0.002
    const g = rimGrip(rt)
    // The hand takes the rim the one way round that lets it then carry the plate with the elbow
    // down at the side (PLATE_CARRY_Q), not either way.
    g.close.multiplyScalar(PLATE_SIGN)
    return [
      { k: 'grip', side: 'R', g: 0.75 },
      // In from above and outside the rim, along the fingers.
      { k: 'move', arms: { R: H(P.clone().addScaledVector(g.approach, -0.12), g.approach, g.close) } },
      { k: 'move', arms: { R: H(P, g.approach, g.close) }, slow: true },
      { k: 'grip', side: 'R', g: gripFor(0.02) },
      { k: 'grab', side: 'R', item: id },
      ...cutleryOn(id).map((cid): Act => ({ k: 'glue', item: cid, onto: id })),
      {
        k: 'plan',
        make: (): Act[] => {
          const sg = signOf(g.close)
          return [
            { k: 'move', arms: { R: H(up(P, 0.15).addScaledVector(f, -0.06), g.approach, g.close.clone().multiplyScalar(sg)) }, slow: true },
            // Torso up first: the carrying hand rides low, and the plate must clear the table edge.
            { k: 'lift', h: PLATE_LIFT },
            { k: 'move', arms: { R: carryFlat('R', sg * PLATE_SIGN) } },
          ]
        },
      },
    ]
  }

  /** At a place: take the plate by its right-hand rim, the cutlery on it riding along. */
  const pickPlate = (id: string): Act[] => [
    { k: 'say', text: 'Picking up a plate (the cutlery stays on it)' },
    { k: 'lift', h: 1.08 },
    ...takeRim(id),
  ]
  /** Back at the stack: set the plate square on top, let go, and it becomes part of the stack. */
  const putOnStack = (id: string): Act[] => {
    const below = stack[stack.length - 1]
    const centre = ctx.item(below).pos.clone().add(new THREE.Vector3(0, STACK_STEP + 0.004, 0))
    // The wrist turns the plate level again over the stack, keeping the hand's side of it.
    const out = outFrom(id)
    const g = rimGrip(out)
    g.close.multiplyScalar(signOf(g.close))
    const hand = centre.clone().addScaledVector(out, RIM_IN)
    stack.push(id)
    return [
      { k: 'say', text: `Onto the stack (${stack.length} plates)` },
      // Over the stack first with the torso still up (the carried plate rides low: lowered first,
      // it would sweep into the stack), then down.
      { k: 'move', arms: { R: H(up(hand, 0.12), g.approach, g.close) } },
      { k: 'lift', h: 1.08 },
      // Measured again once level over the stack (the waist may have turned getting there),
      // then straight down: square on the one below.
      {
        k: 'plan',
        make: (): Act[] => {
          const h2 = centre.clone().add(ctx.tcp('R').sub(ctx.item(id).pos))
          return [
            { k: 'move', arms: { R: H(up(h2, 0.04), g.approach, g.close) }, slow: true },
            { k: 'move', arms: { R: H(h2, g.approach, g.close) }, slow: true },
            { k: 'grip', side: 'R', g: 0.7 },
            { k: 'release', side: 'R', item: id, container: 'kitchen', pos: [0, 0, 0], yaw: 0, where: 'stack' },
            { k: 'wait', t: 0.4 },
            { k: 'glue', item: id, onto: below },
            // Back out along the fingers, off the rim, then up.
            { k: 'move', arms: { R: T(h2.clone().addScaledVector(g.approach, -0.06), g.approach, g.close) } },
            { k: 'move', arms: { R: T(up(h2, 0.14), DOWN) } },
            { k: 'move', arms: { R: TUCK_Q } },
          ]
        },
      },
    ]
  }

  // The first plate stays put as the bottom of the stack: its cutlery goes with it from the start.
  const acts: Act[] = [
    { k: 'say', text: 'Clearing the table' },
    ...cutleryOn(basePlate.id).map((cid): Act => ({ k: 'glue', item: cid, onto: basePlate.id })),
    tuckBoth(),
    { k: 'lift', h: CARRY_LIFT },
  ]
  for (const i of order) {
    const id = plateAt(i).id
    acts.push(
      bayGoto(i),
      { k: 'plan', make: () => pickPlate(id) },
      { k: 'lift', h: PLATE_LIFT },
      bayGoto(base),
      { k: 'plan', make: () => putOnStack(id) },
    )
  }
  if (!order.length) acts.push(bayGoto(base))

  // Lift the whole stack by the top plate's rim (everything under it is fixed to it) and carry it to the kitchen.
  acts.push({
    k: 'plan',
    make: (): Act[] => {
      const top = stack[stack.length - 1]
      return [{ k: 'say', text: `Taking the stack of ${stack.length} plates to the kitchen` }, { k: 'lift', h: 1.08 }, ...takeRim(top)]
    },
  })
  const sp = toWorld('kitchen', KITCHEN.stackPose.a, 0, KITCHEN.stackPose.b)
  acts.push({ k: 'lift', h: PLATE_LIFT }, { k: 'goto', x: sp.x, z: sp.z, yaw: DOCKS.kitchen.yaw, what: 'the counter' })

  // Set the stack down on the counter in front. The torso rises
  // first: the hand (and the stack) is measured after that. (The left arm folds away only once
  // the stack is down: folded against the chest it would hit the stack held in front.)
  const { fwd, W } = kitchenFrame()
  acts.push({ k: 'say', text: 'Stack down on the counter' }, { k: 'lift', h: PLATE_LIFT })
  acts.push({
    k: 'plan',
    make: (): Act[] => {
      const top = stack[stack.length - 1]
      const S = W(K.stackAt.a, K.counterH + 0.01 + 0.004, K.stackAt.b)
      // The wrist turns the stack level again on the way over, the hand on the same side of it.
      const out = outFrom(top)
      const g = rimGrip(out)
      g.close.multiplyScalar(signOf(g.close))
      const hand = up(S, (stack.length - 1) * STACK_STEP + 0.002).addScaledVector(out, RIM_IN)
      const at = (p: THREE.Vector3) => H(p, g.approach, g.close)
      return [
        // High over the drying rack and the mugs at the front of the counter, then down at the back.
        { k: 'move', arms: { R: at(up(hand, 0.32).addScaledVector(fwd, -0.3)) } },
        { k: 'move', arms: { R: at(up(hand, 0.32)) } },
        // Measured again over the spot (the waist may have turned on the way), then straight down.
        {
          k: 'plan',
          make: (): Act[] => {
            const h2 = S.clone().add(ctx.tcp('R').sub(ctx.item(stack[0]).pos))
            return [
              { k: 'move', arms: { R: at(up(h2, 0.03)) }, slow: true },
              { k: 'move', arms: { R: at(h2) }, slow: true },
            ]
          },
        },
        { k: 'grip', side: 'R', g: 0.7 },
        { k: 'release', side: 'R', item: top, container: 'kitchen', pos: [0, 0, 0], yaw: 0, where: 'counter' },
        // Out along the fingers, off the rim, then up.
        {
          k: 'plan',
          make: (): Act[] => [{ k: 'move', arms: { R: at(ctx.tcp('R').addScaledVector(g.approach, -0.06)) } }],
        },
        { k: 'move', arms: { R: at(up(hand, 0.2)) } },
        // Back out the way it came, high over the mugs and the drying rack, before the hand turns to the ready pose.
        { k: 'move', arms: { R: at(up(hand, 0.32).addScaledVector(fwd, -0.3)) } },
        { k: 'move', arms: { R: tuck('R') } },
        { k: 'move', arms: { L: fold('L') } },
      ]
    },
  })

  // Back off from the counter, so there's room to turn for whatever comes next.
  acts.push(
    { k: 'move', arms: { L: tuck('L'), R: tuck('R') } },
    { k: 'nudge', d: -0.3 },
    { k: 'say', text: 'Table cleared: the plates are stacked on the counter, ready for the dishwasher' },
  )
  return acts
}

export function planTask(id: TaskId, ctx: PlanCtx, opts: { towelsLeft: number }): Act[] {
  // Hands left low (resting on a table after the last task)? Raise the torso and lift the hands
  // straight up off it first, so the arms swing in to tuck from well above the table.
  const r = ctx.robot()
  // Only when parked at a station (out in the open the hands just hang, and there may be no room to back off).
  const dish = toWorld('kitchen', KITCHEN.dishPose.a, 0, KITCHEN.dishPose.b)
  const parked = [...Object.values(DOCKS), dish].some((d) => Math.hypot(r.x - d.x, r.z - d.z) < 0.2)
  const low = parked && Math.min(ctx.tcp('L').y, ctx.tcp('R').y) < 1.0
  const fwd = new THREE.Vector3(Math.cos(r.yaw), 0, -Math.sin(r.yaw))
  const right = new THREE.Vector3(Math.sin(r.yaw), 0, Math.cos(r.yaw))
  const close = (s: number) => new THREE.Vector3(r.x, 1.15, r.z).addScaledVector(fwd, 0.22).addScaledVector(right, s * 0.26)
  const handsUp: Act[] = low
    ? [
        { k: 'lift', h: 1.18 },
        // Up and back toward the body, clear of the edge of whatever is in front (a plate in the drying rack).
        {
          k: 'move',
          arms: { L: T(up(ctx.tcp('L'), 0.3).addScaledVector(fwd, -0.12), DOWN), R: T(up(ctx.tcp('R'), 0.3).addScaledVector(fwd, -0.12), DOWN) },
          slow: true,
        },
        // Then in close over the base, clear of whatever they were working on, and back off a
        // little so the arms have room to tuck.
        { k: 'move', arms: { L: T(close(-1), DOWN), R: T(close(1), DOWN) } },
        { k: 'nudge', d: -0.25 },
      ]
    : []
  // Cleaning up works low and carries things, so its arms never hang (they'd hit the floor);
  // loading the dishwasher happens right beside its open door, where a hanging arm would hit it.
  if (id === 'cleanup') return [...handsUp, ...planCleanup(ctx)]
  if (id === 'dishes') return [...handsUp, ...planDishes(ctx)]
  if (id === 'table') return [...handsUp, ...planTable(ctx)]
  const acts = id === 'sort' ? planSort(ctx) : id === 'fold' ? planFold(ctx, opts.towelsLeft) : planWash(ctx)
  return [...handsUp, ...hangIdleArms(acts, true)]
}

const isTuck = (g: ArmGoal | undefined) => !!g && (g === TUCK_Q || (!Array.isArray(g) && !!g.keepWaist))

/**
 * Arms that aren't working hang straight down at the sides, like a person's. Each
 * "park" in a plan (moving an arm to the ready pose) becomes: lift the hand clear of
 * the table to the ready pose, then lower the arm to hang. An arm that is already
 * hanging isn't moved again. `during`: also do this between items, not just at the
 * start and end.
 */
function hangIdleArms(acts: Act[], during: boolean): Act[] {
  const out: Act[] = []
  const hanging: Record<Side, boolean> = { L: false, R: false }
  const lastPark = acts.reduce((n, a, i) => (a.k === 'move' && SIDES.some((s) => isTuck(a.arms[s])) ? i : n), -1)
  const moveOf = (sides: Side[], goal: (s: Side) => ArmGoal): Act => ({ k: 'move', arms: Object.fromEntries(sides.map((s) => [s, goal(s)])) })
  acts.forEach((a, i) => {
    if (a.k !== 'move') return void out.push(a)
    const parks = SIDES.filter((s) => isTuck(a.arms[s]))
    const first = !out.some((x) => x.k === 'drive')
    if (parks.length && first) {
      // Before driving off: just let the arms hang (away from any table).
      out.push(moveOf(parks, () => REST_Q))
      for (const s of parks) hanging[s] = true
      return
    }
    if (parks.length && (during || i === lastPark)) {
      // Park: lift the hand clear to the ready pose, then lower it straight down and
      // back to hang (bending the elbow open in one swing would arc the hand forward
      // through the table edge).
      const lift = parks.filter((s) => !hanging[s])
      if (lift.length) {
        out.push(moveOf(lift, tuck))
        out.push(moveOf(lift, drawBack))
        out.push(moveOf(lift, hang))
      }
      for (const s of parks) hanging[s] = true
      const others = SIDES.filter((s) => a.arms[s] && !parks.includes(s))
      if (others.length) out.push({ ...a, arms: Object.fromEntries(others.map((s) => [s, a.arms[s]!])) })
      return
    }
    // Work: a hanging arm first comes up to the ready pose (straight up beside the
    // body), so it never sweeps up through the table edge.
    const unhang = SIDES.filter((s) => a.arms[s] && hanging[s] && !isTuck(a.arms[s]))
    if (unhang.length) out.push(moveOf(unhang, drawBack), moveOf(unhang, tuck))
    for (const s of SIDES) if (a.arms[s]) hanging[s] = false
    out.push(a)
  })
  return out
}
const SIDES: Side[] = ['L', 'R']
/**
 * Between the ready pose and hanging: the hand drawn back beside the hip, above table
 * height, so lowering the arm (or raising it again) never sweeps the fingers over the table edge.
 */
const drawBack = (side: Side): Target => ({
  frame: 'torso',
  pos: new THREE.Vector3(0.13, -0.27, side === 'L' ? -0.34 : 0.34),
  approach: new THREE.Vector3(0.3, -0.95, 0).normalize(),
  keepWaist: true,
})
