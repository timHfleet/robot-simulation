/**
 * Task planners. Each turns the current state of a station into a list of steps
 * (drive, lift, arm moves with IK targets, gripper, grab/release, mechanisms),
 * which the runner in Humanoid.tsx then executes one by one.
 */
import * as THREE from 'three'
import { fk, torsoBase, type Side } from './kinematics'
import { rad } from './math'
import {
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

export type TaskId = 'sort' | 'dishes' | 'fold' | 'wash'
export const TASKS: { id: TaskId; label: string; station: StationId; blurb: string }[] = [
  { id: 'sort', label: 'Sort blocks', station: 'sort', blurb: 'Pick each block and drop it in the bin of its colour, turning at the waist to reach across.' },
  { id: 'dishes', label: 'Load dishwasher', station: 'kitchen', blurb: 'Open the door, pull out the racks, load plates and cups, close up and start it.' },
  { id: 'fold', label: 'Fold towel', station: 'laundry', blurb: 'Spread the crumpled towel flat by two corners, then fold it in half and in half again.' },
  { id: 'wash', label: 'Wash clothes', station: 'washer', blurb: 'Open the lid, move the clothes from the basket into the drum, close it and press start.' },
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
  /** Plan the next steps now, from what the scene looks like at this point in the task. */
  | { k: 'plan'; make: () => Act[] }
  | { k: 'clothFolded' }

export interface PlanCtx {
  /** Item world pose and where it is ('home', 'bin', 'rack', 'drum', ...). */
  item: (id: string) => { pos: THREE.Vector3; quat: THREE.Quaternion; where: string }
  ids: (kind: string) => string[]
  /** World position of a named point on the shirt, as the head camera sees it now. */
  cloth: (feature: string) => THREE.Vector3
}

/** Arms hanging beside the body (used before driving off and at the end). */
export const REST_Q = [0, 8, 0, 15, 0, 0, 0].map(rad)
/**
 * Ready pose between items: elbows at the sides, hands up in front with the
 * fingers pointing down, above table height but short of the table edge.
 */
export const TUCK_Q = [15, 8, 0, 125, 0, -130, 0].map(rad)
/** The same ready pose as a hand target, so getting there is a smooth straight-line move. */
/** Where the hand is when the arm hangs (REST_Q), relative to the shoulders, as a hand target. */
export const hang = (side: Side): Target => {
  const f = fk(side, REST_Q, torsoBase(0, 0))
  return { frame: 'torso', pos: f.tcp, approach: f.dir, keepWaist: true }
}
export const tuck = (side: Side): Target => ({
  frame: 'torso',
  pos: new THREE.Vector3(0.29, -0.25, side === 'L' ? -0.31 : 0.31),
  approach: new THREE.Vector3(0.17, -0.98, side === 'L' ? -0.14 : 0.14).normalize(),
  keepWaist: true,
})

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
function downish(st: StationId, side: Side, p: THREE.Vector3, maxTilt = 55, shift = 0) {
  const l = toLocal(st, p)
  const h = new THREE.Vector3(l.x - shift, 0, l.z - (side === 'L' ? -0.265 : 0.265))
  const d = h.length()
  const a = rad(maxTilt) * THREE.MathUtils.clamp((d - 0.36) / 0.24, 0, 1)
  const local = new THREE.Vector3(0, -Math.cos(a), 0).addScaledVector(h.normalize(), Math.sin(a))
  return dirWorld(st, local.x, local.y, local.z)
}

/** Pick up an item from above (hand leaning toward it if far) and lift it clear. */
function pickFromAbove(st: StationId, side: Side, id: string, tcp: THREE.Vector3, close: THREE.Vector3 | undefined, width: number, clear = 0.12, lift = 0.15, maxTilt = 55): Act[] {
  const at = (p: THREE.Vector3) => T(p, downish(st, side, p, maxTilt), close)
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
): Act[] {
  const at = (p: THREE.Vector3) => T(p, downish(st, side, p, maxTilt), close)
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
function planDishes(ctx: PlanCtx): Act[] {
  const st: StationId = 'kitchen'
  const plates = ctx.ids('plate').filter((id) => ctx.item(id).where === 'home')
  const cups = ctx.ids('cup').filter((id) => ctx.item(id).where === 'home')
  if (!plates.length && !cups.length) return []
  const K = KITCHEN
  const fwd = dirWorld(st, 1, 0, 0)
  const right = dirWorld(st, 0, 0, 1)
  const W = (a: number, y: number, b: number) => toWorld(st, a, y, b)
  const acts = arrive(st, 'Loading the dishwasher')

  // Open the door: hook the handle, pull it down a little, then let it drop open.
  const handleAt = (phi: number) => {
    const [a, y] = doorPoint(K.handle.u, K.handle.s, phi)
    return W(a, y, 0.04)
  }
  acts.push(
    { k: 'say', text: 'Opening the door' },
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'move', arms: { R: T(handleAt(0).addScaledVector(fwd, -0.08), fwd, UP) } },
    { k: 'move', arms: { R: T(handleAt(0), fwd, UP) }, slow: true },
    { k: 'grip', side: 'R', g: gripFor(0.02) },
    { k: 'bind', side: 'R', mech: { kind: 'door', u: K.handle.u, s: K.handle.s } },
  )
  const openDeg = Math.round((K.openByHand * 180) / Math.PI)
  for (let d = 10; d <= openDeg; d += 10) acts.push({ k: 'move', arms: { R: T(handleAt(rad(d)), fwd, UP) }, via: d < openDeg, slow: true })
  acts.push(
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'unbind', side: 'R' },
    { k: 'move', arms: { R: T(handleAt(K.openByHand).addScaledVector(UP, 0.06).addScaledVector(fwd, -0.04), fwd, UP) } },
    { k: 'animate', key: 'door', to: Math.PI / 2, dur: 1.1 },
    { k: 'move', arms: { R: tuck('R') } },
  )

  // Pull a rack out (or push it back in) by its front handle.
  const rackHandle = (rack: RackId, out: number) => rackOrigin(rack, out).add(new THREE.Vector3(-0.015, K.racks[rack].wall + 0.02, 0))
  const moveRack = (rack: RackId, from: number, to: number): Act[] => {
    const h0 = rackHandle(rack, from)
    const h1 = rackHandle(rack, to)
    return [
      { k: 'say', text: `${to > from ? 'Pulling out' : 'Pushing in'} the ${rack} rack` },
      { k: 'lift', h: rack === 'lower' ? 0.86 : 1.0 },
      { k: 'grip', side: 'R', g: 0.7 },
      { k: 'move', arms: { R: T(W(h0.x - 0.08, h0.y + 0.04, 0.05), fwd, UP) } },
      { k: 'move', arms: { R: T(W(h0.x, h0.y, 0.05), fwd, UP) }, slow: true },
      { k: 'grip', side: 'R', g: gripFor(0.012) },
      { k: 'bind', side: 'R', mech: { kind: 'rack', rack } },
      { k: 'move', arms: { R: T(W(h1.x, h1.y, 0.05), fwd, UP) }, slow: true },
      { k: 'grip', side: 'R', g: 0.7 },
      { k: 'unbind', side: 'R' },
      { k: 'move', arms: { R: T(W(h1.x - 0.08, h1.y + 0.1, 0.05), fwd, UP) }, via: true },
      { k: 'move', arms: { R: tuck('R') } },
    ]
  }

  const out = K.racks.lower.travel
  if (plates.length) {
    acts.push(...moveRack('lower', 0, out), { k: 'lift', h: 1.1 })
    plates.forEach((id, i) => {
      const { pos } = ctx.item(id)
      const rim = up(pos, K.plate.r - 0.018)
      const o = rackOrigin('lower', out)
      const slotLocal: [number, number, number] = [K.plateSlotA, 0.01 + K.plate.r, K.plateSlots[i % K.plateSlots.length]]
      const slotRim = up(W(o.x + slotLocal[0], o.y + slotLocal[1], slotLocal[2]), K.plate.r - 0.018)
      acts.push(
        { k: 'say', text: `Plate ${i + 1}: left hand pinches the rim` },
        ...pickFromAbove(st, 'L', id, rim, right, K.plate.t, 0.1, 0.16, 30),
        ...placeFromAbove(st, 'L', id, slotRim, right, 0.45, { container: 'kitchen.lower', pos: slotLocal, yaw: 0, where: 'rack' }, 0.14, 30),
      )
    })
    acts.push({ k: 'move', arms: { L: tuck('L') } }, ...moveRack('lower', out, 0))
  }
  if (cups.length) {
    acts.push(...moveRack('upper', 0, out), { k: 'lift', h: 1.12 })
    cups.forEach((id, i) => {
      const { pos } = ctx.item(id)
      const o = rackOrigin('upper', out)
      const sl = K.cupSlots[i % K.cupSlots.length]
      const slotLocal: [number, number, number] = [sl.a, K.cup.h / 2 + 0.004, sl.b]
      const slot = W(o.x + slotLocal[0], o.y + slotLocal[1], slotLocal[2])
      acts.push(
        { k: 'say', text: `Cup ${i + 1}: right hand grips it from above` },
        ...pickFromAbove(st, 'R', id, pos, undefined, 2 * K.cup.r, 0.1, 0.16, 30),
        ...placeFromAbove(st, 'R', id, slot, undefined, Math.min(1, gripFor(2 * K.cup.r) + 0.25), { container: 'kitchen.upper', pos: slotLocal, yaw: 0, where: 'rack' }, 0.15, 30),
      )
    })
    acts.push({ k: 'move', arms: { R: tuck('R') } }, ...moveRack('upper', out, 0))
  }

  // Close the door: back up, lift it by its free edge, drive back in, push it shut.
  const edgeAt = (phi: number) => {
    const [a, y] = doorPoint(K.edge.u, K.edge.s, phi)
    return W(a, y, 0.04)
  }
  const pushAt = (phi: number) => {
    const [a, y] = doorPoint(K.push.u, K.push.s, phi)
    return W(a, y, 0.04)
  }
  const liftDeg = Math.round((K.liftTo * 180) / Math.PI)
  acts.push(
    { k: 'say', text: 'Closing the door: backing up to reach its edge' },
    tuckHands(),
    { k: 'nudge', d: -0.3 },
    { k: 'lift', h: 0.95 },
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'move', arms: { R: T(edgeAt(Math.PI / 2).addScaledVector(fwd, -0.08), fwd, UP) } },
    { k: 'move', arms: { R: T(edgeAt(Math.PI / 2), fwd, UP) }, slow: true },
    { k: 'grip', side: 'R', g: gripFor(K.door.thick) },
    { k: 'bind', side: 'R', mech: { kind: 'door', u: K.edge.u, s: K.edge.s } },
  )
  for (let d = 80; d >= liftDeg; d -= 10) acts.push({ k: 'move', arms: { R: T(edgeAt(rad(d)), fwd, UP) }, via: d > liftDeg, slow: true })
  acts.push(
    { k: 'grip', side: 'R', g: 0.75 },
    { k: 'unbind', side: 'R' },
    { k: 'move', arms: { R: T(edgeAt(K.liftTo).addScaledVector(fwd, -0.12), fwd, UP) } },
    { k: 'move', arms: { R: tuck('R') } },
    { k: 'nudge', d: 0.3 },
    { k: 'say', text: 'Pushing the door shut' },
    { k: 'lift', h: 1.0 },
    { k: 'grip', side: 'R', g: 0 },
    { k: 'move', arms: { R: T(pushAt(K.liftTo).addScaledVector(fwd, -0.06), fwd) } },
    { k: 'bind', side: 'R', mech: { kind: 'door', u: K.push.u, s: K.push.s, pushOnly: true } },
  )
  for (let d = liftDeg - 10; d >= 0; d -= 10) acts.push({ k: 'move', arms: { R: T(pushAt(rad(d)), fwd) }, via: d > 0, slow: true })
  const btn = W(K.button.a, K.button.y, K.button.b)
  acts.push(
    { k: 'unbind', side: 'R' },
    { k: 'animate', key: 'door', to: 0, dur: 0.15 },
    { k: 'move', arms: { R: T(btn.clone().addScaledVector(fwd, -0.08), fwd) } },
    { k: 'move', arms: { R: T(btn.clone().addScaledVector(fwd, 0.004), fwd) }, slow: true },
    { k: 'set', key: 'dishOn', value: true },
    { k: 'move', arms: { R: T(btn.clone().addScaledVector(fwd, -0.08), fwd) }, via: true },
    tuckHands(),
    { k: 'say', text: 'Dishwasher loaded and running' },
  )
  return acts
}

// ---------------------------------------------------------------------------
// Fold the shirt
// ---------------------------------------------------------------------------
function planFold(ctx: PlanCtx, foldsDone: boolean): Act[] {
  if (foldsDone) return []
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
  const acts = arrive(st, 'Folding the towel')
  const steps = 10

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

export function planTask(id: TaskId, ctx: PlanCtx, opts: { foldsDone: boolean }): Act[] {
  const acts = id === 'sort' ? planSort(ctx) : id === 'dishes' ? planDishes(ctx) : id === 'fold' ? planFold(ctx, opts.foldsDone) : planWash(ctx)
  // With the dishwasher door open a hanging arm would pass through it, so there the
  // free arm stays tucked until the end.
  return hangIdleArms(acts, id !== 'dishes')
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
    if (unhang.length) out.push(moveOf(unhang, tuck))
    for (const s of SIDES) if (a.arms[s]) hanging[s] = false
    out.push(a)
  })
  return out
}
const SIDES: Side[] = ['L', 'R']
