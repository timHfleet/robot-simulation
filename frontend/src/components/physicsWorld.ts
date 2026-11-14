/**
 * Rigid-body physics for the simulator (Rapier, compiled to WebAssembly).
 *
 * - Items (blocks, plates, mugs, folded clothes) are dynamic bodies with mass and
 *   friction: they fall, settle, stack, tip over and get knocked about.
 * - The robot (base, column, torso, arm links and palms) and the moving parts of the
 *   stations (dishwasher door and racks, washer lid and drum) are kinematic bodies:
 *   they follow the controller and the mechanisms exactly and push items out of the way.
 * - Furniture is static.
 *
 * Grasping: when a task closes a hand on an item, the grasp holds only if the item is
 * really between the fingers at that moment (a shape query at the fingertips). The
 * item is then held by a fixed joint to the palm, standing in for finger friction, and
 * stops colliding with the robot. Letting go removes the joint and the item drops
 * under gravity, landing wherever physics puts it.
 */
import RAPIER from '@dimforge/rapier3d-compat'
import * as THREE from 'three'
import { ARM, jointOffset, type Side } from '../lib/kinematics'
import { ITEMS, KITCHEN, LAUNDRY, SORT, WASHER, stationQuat, toWorld, type ItemDef, type StationId } from '../lib/stations'
import type { Stations } from './stationProps'

type R = typeof RAPIER

// Collision groups: membership in the high 16 bits, the groups it collides with in the low 16.
const STATIC = 0x1
const ITEM = 0x2
const ROBOT = 0x4
const MECH = 0x8
const HELD = 0x10
const groups = (member: number, filter: number) => ((member & 0xffff) << 16) | (filter & 0xffff)
const G_STATIC = groups(STATIC, ITEM | HELD)
const G_ITEM = groups(ITEM, STATIC | ITEM | ROBOT | MECH)
// A held item doesn't push other items around (it would drag a pile apart); it still hits furniture.
const G_HELD = groups(HELD, STATIC | MECH)
const G_ROBOT = groups(ROBOT, ITEM)
const G_MECH = groups(MECH, ITEM | HELD)

const STEP = 1 / 120
const MAX_STEPS = 24
const MAX_V = 4 // m/s
const MAX_W = 40 // rad/s

export interface RobotParts {
  robot: THREE.Object3D
  waist: THREE.Object3D
  carriage: THREE.Object3D
  arms: Record<Side, { joints: THREE.Object3D[] }>
}

interface Box {
  c: THREE.Vector3
  h: THREE.Vector3
  q?: THREE.Quaternion
}
interface Follower {
  body: RAPIER.RigidBody
  obj: THREE.Object3D
  p0: THREE.Vector3
  q0: THREE.Quaternion
  p1: THREE.Vector3
  q1: THREE.Quaternion
}

export interface PhysicsStats {
  awake: number
  contacts: number
  held: Partial<Record<Side, string>>
}

export interface Physics {
  /** Advance by dt seconds of simulation time (fixed internal steps). */
  step(dt: number): void
  /** Close a hand on an item: true if the item is really between the fingers. */
  grab(side: Side, id: string): boolean
  /** Let go of whatever the hand holds. */
  release(side: Side): string | null
  /** Free the engine's memory. */
  dispose(): void
  /** Put every item back where Stations.reset() placed it, at rest. */
  reset(): void
  /** World position of an item and whether it has come to rest. */
  state(id: string): { pos: THREE.Vector3; resting: boolean }
  stats(): PhysicsStats
}

const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)

/** Station-local slab (a, y, b ranges) as a world-space box. */
function slabW(st: StationId, a0: number, a1: number, y0: number, y1: number, b0: number, b1: number): Box {
  return { c: toWorld(st, (a0 + a1) / 2, (y0 + y1) / 2, (b0 + b1) / 2), h: v((a1 - a0) / 2, (y1 - y0) / 2, (b1 - b0) / 2), q: stationQuat(st) }
}
/** Object-local slab. */
function slabL(a0: number, a1: number, y0: number, y1: number, b0: number, b1: number): Box {
  return { c: v((a0 + a1) / 2, (y0 + y1) / 2, (b0 + b1) / 2), h: v((a1 - a0) / 2, (y1 - y0) / 2, (b1 - b0) / 2) }
}

function staticBoxes(): Box[] {
  const out: Box[] = []
  // Sorting: table and three open tubs.
  const t = SORT.table
  out.push(slabW('sort', t.a0, t.a1, 0, t.h, t.b0, t.b1))
  for (const bin of SORT.bins) {
    const s = SORT.binSize / 2 - 0.004
    const y0 = t.h
    const y1 = t.h + SORT.binWall
    const w = 0.006
    out.push(
      slabW('sort', bin.a - s, bin.a + s, y0, y0 + SORT.binFloor, bin.b - s, bin.b + s),
      slabW('sort', bin.a - s - w, bin.a - s, y0, y1, bin.b - s, bin.b + s),
      slabW('sort', bin.a + s, bin.a + s + w, y0, y1, bin.b - s, bin.b + s),
      slabW('sort', bin.a - s, bin.a + s, y0, y1, bin.b - s - w, bin.b - s),
      slabW('sort', bin.a - s, bin.a + s, y0, y1, bin.b + s, bin.b + s + w),
    )
  }
  // Kitchen: counters, the cabinet around the dishwasher cavity, the plate drying rack.
  const K = KITCHEN
  for (const c of K.counters) out.push(slabW('kitchen', c.a0, c.a1, 0, K.counterH, c.b0, c.b1))
  const cb = K.cabinet
  const d = K.door
  const top = d.hingeY + d.height
  out.push(
    slabW('kitchen', cb.a0, cb.a1, 0, d.hingeY, cb.b0, cb.b1),
    slabW('kitchen', cb.a0, cb.a1, top, cb.top, cb.b0, cb.b1),
    slabW('kitchen', cb.a0, cb.a1, d.hingeY, top, cb.b0, cb.b0 + 0.03),
    slabW('kitchen', cb.a0, cb.a1, d.hingeY, top, cb.b1 - 0.03, cb.b1),
    slabW('kitchen', cb.a1 - 0.03, cb.a1, d.hingeY, top, cb.b0, cb.b1),
  )
  const pr = K.plateRack
  out.push(slabW('kitchen', pr.a - 0.07, pr.a + 0.07, K.counterH, K.counterH + 0.012, pr.b0, pr.b1))
  for (const b of pr.slots)
    for (const side of [-1, 1])
      for (const a of [pr.a - 0.05, pr.a + 0.05]) {
        const pb = b + side * 0.014
        out.push(slabW('kitchen', a - 0.003, a + 0.003, K.counterH + 0.01, K.counterH + 0.08, pb - 0.003, pb + 0.003))
      }
  // Laundry table.
  const lt = LAUNDRY.table
  out.push(slabW('laundry', lt.a0, lt.a1, 0, lt.h, lt.b0, lt.b1))
  // Washer: body around the drum opening, backsplash, the stand and the basket on it.
  const W = WASHER
  const b = W.body
  const r = W.drum.r
  out.push(
    slabW('washer', b.a0, b.a1, 0, W.drum.y0, b.b0, b.b1),
    slabW('washer', b.a0, W.drum.a - r, W.drum.y0, b.top, b.b0, b.b1),
    slabW('washer', W.drum.a + r, b.a1, W.drum.y0, b.top, b.b0, b.b1),
    slabW('washer', W.drum.a - r, W.drum.a + r, W.drum.y0, b.top, b.b0, -r),
    slabW('washer', W.drum.a - r, W.drum.a + r, W.drum.y0, b.top, r, b.b1),
    slabW('washer', W.backsplash.a0, W.backsplash.a1, b.top, W.backsplash.top, b.b0, b.b1),
  )
  const s = W.stand
  const bt = s.top + W.basketWall
  out.push(
    slabW('washer', s.a0, s.a1, 0, s.top + 0.01, s.b0, s.b1),
    slabW('washer', s.a0 + 0.02, s.a0 + 0.03, s.top, bt, s.b0 + 0.02, s.b1 - 0.02),
    slabW('washer', s.a1 - 0.03, s.a1 - 0.02, s.top, bt, s.b0 + 0.02, s.b1 - 0.02),
    slabW('washer', s.a0 + 0.02, s.a1 - 0.02, s.top, bt, s.b0 + 0.02, s.b0 + 0.03),
    slabW('washer', s.a0 + 0.02, s.a1 - 0.02, s.top, bt, s.b1 - 0.03, s.b1 - 0.02),
  )
  return out
}

/**
 * A disk (axis along y) as a convex hull. Rapier's cylinder primitive can tunnel when a
 * thin standing cylinder rests against thin posts, so round items use hulls instead.
 */
function diskHull(R: R, r: number, halfH: number, n = 24) {
  const pts: number[] = []
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2
    for (const y of [-halfH, halfH]) pts.push(r * Math.cos(t), y, r * Math.sin(t))
  }
  return R.ColliderDesc.convexHull(new Float32Array(pts))!
}

function itemCollider(R: R, def: ItemDef) {
  const set = (d: RAPIER.ColliderDesc, density: number, friction: number) =>
    d.setDensity(density).setFriction(friction).setRestitution(0.05).setCollisionGroups(G_ITEM).setActiveEvents(R.ActiveEvents.NONE)
  switch (def.kind) {
    case 'block':
      return set(R.ColliderDesc.roundCuboid(SORT.block / 2 - 0.004, SORT.block / 2 - 0.004, SORT.block / 2 - 0.004, 0.004), 600, 0.6)
    case 'plate': {
      // The plate's axis is its local z.
      const q = new THREE.Quaternion().setFromAxisAngle(v(1, 0, 0), Math.PI / 2)
      return set(diskHull(R, KITCHEN.plate.r, 0.0085), 2400, 0.45).setRotation(q)
    }
    case 'cup':
      return set(diskHull(R, KITCHEN.cup.r, KITCHEN.cup.h / 2, 20), 900, 0.5)
    case 'clothes':
      return set(R.ColliderDesc.roundCuboid(0.06, 0.009, 0.045, 0.004), 700, 0.9)
  }
}

export async function createPhysics(scene: THREE.Scene, stations: Stations, rig: RobotParts): Promise<Physics> {
  await RAPIER.init()
  const R = RAPIER
  const world = new R.World({ x: 0, y: -9.81, z: 0 })
  world.timestep = STEP

  // Floor and furniture.
  const ground = world.createRigidBody(R.RigidBodyDesc.fixed())
  world.createCollider(R.ColliderDesc.cuboid(50, 0.05, 50).setTranslation(0, -0.05, 0).setFriction(0.8).setCollisionGroups(G_STATIC), ground)
  for (const bx of staticBoxes()) {
    const d = R.ColliderDesc.cuboid(bx.h.x, bx.h.y, bx.h.z).setTranslation(bx.c.x, bx.c.y, bx.c.z).setFriction(0.6).setCollisionGroups(G_STATIC)
    if (bx.q) d.setRotation(bx.q)
    world.createCollider(d, ground)
  }

  // Kinematic followers: a body that tracks an Object3D, with boxes/capsules in its local frame.
  const followers: Follower[] = []
  const follow = (obj: THREE.Object3D, shapes: RAPIER.ColliderDesc[], group: number) => {
    obj.updateWorldMatrix(true, false)
    const p = new THREE.Vector3()
    const q = new THREE.Quaternion()
    obj.matrixWorld.decompose(p, q, new THREE.Vector3())
    const body = world.createRigidBody(R.RigidBodyDesc.kinematicPositionBased().setTranslation(p.x, p.y, p.z).setRotation(q))
    for (const s of shapes) world.createCollider(s.setCollisionGroups(group).setFriction(0.7), body)
    followers.push({ body, obj, p0: p.clone(), q0: q.clone(), p1: p.clone(), q1: q.clone() })
    return body
  }
  const box = (b: Box) => {
    const d = R.ColliderDesc.cuboid(b.h.x, b.h.y, b.h.z).setTranslation(b.c.x, b.c.y, b.c.z)
    if (b.q) d.setRotation(b.q)
    return d
  }
  /** Capsule between two local points. */
  const capsule = (a: THREE.Vector3, b: THREE.Vector3, r: number) => {
    const d = b.clone().sub(a)
    const len = d.length()
    const q = new THREE.Quaternion().setFromUnitVectors(v(0, 1, 0), d.normalize())
    const c = a.clone().add(b).multiplyScalar(0.5)
    return R.ColliderDesc.capsule(len / 2, r).setTranslation(c.x, c.y, c.z).setRotation(q)
  }

  // Robot: base, column (turns with the waist), torso (rides the lift), arm links and palms.
  follow(rig.robot, [box(slabL(-0.31, 0.31, 0.06, 0.32, -0.27, 0.27))], G_ROBOT)
  {
    // The column is centred on the waist axis; express it in the waist group's frame.
    rig.robot.updateWorldMatrix(true, true)
    const toWaist = rig.waist.matrixWorld.clone().invert().multiply(rig.robot.matrixWorld)
    const c = v(-0.07, (0.32 + 1.34) / 2, 0).applyMatrix4(toWaist)
    follow(rig.waist, [box({ c, h: v(0.055, 0.51, 0.075) })], G_ROBOT)
  }
  follow(rig.carriage, [box(slabL(-0.115, 0.035, -0.17, 0.08, -0.15, 0.15))], G_ROBOT)
  const LINK_R = [0.05, 0.05, 0.045, 0.042, 0.036, 0.03, 0.03]
  const hands = {} as Record<Side, RAPIER.RigidBody>
  for (const side of ['L', 'R'] as Side[]) {
    const joints = rig.arms[side].joints
    for (let i = 0; i < ARM.length; i++) {
      const shapes: RAPIER.ColliderDesc[] = []
      if (i < ARM.length - 1) {
        const next = jointOffset(side, i + 1)
        if (next.length() > 0.03) shapes.push(capsule(v(0, 0, 0), next, LINK_R[i]))
      } else {
        // Palm above the fingers (the fingers themselves don't push items: the grasp does).
        shapes.push(box(slabL(-0.025, 0.025, -0.085, -0.035, -0.045, 0.045)))
      }
      if (!shapes.length) continue
      const body = follow(joints[i], shapes, G_ROBOT)
      if (i === ARM.length - 1) hands[side] = body
    }
  }

  // Station mechanisms.
  const K = KITCHEN
  const d = K.door
  follow(stations.parts.door, [box(slabL(-d.thick / 2, d.thick / 2, 0, d.height, -d.width / 2, d.width / 2))], G_MECH)
  for (const id of ['lower', 'upper'] as const) {
    const r = K.racks[id]
    const w = r.width / 2
    const shapes = [
      box(slabL(0, r.depth, 0, 0.006, -w, w)),
      box(slabL(0, 0.006, 0, r.wall, -w, w)),
      box(slabL(r.depth - 0.006, r.depth, 0, r.wall, -w, w)),
      box(slabL(0, r.depth, 0, r.wall, -w, -w + 0.006)),
      box(slabL(0, r.depth, 0, r.wall, w - 0.006, w)),
    ]
    if (id === 'lower') {
      for (const b of K.plateSlots)
        for (const side of [-1, 1])
          for (const a of [K.plateSlotA - 0.06, K.plateSlotA + 0.06]) shapes.push(box(slabL(a - 0.0025, a + 0.0025, 0, 0.06, b + side * 0.012 - 0.0025, b + side * 0.012 + 0.0025)))
    } else {
      for (const s of K.cupSlots) shapes.push(box(slabL(s.a - 0.003, s.a + 0.003, 0, 0.03, s.b - 0.003, s.b + 0.003)))
    }
    follow(stations.parts.racks[id], shapes, G_MECH)
  }
  const W = WASHER
  follow(stations.parts.lid, [box(slabL(-W.lid.depth, 0, 0, 0.02, -W.lid.width / 2, W.lid.width / 2))], G_MECH)
  {
    const shapes = [R.ColliderDesc.cuboid(W.drum.r, 0.006, W.drum.r).setTranslation(0, 0.006, 0)]
    const n = 20
    const hgt = W.drum.y1 - W.drum.y0
    for (let k = 0; k < n; k++) {
      const a = (k / n) * Math.PI * 2
      const q = new THREE.Quaternion().setFromAxisAngle(v(0, 1, 0), -a)
      shapes.push(
        R.ColliderDesc.cuboid(0.006, hgt / 2, ((Math.PI * W.drum.r) / n) * 1.15)
          .setTranslation(Math.cos(a) * (W.drum.r + 0.006), hgt / 2, Math.sin(a) * (W.drum.r + 0.006))
          .setRotation(q),
      )
    }
    shapes.push(diskHull(R, 0.035, 0.08, 12).setTranslation(0, 0.08, 0)) // agitator
    follow(stations.parts.drum, shapes, G_MECH)
  }

  // Items.
  const items = new Map<string, { body: RAPIER.RigidBody; collider: RAPIER.Collider; obj: THREE.Object3D }>()
  for (const def of ITEMS) {
    const obj = stations.items.get(def.id)!
    const body = world.createRigidBody(R.RigidBodyDesc.dynamic().setLinearDamping(0.05).setAngularDamping(0.2))
    const collider = world.createCollider(itemCollider(R, def), body)
    items.set(def.id, { body, collider, obj })
  }
  const held: Partial<Record<Side, { id: string; joint: RAPIER.ImpulseJoint }>> = {}

  const _p = new THREE.Vector3()
  const _q = new THREE.Quaternion()
  const _s = new THREE.Vector3()

  const reset = () => {
    for (const side of ['L', 'R'] as Side[]) {
      const h = held[side]
      if (h) world.removeImpulseJoint(h.joint, true)
      delete held[side]
    }
    for (const [, it] of items) {
      it.obj.updateWorldMatrix(true, false)
      it.obj.matrixWorld.decompose(_p, _q, _s)
      it.body.setTranslation(_p, true)
      it.body.setRotation(_q, true)
      it.body.setLinvel({ x: 0, y: 0, z: 0 }, true)
      it.body.setAngvel({ x: 0, y: 0, z: 0 }, true)
      it.collider.setCollisionGroups(G_ITEM)
      scene.attach(it.obj)
    }
  }
  reset()

  let acc = 0
  const step = (dt: number) => {
    acc = Math.min(acc + dt, STEP * MAX_STEPS)
    const n = Math.floor(acc / STEP)
    if (!n) return
    acc -= n * STEP
    // Kinematic targets for this frame, reached gradually over the sub-steps.
    for (const f of followers) {
      f.p0.copy(f.p1)
      f.q0.copy(f.q1)
      f.obj.updateWorldMatrix(true, false)
      f.obj.matrixWorld.decompose(_p, _q, _s)
      // Never hand the engine a non-finite pose.
      if (Number.isFinite(_p.x + _p.y + _p.z + _q.x + _q.y + _q.z + _q.w)) {
        f.p1.copy(_p)
        f.q1.copy(_q)
      }
    }
    for (let k = 1; k <= n; k++) {
      const s = k / n
      for (const f of followers) {
        _p.lerpVectors(f.p0, f.p1, s)
        _q.slerpQuaternions(f.q0, f.q1, s)
        f.body.setNextKinematicTranslation(_p)
        f.body.setNextKinematicRotation(_q)
      }
      world.step()
      // Keep items physical: nothing in a kitchen moves faster than this.
      for (const [, it] of items) {
        const lv = it.body.linvel()
        const sp = Math.hypot(lv.x, lv.y, lv.z)
        if (sp > MAX_V) it.body.setLinvel({ x: (lv.x / sp) * MAX_V, y: (lv.y / sp) * MAX_V, z: (lv.z / sp) * MAX_V }, true)
        const av = it.body.angvel()
        const w = Math.hypot(av.x, av.y, av.z)
        if (w > MAX_W) it.body.setAngvel({ x: (av.x / w) * MAX_W, y: (av.y / w) * MAX_W, z: (av.z / w) * MAX_W }, true)
      }
    }
    for (const [, it] of items) {
      const t = it.body.translation()
      const r = it.body.rotation()
      it.obj.position.set(t.x, t.y, t.z)
      it.obj.quaternion.set(r.x, r.y, r.z, r.w)
    }
  }

  const grab = (side: Side, id: string) => {
    const it = items.get(id)
    const hand = hands[side]
    if (!it || held[side]) return false
    // Is the item really between the fingertips? Query a box spanning the finger gap.
    const g6 = rig.arms[side].joints[ARM.length - 1]
    g6.updateWorldMatrix(true, false)
    g6.matrixWorld.decompose(_p, _q, _s)
    const zone = _p.clone().add(v(0, -0.122, 0).applyQuaternion(_q))
    let hit = false
    const seen: string[] = []
    world.intersectionsWithShape(zone, _q, new R.Cuboid(0.03, 0.03, 0.06), (c) => {
      if (c.handle === it.collider.handle) hit = true
      for (const [k, o] of items) if (o.collider.handle === c.handle) seen.push(k)
      return true
    }, undefined, groups(0xffff, ITEM))
    if (!hit) {
      if (import.meta.env.DEV) {
        const t = it.body.translation()
        console.debug(`[physics] grab ${id} missed: zone at ${zone.toArray().map((x) => x.toFixed(3))}, item at ${[t.x, t.y, t.z].map((x) => x.toFixed(3))}, zone touches [${seen}]`)
      }
      return false
    }
    // Hold it where it is relative to the palm.
    const ht = hand.translation()
    const hr = hand.rotation()
    const hq = new THREE.Quaternion(hr.x, hr.y, hr.z, hr.w)
    const it_t = it.body.translation()
    const it_r = it.body.rotation()
    const rel = v(it_t.x - ht.x, it_t.y - ht.y, it_t.z - ht.z).applyQuaternion(hq.clone().invert())
    const relQ = hq.clone().invert().multiply(new THREE.Quaternion(it_r.x, it_r.y, it_r.z, it_r.w))
    const joint = world.createImpulseJoint(R.JointData.fixed(rel, relQ, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0, w: 1 }), hand, it.body, true)
    it.collider.setCollisionGroups(G_HELD)
    held[side] = { id, joint }
    return true
  }

  const release = (side: Side) => {
    const h = held[side]
    if (!h) return null
    world.removeImpulseJoint(h.joint, true)
    delete held[side]
    const it = items.get(h.id)!
    it.collider.setCollisionGroups(G_ITEM)
    it.body.wakeUp()
    return h.id
  }

  return {
    dispose: () => {
      try {
        world.free()
      } catch {
        // A world that panicked can't be freed cleanly; drop it.
      }
    },
    step,
    grab,
    release,
    reset,
    state: (id) => {
      const it = items.get(id)!
      const t = it.body.translation()
      const lv = it.body.linvel()
      return { pos: v(t.x, t.y, t.z), resting: it.body.isSleeping() || Math.hypot(lv.x, lv.y, lv.z) < 0.01 }
    },
    stats: () => {
      let awake = 0
      for (const [, it] of items) if (!it.body.isSleeping()) awake++
      let contacts = 0
      for (const [, it] of items) world.contactPairsWith(it.collider, () => contacts++)
      return { awake, contacts, held: Object.fromEntries(Object.entries(held).map(([s, h]) => [s, h!.id])) as Partial<Record<Side, string>> }
    },
  }
}
