export type Vec = [number, number]
/** Joint angles in radians: waist, shoulder, elbow, wrist. */
export type Q = [number, number, number, number]
/** Link lengths in metres: upper arm, forearm, wrist + gripper. */
export type Lengths = [number, number, number]
export type Method = 'analytic' | 'dls' | 'ccd'
export type Elbow = 'up' | 'down'

export interface Block {
  id: string
  color: string
  x: number
  y: number
  size: number
  /** Depth toward the viewer (only non-zero while carried). */
  z?: number
  /** Rotation about the vertical axis (radians). */
  yaw?: number
  /** Rotation inside the object's own vertical plane (a cup upside down = pi). */
  tilt?: number
  kind?: 'block' | 'plate' | 'cup' | 'towel' | 'shirt'
  binId: string | null
}

export interface Bin {
  id: string
  color: string
  x: number
  z?: number
  width: number
  wall: number
}

export interface Config {
  base: Vec
  lengths: Lengths
  limitsDeg: [number, number][]
  homeQ: Q
  methods: Method[]
  fps: number
  objects: Block[]
  bins: Bin[]
  blockSize: number
  graspFraction: number
  scenes: Record<SceneId, { objects: Block[]; bins: Bin[]; fixtures: Fixture[]; arms?: { id: string; baseZ: number; park: Q }[] }>
}

export interface IKResult {
  q: Q
  qDeg: Q
  joints: Vec[]
  tool: { x: number; y: number; phiDeg: number }
  method: Method
  reachable: boolean
  withinLimits: boolean
  iterations: number
  positionErrorMm: number
  angleErrorDeg: number
  solveMs: number
  notes: string[]
}

export interface Frame {
  t: number
  q: Q
  grip: number
  /** Wrist roll (rad): the gripper turning about its own axis. */
  roll?: number
  holding: string | null
}

export interface PlanEvent {
  t: number
  text: string
  kind: 'info' | 'pick' | 'place' | 'warn' | 'done' | 'fault' | 'recover' | 'sense' | 'wait'
}

/** A camera detection: a scan from above, or the close-up check before a grasp. */
export interface Detection {
  tStart: number
  tEnd: number
  id: string
  label: string
  confidence: number
  center: [number, number, number]
  size: [number, number, number]
  yaw: number
  phase: 'scan' | 'align'
  arm?: 'A' | 'B'
}

export type FaultMode = 'none' | 'demo' | 'random'

/** A queued pick-and-place command: move a block into a bin or onto a table spot. */
export interface Command {
  id: string
  blockId: string
  binId?: string
  x?: number
  z?: number
  label: string
}

export interface Keyframe {
  t: number
  mode: 'table' | 'held' | 'falling' | 'bin'
  pos?: [number, number, number]
  to?: [number, number, number]
  yaw?: number
  tilt?: number
  binId?: string
  /** For 'held': the object's centre in the tool frame (along tool, across tool, out of plane). */
  local?: [number, number, number]
  tilt0?: number
  /** Object yaw relative to waist + wrist roll at the grasp (it turns with both). */
  yaw0?: number
  /** Two-arm mode: which arm is holding it. */
  arm?: 'A' | 'B'
}

export type SceneId = 'blocks' | 'dual' | 'dishes' | 'dishes2' | 'laundry' | 'laundry2' | 'shirt'

export interface Fixture {
  id: string
  kind: 'plate_holder' | 'plate_rack' | 'cup_rack' | 'basket'
  x: number
  z: number
  width: number
  depth: number
  height: number
  slots?: { x: number; z: number; yaw: number }[]
  label?: string
}

export interface ClothFold {
  F: [number, number]
  n: [number, number]
  w: number
  /** Which arm(s) hold the edge (default: arm A). Two arms = their average. */
  arms?: ('A' | 'B')[]
  tGrasp: number
  tRelease: number
}

export interface ClothPlan {
  shape?: 'towel' | 'shirt'
  dims?: { length: number; half: number; sleeve: number }
  towelId: string
  center: [number, number]
  half: number
  /** Towel orientation (radians); default: facing arm A's base. */
  yaw?: number
  eps: number
  folds: ClothFold[]
  /** Carried to the basket by arm A (default) or arm B. */
  carry: { tGrasp: number; tRelease: number; floorY: number; basketId: string; arm?: 'A' | 'B' } | null
}

export interface Plan {
  fps: number
  duration: number
  frames: Frame[]
  events: PlanEvent[]
  /** Where each block is over time. */
  timeline: Record<string, Keyframe[]>
  /** Towel folding: fold lines and timing, for the cloth model. */
  cloth?: ClothPlan
  /** Force sensors, one sample per frame (N): wrist vertical load and gripper force. */
  sensors?: { fz: number[]; grip: number[] }
  /** What the wrist camera saw and when. */
  perception?: Detection[]
  /** Two-arm mode: the second arm's motion and sensors (same timing as frames). */
  armB?: { baseZ: number; frames: Frame[]; sensors?: { fz: number[]; grip: number[] } }
  dual?: {
    singleArmS: number | null
    speedup: number | null
    minArmGapMm: number
    clearanceMm: number
    arms: { id: string; blocks: number; busyS: number; waitS: number; waits: number; utilization: number }[]
  }
  placements: { id: string; binId: string; t: number; x: number; y: number }[]
  warnings: string[]
  stats: {
    sorted: number
    skipped: number
    slips: number
    drops: number
    recovered: number
    durationS: number
    toolPathM: number
    maxJointSpeedDegS: number
    minClearanceMm: number
    frames: number
  }
}
