import type { Q } from './types'

const r = (d: number) => (d * Math.PI) / 180

// Motor limits per joint: waist, shoulder, elbow, wrist. The waist and shoulder
// move the most mass, so they are the slowest.
export const MAX_SPEED: Q = [r(90), r(80), r(110), r(150)] // rad/s
export const MAX_ACCEL: Q = [r(240), r(220), r(320), r(480)] // rad/s^2

export interface ServoState {
  q: Q
  v: Q
}

/**
 * One control step of a simple motor model: each joint accelerates toward the
 * commanded angle, cruises at its top speed and brakes in time to stop exactly
 * on target (a trapezoidal velocity profile). This is what keeps the arm from
 * snapping to a new pose: it has to actually travel there.
 */
export function stepServo(state: ServoState, cmd: Q, dt: number): ServoState & { moving: boolean } {
  const q = [...state.q] as Q
  const v = [...state.v] as Q
  let moving = false
  for (let j = 0; j < 4; j++) {
    const d = cmd[j] - q[j]
    if (Math.abs(d) < 1e-4 && Math.abs(v[j]) < 1e-3) {
      q[j] = cmd[j]
      v[j] = 0
      continue
    }
    moving = true
    // Fastest speed that can still brake to zero before the target.
    const desired = Math.sign(d) * Math.min(MAX_SPEED[j], Math.sqrt(2 * MAX_ACCEL[j] * Math.abs(d)))
    const dv = Math.max(-MAX_ACCEL[j] * dt, Math.min(MAX_ACCEL[j] * dt, desired - v[j]))
    v[j] += dv
    const step = v[j] * dt
    if (Math.abs(step) >= Math.abs(d) && Math.sign(step) === Math.sign(d)) {
      q[j] = cmd[j]
      v[j] = 0
    } else {
      q[j] += step
    }
  }
  return { q, v, moving }
}
