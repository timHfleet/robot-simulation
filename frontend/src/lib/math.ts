export const deg = (r: number) => (r * 180) / Math.PI
export const rad = (d: number) => (d * Math.PI) / 180
export const fmt = (v: number, digits = 1) => (Number.isFinite(v) ? v.toFixed(digits) : '–')
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
export const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a))
export const approach = (v: number, target: number, step: number) => v + clamp(target - v, -step, step)

/**
 * One control step of a motor moving to a target with a trapezoidal velocity
 * profile: accelerate at amax up to vmax, then brake so it stops exactly on target.
 */
export function servoStep(q: number, v: number, target: number, vmax: number, amax: number, h: number): [number, number] {
  const e = target - q
  if (Math.abs(e) < 1e-5 && Math.abs(v) <= amax * h) return [target, 0]
  const vWant = Math.sign(e) * Math.min(vmax, Math.sqrt(2 * amax * Math.abs(e)))
  const vn = approach(v, vWant, amax * h)
  const qn = q + vn * h
  if ((target - qn) * e < 0) return [target, 0] // would overshoot: land on target
  return [qn, vn]
}
