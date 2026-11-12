import { useMemo, useRef } from 'react'
import type { Plan } from '../lib/types'

const W = 900
const H = 150
const PL = 44
const PR = 12
const PT = 10
const PB = 22

export const SENSOR_COLORS = { grip: '#34d399', fz: '#f472b6' }

/**
 * Gripper force and wrist load over the run, as the robot's sensors read
 * them (with noise). Squeezes, carried weight, touch-down spikes and slips
 * are all visible here. Click or drag to scrub.
 */
export default function SensorChart({ plan, t, onSeek }: { plan: Plan; t: number; onSeek: (t: number) => void }) {
  const ref = useRef<SVGSVGElement>(null)
  const dragging = useRef(false)
  const sensors = plan.sensors
  const top = useMemo(() => {
    if (!sensors) return 16
    return Math.max(16, Math.ceil(Math.max(...sensors.grip, ...sensors.fz) / 4) * 4)
  }, [sensors])
  const x = (time: number) => PL + (time / plan.duration) * (W - PL - PR)
  const y = (v: number) => PT + ((top - v) / top) * (H - PT - PB)

  const lines = useMemo(() => {
    if (!sensors) return null
    const n = plan.frames.length
    const step = Math.max(1, Math.ceil(n / 600))
    const pts = (arr: number[]) => {
      const out: string[] = []
      for (let i = 0; i < n; i += step) out.push(`${x(plan.frames[i].t).toFixed(1)},${y(arr[i]).toFixed(1)}`)
      return out.join(' ')
    }
    return { grip: pts(sensors.grip), fz: pts(sensors.fz) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plan, top])

  if (!sensors || !lines) return null

  const seek = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    const px = ((clientX - r.left) / r.width) * W
    onSeek(Math.max(0, Math.min(plan.duration, ((px - PL) / (W - PL - PR)) * plan.duration)))
  }

  const ticks = [0, top / 4, top / 2, (3 * top) / 4, top]
  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full cursor-ew-resize touch-none select-none"
      onPointerDown={(e) => {
        dragging.current = true
        e.currentTarget.setPointerCapture(e.pointerId)
        seek(e.clientX)
      }}
      onPointerMove={(e) => dragging.current && seek(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      role="img"
      aria-label="Force sensor readings over time"
    >
      {ticks.map((v) => (
        <g key={v}>
          <line x1={PL} x2={W - PR} y1={y(v)} y2={y(v)} stroke="#1e293b" strokeDasharray={v === 0 ? undefined : '3 5'} />
          <text x={PL - 8} y={y(v) + 4} fontSize={11} textAnchor="end" className="fill-slate-500 font-mono">
            {v} N
          </text>
        </g>
      ))}
      {plan.events
        .filter((e) => e.kind === 'fault')
        .map((e, i) => (
          <line key={i} x1={x(e.t)} x2={x(e.t)} y1={PT} y2={H - PB} stroke="#f43f5e" strokeWidth={1.5} strokeDasharray="4 3" />
        ))}
      <polyline points={lines.grip} fill="none" stroke={SENSOR_COLORS.grip} strokeWidth={1.6} strokeLinejoin="round" />
      <polyline points={lines.fz} fill="none" stroke={SENSOR_COLORS.fz} strokeWidth={1.6} strokeLinejoin="round" />
      <line x1={x(t)} x2={x(t)} y1={PT - 4} y2={H - PB} stroke="#f8fafc" strokeWidth={1.5} />
    </svg>
  )
}
