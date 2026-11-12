import { useMemo, useRef } from 'react'
import type { Plan } from '../lib/types'
import { deg } from '../lib/geom'

const W = 900
const H = 200
const PL = 44
const PR = 12
const PT = 14
const PB = 24
const YMIN = -180
const YMAX = 200

export const JOINT_COLORS = ['#f472b6', '#22d3ee', '#a78bfa', '#fbbf24']
export const JOINT_LABELS = ['θ1 waist', 'θ2 shoulder', 'θ3 elbow', 'θ4 wrist']
export const ROLL_COLOR = '#94a3b8'
export const ROLL_LABEL = 'θ5 wrist roll'

/** Joint angles over the planned trajectory. Click or drag to scrub. */
export default function JointChart({ plan, t, onSeek }: { plan: Plan; t: number; onSeek: (t: number) => void }) {
  const ref = useRef<SVGSVGElement>(null)
  const dragging = useRef(false)
  const x = (time: number) => PL + (time / plan.duration) * (W - PL - PR)
  const y = (d: number) => PT + ((YMAX - d) / (YMAX - YMIN)) * (H - PT - PB)

  const lines = useMemo(() => {
    const step = Math.max(1, Math.ceil(plan.frames.length / 450))
    const sample = plan.frames.filter((_, i) => i % step === 0 || i === plan.frames.length - 1)
    const joints = [0, 1, 2, 3].map((j) => sample.map((f) => `${x(f.t).toFixed(1)},${y(deg(f.q[j])).toFixed(1)}`).join(' '))
    return [...joints, sample.map((f) => `${x(f.t).toFixed(1)},${y(deg(f.roll ?? 0)).toFixed(1)}`).join(' ')]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plan])

  // Shade every stretch of time a block is in the gripper.
  const holds: (readonly [number, number])[] = []
  for (const kfs of Object.values(plan.timeline)) {
    kfs.forEach((k, i) => {
      if (k.mode === 'held') holds.push([k.t, kfs[i + 1]?.t ?? plan.duration] as const)
    })
  }

  const seek = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    const px = ((clientX - r.left) / r.width) * W
    onSeek(Math.max(0, Math.min(plan.duration, ((px - PL) / (W - PL - PR)) * plan.duration)))
  }

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
      aria-label="Joint angles over time"
    >
      {holds.map(([a, b], i) => (
        <rect key={i} x={x(a)} y={PT} width={Math.max(0, x(b) - x(a))} height={H - PT - PB} fill="#fbbf24" fillOpacity={0.06} />
      ))}
      {[-180, -90, 0, 90, 180].map((d) => (
        <g key={d}>
          <line x1={PL} x2={W - PR} y1={y(d)} y2={y(d)} stroke="#1e293b" strokeDasharray={d === 0 ? undefined : '3 5'} />
          <text x={PL - 8} y={y(d) + 4} fontSize={11} textAnchor="end" className="fill-slate-500 font-mono">
            {d}°
          </text>
        </g>
      ))}
      {Array.from({ length: Math.floor(plan.duration / 5) + 1 }, (_, i) => i * 5).map((s) => (
        <text key={s} x={x(s)} y={H - 6} fontSize={11} textAnchor="middle" className="fill-slate-500 font-mono">
          {s}s
        </text>
      ))}
      {lines.map((pts, j) => (
        <polyline key={j} points={pts} fill="none" stroke={JOINT_COLORS[j] ?? ROLL_COLOR} strokeWidth={2} strokeLinejoin="round" strokeDasharray={j === 4 ? '6 3' : undefined} />
      ))}
      {plan.events
        .filter((e) => e.kind === 'pick' || e.kind === 'place' || e.kind === 'fault' || e.kind === 'recover')
        .map((e, i) =>
          e.kind === 'fault' ? (
            <g key={i}>
              <line x1={x(e.t)} x2={x(e.t)} y1={PT} y2={H - PB} stroke="#f43f5e" strokeWidth={1.5} strokeDasharray="4 3" />
              <path d={`M ${x(e.t) - 6} ${PT - 2} L ${x(e.t) + 6} ${PT - 2} L ${x(e.t)} ${PT + 8} Z`} fill="#f43f5e" />
            </g>
          ) : (
            <line key={i} x1={x(e.t)} x2={x(e.t)} y1={PT} y2={PT + 6} stroke={e.kind === 'pick' ? '#fbbf24' : e.kind === 'recover' ? '#22d3ee' : '#34d399'} strokeWidth={2} />
          ),
        )}
      <line x1={x(t)} x2={x(t)} y1={PT - 4} y2={H - PB} stroke="#f8fafc" strokeWidth={1.5} />
      <circle cx={x(t)} cy={PT - 4} r={4} fill="#f8fafc" />
    </svg>
  )
}
