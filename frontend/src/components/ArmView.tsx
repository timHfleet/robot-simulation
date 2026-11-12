import { useRef } from 'react'
import type { Bin, Block, Lengths, Q, Vec } from '../lib/types'
import { deg, fk, fmt } from '../lib/geom'
import type { PickTarget } from './Arm3D'

// World window shown on screen (metres) and pixels per metre.
const X0 = -0.8
const X1 = 0.8
const Y0 = -0.09
const Y1 = 0.9
const S = 600
const W = (X1 - X0) * S
const H = (Y1 - Y0) * S
const sx = (x: number) => (x - X0) * S
const sy = (y: number) => (Y1 - y) * S

export const COLOR: Record<string, string> = { red: '#f87171', blue: '#60a5fa', green: '#4ade80' }

const FINGER = 0.05 // finger length, m
const OPEN = 0.058 // finger spacing when open, m
const CLOSED = 0.042 // finger spacing when closed on a 40 mm block, m

interface Props {
  q: Q
  lengths: Lengths
  base: Vec
  grip: number
  blocks: Block[]
  bins: Bin[]
  target: Vec | null
  targetOk: boolean
  trail: Vec[]
  interactive: boolean
  hint?: string
  onTarget?: (p: Vec) => void
  pickMode?: boolean
  selectedId?: string | null
  onPick?: (t: PickTarget) => void
  /** Wrist roll (rad). */
  roll?: number
  /** Shown over the view instead of the scene objects (e.g. 3D-only tasks). */
  note?: string
}

export default function ArmView(props: Props) {
  const { q, lengths, base, grip, blocks, bins, target, targetOk, trail, interactive, hint, onTarget } = props
  const svgRef = useRef<SVGSVGElement>(null)
  const dragging = useRef(false)

  const pose = fk(q, lengths, base)
  const { pts, phi, c } = pose
  const [p0, p1, p2, tip] = pts
  const reach = lengths[0] + lengths[1] + lengths[2]

  // Gripper geometry: palm crossbar, then two fingers ending at the tool point.
  // In-plane directions are projected into the side view (x shrinks by cos(waist)).
  const dir: Vec = [Math.cos(phi) * c, Math.sin(phi)]
  // The fingers close across the arm's plane, so in side view they only
  // separate when the waist has turned the plane away from the viewer.
  // With the wrist roll r they close along n cos r + (d x n) sin r, where d is
  // the tool direction and n the plane normal; this is that direction's side view.
  const roll = props.roll ?? 0
  const d3 = [Math.cos(phi) * Math.cos(q[0]), Math.sin(phi), Math.cos(phi) * Math.sin(q[0])]
  const n3 = [-Math.sin(q[0]), 0, Math.cos(q[0])]
  const dxn = [d3[1] * n3[2] - d3[2] * n3[1], d3[2] * n3[0] - d3[0] * n3[2], d3[0] * n3[1] - d3[1] * n3[0]]
  const perp: Vec = [n3[0] * Math.cos(roll) + dxn[0] * Math.sin(roll), n3[1] * Math.cos(roll) + dxn[1] * Math.sin(roll)]
  const waistDeg = deg(q[0])
  const palm: Vec = [tip[0] - dir[0] * FINGER, tip[1] - dir[1] * FINGER]
  const half = Math.max(0.0035, (CLOSED + (OPEN - CLOSED) * grip) / 2)
  const off = (p: Vec, k: number, d = 0): Vec => [p[0] + perp[0] * k + dir[0] * d, p[1] + perp[1] * k + dir[1] * d]

  function toWorld(e: React.PointerEvent): Vec | null {
    const svg = svgRef.current
    const m = svg?.getScreenCTM()
    if (!svg || !m) return null
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const p = pt.matrixTransform(m.inverse())
    return [p.x / S + X0, Y1 - p.y / S]
  }

  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (props.pickMode && props.onPick) {
      const p = toWorld(e)
      if (!p) return
      const [x, y] = p
      const block = blocks.find((b) => Math.abs(x - b.x) < b.size / 2 + 0.012 && Math.abs(y - b.y) < b.size / 2 + 0.012)
      if (block) return props.onPick({ kind: 'block', id: block.id })
      const bin = bins.find((b) => Math.abs(x - b.x) < b.width / 2 && y < b.wall + 0.03 && y > -0.02)
      if (bin) return props.onPick({ kind: 'bin', id: bin.id })
      if (y < 0.06 && y > -0.05) props.onPick({ kind: 'table', x, z: 0 })
      return
    }
    if (!interactive || !onTarget) return
    dragging.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = toWorld(e)
    if (p) onTarget(p)
  }
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging.current || !onTarget) return
    const p = toWorld(e)
    if (p) onTarget(p)
  }
  const onUp = () => {
    dragging.current = false
  }

  const link = (a: Vec, b: Vec, w: number) => (
    <>
      <line x1={sx(a[0])} y1={sy(a[1])} x2={sx(b[0])} y2={sy(b[1])} stroke="#0b1220" strokeWidth={w + 5} strokeLinecap="round" />
      <line x1={sx(a[0])} y1={sy(a[1])} x2={sx(b[0])} y2={sy(b[1])} stroke="url(#metal)" strokeWidth={w} strokeLinecap="round" />
    </>
  )

  const jointLabel = (p: Vec, text: string, dx: number, dy: number) => (
    <text x={sx(p[0]) + dx} y={sy(p[1]) + dy} className="fill-slate-400 font-mono" fontSize={13}>
      {text}
    </text>
  )

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className={`block w-full touch-none select-none ${interactive ? 'cursor-crosshair' : ''}`}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        role="img"
        aria-label="Robot arm workspace"
      >
        <defs>
          <linearGradient id="metal" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#e2e8f0" />
            <stop offset="1" stopColor="#94a3b8" />
          </linearGradient>
          <radialGradient id="reach" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0.6" stopColor="#22d3ee" stopOpacity="0" />
            <stop offset="1" stopColor="#22d3ee" stopOpacity="0.07" />
          </radialGradient>
          <pattern id="grid" width={S / 10} height={S / 10} patternUnits="userSpaceOnUse" x={sx(0)} y={sy(0)}>
            <path d={`M ${S / 10} 0 L 0 0 0 ${S / 10}`} fill="none" stroke="#1e293b" strokeWidth={1} />
          </pattern>
        </defs>

        <rect width={W} height={H} fill="#0b1120" />
        <rect width={W} height={sy(0)} fill="url(#grid)" />

        {/* Reach envelope */}
        <circle cx={sx(base[0])} cy={sy(base[1])} r={reach * S} fill="url(#reach)" stroke="#22d3ee" strokeOpacity={0.25} strokeDasharray="6 8" />
        <text x={sx(base[0] - reach) + 8} y={sy(base[1]) - 8} fontSize={12} className="fill-cyan-300/60">
          max reach {fmt(reach * 100, 0)} cm
        </text>

        {/* Table */}
        <rect x={0} y={sy(0)} width={W} height={H - sy(0)} fill="#1e293b" />
        <line x1={0} y1={sy(0)} x2={W} y2={sy(0)} stroke="#475569" strokeWidth={2} />
        {[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6].map((x) => (
          <text key={x} x={sx(x)} y={sy(0) + 22} fontSize={12} textAnchor="middle" className="fill-slate-500 font-mono">
            {x.toFixed(1)}
          </text>
        ))}
        <text x={W - 10} y={sy(0) + 22} fontSize={12} textAnchor="end" className="fill-slate-600">
          x (m)
        </text>

        {/* Bins */}
        {bins.map((b) => {
          const l = b.x - b.width / 2
          const r = b.x + b.width / 2
          const c = COLOR[b.color] ?? '#cbd5e1'
          return (
            <g key={b.id}>
              <rect x={sx(l)} y={sy(b.wall)} width={b.width * S} height={b.wall * S} fill={c} fillOpacity={0.08} />
              <path
                d={`M ${sx(l)} ${sy(b.wall)} L ${sx(l)} ${sy(0)} L ${sx(r)} ${sy(0)} L ${sx(r)} ${sy(b.wall)}`}
                fill="none"
                stroke={c}
                strokeWidth={4}
                strokeLinejoin="round"
              />
            </g>
          )
        })}

        {/* Pedestal and waist turntable */}
        <rect x={sx(base[0] - 0.05)} y={sy(base[1] - 0.012)} width={0.1 * S} height={(base[1] - 0.012) * S} rx={6} fill="#334155" stroke="#475569" />
        <ellipse cx={sx(0)} cy={sy(base[1] - 0.012)} rx={0.075 * S} ry={0.02 * S} fill="#1e293b" stroke="#f472b6" strokeWidth={2.5} />
        <circle cx={sx(0.06 * Math.cos(q[0]))} cy={sy(base[1] - 0.012) + 0.016 * S * Math.sin(q[0])} r={5} fill="#f472b6" />
        <text x={sx(0)} y={sy(0) - 10} fontSize={13} textAnchor="middle" className="font-mono" fill="#f9a8d4">
          θ1 waist {fmt(waistDeg, 0)}°
        </text>

        {/* Tool path trail */}
        {trail.length > 1 && (
          <polyline
            points={trail.map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')}
            fill="none"
            stroke="#fbbf24"
            strokeOpacity={0.55}
            strokeWidth={2}
            strokeDasharray="2 5"
            strokeLinecap="round"
          />
        )}

        {/* Blocks */}
        {blocks.map((b) => (
          <rect
            key={b.id}
            x={sx(b.x - b.size / 2)}
            y={sy(b.y + b.size / 2)}
            width={b.size * S}
            height={b.size * S}
            rx={3}
            fill={COLOR[b.color] ?? '#cbd5e1'}
            stroke={props.selectedId === b.id ? '#fbbf24' : '#0b1120'}
            strokeWidth={props.selectedId === b.id ? 4 : 1.5}
          />
        ))}

        {/* Arm */}
        {link(p0, p1, 22)}
        {link(p1, p2, 17)}
        {link(p2, palm, 12)}
        {/* Gripper */}
        <line {...seg(off(palm, -half - 0.006), off(palm, half + 0.006))} stroke="#cbd5e1" strokeWidth={7} strokeLinecap="round" />
        <line {...seg(off(palm, -half), off(tip, -half))} stroke="#e2e8f0" strokeWidth={5} strokeLinecap="round" />
        <line {...seg(off(palm, half), off(tip, half))} stroke="#e2e8f0" strokeWidth={5} strokeLinecap="round" />

        {[p0, p1, p2].map((p, i) => (
          <g key={i}>
            <circle cx={sx(p[0])} cy={sy(p[1])} r={i === 0 ? 14 : 11} fill="#0f172a" stroke="#22d3ee" strokeWidth={3} />
            <circle cx={sx(p[0])} cy={sy(p[1])} r={3} fill="#22d3ee" />
          </g>
        ))}
        {jointLabel(p0, `θ2 ${fmt(deg(q[1]), 0)}°`, c >= 0 ? -84 : 20, -18)}
        {jointLabel(p1, `θ3 ${fmt(deg(q[2]), 0)}°`, 16, -14)}
        {jointLabel(p2, `θ4 ${fmt(deg(q[3]), 0)}°`, 16, -14)}

        {/* Tool point */}
        <circle cx={sx(tip[0])} cy={sy(tip[1])} r={3.5} fill="#fbbf24" />

        {/* IK target */}
        {target && (
          <g>
            <circle cx={sx(target[0])} cy={sy(target[1])} r={13} fill="none" stroke={targetOk ? '#34d399' : '#f87171'} strokeWidth={2.5} />
            <line {...seg([target[0] - 0.035, target[1]], [target[0] + 0.035, target[1]])} stroke={targetOk ? '#34d399' : '#f87171'} strokeWidth={1.5} />
            <line {...seg([target[0], target[1] - 0.035], [target[0], target[1] + 0.035])} stroke={targetOk ? '#34d399' : '#f87171'} strokeWidth={1.5} />
            <text x={sx(target[0]) + 18} y={sy(target[1]) - 14} fontSize={13} className="font-mono" fill={targetOk ? '#6ee7b7' : '#fca5a5'}>
              ({fmt(target[0], 2)}, {fmt(target[1], 2)}){targetOk ? '' : ' out of reach'}
            </text>
          </g>
        )}
      </svg>
      <TopView pose={pose} blocks={blocks} bins={bins} />
      {props.note && (
        <div className="pointer-events-none absolute inset-x-0 bottom-10 mx-auto w-fit rounded-md bg-slate-900/90 px-3 py-2 text-xs text-amber-200 ring-1 ring-amber-400/30">
          {props.note}
        </div>
      )}
      {hint && (
        <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-slate-900/80 px-2.5 py-1.5 text-xs text-slate-300 ring-1 ring-slate-700">
          {hint}
        </div>
      )}
    </div>
  )
}

function seg(a: Vec, b: Vec) {
  return { x1: sx(a[0]), y1: sy(a[1]), x2: sx(b[0]), y2: sy(b[1]) }
}

// ---- Top view inset: shows where the waist is pointing --------------------
const TS = 150 // px per metre
const TX0 = -0.8
const TZ0 = -0.42
const TW = 1.6 * TS
const TH = 0.84 * TS
const tx = (x: number) => (x - TX0) * TS
const tz = (z: number) => (z - TZ0) * TS

function TopView({ pose, blocks, bins }: { pose: ReturnType<typeof fk>; blocks: Block[]; bins: Bin[] }) {
  const p = pose.pts3
  return (
    <div className="pointer-events-none absolute right-3 top-3 hidden rounded-lg bg-slate-950/85 p-2 ring-1 ring-slate-700 sm:block">
      <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-400">
        <span>Top view</span>
        <span className="text-pink-300">waist</span>
      </div>
      <svg viewBox={`0 0 ${TW} ${TH}`} width={TW} height={TH} aria-label="Top view of the table">
        <rect width={TW} height={TH} rx={6} fill="#1e293b" />
        <line x1={0} x2={TW} y1={tz(0)} y2={tz(0)} stroke="#334155" strokeDasharray="3 4" />
        {bins.map((b) => (
          <rect
            key={b.id}
            x={tx(b.x - b.width / 2)}
            y={tz(-0.07)}
            width={b.width * TS}
            height={0.14 * TS}
            rx={3}
            fill={COLOR[b.color]}
            fillOpacity={0.12}
            stroke={COLOR[b.color]}
            strokeWidth={2}
          />
        ))}
        {blocks.map((b) => (
          <rect
            key={b.id}
            x={tx(b.x - b.size / 2)}
            y={tz((b.z ?? 0) - b.size / 2)}
            width={b.size * TS}
            height={b.size * TS}
            rx={1.5}
            fill={COLOR[b.color]}
          />
        ))}
        <circle cx={tx(0)} cy={tz(0)} r={0.075 * TS} fill="#0f172a" stroke="#f472b6" strokeWidth={2} />
        <polyline
          points={p.map((v) => `${tx(v[0])},${tz(v[2])}`).join(' ')}
          fill="none"
          stroke="#cbd5e1"
          strokeWidth={6}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {p.slice(1, 3).map((v, i) => (
          <circle key={i} cx={tx(v[0])} cy={tz(v[2])} r={3.5} fill="#22d3ee" />
        ))}
        <circle cx={tx(p[3][0])} cy={tz(p[3][2])} r={3.5} fill="#fbbf24" />
      </svg>
    </div>
  )
}
