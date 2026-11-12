import clsx from 'clsx'
import type { ReactNode } from 'react'

export function Card({ title, right, children, className }: { title?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={clsx('rounded-xl border border-slate-800 bg-slate-900/70 shadow-lg shadow-black/20', className)}>
      {title && (
        <header className="flex items-center justify-between gap-2 border-b border-slate-800 px-4 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">{title}</h2>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  size = 'md',
}: {
  value: T
  options: { value: T; label: ReactNode; title?: string }[]
  onChange: (v: T) => void
  size?: 'sm' | 'md'
}) {
  return (
    <div className="flex rounded-lg bg-slate-950/70 p-1 ring-1 ring-slate-800">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          title={o.title}
          onClick={() => onChange(o.value)}
          className={clsx(
            'flex-1 whitespace-nowrap rounded-md font-medium transition-colors',
            size === 'sm' ? 'px-1.5 py-1 text-xs' : 'px-2.5 py-1.5 text-sm',
            value === o.value ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-200',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  unit = '',
  color,
  disabled,
  onChange,
}: {
  label: ReactNode
  value: number
  min: number
  max: number
  step?: number
  unit?: string
  color?: string
  disabled?: boolean
  onChange: (v: number) => void
}) {
  return (
    <label className={clsx('block', disabled && 'opacity-50')}>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono text-xs text-slate-400">
          {Number.isInteger(step) ? Math.round(value) : value.toFixed(2)}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={color ? { accentColor: color } : undefined}
        className="w-full"
      />
    </label>
  )
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: 'good' | 'bad' }) {
  return (
    <div className="rounded-lg bg-slate-950/60 px-3 py-2 ring-1 ring-slate-800">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={clsx('mt-0.5 font-mono text-sm', tone === 'good' ? 'text-emerald-300' : tone === 'bad' ? 'text-rose-300' : 'text-slate-100')}>
        {value}
      </div>
    </div>
  )
}

export function Button({
  children,
  onClick,
  variant = 'secondary',
  disabled,
  title,
}: {
  children: ReactNode
  onClick: () => void
  variant?: 'primary' | 'secondary'
  disabled?: boolean
  title?: string
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={clsx(
        'inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50',
        variant === 'primary'
          ? 'bg-cyan-400 text-slate-950 hover:bg-cyan-300'
          : 'bg-slate-800 text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700',
      )}
    >
      {children}
    </button>
  )
}
