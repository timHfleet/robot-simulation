import type { Block, Bin, Config, FaultMode, Fixture, IKResult, Method, Elbow, Lengths, Plan, Q } from './types'

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const j = await res.json()
      detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* keep statusText */
    }
    throw new Error(`${path}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  config: () => call<Config>('config'),
  ik: (body: {
    target: { x: number; y: number }
    method: Method
    lengths: Lengths
    q0: Q
    toolAngleDeg: number | null
    elbow: Elbow
  }) => call<IKResult>('ik', body),
  plan: (body: { lengths: Lengths; q0: Q; objects: Block[]; bins: Bin[]; faults: FaultMode; seed?: number }) => call<Plan>('plan', body),
  commands: (body: {
    lengths: Lengths
    q0: Q
    objects: Block[]
    bins: Bin[]
    commands: { blockId: string; binId?: string; x?: number; z?: number }[]
    faults: FaultMode
    seed?: number
    faultOffset: number
  }) => call<Plan>('commands', body),
  task: (body: { task: 'dishes' | 'laundry'; lengths: Lengths; q0: Q; objects: Block[]; fixtures: Fixture[] }) =>
    call<Plan>('task', body),
  dual: (body: { lengths: Lengths; objects: Block[]; bins: Bin[] }) => call<Plan>('dual', body),
  shirt: (body: { lengths: Lengths; objects: Block[] }) => call<Plan>('shirt', body),
  scene: (seed?: number) => call<{ objects: Block[] }>('scene', { seed: seed ?? null }),
}
