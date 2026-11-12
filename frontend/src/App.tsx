import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import clsx from 'clsx'
import { Crosshair, Github, Pause, Play, RotateCcw, Shuffle, SlidersHorizontal, Boxes, Loader2, Home, Hand, X } from 'lucide-react'
import ArmView, { COLOR } from './components/ArmView'
import Arm3D, { type PickTarget } from './components/Arm3D'
import JointChart, { JOINT_COLORS, JOINT_LABELS, ROLL_COLOR, ROLL_LABEL } from './components/JointChart'
import SensorChart, { SENSOR_COLORS } from './components/SensorChart'
import { Button, Card, Segmented, Slider, Stat } from './components/ui'
import { api } from './lib/api'
import { stepServo, type ServoState } from './lib/servo'
import { tableReach } from './components/Arm3D'
import type { Vec3 } from './lib/geom'
import { blocksAt, clothAt, deg, fk, flatCloth, flatShirt, fmt, frameAt, frameAtFrames, rad, shirtGrid, towelGrid, wrapDeg } from './lib/geom'
import type { Bin, Block, Command, Config, Elbow, FaultMode, Fixture, IKResult, Lengths, Method, Plan, Q, SceneId, Vec } from './lib/types'

type Mode = 'ik' | 'joints' | 'command' | 'task'

const METHOD_INFO: Record<Method, { label: string; blurb: string }> = {
  analytic: {
    label: 'Analytic',
    blurb: 'Closed-form geometry: subtract the last link to find the wrist, then solve the 2-link triangle with the law of cosines. Exact and instant, picks elbow up or down.',
  },
  dls: {
    label: 'Jacobian',
    blurb: 'Damped least squares: repeatedly nudges the joints along the Jacobian toward the target. Damping keeps it stable near full stretch; joints that hit a limit are frozen.',
  },
  ccd: {
    label: 'CCD',
    blurb: 'Cyclic coordinate descent: rotates one joint at a time, wrist first, to point the tool at the target. Simple and robust, but ignores the tool angle.',
  },
}

const TASK_INFO: Record<SceneId, { run: string; idle: string; blurb: string; doneLabel: string; doneValue: (n: number) => string }> = {
  blocks: {
    run: 'Run sort task',
    idle: 'Press Run to sort the blocks into matching bins',
    blurb:
      'Sort colored blocks into the matching bins. The waist turns the arm between the blocks and the bins (no swinging over the top), with straight vertical moves to grasp and release.',
    doneLabel: 'Sorted',
    doneValue: (n) => `${n} blocks`,
  },
  dishes: {
    run: 'Load dishwasher',
    idle: 'Press Run to load the plates and cups into the dishwasher',
    blurb:
      'Plates: pinch the top rim, lift straight out of the holder and slide into a bottom-rack slot. Cups: grab from the side with the gripper level, flip mouth-down with a wrist turn, set on the top rack. Bottom rack first, like a person would.',
    doneLabel: 'Loaded',
    doneValue: (n) => `${n} items`,
  },
  dual: {
    run: 'Run with 2 arms',
    idle: 'Press Run: two arms share the sorting',
    blurb:
      'Two arms face each other across a shared work line. Whichever arm is free takes the nearest block. Before a job starts, the scheduler checks every frame of it against the other arm (link by link, in 3D); if they would come within 8.5 cm, the arm waits. Arms pull in close to their base for big turns and park on their own side after each job.',
    doneLabel: 'Sorted',
    doneValue: (n) => `${n} blocks`,
  },
  shirt: {
    run: 'Fold shirt (2 arms)',
    idle: 'Press Run: two arms fold the T-shirt together',
    blurb:
      'Both wrist cameras find the collar, sleeves and hem corners. Then each arm folds its side and sleeve in toward the middle at the same time, both arms pinch the hem corners and fold the bottom up in lockstep (same timing, so the hem stays straight), and both press the crease. The shirt bends with the grippers.',
    doneLabel: 'Folded',
    doneValue: (n) => (n ? 'yes, in thirds and half' : 'no'),
  },
  laundry: {
    run: 'Fold towel',
    idle: 'Press Run to fold the towel and put it in the basket',
    blurb:
      'Grab the far edge and swing it over toward the robot to fold the towel in half, grab a side edge and fold it again across the first fold, then drop the folded towel in the laundry basket. The cloth bends with the gripper.',
    doneLabel: 'Folded',
    doneValue: (n) => (n ? 'in quarters' : 'no'),
  },
}

const REPO_URL = 'https://github.com/timHfleet/robot-simulation'

export default function App() {
  const [cfg, setCfg] = useState<Config | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Live arm state (used whenever no planned trajectory is loaded).
  const [lengths, setLengths] = useState<Lengths>([0.34, 0.3, 0.12])
  const [q, setQ] = useState<Q>([0, 0, 0, 0])
  const [grip, setGrip] = useState(1)
  /** Wrist roll (rad) when no plan is playing. */
  const [roll, setRoll] = useState(0)
  const [blocks, setBlocks] = useState<Block[]>([])
  const [bins, setBins] = useState<Bin[]>([])
  const [fixtures, setFixtures] = useState<Fixture[]>([])
  const [scene, setScene] = useState<SceneId>('blocks')
  /** Towel shape after a folding run (null = lying flat). */
  const [clothVerts, setClothVerts] = useState<Float32Array | null>(null)
  /** Two-arm mode: the second arm's pose when no plan is playing, and which arm the charts show. */
  const [armBq, setArmBq] = useState<Q>([-Math.PI / 2, 1.658, -2.007, -1.047])
  const [chartArm, setChartArm] = useState<'A' | 'B'>('A')
  const twoArms = scene === 'dual' || scene === 'shirt'

  const [mode, setMode] = useState<Mode>('ik')
  const [method, setMethod] = useState<Method>('analytic')
  const [elbow, setElbow] = useState<Elbow>('up')
  const [toolAngle, setToolAngle] = useState(-90)
  const [lockTool, setLockTool] = useState(true)
  const [target, setTarget] = useState<Vec | null>(null)
  const [ik, setIk] = useState<IKResult | null>(null)
  const [roundTrip, setRoundTrip] = useState<number | null>(null)

  const [plan, setPlan] = useState<Plan | null>(null)
  const [planning, setPlanning] = useState(false)
  const [playT, setPlayT] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [faults, setFaults] = useState<FaultMode>('demo')

  // ---- pick & place commands ------------------------------------------------
  const [selected, setSelected] = useState<string | null>(null)
  const [queue, setQueue] = useState<Command[]>([])
  const [running, setRunning] = useState<Command | null>(null)
  const [history, setHistory] = useState<{ cmd: Command; outcome: 'done' | 'recovered' | 'skipped' | 'stopped'; note: string }[]>([])
  const [notice, setNotice] = useState<string | null>(null)
  const cmdCount = useRef(0)

  // ---- load config --------------------------------------------------------
  useEffect(() => {
    api
      .config()
      .then((c) => {
        setCfg(c)
        setLengths(c.lengths)
        setQ(c.homeQ)
        servo.current = { q: c.homeQ, v: [0, 0, 0, 0] }
        setArmQ(c.homeQ)
        setBlocks(c.objects)
        setBins(c.bins)
        setFixtures([])
      })
      .catch((e) => setError(String(e)))
  }, [])

  const base: Vec = cfg?.base ?? [0, 0.12]
  const graspFraction = cfg?.graspFraction ?? 0.3

  // ---- motors: the arm travels to the commanded pose instead of snapping ----
  // `q` is the command (from IK or the sliders); `armQ` is where the motors
  // actually are. A rate- and acceleration-limited servo loop closes the gap.
  const [armQ, setArmQ] = useState<Q>([0, 0, 0, 0])
  const servo = useRef<ServoState>({ q: [0, 0, 0, 0], v: [0, 0, 0, 0] })
  const snapArm = (to: Q) => {
    servo.current = { q: to, v: [0, 0, 0, 0] }
    setArmQ(to)
  }
  useEffect(() => {
    if (plan) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const next = stepServo(servo.current, q, dt)
      servo.current = { q: next.q, v: next.v }
      setArmQ(next.q)
      if (next.moving) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [q, plan])

  const [view, setView] = useState<'3d' | '2d'>('3d')

  // ---- what is on screen right now ----------------------------------------
  const frame = plan ? frameAt(plan, playT) : null
  const shownQ = frame?.q ?? armQ
  const shownGrip = frame?.grip ?? grip
  const shownRoll = frame?.roll ?? roll
  const shownBlocks = plan ? blocksAt(plan, blocks, playT, lengths, base, graspFraction) : blocks
  const pose = fk(shownQ, lengths, base)
  const tool = pose.pts[3]

  const toolPath = useMemo<Vec3[]>(() => {
    if (!plan) return []
    return plan.frames.filter((_, i) => i % 2 === 0).map((f) => fk(f.q, lengths, base).pts3[3])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plan])
  // Show the last ~3 seconds of tool motion (points are every 2nd frame).
  const trailEnd = plan ? Math.floor((playT * plan.fps) / 2) + 1 : 0
  const trail3 = plan ? toolPath.slice(Math.max(0, trailEnd - plan.fps * 1.5), trailEnd) : []
  const trail = trail3.map(([x, y]) => [x, y] as Vec)

  /** Leave trajectory playback and continue from the pose currently shown. */
  const goLive = useCallback((): { q: Q; arm: Q; blocks: Block[] } => {
    if (!plan) return { q, arm: servo.current.q, blocks }
    const f = frameAt(plan, playT)
    // A block that was in the gripper goes back to where it started on the table.
    const b = blocksAt(plan, blocks, playT, lengths, base, graspFraction).map((blk, i) =>
      // A block caught mid-fall lands where it was falling.
      blk.id === f.holding
        ? blocks[i]
        : (blk.kind ?? 'block') === 'block'
          ? { ...blk, y: blk.size / 2 }
          : // A towel that made it into the basket stays there.
            plan.cloth && blk.id === plan.cloth.towelId && playT >= (plan.cloth.carry?.tRelease ?? plan.cloth.folds[plan.cloth.folds.length - 1].tRelease)
            ? { ...blk, binId: plan.cloth.carry?.basketId ?? 'folded' }
            : blk,
    )
    setQ(f.q)
    qRef.current = f.q
    snapArm(f.q)
    setGrip(f.grip)
    setRoll(f.roll ?? 0)
    setBlocks(b)
    if (plan.cloth) setClothVerts(clothAt(plan, plan.cloth, playT, lengths, base))
    if (plan.armB) setArmBq(frameAtFrames(plan.armB.frames, plan.fps, playT).q)
    setPlan(null)
    setPlaying(false)
    return { q: f.q, arm: f.q, blocks: b }
  }, [plan, playT, q, blocks, lengths, base, graspFraction])

  // ---- inverse kinematics (one request in flight, latest target wins) ----
  const qRef = useRef(q)
  qRef.current = frame ? frame.q : q
  const params = useRef({ method, elbow, toolAngle, lockTool, lengths })
  params.current = { method, elbow, toolAngle, lockTool, lengths }
  const ikQueue = useRef<{ busy: boolean; next: Vec | null }>({ busy: false, next: null })

  const runIK = useCallback(async (t: Vec) => {
    const s = ikQueue.current
    s.next = t
    if (s.busy) return
    s.busy = true
    while (s.next) {
      const cur = s.next
      s.next = null
      const p = params.current
      const useAngle = p.method === 'analytic' || (p.method === 'dls' && p.lockTool)
      const started = performance.now()
      try {
        const res = await api.ik({
          target: { x: cur[0], y: cur[1] },
          method: p.method,
          lengths: p.lengths,
          q0: qRef.current,
          toolAngleDeg: useAngle ? p.toolAngle : null,
          elbow: p.elbow,
        })
        setIk(res)
        setQ(res.q)
        qRef.current = res.q
        setRoundTrip(performance.now() - started)
        setError(null)
      } catch (e) {
        setError(String(e))
      }
    }
    s.busy = false
  }, [])

  const onTarget = (p: Vec) => {
    goLive()
    if (mode !== 'ik') setMode('ik')
    setTarget(p)
    runIK(p)
  }

  // Re-solve when the solver settings or the arm itself change.
  useEffect(() => {
    if (target && !plan) runIK(target)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [method, elbow, toolAngle, lockTool, lengths])

  // ---- playback -------------------------------------------------------------
  const speedRef = useRef(speed)
  speedRef.current = speed
  useEffect(() => {
    if (!playing || !plan) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      setPlayT((t) => Math.min(plan.duration, t + dt * speedRef.current))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, plan])
  useEffect(() => {
    if (plan && playing && playT >= plan.duration) setPlaying(false)
  }, [plan, playing, playT])

  /** Commit the pose on screen, ask the Python planner for a trajectory, then play it. */
  const startPlan = async (fetchPlan: (live: { arm: Q; blocks: Block[] }) => Promise<Plan>) => {
    const live = goLive()
    setPlanning(true)
    setTarget(null)
    setIk(null)
    try {
      // Plan from where the motors actually are, and stop chasing the old command.
      setQ(live.arm)
      qRef.current = live.arm
      snapArm(live.arm)
      const res = await fetchPlan(live)
      setBlocks(live.blocks)
      setPlan(res)
      setPlayT(0)
      setPlaying(true)
      setError(null)
      return true
    } catch (e) {
      setError(String(e))
      return false
    } finally {
      setPlanning(false)
    }
  }

  const randomSeed = () => (faults === 'random' ? Math.floor(Math.random() * 1e6) : undefined)

  const runTask = () =>
    startPlan((live) =>
      scene === 'blocks'
        ? api.plan({ lengths, q0: live.arm, objects: live.blocks, bins, faults, seed: randomSeed() })
        : scene === 'dual'
          ? api.dual({ lengths, objects: live.blocks, bins })
          : scene === 'shirt'
          ? api.shirt({ lengths, objects: live.blocks })
          : api.task({ task: scene, lengths, q0: live.arm, objects: live.blocks, fixtures }),
    )

  /** Switch to (or reset) a task's scene: its objects, bins and fixtures. */
  const loadScene = (id: SceneId) => {
    if (!cfg) return
    goLive()
    const sc = cfg.scenes[id]
    setScene(id)
    setBlocks(sc.objects)
    setBins(sc.bins)
    setFixtures(sc.fixtures)
    setClothVerts(null)
    setPlan(null)
    setPlaying(false)
    setPlayT(0)
    setSelected(null)
    setQueue([])
    setGrip(1)
    setRoll(0)
    // In two-arm mode both arms start parked on their own side of the table.
    const startQ = sc.arms ? sc.arms[0].park : cfg.homeQ
    setQ(startQ)
    qRef.current = startQ
    if (sc.arms) setArmBq(sc.arms[1].park)
    setChartArm('A')
  }

  // A command finished (or was interrupted): record how it went.
  useEffect(() => {
    if (!running) return
    if (!plan && !planning) {
      setHistory((h) => [{ cmd: running, outcome: 'stopped' as const, note: 'Interrupted' }, ...h].slice(0, 8))
      setRunning(null)
      return
    }
    if (!plan || playing || playT < plan.duration) return
    const s = plan.stats
    const warn = plan.events.find((e) => e.kind === 'warn')
    const faulted = s.drops + s.slips > 0
    const outcome = s.skipped ? ('skipped' as const) : faulted ? ('recovered' as const) : ('done' as const)
    const note = s.skipped
      ? (warn?.text.replace(/^Skipped [a-z]+ block: /, '') ?? 'Skipped')
      : faulted
        ? `${s.drops ? 'Dropped' : 'Slipped'}, recovered · ${fmt(s.durationS, 1)} s`
        : `${fmt(s.durationS, 1)} s`
    setHistory((h) => [{ cmd: running, outcome, note }, ...h].slice(0, 8))
    setRunning(null)
  }, [running, plan, planning, playing, playT])

  // Idle with commands waiting: start the next one.
  useEffect(() => {
    if (running || planning || queue.length === 0) return
    if (plan && (playing || playT < plan.duration)) return // a trajectory is playing or paused
    const [next, ...rest] = queue
    setQueue(rest)
    setRunning(next)
    const offset = cmdCount.current++
    void startPlan((live) =>
      api.commands({
        lengths,
        q0: live.arm,
        objects: live.blocks,
        bins,
        commands: [{ blockId: next.blockId, binId: next.binId, x: next.x, z: next.z }],
        faults,
        seed: randomSeed(),
        faultOffset: offset,
      }),
    ).then((ok) => {
      if (!ok) setRunning(null)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue, running, planning, plan, playing, playT])

  const blockName = (id: string) => {
    const b = shownBlocks.find((x) => x.id === id)
    return b ? `${b.color} block` : 'block'
  }

  const onPick = (t: PickTarget) => {
    setNotice(null)
    if (t.kind === 'block') {
      setSelected((cur) => (cur === t.id ? null : t.id))
      return
    }
    if (!selected) {
      setNotice('Click a block first, then click where it should go.')
      return
    }
    if (t.kind === 'table' && Math.hypot(t.x, t.z) > tableReach(lengths)) {
      setNotice(`That spot is ${fmt(Math.hypot(t.x, t.z) * 100, 0)} cm from the base; the arm reaches about ${fmt(tableReach(lengths) * 100, 0)} cm. Pick a spot inside the dashed ring.`)
      return
    }
    queueCommand(selected, t.kind === 'bin' ? { binId: t.id } : { x: t.x, z: t.z })
  }

  const queueCommand = (blockId: string, dest: { binId: string } | { x: number; z: number }) => {
    const name = blockName(blockId)
    const id = `cmd-${Date.now()}-${Math.round(Math.random() * 1e6)}`
    const cmd: Command =
      'binId' in dest
        ? { id, blockId, binId: dest.binId, label: `${name} → ${bins.find((b) => b.id === dest.binId)?.color ?? ''} bin` }
        : {
            id,
            blockId,
            x: Math.round(dest.x * 1000) / 1000,
            z: Math.round(dest.z * 1000) / 1000,
            label: `${name} → table (${fmt(dest.x, 2)}, ${fmt(dest.z, 2)})`,
          }
    setQueue((q) => [...q, cmd])
    setSelected(null)
  }

  const armHome = () => {
    if (!cfg) return
    goLive()
    setTarget(null)
    setIk(null)
    setQ(cfg.homeQ)
    qRef.current = cfg.homeQ
  }

  const shuffle = async () => {
    goLive()
    try {
      const res = await api.scene()
      setBlocks(res.objects)
      setPlan(null)
    } catch (e) {
      setError(String(e))
    }
  }

  const resetAll = () => {
    if (!cfg) return
    setPlan(null)
    setPlaying(false)
    setPlayT(0)
    setLengths(cfg.lengths)
    setQ(cfg.homeQ)
    qRef.current = cfg.homeQ
    setGrip(1)
    setRoll(0)
    setScene('blocks')
    setBlocks(cfg.objects)
    setBins(cfg.bins)
    setFixtures([])
    setClothVerts(null)
    setSelected(null)
    setQueue([])
    setTarget(null)
    setIk(null)
  }

  const setJoint = (i: number, valueDeg: number) => {
    const live = goLive()
    const next = [...live.q] as Q
    next[i] = rad(valueDeg)
    setQ(next)
    qRef.current = next
    setTarget(null)
    setIk(null)
  }

  const setLength = (i: number, v: number) => {
    goLive()
    setLengths((l) => l.map((x, j) => (j === i ? v : x)) as Lengths)
  }

  const togglePlay = () => {
    if (!plan) return
    if (playT >= plan.duration) setPlayT(0)
    setPlaying((p) => !p)
  }

  // Rings on the table for queued "put it here" commands.
  const dropMarkers = [running, ...queue].filter((c): c is Command => !!c && c.x !== undefined).map((c) => ({ x: c.x!, z: c.z! }))

  // Charts follow the arm chosen in two-arm mode.
  const chartPlan: Plan | null =
    plan && chartArm === 'B' && plan.armB ? { ...plan, frames: plan.armB.frames, sensors: plan.armB.sensors } : plan

  // What the wrist camera is looking at right now, and the force sensors.
  const activeDetections = plan?.perception?.filter((d) => playT >= d.tStart && playT <= d.tEnd) ?? []
  const sensorIdx = plan ? Math.min(plan.frames.length - 1, Math.max(0, Math.round(playT * plan.fps))) : 0
  const gripForce = plan?.sensors ? plan.sensors.grip[sensorIdx] : 0
  const wristLoad = plan?.sensors ? plan.sensors.fz[sensorIdx] : 0

  const activeEvent = plan ? [...plan.events].reverse().find((e) => e.t <= playT + 1e-6) : undefined
  const limits = cfg?.limitsDeg ?? [
    [-180, 180],
    [-10, 190],
    [-160, 160],
    [-160, 160],
  ]

  const commandHint = running
    ? (activeEvent?.text ?? 'Planning…')
    : selected
      ? `Now click a bin or a spot on the table for the ${blockName(selected)}`
      : 'Click a block to pick it up'

  const hint =
    mode === 'command'
      ? commandHint
      : mode === 'ik'
      ? 'Click or drag anywhere to move the gripper there'
      : mode === 'joints'
        ? 'Drive each joint with the sliders (forward kinematics)'
        : plan
          ? activeEvent?.text
          : TASK_INFO[scene].idle

  const hint3d =
    mode === 'command'
      ? commandHint
      : mode === 'ik'
        ? 'Click a point on the glowing work plane to move the gripper there'
        : mode === 'task' && !plan
          ? TASK_INFO[scene].idle
          : hint

  // The towel as drawn: folding live, as it was left, or lying flat.
  const towel = blocks.find((b) => b.kind === 'towel' || b.kind === 'shirt')
  const isShirt = towel?.kind === 'shirt'
  const clothGrid = useMemo(
    () => (towel ? (isShirt ? shirtGrid(towel.color) : towelGrid(towel.color)) : null),
    [towel?.kind, towel?.color], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const clothNow =
    towel && clothGrid
      ? {
          verts: plan?.cloth
            ? clothAt(plan, plan.cloth, playT, lengths, base)
            : (clothVerts ?? (isShirt ? flatShirt([towel.x, towel.z ?? 0]) : flatCloth([towel.x, towel.z ?? 0], towel.size))),
          grid: clothGrid,
        }
      : null

  const eventLog = () =>
    plan && (
          <ol className="max-h-56 space-y-0.5 overflow-y-auto pr-1 text-xs">
            {plan.events.map((e, i) => (
              <li key={i}>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false)
                    setPlayT(e.t)
                  }}
                  className={clsx(
                    'flex w-full gap-2 rounded px-2 py-1 text-left hover:bg-slate-800',
                    activeEvent === e ? 'bg-slate-800 text-white' : 'text-slate-400',
                    e.kind === 'warn' && 'text-amber-300',
                    e.kind === 'fault' && 'text-rose-300',
                    e.kind === 'recover' && 'text-cyan-200',
                  )}
                >
                  <span className="w-10 shrink-0 font-mono text-slate-500">{fmt(e.t, 1)}s</span>
                  <span className={clsx('mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full', dotClass(e.kind))} />
                  {e.text}
                </button>
              </li>
            ))}
          </ol>
    )

  if (!cfg && !error) {
    return (
      <div className="flex min-h-screen items-center justify-center gap-2 bg-slate-950 text-slate-400">
        <Loader2 className="h-5 w-5 animate-spin" /> Starting the simulator… (the free server can take about a minute to wake up)
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-950/80">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-cyan-400/10 ring-1 ring-cyan-400/30">
              <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden>
                <path d="M7 26 L13 14 L23 9" stroke="#22d3ee" strokeWidth="3.5" fill="none" strokeLinecap="round" />
                <circle cx="13" cy="14" r="2.8" fill="#f8fafc" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-semibold leading-tight">Robot Arm Simulator</h1>
              <p className="text-xs text-slate-400">4-joint arm (waist + 3-joint arm) · inverse kinematics · pick-and-place planning</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={resetAll} title="Reset arm, blocks and link lengths">
              <Home className="h-4 w-4" /> Reset
            </Button>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700"
            >
              <Github className="h-4 w-4" /> Code
            </a>
          </div>
        </div>
      </header>

      {error && (
        <div className="mx-auto mt-3 max-w-[1400px] px-4 sm:px-6">
          <div className="rounded-lg border border-rose-900 bg-rose-950/50 px-3 py-2 text-sm text-rose-200">{error}</div>
        </div>
      )}

      <main className="mx-auto grid max-w-[1400px] gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-4">
          <Card
            className="overflow-hidden"
            title={view === '3d' ? 'Workspace (3D)' : 'Workspace (side view)'}
            right={
              <div className="flex items-center gap-2">
                <div className="w-32" title="One robot, or two sharing the block sorting">
                  <Segmented
                    size="sm"
                    value={twoArms ? 2 : 1}
                    onChange={(n) => {
                      if (n === 2 && !twoArms) {
                        loadScene('dual')
                        setMode('task')
                        setView('3d')
                      } else if (n === 1 && twoArms) {
                        loadScene('blocks')
                      }
                    }}
                    options={[
                      { value: 1, label: '1 arm' },
                      { value: 2, label: '2 arms' },
                    ]}
                  />
                </div>
                <div className="w-40">
                  <Segmented
                    size="sm"
                    value={view}
                    onChange={setView}
                    options={[
                      { value: '3d', label: '3D' },
                      { value: '2d', label: 'Side view' },
                    ]}
                  />
                </div>
              </div>
            }
          >
            <div className="-m-4">
              {view === '3d' ? (
                <Arm3D
                  q={shownQ}
                  lengths={lengths}
                  base={base}
                  grip={shownGrip}
                  roll={shownRoll}
                  blocks={shownBlocks}
                  bins={bins}
                  target={mode === 'ik' ? target : null}
                  targetOk={ik?.reachable ?? true}
                  trail={trail3}
                  showReachPlane={mode === 'ik' && !plan}
                  interactive={!playing && mode === 'ik'}
                  hint={hint3d}
                  onTarget={onTarget}
                  pickMode={mode === 'command' && scene === 'blocks'}
                  selectedId={selected}
                  markers={dropMarkers}
                  onPick={onPick}
                  fixtures={fixtures}
                  cloth={clothNow}
                  detections={activeDetections}
                  second={
                    twoArms
                      ? {
                          q: plan?.armB ? frameAtFrames(plan.armB.frames, plan.fps, playT).q : armBq,
                          grip: plan?.armB ? frameAtFrames(plan.armB.frames, plan.fps, playT).grip : 1,
                          roll: plan?.armB ? frameAtFrames(plan.armB.frames, plan.fps, playT).roll : 0,
                          baseZ: plan?.armB?.baseZ ?? cfg?.scenes.dual.arms?.[1].baseZ ?? -0.44,
                        }
                      : null
                  }
                />
              ) : (
              <ArmView
                q={shownQ}
                lengths={lengths}
                base={base}
                grip={shownGrip}
                roll={shownRoll}
                blocks={shownBlocks}
                bins={bins}
                target={mode === 'ik' ? target : null}
                targetOk={ik?.reachable ?? true}
                trail={trail}
                interactive={!playing}
                hint={hint}
                onTarget={onTarget}
                pickMode={mode === 'command' && scene === 'blocks'}
                selectedId={selected}
                onPick={onPick}
                note={scene === 'blocks' ? undefined : 'The side view only draws the one-arm block scene. Switch to 3D to watch this task.'}
              />
              )}
            </div>
          </Card>

          <Card
            title="Trajectory"
            right={
              plan && (
                <div className="flex items-center gap-3 text-xs">
                  {JOINT_LABELS.map((l, i) => (
                    <span key={l} className="flex items-center gap-1.5 text-slate-400">
                      <span className="h-2 w-2 rounded-full" style={{ background: JOINT_COLORS[i] }} />
                      {l}
                    </span>
                  ))}
                  <span className="flex items-center gap-1.5 text-slate-400">
                    <span className="h-0.5 w-3 rounded-full" style={{ background: ROLL_COLOR }} />
                    {ROLL_LABEL}
                  </span>
                  <span className="hidden items-center gap-1.5 text-slate-400 sm:flex">
                    <span className="h-2 w-3 rounded-sm bg-amber-300/30" /> holding a block
                  </span>
                </div>
              )
            }
          >
            {plan ? (
              <div className="space-y-3">
                {plan.armB && (
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span>Charts for</span>
                    <div className="w-36">
                      <Segmented
                        size="sm"
                        value={chartArm}
                        onChange={setChartArm}
                        options={[
                          { value: 'A', label: 'Arm A' },
                          { value: 'B', label: 'Arm B' },
                        ]}
                      />
                    </div>
                  </div>
                )}
                <JointChart
                  plan={chartPlan!}
                  t={playT}
                  onSeek={(t) => {
                    setPlaying(false)
                    setPlayT(t)
                  }}
                />
                {plan.sensors && (
                  <div>
                    <div className="mb-1 flex flex-wrap items-center gap-3 text-xs text-slate-400">
                      <span className="font-semibold uppercase tracking-wider text-slate-500">Force sensors</span>
                      <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full" style={{ background: SENSOR_COLORS.grip }} />
                        gripper force
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full" style={{ background: SENSOR_COLORS.fz }} />
                        wrist load (weight + contact)
                      </span>
                    </div>
                    <SensorChart
                      plan={chartPlan!}
                      t={playT}
                      onSeek={(t) => {
                        setPlaying(false)
                        setPlayT(t)
                      }}
                    />
                  </div>
                )}
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant="primary" onClick={togglePlay}>
                    {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {playing ? 'Pause' : playT >= plan.duration ? 'Replay' : 'Play'}
                  </Button>
                  <Button
                    onClick={() => {
                      setPlayT(0)
                      setPlaying(false)
                    }}
                    title="Back to the start"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                  <div className="w-48">
                    <Segmented
                      size="sm"
                      value={speed}
                      onChange={setSpeed}
                      options={[0.5, 1, 2, 4].map((s) => ({ value: s, label: `${s}×` }))}
                    />
                  </div>
                  <span className="ml-auto font-mono text-sm text-slate-400">
                    {fmt(playT, 1)} / {fmt(plan.duration, 1)} s
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-400">
                Run the <span className="text-slate-200">Sort task</span> to plan a full pick-and-place trajectory. You'll see the shoulder, elbow and
                wrist angles over time here and can scrub through it.
              </p>
            )}
          </Card>
        </div>

        <aside className="space-y-4">
          <Card>
            <Segmented
              size="sm"
              value={mode}
              onChange={(m) => setMode(m)}
              options={[
                { value: 'ik', label: <span className="inline-flex items-center gap-1"><Crosshair className="h-3.5 w-3.5" />Reach</span> },
                { value: 'joints', label: <span className="inline-flex items-center gap-1"><SlidersHorizontal className="h-3.5 w-3.5" />Joints</span> },
                { value: 'command', label: <span className="inline-flex items-center gap-1"><Hand className="h-3.5 w-3.5" />Pick & place</span> },
                { value: 'task', label: <span className="inline-flex items-center gap-1"><Boxes className="h-3.5 w-3.5" />Tasks</span> },
              ]}
            />

            {mode === 'command' && scene !== 'blocks' && (
              <div className="mt-4 space-y-3">
                <p className="text-sm text-slate-300">Pick & place commands work in the block scene.</p>
                <Button variant="primary" onClick={() => loadScene('blocks')}>
                  <Boxes className="h-4 w-4" /> Switch to the blocks
                </Button>
              </div>
            )}

            {mode === 'command' && scene === 'blocks' && (
              <div className="mt-4 space-y-4">
                <p className="text-xs leading-relaxed text-slate-400">
                  Click a block, then click a bin or any free spot on the table. Commands queue up and run in order; the Python planner checks
                  reach and clearance first and recovers from drops on the way.
                </p>

                <div className="rounded-lg bg-slate-950/60 p-3 ring-1 ring-slate-800">
                  {selected ? (
                    <div className="space-y-2.5">
                      <div className="flex items-center justify-between text-sm">
                        <span className="inline-flex items-center gap-2 text-slate-200">
                          <span className="h-3 w-3 rounded-sm" style={{ background: COLOR[shownBlocks.find((b) => b.id === selected)?.color ?? ''] }} />
                          {blockName(selected)} selected
                        </span>
                        <button type="button" onClick={() => setSelected(null)} className="text-xs text-slate-400 hover:text-slate-200">
                          Cancel
                        </button>
                      </div>
                      <div className="text-xs text-slate-400">Place it in a bin, or click a spot on the table:</div>
                      <div className="flex gap-1.5">
                        {bins.map((b) => (
                          <button
                            key={b.id}
                            type="button"
                            onClick={() => queueCommand(selected, { binId: b.id })}
                            className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-slate-800 px-2 py-1.5 text-xs text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700"
                          >
                            <span className="h-2 w-2 rounded-sm" style={{ background: COLOR[b.color] }} />
                            {b.color} bin
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-400">No block selected. Click one in the view.</div>
                  )}
                  {notice && <p className="mt-2 text-xs text-amber-300">{notice}</p>}
                </div>

                <div>
                  <div className="mb-1.5 text-xs font-medium text-slate-400">Failures to inject</div>
                  <Segmented
                    size="sm"
                    value={faults}
                    onChange={setFaults}
                    options={[
                      { value: 'none', label: 'None' },
                      { value: 'demo', label: 'Drop + slip', title: 'The 2nd command slips while lifting, the 4th is dropped mid-move' },
                      { value: 'random', label: 'Random', title: '20% slip, 20% drop per command' },
                    ]}
                  />
                </div>

                <div>
                  <div className="mb-1.5 flex items-center justify-between text-xs font-medium text-slate-400">
                    <span>Command queue</span>
                    {queue.length > 0 && (
                      <button type="button" onClick={() => setQueue([])} className="text-slate-500 hover:text-slate-300">
                        Clear
                      </button>
                    )}
                  </div>
                  {!running && queue.length === 0 ? (
                    <p className="text-xs text-slate-500">Empty.</p>
                  ) : (
                    <ol className="space-y-1 text-xs">
                      {running && (
                        <li className="flex items-center gap-2 rounded-md bg-cyan-400/10 px-2.5 py-1.5 text-cyan-100 ring-1 ring-cyan-400/30">
                          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                          <span className="flex-1">{running.label}</span>
                          <span className="text-cyan-300/80">{playing ? 'running' : planning ? 'planning' : 'paused'}</span>
                        </li>
                      )}
                      {queue.map((c, i) => (
                        <li key={c.id} className="flex items-center gap-2 rounded-md bg-slate-800/70 px-2.5 py-1.5 text-slate-300">
                          <span className="w-4 shrink-0 font-mono text-slate-500">{i + 1}</span>
                          <span className="flex-1">{c.label}</span>
                          <button type="button" title="Remove" onClick={() => setQueue((q) => q.filter((x) => x.id !== c.id))} className="text-slate-500 hover:text-slate-200">
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>

                {history.length > 0 && (
                  <div>
                    <div className="mb-1.5 text-xs font-medium text-slate-400">Finished</div>
                    <ol className="space-y-1 text-xs">
                      {history.map((h, i) => (
                        <li key={h.cmd.id + i} className="flex items-start gap-2 px-1">
                          <span
                            className={clsx(
                              'mt-1 h-1.5 w-1.5 shrink-0 rounded-full',
                              h.outcome === 'done' ? 'bg-emerald-400' : h.outcome === 'recovered' ? 'bg-cyan-300' : h.outcome === 'skipped' ? 'bg-amber-500' : 'bg-slate-500',
                            )}
                          />
                          <span className="flex-1 text-slate-300">{h.cmd.label}</span>
                          <span className={clsx('text-right', h.outcome === 'skipped' ? 'text-amber-300' : h.outcome === 'recovered' ? 'text-cyan-200' : 'text-slate-500')}>
                            {h.note}
                          </span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}

                <div className="flex gap-2">
                  <Button onClick={armHome} disabled={!!running || planning} title="Send the arm back to its home pose">
                    <Home className="h-4 w-4" /> Arm home
                  </Button>
                  <Button onClick={shuffle} disabled={!!running || planning} title="New random block layout">
                    <Shuffle className="h-4 w-4" /> Shuffle
                  </Button>
                </div>

                {plan && running && eventLog()}
              </div>
            )}

            {mode === 'ik' && (
              <div className="mt-4 space-y-4">
                <div>
                  <div className="mb-1.5 text-xs font-medium text-slate-400">IK solver</div>
                  <Segmented
                    value={method}
                    onChange={setMethod}
                    options={(Object.keys(METHOD_INFO) as Method[]).map((m) => ({ value: m, label: METHOD_INFO[m].label }))}
                  />
                  <p className="mt-2 text-xs leading-relaxed text-slate-400">{METHOD_INFO[method].blurb}</p>
                </div>

                {method === 'analytic' && (
                  <div>
                    <div className="mb-1.5 text-xs font-medium text-slate-400">Elbow</div>
                    <Segmented size="sm" value={elbow} onChange={setElbow} options={[{ value: 'up', label: 'Up' }, { value: 'down', label: 'Down' }]} />
                  </div>
                )}

                {method !== 'ccd' && (
                  <div className="space-y-2">
                    {method === 'dls' && (
                      <label className="flex items-center gap-2 text-sm text-slate-300">
                        <input type="checkbox" checked={lockTool} onChange={(e) => setLockTool(e.target.checked)} className="accent-cyan-400" />
                        Also match the tool angle
                      </label>
                    )}
                    <Slider
                      label="Tool angle φ"
                      value={toolAngle}
                      min={-180}
                      max={180}
                      unit="°"
                      disabled={method === 'dls' && !lockTool}
                      onChange={setToolAngle}
                    />
                    <div className="flex gap-1.5">
                      {[
                        { v: -90, l: 'Down' },
                        { v: 0, l: 'Right' },
                        { v: 90, l: 'Up' },
                        { v: 180, l: 'Left' },
                      ].map((o) => (
                        <button
                          key={o.v}
                          type="button"
                          disabled={method === 'dls' && !lockTool}
                          onClick={() => setToolAngle(o.v)}
                          className="flex-1 rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300 ring-1 ring-slate-700 hover:bg-slate-700 disabled:opacity-50"
                        >
                          {o.l}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {ik ? (
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-2">
                      <Stat label="Result" value={ik.reachable ? 'Reached' : 'Not reached'} tone={ik.reachable ? 'good' : 'bad'} />
                      <Stat label="Position error" value={`${fmt(ik.positionErrorMm, ik.positionErrorMm < 1 ? 3 : 1)} mm`} />
                      <Stat label="Iterations" value={ik.iterations} />
                      <Stat label="Solve time" value={`${fmt(ik.solveMs, 2)} ms`} />
                    </div>
                    {roundTrip !== null && <p className="text-[11px] text-slate-500">Round trip to the Python API: {fmt(roundTrip, 0)} ms</p>}
                    {ik.notes.map((n) => (
                      <p key={n} className="rounded-md bg-amber-400/10 px-2.5 py-1.5 text-xs text-amber-200 ring-1 ring-amber-400/20">
                        {n}
                      </p>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-slate-400">Click anywhere in the workspace to set a target.</p>
                )}
              </div>
            )}

            {mode === 'joints' && (
              <div className="mt-4 space-y-4">
                {JOINT_LABELS.map((label, i) => (
                  <Slider
                    key={label}
                    label={label}
                    value={Math.round(deg(shownQ[i]))}
                    min={limits[i][0]}
                    max={limits[i][1]}
                    unit="°"
                    color={JOINT_COLORS[i]}
                    onChange={(v) => setJoint(i, v)}
                  />
                ))}
                <Slider
                  label={ROLL_LABEL}
                  value={Math.round(deg(shownRoll))}
                  min={-180}
                  max={180}
                  unit="°"
                  color={ROLL_COLOR}
                  onChange={(v) => {
                    goLive()
                    setRoll(rad(v))
                  }}
                />
                <Slider
                  label="Gripper"
                  value={Math.round(shownGrip * 100)}
                  min={0}
                  max={100}
                  unit="% open"
                  onChange={(v) => {
                    goLive()
                    setGrip(v / 100)
                  }}
                />
                <p className="text-xs leading-relaxed text-slate-400">
                  This is forward kinematics: you set the angles and the tool position follows from summing each link's vector. The waist turns the
                  whole arm about the vertical axis; watch the top view.
                </p>
              </div>
            )}

            {mode === 'task' && (
              <div className="mt-4 space-y-4">
                <div>
                  <div className="mb-1.5 text-xs font-medium text-slate-400">Task</div>
                  <Segmented
                    size="sm"
                    value={scene}
                    onChange={(id) => id !== scene && loadScene(id)}
                    options={[
                      { value: 'blocks', label: 'Blocks' },
                      { value: 'dual', label: '2 arms' },
                      { value: 'dishes', label: 'Dishes' },
                      { value: 'laundry', label: 'Towel' },
                      { value: 'shirt', label: 'Shirt' },
                    ]}
                  />
                  <p className="mt-2 text-xs leading-relaxed text-slate-400">{TASK_INFO[scene].blurb}</p>
                </div>
                {scene === 'blocks' && (
                <div>
                  <div className="mb-1.5 text-xs font-medium text-slate-400">Failures to inject</div>
                  <Segmented
                    size="sm"
                    value={faults}
                    onChange={setFaults}
                    options={[
                      { value: 'none', label: 'None', title: 'A clean run' },
                      { value: 'demo', label: 'Drop + slip', title: 'One block slips while lifting, one is dropped mid-swing' },
                      { value: 'random', label: 'Random', title: 'Each pick has a 20% chance to slip and a 20% chance to drop' },
                    ]}
                  />
                  <p className="mt-2 text-xs leading-relaxed text-slate-400">
                    {faults === 'none'
                      ? 'Every grasp holds.'
                      : faults === 'demo'
                        ? 'The 2nd block slips out while lifting and is re-grasped with more force. The 4th is dropped mid-swing: the arm brakes, the camera relocates the block, and the arm turns to pick it up where it landed.'
                        : 'Each pick has a 20% chance to slip and a 20% chance to be dropped in transit. Every failure is detected and recovered.'}
                  </p>
                </div>
                )}
                <div className="flex gap-2">
                  <Button variant="primary" onClick={runTask} disabled={planning || playing || !!running || queue.length > 0}>
                    {planning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} {TASK_INFO[scene].run}
                  </Button>
                  {scene === 'blocks' ? (
                    <Button onClick={shuffle} disabled={planning || playing} title="New random block layout">
                      <Shuffle className="h-4 w-4" /> Shuffle
                    </Button>
                  ) : (
                    <Button onClick={() => loadScene(scene)} disabled={planning || playing} title="Put everything back where it started">
                      <RotateCcw className="h-4 w-4" /> Reset scene
                    </Button>
                  )}
                </div>
                {plan && (
                  <>
                    <div className="grid grid-cols-2 gap-2">
                      {plan.dual && (
                        <div className="col-span-2 space-y-1.5 rounded-lg bg-slate-950/60 p-3 ring-1 ring-slate-800">
                          <div className="flex items-baseline justify-between text-xs">
                            <span className="uppercase tracking-wide text-slate-500">Two arms</span>
                            {plan.dual.speedup && (
                              <span className="font-mono text-emerald-300">
                                {plan.dual.speedup}× faster than one arm ({fmt(plan.dual.singleArmS ?? 0, 0)} s)
                              </span>
                            )}
                          </div>
                          {plan.dual.arms.map((a) => (
                            <div key={a.id} className="flex items-center gap-2 text-xs">
                              <span className={`w-12 font-medium ${a.id === 'A' ? 'text-orange-300' : 'text-sky-300'}`}>Arm {a.id}</span>
                              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-800">
                                <div className={`h-full ${a.id === 'A' ? 'bg-orange-400' : 'bg-sky-400'}`} style={{ width: `${Math.round(a.utilization * 100)}%` }} />
                              </div>
                              <span className="w-40 text-right font-mono text-slate-400">
                                {scene === 'shirt' ? `${a.blocks} grasps · in lockstep` : `${a.blocks} blocks · ${Math.round(a.utilization * 100)}% busy · waited ${fmt(a.waitS, 1)} s`}
                              </span>
                            </div>
                          ))}
                          <div className="text-[11px] text-slate-500">
                            Closest the arms came: {fmt(plan.dual.minArmGapMm, 0)} mm (limit {fmt(plan.dual.clearanceMm, 0)} mm)
                          </div>
                        </div>
                      )}
                      <Stat label={TASK_INFO[scene].doneLabel} value={TASK_INFO[scene].doneValue(plan.stats.sorted)} tone="good" />
                      <Stat label="Skipped" value={plan.stats.skipped} tone={plan.stats.skipped ? 'bad' : undefined} />
                      <Stat label="Cycle time" value={`${fmt(plan.stats.durationS, 1)} s`} />
                      <Stat label="Tool travel" value={`${fmt(plan.stats.toolPathM, 2)} m`} />
                      <Stat label="Peak joint speed" value={`${fmt(plan.stats.maxJointSpeedDegS, 0)}°/s`} />
                      <Stat label="Min clearance" value={`${fmt(plan.stats.minClearanceMm, 0)} mm`} />
                      {scene === 'blocks' && (
                      <>
                      <Stat
                        label="Faults"
                        value={`${plan.stats.drops} drop${plan.stats.drops === 1 ? '' : 's'} · ${plan.stats.slips} slip${plan.stats.slips === 1 ? '' : 's'}`}
                        tone={plan.stats.drops + plan.stats.slips ? 'bad' : undefined}
                      />
                      <Stat
                        label="Recovered"
                        value={`${plan.stats.recovered} / ${plan.stats.drops + plan.stats.slips}`}
                        tone={plan.stats.recovered === plan.stats.drops + plan.stats.slips ? 'good' : 'bad'}
                      />
                      </>
                      )}
                    </div>
                    {eventLog()}
                  </>
                )}
                <div className="flex flex-wrap gap-2">
                  {fixtures
                    .filter((fx) => fx.kind !== 'plate_holder')
                    .map((fx) => (
                      <span key={fx.id} className="inline-flex items-center gap-1.5 rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
                        {fx.kind === 'basket'
                          ? `${shownBlocks.some((x) => x.kind === 'towel') && (clothNow && plan?.cloth ? playT >= (plan.cloth.carry?.tRelease ?? Infinity) : !!clothVerts) ? 1 : 0} towel in the basket`
                          : `${shownBlocks.filter((x) => x.binId === fx.id).length} / ${fx.slots?.length ?? 0} in ${(fx.label ?? fx.id).toLowerCase()}`}
                      </span>
                    ))}
                  {bins.map((b) => (
                    <span key={b.id} className="inline-flex items-center gap-1.5 rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
                      <span className="h-2 w-2 rounded-sm" style={{ background: COLOR[b.color] }} />
                      {shownBlocks.filter((x) => x.binId === b.id).length} in {b.color} bin
                    </span>
                  ))}
                </div>
              </div>
            )}
          </Card>

          <Card title="Live readout">
            <div className="mb-2 grid grid-cols-4 gap-2">
              {[0, 1, 2, 3].map((i) => (
                <Stat key={i} label={JOINT_LABELS[i].split(' ')[1]} value={<span style={{ color: JOINT_COLORS[i] }}>{fmt(deg(shownQ[i]), 0)}°</span>} />
              ))}
            </div>
            <div className="grid grid-cols-3 gap-2">
              <Stat label="Tool x" value={`${fmt(tool[0] * 100, 1)} cm`} />
              <Stat label="Tool y" value={`${fmt(tool[1] * 100, 1)} cm`} />
              <Stat label="Tool φ" value={`${fmt(wrapDeg(deg(pose.phiSide)), 1)}°`} />
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2">
              <Stat label="Wrist roll" value={<span style={{ color: ROLL_COLOR }}>{fmt(deg(shownRoll), 0)}°</span>} />
              <Stat label="Grip force" value={<span style={{ color: SENSOR_COLORS.grip }}>{fmt(gripForce, 1)} N</span>} />
              <Stat label="Wrist load" value={<span style={{ color: SENSOR_COLORS.fz }}>{fmt(wristLoad, 2)} N</span>} />
</div>
          </Card>

          <Card title="Arm setup">
            <div className="space-y-3">
              {['Upper arm L1', 'Forearm L2', 'Wrist + gripper L3'].map((label, i) => (
                <Slider key={label} label={label} value={lengths[i] * 100} min={i === 2 ? 8 : 15} max={i === 2 ? 20 : 45} unit=" cm" onChange={(v) => setLength(i, v / 100)} />
              ))}
              <p className="text-xs text-slate-400">Shorter links shrink the reach circle. Run the sort task again to see blocks the arm can no longer reach get skipped.</p>
            </div>
          </Card>
        </aside>
      </main>

      <footer className="mx-auto max-w-[1400px] px-4 pb-6 text-xs text-slate-500 sm:px-6">
        Python (FastAPI + NumPy) kinematics and motion planning · React + TypeScript front end
      </footer>
    </div>
  )
}

function dotClass(kind: string) {
  switch (kind) {
    case 'pick':
      return 'bg-amber-300'
    case 'place':
      return 'bg-emerald-400'
    case 'warn':
      return 'bg-amber-500'
    case 'fault':
      return 'bg-rose-500'
    case 'recover':
      return 'bg-cyan-300'
    case 'wait':
      return 'bg-amber-300'
    case 'sense':
      return 'bg-violet-400'
    case 'done':
      return 'bg-cyan-400'
    default:
      return 'bg-slate-500'
  }
}
