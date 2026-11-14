import { ArrowRight, ExternalLink } from 'lucide-react'
import type { ReactNode } from 'react'
import { Card } from './ui'

const REPO = 'https://github.com/timHfleet/robot-simulation/tree/main/physics'
const FILE = (f: string) => `https://github.com/timHfleet/robot-simulation/blob/main/physics/${f}`
const media = (f: string) => `${import.meta.env.BASE_URL}physics/${f}`

/** From `python demo.py --record` (the run that made the video on this page). */
const RESULTS: { label: string; value: string; note: string }[] = [
  { label: 'Cup lifted', value: '11.6 cm', note: 'off the counter, held only by grip and friction' },
  { label: 'Slip in the hand', value: '1.6 mm', note: 'largest, while carried' },
  { label: 'Grip force', value: '40 N', note: 'the gripper motor’s limit' },
  { label: 'Cup placed', value: '0.6 cm', note: 'from the mat’s centre, upright' },
  { label: 'Dishwasher door', value: '91°', note: 'open, against its spring and damper' },
  { label: 'Unplanned contacts', value: 'None', note: 'robot never bumped the furniture' },
]

const COMPARE: [string, string, string][] = [
  ['Runs', 'Live in your browser (Rapier, WebAssembly, 120 Hz)', 'On your computer (Python), recorded here'],
  ['Items', 'Rigid bodies with mass and friction: they fall, stack, tip over and tumble in the drum', 'Same, for the cup'],
  ['Arm motion', 'Follows the controller exactly and shoves items it touches', 'Torque-limited motors push against mass and gravity'],
  ['Holding an object', 'Holds only if the item is really between the fingers; let go and it drops', 'Finger pads squeeze; friction holds it, or it slips'],
  ['The T-shirt', 'Simulated cloth (about 860 particles): pinched, lifted, draped and folded; layers grip each other', 'Not modelled'],
  ['Doors and racks', 'Move with the hand; solid, and carry what sits on them', 'Hinge with a spring and damper; the hand must push and pull'],
]

const LOOP: { title: string; text: string; tone: 'plan' | 'physics' }[] = [
  { title: 'Task script', text: 'Sets a target for the base, the lift and each hand', tone: 'plan' },
  { title: 'Inverse kinematics', text: 'Turns each hand target into 7 joint angles', tone: 'plan' },
  { title: 'Joint servos', text: 'Drive each joint, capped at its rated torque', tone: 'physics' },
  { title: 'MuJoCo step', text: '5 steps of 2 ms: gravity, contact, friction', tone: 'physics' },
]

const STILLS: { src: string; title: string; text: string }[] = [
  { src: '2_lifted.jpg', title: 'Grip', text: 'The fingers close with up to 40 N; nothing glues the cup to the hand.' },
  { src: '3_handle.jpg', title: 'Handle', text: 'The hand closes on the handle bar and pulls along the hinge arc.' },
  { src: '4_press_side.jpg', title: 'Press', text: 'Over the top edge, the hand presses the door down as the base backs away.' },
  { src: '6_head_cam.jpg', title: 'Head camera', text: 'What the robot sees at the end: cup on the mat, door open.' },
]

const CHECKS = [
  'Every model file compiles (MJCF and URDF)',
  'Joint limits and rated torques match the web app exactly',
  'Every moving part has mass and collision geometry',
  'Visual geometry never collides',
  'No self-contact in the rest and tuck poses',
  'Nothing overlaps at the start; the scene holds still',
]

function Code({ children }: { children: ReactNode }) {
  return <code className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[12px] text-slate-200">{children}</code>
}

export default function PhysicsPage() {
  return (
    <main className="mx-auto max-w-[1400px] space-y-4 px-4 py-4 sm:px-6">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card title="Physics simulation · MuJoCo">
          <video
            className="aspect-[16/10] w-full rounded-lg bg-slate-950 ring-1 ring-slate-800"
            src={media('demo.mp4')}
            poster={media('demo_poster.jpg')}
            autoPlay
            muted
            loop
            playsInline
            controls
          />
          <p className="mt-3 text-sm leading-relaxed text-slate-300">
            The Simulator tab now runs live physics: rigid bodies (Rapier) for every item, the robot and the appliances, and cloth for the
            T-shirt. This page shows the
            deeper version: the same robot, rebuilt in the <a className="text-cyan-300 hover:underline" href="https://mujoco.org" target="_blank" rel="noreferrer">MuJoCo</a> physics
            engine. It picks up a cup and opens a dishwasher with real mass, motor torque limits, contact and friction. The models are generated
            from this simulator’s own joint table, so both always describe the same arm. Recorded in real time from{' '}
            <a className="text-cyan-300 hover:underline" href={FILE('demo.py')} target="_blank" rel="noreferrer">
              demo.py
            </a>
            .
          </p>
        </Card>

        <Card title="What the run measured">
          <dl className="divide-y divide-slate-800">
            {RESULTS.map((r) => (
              <div key={r.label} className="flex items-baseline justify-between gap-4 py-2.5 first:pt-0">
                <div>
                  <dt className="text-sm text-slate-200">{r.label}</dt>
                  <dd className="text-xs text-slate-500">{r.note}</dd>
                </div>
                <dd className="whitespace-nowrap font-mono text-lg font-semibold text-white">{r.value}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 rounded-lg bg-emerald-400/10 px-3 py-2 text-xs text-emerald-200 ring-1 ring-emerald-400/20">
            Both tasks pass. The script exits with an error if any measure misses its threshold.
          </p>
        </Card>
      </div>

      <Card title="Physics in the simulator vs. MuJoCo">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wider text-slate-500">
                <th className="pb-2 pr-4 font-medium" />
                <th className="pb-2 pr-4 font-medium">Simulator tab · Rapier</th>
                <th className="pb-2 font-medium text-cyan-300">MuJoCo physics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {COMPARE.map(([k, a, b]) => (
                <tr key={k}>
                  <td className="py-2.5 pr-4 font-medium text-slate-200">{k}</td>
                  <td className="py-2.5 pr-4 text-slate-400">{a}</td>
                  <td className="py-2.5 text-slate-200">{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Control loop · every 10 ms">
        <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {LOOP.map((s, i) => (
            <li key={s.title} className="relative flex items-stretch">
              <div
                className={
                  'flex-1 rounded-lg px-3 py-2.5 ring-1 ' +
                  (s.tone === 'plan' ? 'bg-violet-400/10 ring-violet-400/25' : 'bg-orange-400/10 ring-orange-400/25')
                }
              >
                <div className={'text-sm font-semibold ' + (s.tone === 'plan' ? 'text-violet-200' : 'text-orange-200')}>{s.title}</div>
                <div className="mt-0.5 text-xs text-slate-400">{s.text}</div>
              </div>
              {i < LOOP.length - 1 && <ArrowRight className="mx-1 hidden h-4 w-4 shrink-0 self-center text-slate-600 lg:block" />}
            </li>
          ))}
        </ol>
        <p className="mt-3 text-xs text-slate-500">
          <span className="text-violet-300">Violet</span> is what the robot decides; <span className="text-orange-300">orange</span> is what physics
          allows. The new positions, contacts and forces feed the next tick.
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <Card title="Visual vs. collision geometry">
          <img className="w-full rounded-lg ring-1 ring-slate-800" src={media('visual_vs_collision.jpg')} alt="The robot drawn with its visual geometry on the left and its simplified collision geometry on the right" loading="lazy" />
          <p className="mt-3 text-sm leading-relaxed text-slate-300">
            Every part has two kinds of shapes. The detailed ones are only drawn. The physics engine uses simple capsules and boxes, which keep
            contact fast and stable. The finger pads are the exception: they use high-friction rubber with torsional friction, so a held cup can’t spin.
          </p>
        </Card>
        <Card title="Model checks · 15 of 15 pass">
          <ul className="space-y-2 text-sm text-slate-300">
            {CHECKS.map((c) => (
              <li key={c} className="flex gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
                {c}
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-slate-500">
            Run before anything uses a model, by{' '}
            <a className="text-cyan-300 hover:underline" href={FILE('check_models.py')} target="_blank" rel="noreferrer">
              check_models.py
            </a>
            .
          </p>
        </Card>
      </div>

      <Card title="Moments from the run">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {STILLS.map((s) => (
            <figure key={s.src}>
              <img className="aspect-[8/5] w-full rounded-lg object-cover ring-1 ring-slate-800" src={media(s.src)} alt={s.text} loading="lazy" />
              <figcaption className="mt-2 text-sm">
                <span className="font-medium text-slate-200">{s.title}.</span> <span className="text-slate-400">{s.text}</span>
              </figcaption>
            </figure>
          ))}
        </div>
      </Card>

      <Card
        title="Run it yourself"
        right={
          <a className="inline-flex items-center gap-1 text-xs text-cyan-300 hover:underline" href={REPO} target="_blank" rel="noreferrer">
            physics/ on GitHub <ExternalLink className="h-3 w-3" />
          </a>
        }
      >
        <div className="grid gap-4 md:grid-cols-2">
          <pre className="overflow-x-auto rounded-lg bg-slate-950 p-3 font-mono text-[12px] leading-relaxed text-slate-200 ring-1 ring-slate-800">
            {`cd physics
pip install -r requirements.txt
python build_models.py   # joint table -> MJCF + URDF
python check_models.py   # 15 model checks
python demo.py --view    # watch it live`}
          </pre>
          <ul className="space-y-2 text-sm text-slate-300">
            <li>
              <Code>models/robot.xml</Code> and <Code>models/scene.xml</Code>: MuJoCo (MJCF) models of the robot and the kitchen.
            </li>
            <li>
              <Code>models/robot.urdf</Code>: the same robot for Drake, Isaac Sim and ROS.
            </li>
            <li>
              Simplifications: the base slides instead of rolling on its wheels, the fingers are parallel sliders, and the arms cancel their own weight as a
              real arm controller does.
            </li>
          </ul>
        </div>
      </Card>
    </main>
  )
}
