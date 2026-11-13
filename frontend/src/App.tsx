import { Github } from 'lucide-react'
import Humanoid from './components/Humanoid'

const REPO_URL = 'https://github.com/timHfleet/robot-simulation'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-950/80">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-cyan-400/10 ring-1 ring-cyan-400/30">
              <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden>
                <rect x="12" y="4" width="8" height="6" rx="2" fill="#22d3ee" />
                <rect x="14.5" y="10" width="3" height="14" fill="#f8fafc" />
                <path d="M9 14 L23 14 M9 14 L7 21 M23 14 L25 21" stroke="#f8fafc" strokeWidth="2.4" fill="none" strokeLinecap="round" />
                <rect x="8" y="24" width="16" height="5" rx="2" fill="#f8fafc" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-semibold leading-tight">Robot Simulator</h1>
              <p className="text-xs text-slate-400">Mobile manipulator · two 7-joint arms · forward and inverse kinematics</p>
            </div>
          </div>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700"
          >
            <Github className="h-4 w-4" /> Code
          </a>
        </div>
      </header>
      <Humanoid />
    </div>
  )
}
