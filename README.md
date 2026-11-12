# Robot Arm Simulator

An interactive simulator for a 4-joint robot arm (a rotating waist plus shoulder, elbow and wrist), shown in a 3D scene you can orbit. Drag a target and watch three different inverse kinematics solvers reach for it, drive the joints by hand, or have the arm plan and run a pick-and-place job that sorts colored blocks into matching bins.

**Stack:** Python (FastAPI + NumPy) for the kinematics and motion planning · React + TypeScript (TSX) + Tailwind front end, with three.js for the 3D view and an SVG side view.

## What it does

| Mode | What you can do |
| --- | --- |
| **Reach (IK)** | Click or drag anywhere. The Python API solves for the joint angles and reports whether the target was reached, the position error, the iteration count and the solve time. Switch between the three solvers, pick elbow up or down, and set the tool angle. |
| **Joints (FK)** | Move the shoulder, elbow and wrist with sliders inside their joint limits and see the tool pose update (forward kinematics). |
| **Pick & place** | Click a block, then click a bin or any free spot on the table. Commands queue up and run one after another. The planner rejects spots that are out of reach, on a bin, on the base or too close to another block, and says why. A dashed ring on the table shows how far the arm can reach. |
| **2 arms** | Two arms face each other across a shared work line and split the block sorting. Whichever arm is free takes the nearest block. A scheduler checks every frame of a new job against the other arm's committed motion (points along every link, in 3D) and delays the job until the arms stay at least 8.5 cm apart. Arms tuck in close to their base for big waist turns and park on their own side after each job. Shows per-arm utilization, waiting time and the speed-up over one arm. |
| **Fold shirt (2 arms)** | Both wrist cameras find the collar, sleeve tips and hem corners. Each arm folds its side and sleeve in toward the middle at the same time, then both pinch the hem corners and fold the bottom up in lockstep (identical timing every frame, so the hem stays straight), press the crease (felt by the wrist force sensors) and park. The T-shirt is a masked cloth grid (body, sleeves, neck opening) that bends with the grippers; cloth past a pinch point, like a sleeve, hangs and then lies flat. |
| **Load dishwasher** | Plates stand in a holder; the arm pinches each top rim, lifts it straight out and slides it into a bottom-rack slot. Cups are grabbed from the side with the gripper level, flipped mouth-down by a wrist turn (with the cup turning about its own centre), and set on the top rack. Bottom rack first. |
| **Fold towel** | Grab the far edge and swing it over in an arc to fold the towel in half, fold again across the first fold, then drop the folded towel in a laundry basket. The cloth is a 25x25 vertex grid that bends with the gripper. |
| **Sort task** | The planner sorts the blocks into the bin of their color, then the front end plays back the trajectory with a joint-angle chart you can scrub, an event log, and stats (cycle time, tool travel, peak joint speed, minimum clearance). Shuffle the blocks or shorten the links to see unreachable blocks get skipped. |

## The robotics

* **Wrist roll (5th joint):** the gripper can turn about its own axis. Blocks sit at random angles; the camera estimates each angle, the wrist rolls so the fingers close square on the block's faces (fine-tuned after the close-up look), and rolls again on the way to the bin so the block lands square. For cloth the wrist rolls so the fingers close straight across the edge being pinched: almost no roll for the shirt's side seams, about 90 degrees for the hem corners and the towel's far edge. The roll is a frame channel next to the 4 arm joints and turns held objects with it.

* **Robot behaviour:** before each pick the wrist camera scans the object (a detection box and confidence appear in 3D); the estimate is a few mm off, so a close-up look just above the object measures the error and the arm corrects sideways. It moves fast in open space but creeps the last 3 cm down and lifts off gently. The gripper closes fast, slows near the object and squeezes to a target force. A simulated wrist force sensor and gripper force sensor (with noise) show the squeeze, the carried weight, the touch-down spike when placing, and the grip force dropping to 0 N when something slips. After each placement a quick look confirms it landed.

* **Failures and recovery:** the demo data injects real-world gripper faults. A *slip* (the block slides out while lifting) is caught by the grip-force sensor; the block falls back and the arm re-grasps with more force. A *drop* (the block falls mid-move) triggers an emergency stop with a braking ramp; the camera relocates the block where it landed, the waist turns toward it and the arm picks it up again. Modes: none, a fixed "drop + slip" demo, or random (20% slip / 20% drop per pick). Every fault is shown in the event log, marked on the joint chart and counted in the stats.

* **Waist + planar arm:** the waist turns the arm's vertical plane about the base. To reach the other side of the table the arm turns around instead of flipping its elbow over the top, which keeps it low and predictable (tested: the elbow never changes sign during a run).
* **Motor model:** in Reach and Joints modes the arm doesn't snap to a new pose. Each joint has a top speed and acceleration, and a trapezoidal velocity profile brings it to a stop exactly on target.

* **Forward kinematics:** each joint angle is relative to the previous link. The tool position is the sum of the link vectors.
* **Analytic IK:** subtract the last link (along the requested tool angle) to get the wrist point, then solve the 2-link triangle with the law of cosines. There are two solutions (elbow up and elbow down). The solver keeps the one inside the joint limits, or the one closest to the current pose so paths stay smooth.
* **Jacobian IK (damped least squares):** `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`. Damping keeps it stable near full stretch and out of reach. Joints pinned at a limit are frozen so the others take over. Because it is a local method, it retries from a few start poses if it gets stuck: on random reachable poses that took the success rate from 57% to 96%.
* **CCD (cyclic coordinate descent):** rotate one joint at a time, wrist first, to point the tool at the target.
* **Motion planning:** big moves are joint-space moves with a quintic time profile (zero velocity and acceleration at both ends). Grasp and release are straight vertical Cartesian moves solved with IK every frame. Every waypoint is checked for reachability before the arm commits to a block. If a path still leaves the workspace, the partial motion is rolled back and the block is skipped.

## Run locally

Requires Python 3.11+ and Node 20+.

```bash
# API on http://127.0.0.1:8000
cd backend
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
# Front end on http://localhost:5173 (proxies /api to the Python server)
cd frontend
npm install
npm run dev
```

Tests (FK/IK round trips, a Jacobian check against finite differences, and planner checks for smooth paths, joint limits and correct bins):

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

## Deploy

The `Dockerfile` builds the front end and serves it from the Python server as a single web service. On [Render](https://render.com), choose **New → Blueprint** and pick this repo (`render.yaml` is included).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/config` | Arm geometry, joint limits, home pose, default scene |
| `POST /api/fk` | Joint angles → joint positions and tool angle |
| `POST /api/ik` | Target (+ tool angle, solver, elbow) → joint angles and solver stats |
| `POST /api/plan` | Blocks + bins (+ failure mode) → full sorting trajectory (60 fps frames), block timelines, events, stats |
| `POST /api/task` | Household task (`dishes` or `laundry`) → trajectory, object timelines and (for the towel) the fold lines for the cloth model |
| `POST /api/dual` | Two-arm block sorting → both arms' trajectories and sensors, merged events, per-arm stats |
| `POST /api/shirt` | Two arms folding a T-shirt → both arms' trajectories and sensors, fold lines for the cloth model |
| `POST /api/commands` | Pick-and-place commands (block → bin or table spot) → trajectory, with the same failure handling |
| `POST /api/scene` | Random block layout |

## Project layout

```
backend/app/
  kinematics.py   FK, Jacobian, analytic / DLS / CCD inverse kinematics
  planner.py      pick-and-place planning, failure recovery, motion primitives
  tasks.py        dishwasher loading and towel folding
  dual.py         two-arm scheduling with collision checks
  bimanual.py     two arms folding a shirt together (lockstep moves)
  main.py         FastAPI routes + static front end
backend/tests/    pytest suite
frontend/src/
  App.tsx                 modes, IK requests, playback
  components/ArmView.tsx  SVG workspace (arm, gripper, bins, blocks, target)
  components/JointChart.tsx  joint angles over time, scrubbable
  lib/                    API client, types, geometry helpers
```
