# Robot Simulator

An interactive 3D simulator of a humanoid mobile manipulator: a wheeled base, a lift column with a torso that rides up and down it, a camera head, and two 7-joint arms. Drive it around a warehouse floor, pose the arms joint by joint, or drag a target and let inverse kinematics move the hand there. Each joint shows its angle, speed and the torque its motor needs to hold the arm against gravity.

**Stack:** React + TypeScript + Tailwind, with three.js for the 3D view. Everything runs in the browser; there is no server.

## What you can do

| Panel | What it does |
| --- | --- |
| **Arms · Joints (FK)** | Move all 7 joints of either arm (or both, mirrored) with sliders inside their limits, open and close the gripper, or pick a preset pose (Ready, Rest, Reach, Carry, Arms up, T-pose). **Wave** waves the right arm. |
| **Arms · Reach (IK)** | Drag the arrows on the hand target, set it with sliders, or click a crate, barrel or the floor. The solver moves the arm so the fingertips reach it, optionally with the fingers pointing forward or down. It shows whether the target was reached, the remaining error, iterations and solve time. Out-of-reach targets get the closest pose, with a red dashed line to the target. |
| **Joints table** | Angle within limits, speed, and gravity holding torque against the motor's rating, live for the selected arm, plus the hand position. A **payload** slider puts weight in the hand. |
| **Drive** | WASD / arrow keys or the on-screen pad, or click the floor to send the robot there. Crates and barrels block it. The camera can follow the robot. |
| **Lift** | Shoulder height on the column (78 to 118 cm). |
| **Head** | Looks where the robot is going, at the IK target, or around when idle; or pan and tilt it by hand. |

## The robotics

* **Arm chain:** shoulder pitch, shoulder roll, upper-arm twist, elbow, forearm twist, wrist pitch, wrist yaw (7 joints), then a two-finger gripper. One table in `src/lib/kinematics.ts` gives each joint's position in the previous joint's frame, its axis, limits, motor speed and acceleration, rated torque and the mass of the link after it. The 3D model, forward kinematics, IK and torques are all built from that table. The left arm is the right arm mirrored.
* **Realistic joints:** every joint is a rotary actuator on its real axis: a finned motor housing, an aluminium output flange with bolts, an end cap, cables, clevis brackets at the elbow and wrist, and a status LED that turns amber at half its rated torque and red near the limit. Joints (and the lift and head) follow targets with a **trapezoidal velocity profile**: they accelerate, cruise at their top speed, then brake to stop exactly on target.
* **Forward kinematics:** chain the joint transforms `T = base · Π Trans(offsetᵢ) · Rot(axisᵢ, qᵢ) · Trans(tool)`.
* **Jacobian:** for a revolute joint, column `i` is `aᵢ × (p_tool − pᵢ)` for position and `aᵢ` for orientation, with `aᵢ` and `pᵢ` the joint's world axis and position.
* **Inverse kinematics (damped least squares):** `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`, where `e` is the position error (and, if requested, the error in the fingers' pointing direction). Seven joints is more than a hand position needs, so the spare freedom (null space) pulls the arm toward a natural rest pose: `dq += (I − J⁺J)·k·(q_rest − q)`. Joints pinned at a limit are frozen so the others take over, and if a solve gets stuck it retries from a few start poses. On random reachable targets it succeeds 100% of the time in about 0.3 ms.
* **Gravity torque:** each motor holds `τᵢ = −aᵢ · Σ (p_m − pᵢ) × m·g` over every link (and payload) further down the arm.
* **Base:** differential drive with acceleration limits; it turns toward a clicked goal, then drives, slowing on arrival.

## Run locally

Requires Node 20+.

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173.

## Deploy

`render.yaml` deploys the built front end as a static site on [Render](https://render.com) (**New → Blueprint**, pick this repo). Any static host works: `npm run build` writes the site to `frontend/dist`.

## Project layout

```
frontend/src/
  App.tsx                    page header + simulator
  components/Humanoid.tsx    3D scene, robot model, driving, arm control, panels
  components/ui.tsx          cards, sliders, buttons
  lib/kinematics.ts          joint table, FK, Jacobian, IK, gravity torques
  lib/math.ts                helpers and the trapezoidal motor profile
```
