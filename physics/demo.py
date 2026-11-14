"""
Physics demo: the robot picks up a cup and opens a dishwasher in MuJoCo.

In the web simulator a grasped item is simply attached to the hand. Here nothing
is attached: the cup stays in the hand only because the finger pads squeeze it
and friction holds it, and the dishwasher door moves only because the hand
pushes and pulls on it against its hinge spring, damping and gravity.

Each control tick (every 10 ms):
  1. the task script gives a target for the base, lift and each hand,
  2. inverse kinematics (damped least squares on MuJoCo's Jacobian) turns each
     hand target into joint angles,
  3. those angles go to the joint motors (position servos with torque limits),
  4. MuJoCo steps the physics: motor torques, gravity, contacts and friction.

Usage:
  python demo.py              run headless and print a report
  python demo.py --record     also save a GIF and still images to media/
  python demo.py --view       watch it live in the MuJoCo viewer
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE / 'models' / 'scene.xml'
MEDIA = HERE / 'media'

ARM_KEYS = ['sp', 'sr', 'sy', 'el', 'fy', 'wp', 'wy']
REST_DEG = [0, 8, 0, 15, 0, 0, 0]  # the web app's hanging rest pose
TUCK_DEG = [15, 8, 0, 125, 0, -130, 0]  # forearm folded up in front of the chest, for driving
LIFT_MIN = 0.78
CTRL_DT = 0.01
GRIP_OPEN, GRIP_CLOSED = 0.0, 0.05
DEBUG = bool(__import__('os').environ.get('DEBUG'))


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m, self.d = model, data
        self.ik = mujoco.MjData(model)  # scratch copy for inverse kinematics
        j = lambda n: model.joint(n)
        self.qadr = {n: j(n).qposadr[0] for n in [model.joint(i).name for i in range(model.njnt)]}
        self.dadr = {n: j(n).dofadr[0] for n in self.qadr}
        self.arm = {s: [f'{s}_{k}' for k in ARM_KEYS] for s in 'LR'}
        self.arm_q = {s: np.array([self.qadr[n] for n in self.arm[s]]) for s in 'LR'}
        self.arm_v = {s: np.array([self.dadr[n] for n in self.arm[s]]) for s in 'LR'}
        self.lim = {s: model.jnt_range[[model.joint(n).id for n in self.arm[s]]] for s in 'LR'}
        self.act = {model.actuator(i).name: i for i in range(model.nu)}
        self.site = {s: model.site(f'{s}_tcp').id for s in 'LR'}
        self.wrist = {s: model.body(f'{s}_wy').id for s in 'LR'}
        # Commanded state.
        self.cmd = dict(base_x=0.0, base_y=0.0, base_yaw=0.0, waist=0.0, lift=0.3)
        self.q = {s: np.radians(REST_DEG) for s in 'LR'}
        self.grip = {'L': GRIP_OPEN, 'R': GRIP_OPEN}

    # -- state ---------------------------------------------------------------
    def tcp(self, side, d=None):
        d = d or self.d
        R = d.xmat[self.wrist[side]].reshape(3, 3)
        return d.site_xpos[self.site[side]].copy(), -R[:, 2].copy(), R[:, 1].copy()

    def lift_z(self):
        return LIFT_MIN + self.cmd['lift']

    # -- inverse kinematics --------------------------------------------------
    def solve(self, side, pos, direction=None, close=None, iters=60):
        """Joint angles that put the tool centre point at pos, fingers pointing along
        `direction` and closing along `close` (either sign: the gripper is symmetric)."""
        m, d = self.m, self.ik
        d.qpos[:] = self.d.qpos
        for n in ('base_x', 'base_y', 'base_yaw', 'waist', 'lift'):
            d.qpos[self.qadr[n]] = self.cmd[n]
        q = self.q[side].copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        cols = self.arm_v[side]
        lo, hi = self.lim[side][:, 0], self.lim[side][:, 1]
        for _ in range(iters):
            d.qpos[self.arm_q[side]] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p, fdir, fclose = self.tcp(side, d)
            err = [pos - p]
            rot = np.zeros(3)
            if direction is not None:
                rot += np.cross(fdir, direction)
            if close is not None:
                c = close if np.dot(close, fclose) >= 0 else -close
                rot += 0.7 * np.cross(fclose, c)
            err.append(0.35 * rot)
            e = np.concatenate(err)
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(rot) < 5e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site[side])
            J = np.vstack([jacp[:, cols], 0.35 * jacr[:, cols]])
            lam = 0.02
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            dq = np.clip(dq, -0.15, 0.15)
            q = np.clip(q + dq, lo + 1e-3, hi - 1e-3)
        return q

    # -- commands ------------------------------------------------------------
    def apply(self):
        c = self.d.ctrl
        for n in ('base_x', 'base_y', 'base_yaw', 'waist', 'lift'):
            c[self.act[n]] = self.cmd[n]
        for s in 'LR':
            for n, v in zip(self.arm[s], self.q[s]):
                c[self.act[n]] = v
            c[self.act[f'{s}_grip']] = self.grip[s]


class Demo:
    def __init__(self, record=False, viewer=None):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.r = Robot(self.m, self.d)
        self.viewer = viewer
        self.record = record
        self.frames: list[np.ndarray] = []
        self.stills: dict[str, np.ndarray] = {}
        self.log: list[str] = []
        self.stats = dict(robot_hits=set(), self_hits=set(), max_slip=0.0, peak_grip=0.0)
        if record:
            self.renderer = mujoco.Renderer(self.m, 400, 640)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        r = self.r
        for s in 'LR':
            self.d.qpos[r.arm_q[s]] = r.q[s]
        self.d.qpos[r.qadr['lift']] = r.cmd['lift']
        mujoco.mj_forward(self.m, self.d)
        self.robot_bodies = {i for i in range(self.m.nbody) if self._root_name(i) == 'base'}
        self.env_bodies = {self.m.body(n).id for n in ('counter', 'dishwasher', 'door')}
        self.held = None
        self.last_target = {}

    def _root_name(self, b):
        while self.m.body_parentid[b] != 0:
            b = self.m.body_parentid[b]
        return self.m.body(b).name

    def say(self, msg):
        self.log.append(f'{self.d.time:5.1f}s  {msg}')
        print(self.log[-1])

    # -- simulation loop ----------------------------------------------------------
    def step(self, seconds):
        n = max(1, round(seconds / CTRL_DT))
        sub = round(CTRL_DT / self.m.opt.timestep)
        for _ in range(n):
            self.r.apply()
            mujoco.mj_step(self.m, self.d, nstep=sub)
            self.watch()
            if self.record and len(self.frames) < self.d.time * 12.5:
                self.renderer.update_scene(self.d, camera='overview')
                self.frames.append(self.renderer.render())
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(CTRL_DT)

    def watch(self):
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self.robot_bodies and b2 in self.robot_bodies:
                key = f'{m.body(b1).name} touched {m.body(b2).name}'
                if key not in self.stats['self_hits'] and DEBUG:
                    print(f'        self-contact at {d.time:.2f}s: {key}')
                self.stats['self_hits'].add(key)
            for a, b in ((b1, b2), (b2, b1)):
                if a in self.robot_bodies and b in self.env_bodies:
                    name = m.body(a).name
                    if 'finger' not in name and not (b == m.body('door').id and name.endswith('_wy')):
                        key = f'{name} touched {m.body(b).name}'
                        if key not in self.stats['robot_hits'] and DEBUG:
                            print(f'        contact at {d.time:.2f}s: {key} at {np.round(c.pos, 3)}, '
                                  f'base x {d.qpos[self.r.qadr["base_x"]]:.3f}, door {math.degrees(d.qpos[self.r.qadr["door"]]):.0f} deg')
                        self.stats['robot_hits'].add(key)
        if self.held:
            side, offset = self.held
            p, _, _ = self.r.tcp(side)
            slip = np.linalg.norm(d.xpos[m.body('cup').id] - p - offset)
            self.stats['max_slip'] = max(self.stats['max_slip'], slip)
            self.stats['peak_grip'] = max(self.stats['peak_grip'], abs(d.actuator_force[self.r.act[f'{side}_grip']]))

    def move(self, seconds, base=None, lift=None, hands=None, grips=None):
        """Move smoothly over `seconds`. `hands` maps a side to a function s -> (pos, dir, close)
        or to a fixed target tuple; base, lift and grips go from their current values to the given ones."""
        r = self.r
        b0 = (r.cmd['base_x'], r.cmd['base_y'])
        l0 = r.cmd['lift']
        g0 = dict(r.grip)
        n = max(1, round(seconds / CTRL_DT))
        starts = {s: r.tcp(s) for s in (hands or {})}
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            if base is not None:
                r.cmd['base_x'] = b0[0] + (base[0] - b0[0]) * s
                r.cmd['base_y'] = b0[1] + (base[1] - b0[1]) * s
            if lift is not None:
                r.cmd['lift'] = l0 + (lift - LIFT_MIN - l0) * s
            for side, g in (grips or {}).items():
                r.grip[side] = g0[side] + (g - g0[side]) * s
            for side, h in (hands or {}).items():
                if callable(h):
                    pos, dr, cl = h(s)
                else:
                    p0, d0, _ = starts[side]
                    pos = p0 + (np.asarray(h[0]) - p0) * s
                    dr = h[1]
                    if dr is not None:
                        dr = d0 + (np.asarray(dr) - d0) * s
                        dr = dr / np.linalg.norm(dr)
                    cl = h[2]
                r.q[side] = r.solve(side, np.asarray(pos), dr, None if cl is None else np.asarray(cl, float))
                self.last_target[side] = np.asarray(pos)
            self.step(CTRL_DT)
        if DEBUG:
            for side in hands or {}:
                r.ik.qpos[r.arm_q[side]] = r.q[side]
                mujoco.mj_kinematics(self.m, r.ik)
                ik_p = r.tcp(side, r.ik)[0]
                act_p = r.tcp(side)[0]
                t = self.last_target[side]
                print(f'        {side}: IK miss {np.linalg.norm(ik_p - t) * 1000:5.1f} mm, tracking miss {np.linalg.norm(act_p - t) * 1000:5.1f} mm, '
                      f'q={np.degrees(r.q[side]).round(0)}')

    def snap(self, name, camera='overview', **opts):
        if not self.record:
            return
        rend = mujoco.Renderer(self.m, 600, 960)
        opt = mujoco.MjvOption()
        for g, on in opts.get('groups', {}).items():
            opt.geomgroup[g] = on
        if opts.get('contacts'):
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.m.vis.scale.forcewidth = 0.02
            self.m.vis.map.force = 0.01
        rend.update_scene(self.d, camera=camera, scene_option=opt)
        self.stills[name] = rend.render()
        rend.close()

    # -- tasks --------------------------------------------------------------
    def pick_cup(self):
        m, d, r = self.m, self.d, self.r
        cup = d.xpos[m.body('cup').id].copy()
        goal = d.site_xpos[m.site('cup_goal').id].copy()
        # Fingers point forward and 30 degrees down, closing sideways: the forearm then comes
        # in from above the counter instead of scraping its front edge.
        fwd = np.array([math.cos(math.radians(30)), 0, -math.sin(math.radians(30))])
        side_axis = np.array([0, 1.0, 0])
        grasp = cup + [0, 0, 0.015]  # just above the cup's middle
        self.say('Cup task: drive up to the counter, raise the hand above the counter edge')
        self.move(2.0, base=(0.1, 0.0), lift=1.15,
                  hands={'R': (np.array([0.32, cup[1], 1.02]), fwd, side_axis)}, grips={'R': GRIP_OPEN})
        self.move(1.2, hands={'R': (grasp - fwd * 0.12, fwd, side_axis)})
        self.snap('1_reach')
        self.say('Slide the open hand around the cup')
        self.move(1.2, hands={'R': (grasp, fwd, side_axis)})
        self.say('Close the fingers: grip force and friction, not glue, will hold it')
        self.move(0.8, grips={'R': GRIP_CLOSED})
        self.step(0.3)
        p, _, _ = r.tcp('R')
        self.held = ('R', d.xpos[m.body('cup').id] - p)
        self.say('Lift, carry to the mat and set it down')
        z0 = cup[2]
        self.move(1.2, hands={'R': (grasp + [0, 0, 0.12], fwd, side_axis)})
        lifted = d.xpos[m.body('cup').id][2] - z0
        self.snap('2_lifted')
        self.snap('2_lifted_collision', groups={1: 0, 3: 1})
        place = np.array([goal[0], goal[1], grasp[2]])
        self.move(2.0, hands={'R': (place + [0, 0, 0.12], fwd, side_axis)})
        self.move(1.2, hands={'R': (place + [0, 0, 0.004], fwd, side_axis)})
        self.held = None
        self.move(0.8, grips={'R': GRIP_OPEN})
        self.say('Back the hand out the way it came in, then lower the arm')
        self.move(1.0, hands={'R': (place - fwd * 0.14 + [0, 0, 0.04], fwd, side_axis)})
        self.move(1.0, hands={'R': (place - fwd * 0.14 + [0, 0, 0.16], fwd, side_axis)})
        self.move(1.0, hands={'R': (np.array([0.36, goal[1] - 0.12, 1.0]), None, None)})
        self.step(0.5)
        final = d.xpos[m.body('cup').id]
        self.result_cup = dict(lifted=lifted, error=float(np.linalg.norm(final[:2] - goal[:2])),
                               upright=float(d.xmat[m.body('cup').id].reshape(3, 3)[2, 2]))
        self.say(f'Cup placed {self.result_cup["error"] * 100:.1f} cm from the mat centre')
        self.move(2.0, hands={'R': self._rest_hand('R', TUCK_DEG)})
        self.step(0.5)  # let the arm finish settling before the base moves

    def _rest_hand(self, side, pose=REST_DEG):
        """Joint-space move to a named pose (rest or tuck), as a hand-target function."""
        r = self.r
        q0 = r.q[side].copy()
        rest = np.radians(pose)

        def f(s):
            r.q[side] = q0 + (rest - q0) * s
            r.ik.qpos[:] = self.d.qpos
            r.ik.qpos[r.arm_q[side]] = r.q[side]
            for n in ('base_x', 'base_y', 'base_yaw', 'waist', 'lift'):
                r.ik.qpos[r.qadr[n]] = r.cmd[n]
            mujoco.mj_kinematics(self.m, r.ik)
            return r.tcp(side, r.ik)
        return f

    def door_frame(self, theta):
        """Handle and top-edge positions and the door's directions at hinge angle theta."""
        m, d = self.m, self.d
        door = m.body('door')
        hinge = d.xpos[m.body('dishwasher').id] + door.pos
        up = np.array([-math.sin(theta), 0, math.cos(theta)])   # along the door, hinge to top
        out = np.array([-math.cos(theta), 0, -math.sin(theta)])  # out of the front face
        handle_local = m.site('handle').pos
        handle = hinge + up * handle_local[2] + out * -handle_local[0]
        top = hinge + up * m.geom('door').size[2] * 2 + out * m.geom('door').size[0]  # middle of the top edge
        inner = -out
        return handle, top, up, out, inner

    def open_dishwasher(self):
        m, d, r = self.m, self.d, self.r
        door_q = r.qadr['door']
        self.say('Dishwasher task: drive over to the dishwasher')
        self.move(1.5, hands={'L': self._rest_hand('L', TUCK_DEG)})
        self.move(2.5, base=(0.1, 0.24))
        self.move(2.0, base=(0.56, 0.24), lift=0.98)
        handle, _, up, out, _ = self.door_frame(0.0)
        self.say('Reach the handle, fingers pointing into the door, closing top-to-bottom')
        self.move(1.6, hands={'L': (handle + out * 0.1, -out, up)}, grips={'L': 0.0})
        self.move(1.0, hands={'L': (handle + out * 0.0, -out, up)})
        self.move(0.6, grips={'L': GRIP_CLOSED})
        self.snap('3_handle')
        self.say('Pull: the hand follows the handle around the hinge, against the spring')
        th_end = 0.62

        def pull(s):
            th = th_end * s
            h, _, u, o, _ = self.door_frame(th)
            return h, -o, u
        self.move(2.5, base=(0.42, 0.24), hands={'L': pull})
        self.say(f'Door at {math.degrees(d.qpos[door_q]):.0f} degrees; let go of the handle')
        self.move(0.5, grips={'L': GRIP_OPEN})
        h, top, u, o, inner = self.door_frame(d.qpos[door_q])
        self.move(1.0, hands={'L': (h + o * 0.1, -o, u)})
        self.say('Hand up and over the top edge, then press down on it')
        th0 = float(d.qpos[door_q])
        _, top, u, o, inner = self.door_frame(th0)
        # Press 10 cm left of the door's centre, in front of the left shoulder: the left arm
        # cannot reach far across the body (its shoulder roll stops at -15 degrees).
        off = np.array([0, 0.1, 0])
        top = top + off
        down = np.array([0, 0, -1.0])
        self.move(1.2, base=(0.2, 0.24), hands={'L': (top + u * 0.12 + o * 0.06, down, None)}, grips={'L': GRIP_CLOSED})
        self.move(0.8, hands={'L': (top + u * 0.004, down, None)})
        th_push = 1.1

        def press(s):
            # Keep the fingertips on the edge as it swings down; the servo's stiffness
            # turns the 8 mm the target sits inside the edge into a steady push.
            th = th0 + (th_push - th0) * s
            _, tp, uu, _, _ = self.door_frame(th)
            return tp + off - uu * 0.008, down, None
        self.move(3.0, base=(0.16, 0.24), lift=0.88, hands={'L': press})
        self.snap('4_press')
        self.snap('4_press_side', camera='dishwasher')
        self.say('Let go and back off while the door settles open under its own weight')
        p_now = r.tcp('L')[0]
        self.move(1.0, base=(0.06, 0.24), hands={'L': (p_now + [-0.1, 0, 0.18], down, None)})
        self.move(1.6, hands={'L': self._rest_hand('L', TUCK_DEG)})
        self.step(1.5)
        self.result_door = math.degrees(d.qpos[door_q])
        self.say(f'Door open at {self.result_door:.0f} degrees')
        self.snap('5_done')
        self.snap('5_done_collision', groups={1: 0, 3: 1})
        self.snap('6_head_cam', camera='head_cam')

    def run(self):
        self.say('Settle')
        self.snap('0_start')
        self.step(0.5)
        self.pick_cup()
        self.open_dishwasher()
        return self.report()

    def report(self):
        c = self.result_cup
        lines = [
            '',
            'Report',
            f'  cup lifted off the counter : {c["lifted"] * 100:.1f} cm',
            f'  cup slip in the hand       : {self.stats["max_slip"] * 1000:.1f} mm (largest, while carried)',
            f'  peak grip force            : {self.stats["peak_grip"]:.1f} N',
            f'  cup placement error        : {c["error"] * 100:.1f} cm (upright: {"yes" if c["upright"] > 0.95 else "no"})',
            f'  dishwasher door            : {self.result_door:.0f} degrees open',
            f'  unplanned robot contacts   : {len(self.stats["robot_hits"]) or "none"}',
        ]
        lines += [f'    - {h}' for h in sorted(self.stats['robot_hits'])]
        ok = c['lifted'] > 0.08 and c['error'] < 0.05 and c['upright'] > 0.95 and self.result_door > 70 and not self.stats['robot_hits']
        lines.append(f'  result                     : {"PASS" if ok else "CHECK"}')
        print('\n'.join(lines))
        return ok


def save_media(demo: Demo):
    from PIL import Image
    MEDIA.mkdir(exist_ok=True)
    for name, img in demo.stills.items():
        Image.fromarray(img).save(MEDIA / f'{name}.png')
    frames = [Image.fromarray(f).convert('P', palette=Image.Palette.ADAPTIVE, colors=128) for f in demo.frames[::2]]
    if frames:
        frames[0].save(MEDIA / 'demo.gif', save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=True)
    print(f'saved {len(demo.stills)} stills and a {len(frames)}-frame GIF to {MEDIA.relative_to(HERE.parent)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', action='store_true', help='save a GIF and stills to media/')
    ap.add_argument('--view', action='store_true', help='watch in the MuJoCo viewer')
    a = ap.parse_args()
    if a.view:
        import mujoco.viewer
        demo = Demo()
        with mujoco.viewer.launch_passive(demo.m, demo.d) as v:
            demo.viewer = v
            demo.run()
            while v.is_running():
                time.sleep(0.1)
        return
    demo = Demo(record=a.record)
    ok = demo.run()
    if a.record:
        save_media(demo)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
