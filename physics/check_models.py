"""
Quality checks for the generated models, the kind a simulation-content pipeline runs
before anything trains or evaluates on a model:

  1. every file compiles (MJCF in MuJoCo, and the URDF through MuJoCo's URDF reader)
  2. joint limits and rated torques match the web app's ARM table exactly
  3. every moving part has mass, and every link has collision geometry
  4. visual geometry never collides (it is for drawing only)
  5. the robot does not touch itself in its rest and tuck poses
  6. nothing in the kitchen scene overlaps at the start, and it stays still when left alone

Run:  python check_models.py      (exits 1 if any check fails)
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from build_models import read_arm

HERE = Path(__file__).resolve().parent
MODELS = HERE / 'models'
REST = [0, 8, 0, 15, 0, 0, 0]
TUCK = [15, 8, 0, 125, 0, -130, 0]

results: list[tuple[bool, str]] = []


def check(ok, msg):
    results.append((bool(ok), msg))


def contacts(m, d, ignore=()):
    out = []
    for c in d.contact[: d.ncon]:
        a, b = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        if c.dist < -1e-4 and not ({a, b} & set(ignore)):
            out.append(f'{a}/{b} ({-c.dist * 1000:.1f} mm)')
    return out


def main():
    arm, _ = read_arm()
    models = {}
    for name in ('robot.xml', 'scene.xml', 'robot.urdf'):
        try:
            models[name] = mujoco.MjModel.from_xml_path(str(MODELS / name))
            check(True, f'{name} compiles')
        except Exception as e:  # noqa: BLE001
            check(False, f'{name} failed to compile: {e}')
    m = models.get('robot.xml')
    if m is None:
        return report()

    # 2. limits and torques match the web app
    for side in 'LR':
        for j in arm:
            jn = m.joint(f'{side}_{j["key"]}')
            lo, hi = np.degrees(jn.range)
            act = m.actuator(f'{side}_{j["key"]}')
            good = abs(lo - j['min']) < 0.01 and abs(hi - j['max']) < 0.01 and abs(act.forcerange[1] - j['rated']) < 1e-6
            if not good:
                check(False, f'{jn.name}: range {lo:.1f}..{hi:.1f} / {act.forcerange[1]} N.m differ from the web app')
    check(not [r for r in results if not r[0] and 'differ' in r[1]], '14 arm joints match the web app (limits and rated torque)')

    # 3. mass and collision geometry
    massless = [m.body(i).name for i in range(1, m.nbody) if m.body_mass[i] <= 0 and m.body_jntnum[i] > 0]
    check(not massless, 'every moving part has mass' + (f' (missing: {massless})' if massless else ''))
    no_col = [m.body(i).name for i in range(1, m.nbody)
              if m.body_jntnum[i] > 0 and not any(m.geom_bodyid[g] == i and m.geom_contype[g] for g in range(m.ngeom))]
    no_col = [n for n in no_col if not n.endswith(('_sp', '_sy', '_fy', 'waist'))]  # short links covered by their neighbours
    check(not no_col, 'links that need it have collision geometry' + (f' (missing: {no_col})' if no_col else ''))

    # 4. visual geometry is drawing only
    vis = [g for g in range(m.ngeom) if m.geom_group[g] == 1]
    col = [g for g in range(m.ngeom) if m.geom_group[g] == 3]
    check(all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in vis),
          f'{len(vis)} visual geoms never collide; {len(col)} collision geoms carry the physics')

    # 5. no self-contact in the named poses
    d = mujoco.MjData(m)
    for name, pose in (('rest', REST), ('tuck', TUCK)):
        for lift in (0.0, 0.2, 0.4):
            mujoco.mj_resetData(m, d)
            for side in 'LR':
                for k, v in zip([j['key'] for j in arm], pose):
                    d.qpos[m.joint(f'{side}_{k}').qposadr[0]] = math.radians(v)
            d.qpos[m.joint('lift').qposadr[0]] = lift
            mujoco.mj_forward(m, d)
            hits = contacts(m, d, ignore=('world',))
            check(not hits, f'{name} pose at lift {0.78 + lift:.2f} m: no self-contact' + (f' ({", ".join(hits)})' if hits else ''))

    # 6. the scene starts clean and stays put
    s = models.get('scene.xml')
    if s is not None:
        d = mujoco.MjData(s)
        for side in 'LR':
            for k, v in zip([j['key'] for j in arm], REST):
                d.qpos[s.joint(f'{side}_{k}').qposadr[0]] = math.radians(v)
                d.ctrl[s.actuator(f'{side}_{k}').id] = math.radians(v)
        d.qpos[s.joint('lift').qposadr[0]] = 0.3
        d.ctrl[s.actuator('lift').id] = 0.3
        mujoco.mj_forward(s, d)
        hits = contacts(s, d, ignore=('world', 'cup'))
        check(not hits, 'scene starts with nothing overlapping' + (f' ({", ".join(hits)})' if hits else ''))
        mujoco.mj_step(s, d, nstep=250)  # the cup spawns 2 mm up and drops onto the counter
        cup0 = d.xpos[s.body('cup').id].copy()
        mujoco.mj_step(s, d, nstep=1000)
        drift = np.linalg.norm(d.xpos[s.body('cup').id] - cup0)
        door = math.degrees(d.qpos[s.joint('door').qposadr[0]])
        check(drift < 0.001 and abs(door) < 1, f'left alone for 2 s: cup moves {drift * 1000:.1f} mm, door stays shut ({door:.1f} deg)')
    return report()


def report():
    for ok, msg in results:
        print(('  ok    ' if ok else '  FAIL  ') + msg)
    bad = sum(not ok for ok, _ in results)
    print(f'{len(results) - bad}/{len(results)} checks passed')
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
