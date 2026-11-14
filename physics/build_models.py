"""
Build the robot's physics models from the web app's own kinematics table.

The web simulator describes each arm as one table (ARM in frontend/src/lib/kinematics.ts):
where every joint sits, which way it turns, its limits, motor speed, rated torque and
the mass it carries. This script reads that table and writes:

  models/robot.xml   MJCF (MuJoCo) model of the robot
  models/scene.xml   MJCF kitchen scene: the robot, a counter, a cup and a dishwasher
  models/robot.urdf  URDF of the same robot (Drake, Isaac Sim and ROS read URDF)

Frames: the web app is y-up (x forward, z to the robot's right). MuJoCo, URDF and
Drake are z-up (x forward, y to the robot's left), so a web-app point (x, y, z)
becomes (x, -z, y). That mapping is a proper rotation, so joint axes convert the
same way.

Every part gets two kinds of geometry:
  visual    shapes that are drawn but never touch anything (group 1)
  collision simple capsules and boxes the physics engine uses (group 3, hidden by
            default; press 3 in the MuJoCo viewer to show them)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIN = HERE.parent / 'frontend' / 'src' / 'lib' / 'kinematics.ts'
OUT = HERE / 'models'

# Body dimensions from frontend/src/components/Humanoid.tsx (web-app units, metres).
BASE = dict(len=0.62, w=0.54, h=0.24, y0=0.06)
BASE_TOP = BASE['y0'] + BASE['h'] + 0.02
WHEEL_R, WHEEL_Z = 0.075, 0.21
COLUMN = dict(x=-0.07, d=0.11, w=0.15, top=1.34)
LIFT_MIN, LIFT_MAX = 0.78, 1.18
WAIST_X = -0.07
WAIST_LIM = 100  # degrees

# Gripper: two parallel fingers (the web app draws pivoting fingers; a parallel
# gripper is simpler to simulate and grips the same way).
FINGER_OPEN = 0.055  # finger centre from the palm centre when fully open (m)
FINGER_TRAVEL = 0.05
PAD = dict(x=0.013, y=0.004, z=0.03)  # half sizes
GRIP_FORCE = 40.0  # N, maximum squeeze


# ---------------------------------------------------------------------------
# Read the ARM table
# ---------------------------------------------------------------------------
def read_arm():
    src = KIN.read_text(encoding='utf-8')
    consts = {k: float(re.search(rf'export const {k} = ([\d.]+)', src).group(1)) for k in ('UPPER', 'FORE')}
    tool = [float(v) for v in re.search(r'export const TOOL: V3 = \[([^\]]+)\]', src).group(1).split(',')]
    row = re.compile(
        r"\{ key: '(\w+)', label: '([^']+)', offset: \[([^\]]+)\], axis: \[([^\]]+)\], "
        r"min: (-?[\d.]+), max: (-?[\d.]+), vmax: ([\d.]+), amax: ([\d.]+), rated: ([\d.]+), "
        r"mass: \{ m: ([\d.]+), at: \[([^\]]+)\] \}"
    )

    def vec(text):
        # Entries are numbers or simple expressions of UPPER and FORE, e.g. -(UPPER - 0.075).
        return [float(eval(p, {'__builtins__': {}}, consts)) for p in text.split(',')]

    joints = []
    for m in row.finditer(src):
        key, label, off, axis, lo, hi, vmax, amax, rated, mass, at = m.groups()
        joints.append(dict(key=key, label=label, offset=vec(off), axis=vec(axis), min=float(lo), max=float(hi),
                           vmax=float(vmax), amax=float(amax), rated=float(rated), mass=float(mass), at=vec(at)))
    if len(joints) != 7:
        raise SystemExit(f'expected 7 joints in {KIN}, found {len(joints)}')
    return joints, tool


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def to_z_up(v):
    """Web app (x fwd, y up, z right) -> z-up (x fwd, y left, z up)."""
    x, y, z = v
    return [x, -z, y]


def mirror_point(side, v):
    return v if side == 'R' else [v[0], v[1], -v[2]]


def mirror_axis(side, v):
    # An axis is a pseudovector: mirroring z flips its x and y parts.
    return v if side == 'R' else [-v[0], -v[1], v[2]]


def f(*vals):
    return ' '.join(f'{v:.4g}' if abs(v) > 1e-12 else '0' for v in vals)


def norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-12 else v


# Link radii by joint (collision capsules). Index i = the link after joint i.
LINK_R = [0.05, 0.05, 0.045, 0.042, 0.038, 0.032, 0.03]
MOTOR_R = [0.05, 0.048, 0.042, 0.044, 0.036, 0.03, 0.026]
KP = [500, 500, 250, 350, 90, 70, 45]
DAMP = [12, 12, 6, 8, 2.5, 2, 1.2]


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------
def mjcf_arm(side, arm, tool, indent):
    pad = ' ' * indent
    lines = []
    depth = 0
    for i, j in enumerate(arm):
        pos = to_z_up(mirror_point(side, j['offset']))
        axis = to_z_up(mirror_axis(side, j['axis']))
        nxt = to_z_up(mirror_point(side, arm[i + 1]['offset'])) if i + 1 < len(arm) else to_z_up(tool)
        at = to_z_up(mirror_point(side, j['at']))
        name = f'{side}_{j["key"]}'
        p = pad + '  ' * depth
        I = 0.4 * j['mass'] * 0.05 ** 2
        lines += [
            f'{p}<body name="{name}" pos="{f(*pos)}" gravcomp="1">',
            f'{p}  <joint name="{name}" axis="{f(*axis)}" range="{math.radians(j["min"]):.4f} {math.radians(j["max"]):.4f}" '
            f'damping="{DAMP[i]}" armature="0.02" actuatorfrcrange="{-j["rated"]} {j["rated"]}"/>',
            f'{p}  <inertial pos="{f(*at)}" mass="{j["mass"]}" diaginertia="{f(I, I, I)}"/>',
            # Visual: a motor housing on the joint axis and a link toward the next joint.
            f'{p}  <geom class="visual" type="cylinder" size="{MOTOR_R[i]} {MOTOR_R[i] * 0.75:.3f}" '
            f'zaxis="{f(*axis)}" material="{"dark" if i % 2 else "white"}"/>',
        ]
        if i < 6:
            seg = nxt if i + 1 < len(arm) else nxt
            if math.dist(seg, [0, 0, 0]) > 0.02:
                lines += [
                    f'{p}  <geom class="visual" type="capsule" fromto="0 0 0 {f(*seg)}" size="{LINK_R[i] * 0.9:.3f}" material="white"/>',
                    f'{p}  <geom class="collision" type="capsule" fromto="0 0 0 {f(*seg)}" size="{LINK_R[i]}"/>',
                ]
        else:
            lines += mjcf_gripper(side, p + '  ', tool)
        depth += 1
    for d in range(depth - 1, -1, -1):
        lines.append(pad + '  ' * d + '</body>')
    return lines


def mjcf_gripper(side, p, tool):
    tcp = to_z_up(tool)
    palm_h = 0.035
    lines = [
        f'{p}<geom class="visual" type="cylinder" fromto="0 0 -0.02 0 0 -0.04" size="0.032" material="white"/>',
        f'{p}<geom class="visual" type="box" pos="0 0 -0.065" size="0.028 {FINGER_OPEN + 0.012} {palm_h / 2 + 0.005}" material="dark"/>',
        f'{p}<geom class="collision" type="box" pos="0 0 -0.06" size="0.03 {FINGER_OPEN + 0.014} {palm_h / 2 + 0.01}"/>',
        f'{p}<site name="{side}_tcp" pos="{f(*tcp)}" size="0.008" rgba="1 .75 .1 1"/>',
    ]
    for k, s in (('a', 1), ('b', -1)):
        lines += [
            f'{p}<body name="{side}_finger_{k}" pos="0 {s * FINGER_OPEN} -0.11" gravcomp="1">',
            f'{p}  <joint name="{side}_finger_{k}" type="slide" axis="0 {-s} 0" range="0 {FINGER_TRAVEL}" damping="4" armature="0.01"/>',
            f'{p}  <inertial pos="0 0 0" mass="0.05" diaginertia="2e-5 2e-5 2e-5"/>',
            f'{p}  <geom class="visual" type="box" pos="0 {s * 0.003} 0" size="{PAD["x"]} 0.006 {PAD["z"]}" material="dark"/>',
            f'{p}  <geom class="pad" type="box" pos="0 {-s * 0.004} -0.004" size="{PAD["x"]} {PAD["y"]} {PAD["z"] - 0.004}"/>',
            f'{p}</body>',
        ]
    return lines


def mjcf_robot(arm, tool, mobile=True):
    """The robot as an MJCF <body> tree (for including in a scene) plus its actuators and contact rules."""
    col_h = COLUMN['top'] - BASE_TOP
    b = BASE
    body = ['    <body name="base" pos="0 0 0">']
    if mobile:
        # The wheels are drawn, but the base moves as a planar body (x, y, heading):
        # the usual way mobile manipulators are simulated when wheel slip is not the point.
        body += [
            '      <joint name="base_x" type="slide" axis="1 0 0" damping="200"/>',
            '      <joint name="base_y" type="slide" axis="0 1 0" damping="200"/>',
            '      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="100"/>',
        ]
    body += [
        f'      <inertial pos="0 0 0.15" mass="45" diaginertia="2.2 2.6 3.4"/>',
        f'      <geom class="visual" type="box" pos="0 0 {b["y0"] + b["h"] / 2}" size="{b["len"] / 2} {b["w"] / 2} {b["h"] / 2}" material="white"/>',
        f'      <geom class="visual" type="box" pos="0 0 {BASE_TOP - 0.01}" size="{b["len"] / 2 - 0.02} {b["w"] / 2 - 0.02} 0.01" material="dark"/>',
        f'      <geom class="collision" type="box" pos="0 0 {(b["y0"] + BASE_TOP) / 2}" size="{b["len"] / 2} {b["w"] / 2} {(BASE_TOP - b["y0"]) / 2}"/>',
    ]
    for s in (1, -1):
        body.append(f'      <geom class="visual" type="cylinder" pos="0 {s * WHEEL_Z} {WHEEL_R}" zaxis="0 1 0" size="{WHEEL_R} 0.0225" material="rubber"/>')
    body += [
        f'      <body name="waist" pos="{WAIST_X} 0 {BASE_TOP}">',
        f'        <joint name="waist" axis="0 0 1" range="{-math.radians(WAIST_LIM):.4f} {math.radians(WAIST_LIM):.4f}" damping="20" armature="0.2"/>',
        f'        <inertial pos="0 0 {col_h / 2}" mass="8" diaginertia="0.7 0.7 0.05"/>',
        f'        <geom class="visual" type="cylinder" pos="0 0 0.03" size="0.13 0.03" material="dark"/>',
        f'        <geom class="visual" type="box" pos="{COLUMN["x"] - WAIST_X} 0 {col_h / 2}" size="{COLUMN["d"] / 2} {COLUMN["w"] / 2} {col_h / 2}" material="white"/>',
        f'        <geom class="collision" type="box" pos="{COLUMN["x"] - WAIST_X} 0 {col_h / 2}" size="{COLUMN["d"] / 2} {COLUMN["w"] / 2} {col_h / 2}"/>',
        # The carriage frame is the web app's shoulder base: (0, lift, 0) in robot coordinates.
        f'        <body name="carriage" pos="{-WAIST_X} 0 {LIFT_MIN - BASE_TOP}" gravcomp="1">',
        f'          <joint name="lift" type="slide" axis="0 0 1" range="0 {LIFT_MAX - LIFT_MIN:.3f}" damping="400" armature="2"/>',
        '          <inertial pos="-0.04 0 0" mass="6" diaginertia="0.15 0.08 0.12"/>',
        '          <geom class="visual" type="box" pos="-0.04 0 -0.04" size="0.075 0.15 0.11" material="white"/>',
        '          <geom class="visual" type="box" pos="0.036 0 -0.02" size="0.002 0.09 0.06" material="dark"/>',
        '          <geom class="collision" type="box" pos="-0.04 0 -0.04" size="0.075 0.15 0.11"/>',
        '          <body name="head" pos="-0.03 0 0.2">',
        '            <geom class="visual" type="cylinder" pos="0 0 -0.06" size="0.025 0.04" material="dark"/>',
        '            <geom class="visual" type="box" pos="0.01 0 0.03" size="0.07 0.11 0.06" material="white"/>',
        '            <geom class="visual" type="box" pos="0.081 0 0.035" size="0.002 0.09 0.035" material="screen"/>',
        '            <geom class="collision" type="box" pos="0.01 0 0.03" size="0.07 0.11 0.06"/>',
        # Head camera, looking forward and 40 degrees down.
        f'            <camera name="head_cam" pos="0.085 0 0.035" xyaxes="0 -1 0 {math.sin(math.radians(40)):.4f} 0 {math.cos(math.radians(40)):.4f}" fovy="70"/>',
        '          </body>',
    ]
    for side in ('R', 'L'):
        body += mjcf_arm(side, arm, tool, 10)
    body += ['        </body>', '      </body>', '    </body>']

    act = []
    if mobile:
        act += [
            '    <position name="base_x" joint="base_x" kp="4000" forcerange="-600 600"/>',
            '    <position name="base_y" joint="base_y" kp="4000" forcerange="-600 600"/>',
            '    <position name="base_yaw" joint="base_yaw" kp="2000" forcerange="-300 300"/>',
        ]
    act += [
        '    <position name="waist" joint="waist" kp="800" forcerange="-150 150"/>',
        '    <position name="lift" joint="lift" kp="20000" forcerange="-2000 2000"/>',
    ]
    for side in ('R', 'L'):
        for i, j in enumerate(arm):
            n = f'{side}_{j["key"]}'
            act.append(f'    <position name="{n}" joint="{n}" kp="{KP[i]}" forcerange="{-j["rated"]} {j["rated"]}"/>')
        # One motor drives both fingers (the equality below couples them).
        act.append(f'    <position name="{side}_grip" joint="{side}_finger_a" kp="1500" ctrlrange="0 {FINGER_TRAVEL}" forcerange="{-GRIP_FORCE} {GRIP_FORCE}"/>')

    eq = [f'    <joint joint1="{s}_finger_b" joint2="{s}_finger_a" polycoef="0 1 0 0 0"/>' for s in ('R', 'L')]
    # Links next to each other along the chain overlap at their joints by design.
    # Links up to three joints apart on the same arm, the shoulder links and the torso, and
    # the wrist and fingers, overlap around their joints by design and never meet otherwise.
    exclude = []
    keys = [j['key'] for j in arm]
    for side in ('R', 'L'):
        exclude += [f'    <exclude body1="carriage" body2="{side}_{k}"/>' for k in ('sr', 'sy')]
        for i in range(len(keys)):
            for k in range(i + 2, min(i + 4, len(keys))):
                exclude.append(f'    <exclude body1="{side}_{keys[i]}" body2="{side}_{keys[k]}"/>')
        exclude += [f'    <exclude body1="{side}_{w}" body2="{side}_finger_{k}"/>' for w in ('fy', 'wp') for k in ('a', 'b')]
    return body, act, eq, exclude


ASSETS = """  <asset>
    <material name="white" rgba=".93 .94 .96 1" specular=".4" shininess=".5"/>
    <material name="dark" rgba=".16 .17 .2 1" specular=".3"/>
    <material name="rubber" rgba=".08 .08 .09 1"/>
    <material name="screen" rgba=".05 .3 .45 1" emission=".4"/>
    <material name="steel" rgba=".72 .74 .77 1" specular=".8" shininess=".8" reflectance=".1"/>
    <material name="counter" rgba=".55 .55 .53 1" specular=".5"/>
    <material name="cabinet" rgba=".87 .84 .78 1"/>
    <material name="cup" rgba=".75 .25 .2 1" specular=".6"/>
    <material name="plate" rgba=".97 .97 .95 1" specular=".6"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".86 .84 .8" rgb2=".8 .78 .74" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance=".05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".95 .96 .98" rgb2=".75 .78 .82" width="256" height="256"/>
  </asset>
"""

DEFAULTS = """  <default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" density="0"/>
    </default>
    <default class="collision">
      <geom group="3" rgba=".2 .8 .3 .5" density="0" friction="0.8 0.01 0.001"/>
    </default>
    <default class="pad">
      <!-- Rubber finger pads: high friction, 4-D contact (sliding plus torsion) so a held object does not spin. -->
      <geom group="3" rgba=".1 .1 .1 1" density="0" condim="4" friction="1.6 0.02 0.002" solref="0.004 1"/>
    </default>
  </default>
"""


def write_robot_xml(arm, tool):
    body, act, eq, exclude = mjcf_robot(arm, tool, mobile=False)
    xml = '\n'.join([
        '<!-- Generated by physics/build_models.py from frontend/src/lib/kinematics.ts. Do not edit by hand. -->',
        '<mujoco model="robot-simulator">',
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option timestep="0.002" integrator="implicitfast"/>',
        DEFAULTS + ASSETS + '  <worldbody>',
        '    <light pos="0 0 3" dir="0 0 -1" diffuse=".9 .9 .9"/>',
        '    <geom name="floor" type="plane" size="0 0 .05" material="floor"/>',
        *body,
        '  </worldbody>',
        '  <equality>', *eq, '  </equality>',
        '  <contact>', *exclude, '  </contact>',
        '  <actuator>', *act, '  </actuator>',
        '</mujoco>', '',
    ])
    (OUT / 'robot.xml').write_text(xml, encoding='utf-8')


# ---------------------------------------------------------------------------
# Kitchen scene
# ---------------------------------------------------------------------------
# Counter on the robot's right, dishwasher ahead on the left (world frame, z up).
COUNTER = dict(x0=0.48, x1=1.08, y0=-0.85, y1=-0.15, h=0.86)
DW = dict(hinge_x=1.12, y0=0.05, y1=0.65, hinge_z=0.12, door_h=0.62, door_t=0.04, tub_d=0.6)
CUP = dict(x=0.62, y=-0.45, r=0.036, h=0.1, mass=0.3)
CUP_GOAL = (0.62, -0.23)


def scene_world():
    c, d, cup = COUNTER, DW, CUP
    cx, cy = (c['x0'] + c['x1']) / 2, (c['y0'] + c['y1']) / 2
    hx, hy = (c['x1'] - c['x0']) / 2, (c['y1'] - c['y0']) / 2
    dy = (d['y0'] + d['y1']) / 2
    hw = (d['y1'] - d['y0']) / 2
    top = d['hinge_z'] + d['door_h']
    return [
        # Counter: a cabinet with a stone top. Static, so one geom both draws and collides.
        f'    <body name="counter" pos="{cx} {cy} 0">',
        f'      <geom type="box" pos="0 0 {(c["h"] - 0.03) / 2}" size="{hx - 0.02} {hy} {(c["h"] - 0.03) / 2}" material="cabinet"/>',
        f'      <geom name="countertop" type="box" pos="0 0 {c["h"] - 0.015}" size="{hx + 0.02} {hy + 0.01} 0.015" material="counter"/>',
        '    </body>',
        # Target mat for the cup.
        f'    <site name="cup_goal" pos="{CUP_GOAL[0]} {CUP_GOAL[1]} {c["h"] + 0.001}" type="cylinder" size="0.05 0.001" rgba=".2 .7 .4 .6"/>',
        # The cup: a free body with real mass and friction. Visual: body, rim and handle. Collision: one cylinder.
        f'    <body name="cup" pos="{cup["x"]} {cup["y"]} {c["h"] + cup["h"] / 2 + 0.002}">',
        '      <freejoint name="cup"/>',
        f'      <inertial pos="0 0 -0.01" mass="{cup["mass"]}" diaginertia="2.2e-4 2.2e-4 1.9e-4"/>',
        f'      <geom class="visual" type="cylinder" size="{cup["r"]} {cup["h"] / 2}" material="cup"/>',
        f'      <geom class="visual" type="cylinder" pos="0 0 {cup["h"] / 2}" size="{cup["r"] - 0.004} 0.001" material="dark"/>',
        f'      <geom class="visual" type="capsule" fromto="0 {cup["r"]} 0.025 0 {cup["r"] + 0.022} 0" size="0.006" material="cup"/>',
        f'      <geom class="visual" type="capsule" fromto="0 {cup["r"] + 0.022} 0 0 {cup["r"]} -0.025" size="0.006" material="cup"/>',
        f'      <geom name="cup" class="collision" type="cylinder" size="{cup["r"]} {cup["h"] / 2}" condim="4" friction="0.9 0.02 0.002"/>',
        '    </body>',
        # Dishwasher: a fixed tub with a door on a horizontal hinge at the bottom front.
        f'    <body name="dishwasher" pos="{d["hinge_x"]} {dy} 0">',
        f'      <geom type="box" pos="{d["tub_d"] / 2 + 0.01} 0 {top / 2}" size="{d["tub_d"] / 2} {hw + 0.02} {top / 2}" material="steel"/>',
        f'      <geom type="box" pos="{d["tub_d"] / 2 + 0.01} 0 {top + 0.015}" size="{d["tub_d"] / 2 + 0.02} {hw + 0.04} 0.015" material="counter"/>',
        # The open tub: a dark inset with two wire racks, seen once the door is down.
        f'      <geom class="visual" type="box" pos="0.012 0 {d["hinge_z"] + d["door_h"] / 2}" size="0.003 {hw - 0.03} {d["door_h"] / 2 - 0.03}" material="dark"/>',
        *[f'      <geom class="visual" type="box" pos="0.0155 0 {d["hinge_z"] + z}" size="0.002 {hw - 0.05} 0.006" material="steel"/>' for z in (0.14, 0.4)],
        f'      <body name="door" pos="0 0 {d["hinge_z"]}">',
        # Hinge angle 0 = closed (vertical); positive swings the top toward the robot.
        # The spring is the counterbalance real dishwasher doors have: it holds the door shut
        # when nearly closed, and past about 30 degrees gravity wins and the door settles open.
        '        <joint name="door" axis="0 -1 0" range="0 1.6" damping="8" stiffness="4" springref="-0.6" frictionloss="0.3"/>',
        f'        <inertial pos="-0.02 0 {d["door_h"] / 2}" mass="3" diaginertia="0.1 0.08 0.03"/>',
        f'        <geom name="door" type="box" pos="{-d["door_t"] / 2} 0 {d["door_h"] / 2}" size="{d["door_t"] / 2} {hw} {d["door_h"] / 2}" material="steel"/>',
        f'        <geom class="visual" type="capsule" fromto="-0.05 {-hw * 0.6} {d["door_h"] - 0.06} -0.05 {hw * 0.6} {d["door_h"] - 0.06}" size="0.01" material="dark"/>',
        f'        <geom name="handle" class="collision" type="capsule" fromto="-0.05 {-hw * 0.6} {d["door_h"] - 0.06} -0.05 {hw * 0.6} {d["door_h"] - 0.06}" size="0.011" condim="4" friction="1 0.02 0.002"/>',
        f'        <site name="handle" pos="-0.05 0 {d["door_h"] - 0.06}" size="0.01" rgba="1 .3 .3 1"/>',
        f'        <geom class="visual" type="box" pos="-0.042 {-hw * 0.6} {d["door_h"] - 0.06}" size="0.008 0.008 0.008" material="dark"/>',
        f'        <geom class="visual" type="box" pos="-0.042 {hw * 0.6} {d["door_h"] - 0.06}" size="0.008 0.008 0.008" material="dark"/>',
        f'        <site name="door_edge" pos="{-d["door_t"]} 0 {d["door_h"]}" size="0.01" rgba="1 .3 .3 1"/>',
        '      </body>',
        '    </body>',
    ]


def look_at(pos, target, fovy=45, name='cam'):
    fwd = norm([t - p for t, p in zip(target, pos)])
    right = norm([fwd[1], -fwd[0], 0.0])  # fwd x z
    up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]]
    return f'    <camera name="{name}" pos="{f(*pos)}" xyaxes="{f(*right, *up)}" fovy="{fovy}"/>'


def write_scene_xml(arm, tool):
    body, act, eq, exclude = mjcf_robot(arm, tool, mobile=True)
    xml = '\n'.join([
        '<!-- Generated by physics/build_models.py. Kitchen scene for the physics demo. -->',
        '<mujoco model="robot-simulator-kitchen">',
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>',
        '  <visual><global offwidth="1600" offheight="1000"/><quality shadowsize="4096"/><headlight ambient=".35 .35 .35" diffuse=".4 .4 .4"/></visual>',
        DEFAULTS + ASSETS + '  <worldbody>',
        '    <light pos="1 -1 3" dir="-.2 .3 -1" diffuse=".7 .7 .7" castshadow="true"/>',
        '    <geom name="floor" type="plane" size="0 0 .05" material="floor"/>',
        look_at([-0.75, 1.85, 1.75], [0.62, -0.05, 0.55], 48, 'overview'),
        look_at([0.25, -1.9, 1.45], [0.62, -0.25, 0.8], 45, 'counter'),
        look_at([0.3, 1.6, 1.0], [0.85, 0.3, 0.45], 50, 'dishwasher'),
        *body,
        *scene_world(),
        '  </worldbody>',
        '  <equality>', *eq, '  </equality>',
        '  <contact>', *exclude, '    <exclude body1="dishwasher" body2="door"/>', '  </contact>',
        '  <actuator>', *act, '  </actuator>',
        '</mujoco>', '',
    ])
    (OUT / 'scene.xml').write_text(xml, encoding='utf-8')


# ---------------------------------------------------------------------------
# URDF (fixed base: waist, lift, two 7-joint arms and grippers)
# ---------------------------------------------------------------------------
def rpy_from_z(d):
    """Roll-pitch-yaw that turns the z axis (a URDF cylinder's axis) to point along d."""
    x, y, z = norm(d)
    pitch = math.atan2(math.hypot(x, y), z)
    yaw = math.atan2(y, x)
    return 0.0, pitch, yaw


def urdf_link(name, mass, com, visuals, collisions):
    I = 0.4 * mass * 0.05 ** 2 if mass > 0 else 1e-6
    out = [f'  <link name="{name}">']
    if mass > 0:
        out += [
            '    <inertial>',
            f'      <origin xyz="{f(*com)}"/>',
            f'      <mass value="{mass}"/>',
            f'      <inertia ixx="{I:.3g}" iyy="{I:.3g}" izz="{I:.3g}" ixy="0" ixz="0" iyz="0"/>',
            '    </inertial>',
        ]
    for tag, items in (('visual', visuals), ('collision', collisions)):
        for xyz, rpy, geom, mat in items:
            out += [f'    <{tag}>', f'      <origin xyz="{f(*xyz)}" rpy="{f(*rpy)}"/>', f'      <geometry>{geom}</geometry>']
            if tag == 'visual' and mat:
                out.append(f'      <material name="{mat}"/>')
            out.append(f'    </{tag}>')
    out.append('  </link>')
    return out


def urdf_joint(name, kind, parent, child, xyz, axis=None, lo=None, hi=None, effort=100, vel=2, mimic=None):
    out = [f'  <joint name="{name}" type="{kind}">', f'    <parent link="{parent}"/>', f'    <child link="{child}"/>',
           f'    <origin xyz="{f(*xyz)}"/>']
    if axis:
        out.append(f'    <axis xyz="{f(*axis)}"/>')
    if lo is not None:
        out.append(f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="{effort}" velocity="{vel}"/>')
    if mimic:
        out.append(f'    <mimic joint="{mimic}"/>')
    out.append('  </joint>')
    return out


def write_urdf(arm, tool):
    col_h = COLUMN['top'] - BASE_TOP
    b = BASE
    L = ['<?xml version="1.0"?>', '<!-- Generated by physics/build_models.py from frontend/src/lib/kinematics.ts. -->',
         '<robot name="robot_simulator">',
         '  <material name="white"><color rgba=".93 .94 .96 1"/></material>',
         '  <material name="dark"><color rgba=".16 .17 .2 1"/></material>']
    base_box = f'<box size="{b["len"]} {b["w"]} {BASE_TOP - b["y0"]}"/>'
    L += urdf_link('base', 45, [0, 0, 0.15], [([0, 0, (b['y0'] + BASE_TOP) / 2], [0, 0, 0], base_box, 'white')],
                   [([0, 0, (b['y0'] + BASE_TOP) / 2], [0, 0, 0], base_box, None)])
    col_box = f'<box size="{COLUMN["d"]} {COLUMN["w"]} {col_h}"/>'
    L += urdf_link('waist', 8, [0, 0, col_h / 2], [([0, 0, col_h / 2], [0, 0, 0], col_box, 'white')],
                   [([0, 0, col_h / 2], [0, 0, 0], col_box, None)])
    L += urdf_joint('waist', 'revolute', 'base', 'waist', [WAIST_X, 0, BASE_TOP], [0, 0, 1], -math.radians(WAIST_LIM), math.radians(WAIST_LIM), 150, 1.2)
    torso = '<box size="0.15 0.3 0.22"/>'
    L += urdf_link('carriage', 6, [-0.04, 0, 0], [([-0.04, 0, -0.04], [0, 0, 0], torso, 'white')], [([-0.04, 0, -0.04], [0, 0, 0], torso, None)])
    L += urdf_joint('lift', 'prismatic', 'waist', 'carriage', [-WAIST_X, 0, LIFT_MIN - BASE_TOP], [0, 0, 1], 0, LIFT_MAX - LIFT_MIN, 2000, 0.15)
    for side in ('R', 'L'):
        parent = 'carriage'
        for i, j in enumerate(arm):
            name = f'{side}_{j["key"]}'
            pos = to_z_up(mirror_point(side, j['offset']))
            axis = to_z_up(mirror_axis(side, j['axis']))
            nxt = to_z_up(mirror_point(side, arm[i + 1]['offset'])) if i + 1 < len(arm) else [0, 0, -0.06]
            length = math.dist(nxt, [0, 0, 0])
            vis, col = [], []
            if length > 0.02:
                mid = [c / 2 for c in nxt]
                cyl = f'<cylinder radius="{LINK_R[i]}" length="{length:.4f}"/>'
                vis.append((mid, rpy_from_z(nxt), f'<cylinder radius="{LINK_R[i] * 0.9:.3f}" length="{length:.4f}"/>', 'white'))
                col.append((mid, rpy_from_z(nxt), cyl, None))
            if i == 6:
                palm = f'<box size="0.056 {2 * FINGER_OPEN + 0.024:.3f} 0.045"/>'
                vis.append(([0, 0, -0.065], [0, 0, 0], palm, 'dark'))
                col.append(([0, 0, -0.065], [0, 0, 0], palm, None))
            L += urdf_link(name, j['mass'], to_z_up(mirror_point(side, j['at'])), vis, col)
            L += urdf_joint(name, 'revolute', parent, name, pos, axis, math.radians(j['min']), math.radians(j['max']), j['rated'], j['vmax'])
            parent = name
        pad = f'<box size="{2 * PAD["x"]} {2 * PAD["y"]} {2 * PAD["z"]}"/>'
        for k, s in (('a', 1), ('b', -1)):
            fl = f'{side}_finger_{k}'
            L += urdf_link(fl, 0.05, [0, 0, 0], [([0, -s * 0.004, 0], [0, 0, 0], pad, 'dark')], [([0, -s * 0.004, 0], [0, 0, 0], pad, None)])
            L += urdf_joint(fl, 'prismatic', parent, fl, [0, s * FINGER_OPEN, -0.11], [0, -s, 0], 0, FINGER_TRAVEL, GRIP_FORCE, 0.1,
                            mimic=f'{side}_finger_a' if k == 'b' else None)
        L += urdf_link(f'{side}_tcp', 0, [0, 0, 0], [], [])
        L += urdf_joint(f'{side}_tcp', 'fixed', parent, f'{side}_tcp', to_z_up(tool))
    L += ['</robot>', '']
    (OUT / 'robot.urdf').write_text('\n'.join(L), encoding='utf-8')


def main():
    arm, tool = read_arm()
    OUT.mkdir(exist_ok=True)
    write_robot_xml(arm, tool)
    write_scene_xml(arm, tool)
    write_urdf(arm, tool)
    print(f'Read {len(arm)} joints from {KIN.relative_to(HERE.parent)}:')
    for j in arm:
        print(f'  {j["key"]:3} {j["label"]:16} {j["min"]:6.0f} .. {j["max"]:4.0f} deg  {j["rated"]:4.0f} N.m  {j["mass"]:.2f} kg')
    for n in ('robot.xml', 'scene.xml', 'robot.urdf'):
        print('wrote', (OUT / n).relative_to(HERE.parent))


if __name__ == '__main__':
    main()
