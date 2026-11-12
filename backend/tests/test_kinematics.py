import math
import random

import numpy as np
import pytest

from app import kinematics as kin
from app import planner

L = kin.DEFAULT_LENGTHS


def random_q(rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in kin.LIMITS])


def test_fk_straight_up():
    pts, phi = kin.fk(np.radians([90, 0, 0]), L)
    assert pts[-1] == pytest.approx([0.0, kin.BASE[1] + sum(L)], abs=1e-9)
    assert phi == pytest.approx(math.pi / 2)


def test_jacobian_matches_finite_differences():
    rng = random.Random(1)
    for _ in range(20):
        q = random_q(rng)
        J = kin.jacobian(q, L)
        eps = 1e-6
        for i in range(3):
            dq = np.zeros(3)
            dq[i] = eps
            p1, a1 = kin.fk(q + dq, L)
            p0, a0 = kin.fk(q - dq, L)
            num = np.array([*(p1[-1] - p0[-1]), a1 - a0]) / (2 * eps)
            assert J[:, i] == pytest.approx(num, abs=1e-6)


@pytest.mark.parametrize("method", ["analytic", "dls", "ccd"])
def test_ik_round_trip(method):
    """Pick random joint angles, compute the tool point, and ask IK to get back there."""
    rng = random.Random(42)
    solved = 0
    for _ in range(60):
        q_true = random_q(rng)
        pts, phi = kin.fk(q_true, L)
        res = kin.solve(method, pts[-1], L, kin.HOME_Q, phi if method != "ccd" else None)
        if res.reachable:
            solved += 1
            assert res.position_error < 1e-3
            assert res.within_limits
    # Analytic must always solve a pose that came from valid joint angles;
    # the iterative methods can get stuck at joint limits now and then.
    assert solved >= (60 if method == "analytic" else 48)


def test_unreachable_target_is_flagged():
    for method in kin.METHODS:
        res = kin.solve(method, (2.0, 0.1), L, kin.HOME_Q, -math.pi / 2)
        assert not res.reachable
        assert res.notes


def test_plan_sorts_every_block_smoothly():
    objs = planner.default_scene()
    out = planner.plan_sort(L, kin.HOME_Q4, objs, planner.default_bins())
    assert out["stats"]["sorted"] == len(objs)
    assert out["stats"]["skipped"] == 0
    assert out["stats"]["minClearanceMm"] >= 0
    qs = np.array([f["q"] for f in out["frames"]])
    # No teleporting: every frame-to-frame change stays small.
    assert np.max(np.abs(np.diff(qs, axis=0))) < math.radians(6)
    assert all(kin.within_limits(q[1:], tol=1e-4) for q in qs)
    # Each block lands in the bin of its own color.
    bins = {b["id"]: b["color"] for b in planner.default_bins()}
    colors = {o["id"]: o["color"] for o in objs}
    for p in out["placements"]:
        assert bins[p["binId"]] == colors[p["id"]]


@pytest.mark.parametrize("seed", range(10))
def test_random_scenes_are_fully_sortable(seed):
    objs = planner.random_scene(seed)
    out = planner.plan_sort(L, kin.HOME_Q4, objs, planner.default_bins())
    assert out["stats"]["sorted"] == len(objs)


def test_far_block_is_skipped_not_crashed():
    objs = [{"id": "far", "color": "red", "x": 0.95, "y": 0.02, "size": 0.04, "binId": None}]
    out = planner.plan_sort(L, kin.HOME_Q4, objs, planner.default_bins())
    assert out["stats"]["skipped"] == 1
    assert out["frames"]


@pytest.mark.parametrize("seed", range(10))
def test_waist_turns_instead_of_elbow_flipping(seed):
    """Moving blocks from the right to the bins on the left should turn the
    waist, not swing the arm up and over the top."""
    out = planner.plan_sort(L, kin.HOME_Q4, planner.random_scene(seed), planner.default_bins())
    qs = np.array([f["q"] for f in out["frames"]])
    assert np.all(qs[:, 2] < 0), "elbow flipped"  # elbow stays bent the same way
    tool_heights = [kin.fk_world(q, L)[0][-1][1] for q in qs]
    assert max(tool_heights) < 0.35  # never reaches up high
    # Waist swings front <-> back (plus a few degrees of aiming at camera estimates).
    assert np.ptp(qs[:, 0]) == pytest.approx(math.pi, abs=0.1)


def test_world_ik_faces_the_target_side():
    for x in (0.4, -0.4):
        waist, res = kin.solve_world("analytic", (x, 0.1), L, kin.HOME_Q4, -math.pi / 2)
        assert res.reachable
        tool = kin.fk_world([waist, *res.q], L)[0][-1]
        assert tool[0] == pytest.approx(x, abs=1e-6)
        assert tool[2] == pytest.approx(0.0, abs=1e-6)
        assert res.q[1] < 0  # same elbow-up shape on both sides


# ---- Failures and recovery --------------------------------------------------

def test_clean_run_has_no_faults():
    out = planner.plan_sort(L, kin.HOME_Q4, planner.default_scene(), planner.default_bins(), faults="none")
    assert (out["stats"]["slips"], out["stats"]["drops"], out["stats"]["recovered"]) == (0, 0, 0)
    assert not any(e["kind"] in ("fault", "recover") for e in out["events"])


def test_demo_drops_and_recovers_every_block():
    objs = planner.default_scene()
    out = planner.plan_sort(L, kin.HOME_Q4, objs, planner.default_bins(), faults="demo")
    s = out["stats"]
    assert s["slips"] == 1 and s["drops"] == 1
    assert s["recovered"] == 2
    assert s["sorted"] == len(objs)
    # Every block ends up in a bin, and every fall lands on the table top.
    for obj in objs:
        tl = out["timeline"][obj["id"]]
        assert tl[-1]["mode"] == "bin"
        for kf in tl:
            if kf["mode"] == "falling":
                assert kf["to"][1] == pytest.approx(obj["size"] / 2)
                assert abs(kf["to"][2]) < 0.45  # still on the table
    # The arm never teleports, even through the emergency stop.
    qs = np.array([f["q"] for f in out["frames"]])
    assert np.max(np.abs(np.diff(qs, axis=0))) < math.radians(6)


@pytest.mark.parametrize("seed", range(8))
def test_random_faults_are_all_recovered(seed):
    objs = planner.random_scene(seed)
    out = planner.plan_sort(L, kin.HOME_Q4, objs, planner.default_bins(), faults="random", seed=seed)
    s = out["stats"]
    assert s["recovered"] == s["slips"] + s["drops"]
    assert s["sorted"] == len(objs)


# ---- Pick-and-place commands ------------------------------------------------

def test_command_moves_block_to_a_table_spot_then_a_bin():
    objs = planner.default_scene()
    bid = objs[0]["id"]
    out = planner.plan_commands(L, kin.HOME_Q4, objs, planner.default_bins(), [
        {"blockId": bid, "x": 0.3, "z": 0.25},
        {"blockId": bid, "binId": "bin-green"},
    ])
    assert out["stats"]["sorted"] == 2
    tl = out["timeline"][bid]
    table = [k for k in tl if k["mode"] == "table"][-1]
    assert table["pos"][0] == pytest.approx(0.3) and table["pos"][2] == pytest.approx(0.25)
    assert tl[-1]["mode"] == "bin" and tl[-1]["binId"] == "bin-green"


def test_bad_table_spots_are_rejected_with_a_reason():
    objs = planner.default_scene()
    bins = planner.default_bins()
    assert "bin" in planner.table_spot_problem(bins[0]["x"], 0.0, objs, bins)
    assert "taken" in planner.table_spot_problem(objs[1]["x"], 0.0, objs, bins, objs[0]["id"])
    assert "base" in planner.table_spot_problem(0.05, 0.02, objs, bins)
    out = planner.plan_commands(L, kin.HOME_Q4, objs, bins, [{"blockId": objs[0]["id"], "x": objs[1]["x"], "z": 0.0}])
    assert out["stats"]["skipped"] == 1
    assert any("taken" in e["text"] for e in out["events"])


def test_command_api_round_trip():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    cfg = client.get("/api/config").json()
    res = client.post("/api/commands", json={
        "q0": cfg["homeQ"], "objects": cfg["objects"], "bins": cfg["bins"],
        "commands": [{"blockId": cfg["objects"][2]["id"], "binId": "bin-red"}],
        "faults": "demo", "faultOffset": 3,  # 4th command of the session: the demo drop
    })
    assert res.status_code == 200
    s = res.json()["stats"]
    assert s["sorted"] == 1 and s["drops"] == 1 and s["recovered"] == 1


# ---- Household tasks -------------------------------------------------------

from app import tasks  # noqa: E402


def _smooth(out):
    qs = np.array([f["q"] for f in out["frames"]])
    return float(np.max(np.abs(np.diff(qs, axis=0))))


def test_dishwasher_loads_every_plate_and_cup():
    sc = tasks.dishes_scene()
    out = tasks.plan_dishes(L, kin.HOME_Q4, sc["objects"], sc["fixtures"])
    assert out["stats"]["sorted"] == 6 and out["stats"]["skipped"] == 0
    assert _smooth(out) < math.radians(6)
    assert out["stats"]["minClearanceMm"] > 0
    for o in sc["objects"]:
        last = out["timeline"][o["id"]][-1]
        assert last["mode"] == "bin"
        if o["kind"] == "plate":
            assert last["binId"] == "plate-rack"
        else:
            assert last["binId"] == "cup-rack"
            assert abs(abs(last["tilt"]) - math.pi) < 1e-3  # cups go in upside down


def test_cup_flip_keeps_the_cup_in_place():
    """During the wrist flip the cup turns about its own centre, so the tool
    point (between the fingertips) must not move."""
    sc = tasks.dishes_scene()
    out = tasks.plan_dishes(L, kin.HOME_Q4, sc["objects"], sc["fixtures"])
    flip = next(e for e in out["events"] if "Flip" in e["text"])
    nxt = next(e for e in out["events"] if e["t"] > flip["t"])
    tips = [kin.fk_world(f["q"], L)[0][-1] for f in out["frames"] if flip["t"] <= f["t"] < nxt["t"]]
    assert len(tips) > 10
    assert np.max(np.ptp(np.array(tips), axis=0)) < 1e-3


def test_towel_is_folded_twice_and_put_in_the_basket():
    sc = tasks.laundry_scene()
    out = tasks.plan_towel(L, kin.HOME_Q4, sc["objects"], sc["fixtures"])
    assert out["stats"]["sorted"] == 1
    cloth = out["cloth"]
    f1, f2 = cloth["folds"]
    assert f1["tGrasp"] < f1["tRelease"] < f2["tGrasp"] < f2["tRelease"] < cloth["carry"]["tGrasp"] < cloth["carry"]["tRelease"]
    # The second fold is across the first.
    assert abs(np.dot(f1["n"], f2["n"])) < 1e-9
    # When the first fold is released the gripper is over the near edge.
    rel = next(f for f in out["frames"] if f["t"] >= f1["tRelease"])
    tip = kin.fk_world(rel["q"], L)[0][-1]
    edge = np.array(f1["F"]) - np.array(f1["n"]) * f1["w"]
    assert np.hypot(tip[0] - edge[0], tip[2] - edge[1]) < 0.01
    assert _smooth(out) < math.radians(6)


def test_task_api():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    cfg = client.get("/api/config").json()
    for task, key in (("dishes", "dishes"), ("laundry", "laundry")):
        sc = cfg["scenes"][key]
        res = client.post("/api/task", json={"task": task, "q0": cfg["homeQ"], "objects": sc["objects"], "fixtures": sc["fixtures"]})
        assert res.status_code == 200, res.text
        assert res.json()["stats"]["skipped"] == 0


# ---- Realistic robot behaviour: vision, force sensing, careful contact -------

def _all_plans():
    d, l = tasks.dishes_scene(), tasks.laundry_scene()
    return {
        "blocks": planner.plan_sort(L, kin.HOME_Q4, planner.default_scene(), planner.default_bins()),
        "dishes": tasks.plan_dishes(L, kin.HOME_Q4, d["objects"], d["fixtures"]),
        "towel": tasks.plan_towel(L, kin.HOME_Q4, l["objects"], l["fixtures"]),
    }


def test_sensor_streams_cover_every_frame():
    for name, out in _all_plans().items():
        assert len(out["sensors"]["fz"]) == len(out["frames"]), name
        assert len(out["sensors"]["grip"]) == len(out["frames"]), name


def test_every_pick_is_scanned_then_corrected_up_close():
    for name, out in _all_plans().items():
        scans = [p for p in out["perception"] if p["phase"] == "scan"]
        aligns = [p for p in out["perception"] if p["phase"] == "align"]
        picks = [e for e in out["events"] if e["kind"] == "pick"]
        # Every pick is scanned first (one scan of the towel covers both folds).
        assert len(scans) >= (2 if name == "towel" else len(picks)), name
        assert len(aligns) >= 1, name
        for p in scans:
            assert 0.9 <= p["confidence"] <= 0.99
        # The first estimate is a few mm off; the close-up look fixes it.
        for a in aligns:
            scan = [s for s in scans if s["id"] == a["id"] and s["tEnd"] <= a["tStart"] + 1e-6][-1]
            err = math.hypot(a["center"][0] - scan["center"][0], a["center"][2] - scan["center"][2])
            assert 0.002 < err < 0.01


def test_grip_force_while_holding_and_weight_on_the_wrist():
    out = planner.plan_sort(L, kin.HOME_Q4, planner.default_scene(), planner.default_bins(), faults="none")
    grip, fz = np.array(out["sensors"]["grip"]), np.array(out["sensors"]["fz"])
    holding = np.array([f["holding"] is not None for f in out["frames"]])
    assert np.median(grip[holding]) > 8.0  # squeezing a block at ~10 N
    assert np.median(grip[~holding]) < 0.5
    assert np.median(fz[holding]) == pytest.approx(planner.WEIGHT_BLOCK, abs=0.15)  # carrying its weight


def test_touch_down_spikes_the_wrist_sensor():
    out = planner.plan_sort(L, kin.HOME_Q4, planner.default_scene(), planner.default_bins(), faults="none")
    fz = np.array(out["sensors"]["fz"])
    ts = np.array([f["t"] for f in out["frames"]])
    for p in out["placements"]:
        window = fz[(ts > p["t"] - 0.6) & (ts <= p["t"])]
        assert window.max() > planner.WEIGHT_BLOCK + 1.0


def test_slip_shows_up_on_the_grip_sensor():
    out = planner.plan_sort(L, kin.HOME_Q4, planner.default_scene(), planner.default_bins(), faults="demo")
    slip = next(e for e in out["events"] if e["kind"] == "fault" and "slipped" in e["text"])
    i = next(i for i, f in enumerate(out["frames"]) if f["t"] >= slip["t"])
    grip = out["sensors"]["grip"]
    # Squeezing right before the slip, nothing right after (the fingers close on air).
    assert max(grip[i - 25:i - 8]) > 8.0 and grip[i] < 0.5


def test_final_approach_is_slow():
    """The last few cm before a grasp are done at creep speed."""
    out = planner.plan_sort(L, kin.HOME_Q4, planner.default_scene(), planner.default_bins(), faults="none")
    tips = np.array([kin.fk_world(f["q"], L)[0][-1] for f in out["frames"]])
    speed = np.linalg.norm(np.diff(tips, axis=0), axis=1) * planner.FPS
    low = tips[1:, 1] < 0.012 + 0.025  # tool within ~2.5 cm of grasp height on a block
    assert speed[low].max() < 0.1


# ---- Two arms sharing the work ----------------------------------------------

from app import dual  # noqa: E402


@pytest.mark.parametrize("seed", [7, 1, 2])
def test_two_arms_sort_everything_without_touching(seed):
    sc = dual.dual_scene(seed)
    out = dual.plan_dual(L, sc["objects"], sc["bins"], compare=False)
    assert out["stats"]["sorted"] == len(sc["objects"])
    assert len(out["frames"]) == len(out["armB"]["frames"])
    # Check every frame (not just the sampled summary) for the clearance.
    a = np.stack([dual._link_points(f["q"], L, 0.0) for f in out["frames"]])
    b = np.stack([dual._link_points(f["q"], L, dual.ARM_B_Z) for f in out["armB"]["frames"]])
    gap = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1).min()
    assert gap >= dual.CLEARANCE - 1e-6
    # Both arms did some of the work, and the hand-offs between arms are tagged.
    assert all(x["blocks"] >= 2 for x in out["dual"]["arms"])
    held = [kf for kfs in out["timeline"].values() for kf in kfs if kf["mode"] == "held"]
    assert {kf["arm"] for kf in held} == {"A", "B"}


def test_two_arms_are_faster_than_one():
    sc = dual.dual_scene()
    out = dual.plan_dual(L, sc["objects"], sc["bins"])
    assert out["dual"]["speedup"] > 1.1


def test_dual_api():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    sc = client.get("/api/config").json()["scenes"]["dual"]
    res = client.post("/api/dual", json={"objects": sc["objects"], "bins": sc["bins"]})
    assert res.status_code == 200, res.text
    assert res.json()["stats"]["sorted"] == len(sc["objects"])


# ---- Two arms folding a shirt -----------------------------------------------

from app import bimanual  # noqa: E402


def _shirt():
    return bimanual.plan_shirt(L, bimanual.shirt_scene()["objects"])


def test_shirt_fold_has_three_folds_in_the_right_order():
    out = _shirt()
    assert out["stats"]["sorted"] == 1 and not out["warnings"]
    f1, f2, f3 = out["cloth"]["folds"]
    # The two side folds happen at the same time, one arm each...
    assert f1["arms"] == ["A"] and f2["arms"] == ["B"]
    assert f1["tGrasp"] == f2["tGrasp"] and f1["tRelease"] == f2["tRelease"]
    assert f1["n"][1] == -f2["n"][1]  # ...from opposite sides toward the middle
    # ...then both arms fold the bottom up together.
    assert f3["arms"] == ["A", "B"] and f3["tGrasp"] > f1["tRelease"]
    assert abs(np.dot(f1["n"], f3["n"])) < 1e-9  # across the side folds


def test_both_arms_hold_the_hem_in_lockstep():
    """During the bottom fold the two grippers move as a pair: same height,
    same distance along the fold, every frame."""
    out = _shirt()
    f3 = out["cloth"]["folds"][2]
    for fa, fb in zip(out["frames"], out["armB"]["frames"]):
        if f3["tGrasp"] <= fa["t"] <= f3["tRelease"]:
            ta = kin.fk_world(fa["q"], L)[0][-1]
            tb = kin.fk_world(fb["q"], L)[0][-1]
            assert ta[0] == pytest.approx(tb[0], abs=1e-3)
            assert ta[1] == pytest.approx(tb[1], abs=1e-3)


def test_shirt_arms_keep_their_distance_and_move_smoothly():
    out = _shirt()
    assert out["dual"]["minArmGapMm"] >= bimanual.COOP_CLEARANCE * 1000
    for frames in (out["frames"], out["armB"]["frames"]):
        qs = np.array([f["q"] for f in frames])
        assert np.max(np.abs(np.diff(qs, axis=0))) < math.radians(6)
    # Both grippers squeeze twice (side fold, bottom fold).
    assert all(a["blocks"] == 2 for a in out["dual"]["arms"])


def test_shirt_api():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    sc = client.get("/api/config").json()["scenes"]["shirt"]
    res = client.post("/api/shirt", json={"objects": sc["objects"]})
    assert res.status_code == 200, res.text
    assert len(res.json()["cloth"]["folds"]) == 3


# ---- Wrist roll ----------------------------------------------------------------

def _mod(a, period):
    """Distance of angle a from the nearest multiple of period."""
    r = a % period
    return min(r, period - r)


def _frame_at(frames, t):
    return min(frames, key=lambda f: abs(f["t"] - t))


def test_roll_is_smooth_everywhere():
    for name, out in _all_plans().items():
        rolls = np.array([f["roll"] for f in out["frames"]])
        assert np.max(np.abs(np.diff(rolls))) < math.radians(6), name


def test_fingers_line_up_with_each_block_and_blocks_land_square():
    objs = planner.default_scene()
    assert any(abs(o["yaw"]) > math.radians(10) for o in objs)  # blocks really are at angles
    out = planner.plan_sort(L, kin.HOME_Q4, objs, planner.default_bins(), faults="none")
    yaws = {o["id"]: o["yaw"] for o in objs}
    for oid, kfs in out["timeline"].items():
        held = next(k for k in kfs if k["mode"] == "held")
        f = _frame_at(out["frames"], held["t"])
        closing = f["q"][0] + math.pi / 2 + f["roll"]  # direction the fingers close in
        assert _mod(closing - yaws[oid], math.pi / 2) < math.radians(0.6)
        assert _mod(kfs[-1]["yaw"], math.pi / 2) < math.radians(0.6)  # square in the bin


def test_shirt_pinches_close_across_seams_and_hem():
    out = _shirt()
    side, _, hem = out["cloth"]["folds"]
    for frames in (out["frames"], out["armB"]["frames"]):
        f = _frame_at(frames, side["tGrasp"])
        assert _mod(f["q"][0] + math.pi / 2 + f["roll"] - math.pi / 2, math.pi) < math.radians(1.5)  # across the side seam
        f = _frame_at(frames, hem["tGrasp"])
        assert _mod(f["q"][0] + math.pi / 2 + f["roll"], math.pi) < math.radians(1.5)  # across the hem
        assert abs(f["roll"]) > math.radians(60)  # a real twist of the wrist


# ---- two arms on the household tasks ------------------------------------------

def test_dual_short_links_does_not_crash():
    from app import dual
    sc = dual.dual_scene()
    out = dual.plan_dual([0.15, 0.15, 0.17], sc["objects"], sc["bins"])
    assert out["stats"]["sorted"] == 0 and out["stats"]["skipped"] == 6
    assert out["dual"]["speedup"] is None
    assert out["events"][-1]["kind"] == "done"


def test_dishes_with_two_arms_loads_everything_faster():
    from app import teamwork
    sc = teamwork.dishes2_scene()
    out = teamwork.plan_dishes2((0.34, 0.30, 0.12), sc["objects"], sc["fixtures"])
    assert out["stats"]["sorted"] == 6 and out["stats"]["skipped"] == 0
    assert out["dual"]["minArmGapMm"] >= out["dual"]["clearanceMm"]
    assert out["dual"]["speedup"] > 1.3
    a, b = out["dual"]["arms"]
    assert (a["blocks"], b["blocks"]) == (3, 3)  # A plates, B cups


def test_towel_with_two_arms_folds_together_and_b_carries():
    from app import teamwork
    sc = teamwork.laundry2_scene()
    out = teamwork.plan_towel2((0.34, 0.30, 0.12), sc["objects"], sc["fixtures"])
    assert out["stats"]["sorted"] == 1 and not out["warnings"]
    cloth = out["cloth"]
    assert cloth["folds"][0]["arms"] == ["A", "B"] and cloth["folds"][1]["arms"] == ["A"]
    assert cloth["carry"]["arm"] == "B" and cloth["carry"]["tRelease"] > cloth["carry"]["tGrasp"]
    ts = [e["t"] for e in out["events"]]
    assert ts == sorted(ts)


def test_task_api_two_arm_variants():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    cfg = c.get("/api/config").json()
    for name in ("dishes2", "laundry2"):
        sc = cfg["scenes"][name]
        assert sc["arms"]
        r = c.post("/api/task", json={"task": name, "objects": sc["objects"], "fixtures": sc["fixtures"]})
        assert r.status_code == 200 and r.json()["armB"]["frames"]
