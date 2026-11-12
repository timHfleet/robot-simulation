"""FastAPI server: kinematics + planning API, and the built React front end."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import kinematics as kin
from . import planner
from . import tasks
from . import dual
from . import bimanual
from . import teamwork

app = FastAPI(title="Robot Arm Simulator", version="1.0.0")


@app.exception_handler(Exception)
async def planning_failed(request: Request, exc: Exception):
    """Unexpected planner failures come back as a readable message, not a bare 500."""
    return JSONResponse(status_code=500, content={"detail": f"the planner hit a problem ({type(exc).__name__}: {exc}). Try Reset, or longer arm links."})

Lengths = tuple[float, float, float]


def _check_lengths(v):
    if any(not 0.05 <= x <= 0.6 for x in v):
        raise ValueError("each link length must be between 0.05 and 0.6 m")
    return v


class Point(BaseModel):
    x: float = Field(ge=-2, le=2)
    y: float = Field(ge=-2, le=2)


Q4 = tuple[float, float, float, float]


class FKRequest(BaseModel):
    q: Q4  # waist, shoulder, elbow, wrist
    lengths: Lengths = kin.DEFAULT_LENGTHS
    _v = field_validator("lengths")(_check_lengths)


class IKRequest(BaseModel):
    target: Point
    method: Literal["analytic", "dls", "ccd"] = "analytic"
    lengths: Lengths = kin.DEFAULT_LENGTHS
    q0: Q4 = tuple(kin.HOME_Q4)
    toolAngleDeg: Optional[float] = Field(default=-90.0, ge=-360, le=360)
    elbow: Literal["up", "down"] = "up"
    _v = field_validator("lengths")(_check_lengths)


class Block(BaseModel):
    id: str
    color: str
    x: float
    y: float = planner.BLOCK_SIZE / 2
    z: float = 0.0  # depth toward the viewer (non-zero after a drop)
    yaw: float = 0.0
    kind: Literal["block", "plate", "cup", "towel", "shirt"] = "block"
    tilt: float = 0.0  # rotation inside its own vertical plane (a cup upside down = pi)
    size: float = planner.BLOCK_SIZE
    binId: Optional[str] = None


class Bin(BaseModel):
    id: str
    color: str
    x: float
    z: float = 0.0
    width: float = planner.BIN_WIDTH
    wall: float = planner.BIN_WALL


class PlanRequest(BaseModel):
    lengths: Lengths = kin.DEFAULT_LENGTHS
    q0: Q4 = tuple(kin.HOME_Q4)
    objects: list[Block] = Field(max_length=12)
    bins: list[Bin] = Field(max_length=6)
    faults: Literal["none", "demo", "random"] = "demo"
    seed: Optional[int] = None
    _v = field_validator("lengths")(_check_lengths)


class Command(BaseModel):
    """Move one block: into a bin (binId) or onto a table spot (x, z)."""
    blockId: str
    binId: Optional[str] = None
    x: Optional[float] = Field(default=None, ge=-1, le=1)
    z: Optional[float] = Field(default=None, ge=-1, le=1)


class CommandRequest(BaseModel):
    lengths: Lengths = kin.DEFAULT_LENGTHS
    q0: Q4 = tuple(kin.HOME_Q4)
    objects: list[Block] = Field(max_length=12)
    bins: list[Bin] = Field(max_length=6)
    commands: list[Command] = Field(min_length=1, max_length=20)
    faults: Literal["none", "demo", "random"] = "none"
    seed: Optional[int] = None
    faultOffset: int = Field(default=0, ge=0)
    _v = field_validator("lengths")(_check_lengths)


class TaskRequest(BaseModel):
    task: Literal["dishes", "laundry", "dishes2", "laundry2"]
    lengths: Lengths = kin.DEFAULT_LENGTHS
    q0: Q4 = tuple(kin.HOME_Q4)
    objects: list[Block] = Field(max_length=12)
    fixtures: list[dict] = Field(max_length=8)
    _v = field_validator("lengths")(_check_lengths)


class SceneRequest(BaseModel):
    seed: Optional[int] = None
    count: int = Field(default=5, ge=1, le=6)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {
        "base": kin.BASE.tolist(),
        "lengths": list(kin.DEFAULT_LENGTHS),
        "limitsDeg": [list(kin.WAIST_LIMITS_DEG), *kin.LIMITS_DEG.tolist()],
        "homeQ": [float(a) for a in kin.HOME_Q4],
        "methods": list(kin.METHODS),
        "fps": planner.FPS,
        "objects": planner.default_scene(),
        "bins": planner.default_bins(),
        "blockSize": planner.BLOCK_SIZE,
        "graspFraction": planner.GRASP_FRACTION,
        "scenes": {
            "blocks": {"objects": planner.default_scene(), "bins": planner.default_bins(), "fixtures": []},
            "dishes": {**tasks.dishes_scene(), "bins": []},
            "laundry": {**tasks.laundry_scene(), "bins": []},
            "dual": dual.dual_scene(),
            "shirt": bimanual.shirt_scene(),
            "dishes2": teamwork.dishes2_scene(),
            "laundry2": teamwork.laundry2_scene(),
        },
    }


@app.post("/api/fk")
def forward(req: FKRequest):
    pts, phi = kin.fk_world(np.array(req.q), req.lengths)
    return {"joints": pts.round(5).tolist(), "phiDeg": round(math.degrees(kin.wrap(phi)), 3)}


@app.post("/api/ik")
def inverse(req: IKRequest):
    phi = None if req.toolAngleDeg is None else math.radians(req.toolAngleDeg)
    waist, res = kin.solve_world(req.method, (req.target.x, req.target.y), req.lengths, np.array(req.q0), phi, req.elbow)
    return kin.world_result(waist, res, req.lengths)


@app.post("/api/plan")
def plan(req: PlanRequest):
    return planner.plan_sort(
        req.lengths,
        np.array(req.q0),
        [o.model_dump() for o in req.objects],
        [b.model_dump() for b in req.bins],
        faults=req.faults,
        seed=req.seed,
    )


@app.post("/api/commands")
def commands(req: CommandRequest):
    return planner.plan_commands(
        req.lengths,
        np.array(req.q0),
        [o.model_dump() for o in req.objects],
        [b.model_dump() for b in req.bins],
        [c.model_dump() for c in req.commands],
        faults=req.faults,
        seed=req.seed,
        fault_offset=req.faultOffset,
    )


@app.post("/api/task")
def household_task(req: TaskRequest):
    objects = [o.model_dump() for o in req.objects]
    if req.task == "dishes2":
        return teamwork.plan_dishes2(req.lengths, objects, req.fixtures)
    if req.task == "laundry2":
        return teamwork.plan_towel2(req.lengths, objects, req.fixtures)
    return tasks.plan_task(req.task, req.lengths, np.array(req.q0), objects, req.fixtures)


class DualRequest(BaseModel):
    lengths: Lengths = kin.DEFAULT_LENGTHS
    objects: list[Block] = Field(max_length=10)
    bins: list[Bin] = Field(max_length=6)
    seed: Optional[int] = None
    _v = field_validator("lengths")(_check_lengths)


@app.post("/api/dual")
def two_arms(req: DualRequest):
    out = dual.plan_dual(req.lengths, [o.model_dump() for o in req.objects], [b.model_dump() for b in req.bins], seed=req.seed)
    out.pop("_scene", None)
    return out


class ShirtRequest(BaseModel):
    lengths: Lengths = kin.DEFAULT_LENGTHS
    objects: list[Block] = Field(default_factory=list, max_length=4)
    _v = field_validator("lengths")(_check_lengths)


@app.post("/api/shirt")
def fold_shirt(req: ShirtRequest):
    return bimanual.plan_shirt(req.lengths, [o.model_dump() for o in req.objects])


@app.post("/api/scene")
def scene(req: SceneRequest):
    return {"objects": planner.random_scene(req.seed, req.count)}


# --- Static front end (built by `npm run build`) ---------------------------
DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
