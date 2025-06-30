"""TouchCut Plasma‑table **simulator** – expanded

Run:
    uvicorn touchcut_simulator:app --reload

v0.4.0 – June 2025
──────────────────
* Added **REST API parity** with Kinetic TouchCut v1 docs (subset):
  • **auth/change‑password** – simulated change & JWT refresh
  • **stats** – machine, resettable, session & consumable summaries (date‑filtered)
  • **monitor** – rich IO & axis telemetry + field filtering
  • **programs** – pagination & field‑filter support (GET only)
  • **unloader** – demo pallet/unloaded‑parts info
* Machine sensor model extended (air, O₂, temp, voltage)
* Unit conversion helpers (imperial ⇆ metric) on demand
* Utility helpers: wildcard field‑filter, ISO‑date parsing, fake‑data generators
* Existing endpoints remain fully backward‑compatible
"""
from __future__ import annotations

import asyncio
import base64
import fnmatch
import math
import random
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field, validator
from PIL import Image, ImageDraw, ImageFont

###############################################################################
# ─── Settings & helpers ──────────────────────────────────────────────────────
###############################################################################

SECRET_KEY = "super‑secret‑key‑change‑me"  # noqa: S105 – demo only!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/rest/v1/auth")
app = FastAPI(title="TouchCut Simulator", version="0.4.0")

###############################################################################
# ─── Models ──────────────────────────────────────────────────────────────────
###############################################################################

class AuthRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., alias="old-password")
    new_password: str = Field(..., alias="new-password")
    confirm_password: str = Field(..., alias="confirm-password")

    @validator("confirm_password")
    def _match(cls, v, values):  # noqa: N805
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("confirm-password must match new-password")
        return v


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AxisState(BaseModel):
    name: str
    pos: float = 0.0  # inches
    vel: float = 0.0  # inches/sec


class MachineState(BaseModel):
    # motion
    axes: Dict[str, AxisState] = {a: AxisState(name=a) for a in ("X", "Y", "Z")}
    # execution
    torch_on: bool = False
    program_running: bool = False
    program_paused: bool = False
    current_program: Optional[str] = None
    program_progress: float = 0.0  # 0‑100 %
    # overrides & life
    feed_override: float = 1.0
    torch_seconds: int = 0  # accumulative torch‑on time
    nozzle_life_remaining: int = 1000  # arbitrary units
    # simulated sensors
    temperature_c: float = 25.0
    oxygen_pct: float = 21.0
    air_pressure_psi: float = 90.0
    arc_voltage_v: float = 120.0
    # machine on/off cycle
    machine_on: bool = True
    _cycle_time: float = 0.0  # seconds since sim start

    # ── stats accumulation ──
    total_distance_in: float = 0.0
    total_tool_on_time: float = 0.0
    total_starts: int = 0
    resettable_totals: Dict[str, Any] = {
        "distance_in": 0.0,
        "tool_on_s": 0.0,
        "starts": 0,
        "last_reset": datetime.utcnow().isoformat(),
    }

    def step(self, dt: float):
        """Advance simulated physics & bookkeeping."""
        self._cycle_time += dt
        cycle_period = 30 * 60 + 5 * 60  # 30 min on, 5 min off
        self.machine_on = (self._cycle_time % cycle_period) < (30 * 60)

        # sensors
        if self.machine_on:
            self.temperature_c = min(self.temperature_c + 0.01 * dt, 60)
            self.oxygen_pct = max(self.oxygen_pct - 0.0001 * dt, 19)
            self.air_pressure_psi = 90 + 5 * math.sin(self._cycle_time / 60)
            self.arc_voltage_v = 120 + 10 * math.sin(self._cycle_time / 10) if self.torch_on else 0
        else:
            self.temperature_c = max(self.temperature_c - 0.02 * dt, 20)
            self.oxygen_pct = min(self.oxygen_pct + 0.0002 * dt, 21)
            self.air_pressure_psi = 90
            self.arc_voltage_v = 0

        # axis motion
        if self.machine_on:
            for ax in self.axes.values():
                ax.pos += ax.vel * dt
                self.total_distance_in += abs(ax.vel) * dt / 3 if ax.name == "X" else 0  # arbitrary

        # program execution
        if self.program_running and not self.program_paused and self.machine_on:
            self.program_progress += 2.0 * self.feed_override  # finishes in ~50 s
            if self.program_progress >= 100:
                self.finish_program()

        # torch & consumables
        if self.torch_on and self.machine_on:
            self.torch_seconds += int(dt)
            self.total_tool_on_time += int(dt)
            self.nozzle_life_remaining = max(0, 1000 - self.torch_seconds)

    def finish_program(self):
        self.program_running = False
        self.program_paused = False
        self.torch_on = False
        self.current_program = None
        self.program_progress = 0.0
        self.total_starts += 1

###############################################################################
# ─── In‑memory stores ────────────────────────────────────────────────────────
###############################################################################

_fake_users: Dict[str, str] = {"demo": "demo"}
_programs: Dict[str, str] = {}
state = MachineState()

# Demo IO points for /monitor
_io_schema = {
    "din": [
        {"name": "Torch_OK", "address": "DIN0"},
        {"name": "Limit_Pos", "address": "DIN1"},
    ],
    "dout": [
        {"name": "Torch_Relay", "address": "DOUT0"},
    ],
    "ain": [
        {"name": "Arc_V", "address": "AIN0", "scale": 1.0, "unitType": "utVolt"},
    ],
    "aout": [],
}

# Demo unloader info
_unloader_state = {
    "palletStations": [
        {"id": 1, "status": "idle", "loadedParts": 12},
        {"id": 2, "status": "busy", "loadedParts": 3},
    ],
    "unloadedParts": [
        {"partId": 101, "program": "demo.nc", "station": 1, "time": datetime.utcnow().isoformat()},
    ],
}

###############################################################################
# ─── Auth helpers ─────────────────────────────────────────────────────────────
###############################################################################

def _authenticate_user(u: str, p: str) -> bool:
    return _fake_users.get(u) == p


def _create_token(sub: str, expires: timedelta) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + expires}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def current_user(token: str = Depends(oauth2_scheme)) -> str:
    cred_exc = HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW‑Authenticate": "Bearer"})
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = data.get("sub")
    except JWTError:
        raise cred_exc from None
    if user not in _fake_users:
        raise cred_exc
    return user

###############################################################################
# ─── Background ticker – 1 Hz simulation ─────────────────────────────────────
###############################################################################

async def _ticker():
    while True:
        state.step(1.0)
        await asyncio.sleep(1)


@app.on_event("startup")
async def _bg():
    asyncio.create_task(_ticker())

###############################################################################
# ─── Utility functions ───────────────────────────────────────────────────────
###############################################################################

_INCH_TO_MM = 25.4


def _convert_units(obj: Any, units: str):
    """Recursively convert utLength & utThickness fields between imperial <-> metric."""
    if units == "default" or units not in {"metric", "imperial"}:
        return obj

    factor = _INCH_TO_MM if units == "metric" else 1 / _INCH_TO_MM

    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if k in {"pos", "distance", "materialThickness", "commandedPos", "actualPos"} and isinstance(v, (int, float)):
                new[k] = round(v * factor, 4)
            else:
                new[k] = _convert_units(v, units)
        return new
    elif isinstance(obj, list):
        return [_convert_units(v, units) for v in obj]
    else:
        return obj


def _filter_fields(data: Dict[str, Any], patterns: List[str]) -> Dict[str, Any]:
    if not patterns:
        return data

    def match(path: str) -> bool:
        return any(fnmatch.fnmatch(path, pat) for pat in patterns)

    def walk(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                p = f"{prefix}{k}"
                if match(p):
                    out[k] = v
                else:
                    child = walk(v, f"{p}.")
                    if child:
                        out[k] = child
            return out
        elif isinstance(obj, list):
            lst = []
            for i, v in enumerate(obj):
                child = walk(v, f"{prefix}{i}.")
                if child:
                    lst.append(child)
            return lst
        else:
            return None

    return walk(data) or {}


def _parse_iso(d: Optional[str], default: date) -> date:
    if not d:
        return default
    try:
        return datetime.fromisoformat(d).date()
    except ValueError:
        raise HTTPException(400, "Invalid date format, expected ISO‑8601 yyyy-mm-dd") from None

###############################################################################
# ─── REST API ────────────────────────────────────────────────────────────────
###############################################################################
# Auth
@app.post("/rest/v1/auth", response_model=Token, summary="Login to obtain JWT")
async def login(body: AuthRequest):
    if not _authenticate_user(body.username, body.password):
        raise HTTPException(401, "Incorrect username or password")
    tok = _create_token(body.username, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return Token(access_token=tok)


@app.post("/rest/v1/auth/change-password", summary="Change password and get new JWT")
async def change_password(body: ChangePasswordRequest, user: str = Depends(current_user)):
    if not _authenticate_user(user, body.old_password):
        raise HTTPException(400, "Invalid old password")
    _fake_users[user] = body.new_password
    new_tok = _create_token(user, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"success": True, "token": new_tok}

# Info & status
@app.get("/rest/v1/machine-info")
async def machine_info():
    return {
        "programName": state.current_program or "No Active Program",
        "exePath": "C:/TouchCutSim/touchcut_simulator.py",
        "exeName": "touchcut_simulator",
        "exeVersion": app.version,
        "licensedTo": "Simulator User",
        "primeCutInfo": "Simulated",
        "registrationInfo": "Sim-0001",
        "contactInfo": {
            "companyName": "KinetiCode",
            "postalAddr": "123 Plasma Ave",
            "streetAddr": "",
            "countryName": "USA",
            "phone": "+1 555-0100",
            "fax": "",
            "email": "support@example.com",
            "webSite": "https://example.com",
        },
        "tools": [
            {"toolType": "Plasma", "toolName": "Air Plasma", "position": 1},
            {"toolType": "Marker", "toolName": "Laser Pointer", "position": 2},
        ],
    }


@app.get("/rest/v1/status", dependencies=[Depends(current_user)])
async def status(units: str = Query("default", pattern="^(default|metric|imperial)$")):
    d = state.dict()
    d.update(
        {
            "currentUser": "demo",
            "machineState": "Running" if state.program_running else "Idle",
            "programState": "Paused" if state.program_paused else ("Running" if state.program_running else "Idle"),
            "programOverride": "Idle" if not state.program_running else "Running",
            "stateName": "Sim",
            "toolIndex": 0,
            "toolName": "Plasma Torch",
            "toolOn": state.torch_on,
            "spacedCarriages": [1],
            "activeCarriages": [1],
            "heightControlAuto": True,
            "drossBinMode": "Disabled",
            "material": "MS‑10 mm",
            "materialName": "MS",
            "materialThickness": 10,
            "feedrate": 0.0,
            "msgStatus": [],
            "units": units,
        }
    )
    return _convert_units(d, units)

# Jog
@app.post("/rest/v1/jog/{axis}", dependencies=[Depends(current_user)])
async def jog(axis: str, velocity: float = Query(..., description="inches/sec")):
    ax = axis.upper()
    if ax not in state.axes:
        raise HTTPException(404, f"Axis {axis} not found")
    state.axes[ax].vel = velocity
    return {"axis": ax, "vel": velocity}

# Torch
@app.post("/rest/v1/torch/{mode}", dependencies=[Depends(current_user)])
async def torch(mode: str):
    if mode not in ("on", "off"):
        raise HTTPException(400, "mode must be 'on' or 'off'")
    state.torch_on = mode == "on"
    return {"torch_on": state.torch_on}

###############################################################################
# ─── Program management ──────────────────────────────────────────────────────
###############################################################################

class ProgramUpload(BaseModel):
    name: str
    content: str


@app.get("/rest/v1/programs", dependencies=[Depends(current_user)])
async def list_programs(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, gt=0),
    fields: Optional[str] = None,
    units: str = Query("default", pattern="^(default|metric|imperial)$"),
):
    names = list(_programs.keys())[offset : offset + limit]

    items = [
        {
            "name": n,
            "size": len(_programs[n]),
            "modified": datetime.utcnow().isoformat(),
            "nestmapPercentGenerated": 100,
        }
        for n in names
    ]
    data: Dict[str, Any] = {"programs": items, "units": units}
    if fields:
        return _filter_fields(data, fields.split(","))
    return data


@app.post("/rest/v1/programs", dependencies=[Depends(current_user)])
async def upload_program(p: ProgramUpload):
    _programs[p.name] = p.content
    return {"uploaded": p.name, "size": len(p.content)}


@app.delete("/rest/v1/programs/{name}", dependencies=[Depends(current_user)])
async def delete_program(name: str):
    if name not in _programs:
        raise HTTPException(404, "Program not found")
    del _programs[name]
    return {"deleted": name}

# Program control
@app.post("/rest/v1/run/{name}", dependencies=[Depends(current_user)])
async def run_program(name: str):
    if name not in _programs:
        raise HTTPException(404, "Program not found")
    if state.program_running:
        raise HTTPException(409, "A program is already running")
    state.program_running = True
    state.program_paused = False
    state.current_program = name
    state.program_progress = 0.0
    state.torch_on = True
    return {"running": name}


@app.post("/rest/v1/pause", dependencies=[Depends(current_user)])
async def pause():
    if not state.program_running:
        raise HTTPException(409, "No program running")
    state.program_paused = True
    return {"paused": True}


@app.post("/rest/v1/resume", dependencies=[Depends(current_user)])
async def resume():
    if not state.program_running:
        raise HTTPException(409, "No program running")
    state.program_paused = False
    return {"paused": False}


@app.post("/rest/v1/stop", dependencies=[Depends(current_user)])
async def stop():
    if not state.program_running:
        raise HTTPException(409, "No program running")
    state.finish_program()
    return {"stopped": True}

###############################################################################
# ─── Feed override & consumables ─────────────────────────────────────────────
###############################################################################

class FeedOverride(BaseModel):
    override: float = Field(..., ge=0.1, le=2.0)


@app.get("/rest/v1/feedrate", dependencies=[Depends(current_user)])
async def get_feed():
    return {"override": state.feed_override}


@app.post("/rest/v1/feedrate", dependencies=[Depends(current_user)])
async def set_feed(f: FeedOverride):
    state.feed_override = f.override
    return {"override": state.feed_override}


@app.get("/rest/v1/consumables", dependencies=[Depends(current_user)])
async def consumables():
    return {
        "nozzle_life_remaining": state.nozzle_life_remaining,
        "torch_seconds": state.torch_seconds,
    }

###############################################################################
# ─── Monitor & Stats ─────────────────────────────────────────────────────────
###############################################################################

@app.get("/rest/v1/monitor", dependencies=[Depends(current_user)])
async def monitor(
    fields: Optional[str] = None,
    units: str = Query("default", pattern="^(default|metric|imperial)$"),
):
    # Build axis objects
    axes = []
    for ax in state.axes.values():
        axes.append(
            {
                "name": ax.name,
                "address": f"MC,{ax.name}",
                "index": 1,  # demo
                "unitType": "utLength",
                "speedUnitType": "utFeed",
                "scale": 1.0,
                "errorCode": 0,
                "commandedPos": ax.pos,
                "actualPos": ax.pos,
                "posError": 0,
                "errorLimit": 0.1,
                "actualVel": ax.vel,
                "torque": 0,
                "torqueLimit": 150,
                "isEnabled": state.machine_on,
                "isMoving": abs(ax.vel) > 0,
                "isPosErrorInputOn": False,
                "isHomeInputOn": False,
                "isRevLimInputOn": False,
                "isFwdLimInputOn": False,
                "isRevSoftLimOn": False,
                "isFwdSoftLimOn": False,
                "canMoveFwd": True,
                "canMoveRev": True,
                "io": _io_schema,
            }
        )
    data: Dict[str, Any] = {"io": _io_schema, "axes": axes, "units": units}

    if fields:
        data = _filter_fields(data, fields.split(","))
    return _convert_units(data, units)


@app.get("/rest/v1/stats", dependencies=[Depends(current_user)])
async def stats(
    start_date: Optional[str] = Query(None, alias="start-date"),
    end_before: Optional[str] = Query(None, alias="end-before-date"),
    condensed: Optional[bool] = False,
    units: str = Query("default", pattern="^(default|metric|imperial)$"),
):
    today = date.today()
    start = _parse_iso(start_date, today.replace(day=1))
    end = _parse_iso(end_before, today)

    def totals_template(reset: bool = False):
        return {
            "tool": 1,
            "name": "Plasma Torch",
            "time": state.total_tool_on_time if not reset else state.resettable_totals["tool_on_s"],
            "distance": state.total_distance_in if not reset else state.resettable_totals["distance_in"],
            "starts": state.total_starts if not reset else state.resettable_totals["starts"],
            "errors": 0,
            **({"lastReset": state.resettable_totals["last_reset"]} if reset else {}),
            "carriages": [],
        }

    machine = [totals_template(False)]
    machine_reset = [totals_template(True)]

    sessions = [
        {
            "user": "demo",
            "startTime": datetime.combine(start, datetime.min.time()).isoformat(),
            "finishTime": datetime.combine(end, datetime.min.time()).isoformat(),
            "activeTime": (end - start).days * 3600,
            "errorTime": 0,
            "idleTime": 0,
            "xyMovingTime": 0,
            "productiveTime": 0,
            "unproductiveTime": 0,
            "toolOnTime": state.total_tool_on_time,
            "toolTotals": [],
            "programs": [],
            "errors": [],
        }
    ]

    consumables = [
        {
            "date": start.isoformat(),
            "toolName": "Plasma Torch",
            "material": "MS",
            "thickness": 10,
            "process": "O2‑Air",
            "consType": "Nozzle",
            "consID": "NZ-01",
            "pierces": 100,
            "distance": 20,
            "time": 600,
        }
    ]

    data: Dict[str, Any] = {
        "machine": machine,
        "machineResettable": machine_reset,
        "sessions": sessions,
        "consumables": consumables,
        "units": units,
    }

    if condensed:
        if data["machine"][0]["distance"] == 0:
            data["machine"] = []
    return _convert_units(data, units)

###############################################################################
# ─── Unloader ────────────────────────────────────────────────────────────────
###############################################################################

@app.get("/rest/v1/unloader", dependencies=[Depends(current_user)])
async def unloader(
    history: Optional[bool] = False,
    start_date: Optional[str] = Query(None, alias="start-date"),
    end_before: Optional[str] = Query(None, alias="end-before-date"),
    units: str = Query("default", pattern="^(default|metric|imperial)$"),
):
    data = {
        "palletStations": _unloader_state["palletStations"],
        "unloadedParts": _unloader_state["unloadedParts"] if history else [],
        "units": units,
    }
    return data

###############################################################################
# ─── Screenshot ──────────────────────────────────────────────────────────────
###############################################################################

SCREENSHOT_PATH = Path("screenshot.png")


def _make_screenshot(path: Path):
    img = Image.new("RGB", (640, 480), "black")
    draw = ImageDraw.Draw(img)
    txt = (
        f"Program: {state.current_program or '-'}\n"
        f"Progress: {state.program_progress:.0f}%\n"
        f"Torch: {'ON' if state.torch_on else 'OFF'}\n"
        f"Feed‑ovr: {state.feed_override:.2f}×"
    )
    draw.text((10, 10), txt, fill="white", font=ImageFont.load_default())
    img.save(path)


@app.get("/rest/v1/screenshot", dependencies=[Depends(current_user)])
async def screenshot():
    _make_screenshot(SCREENSHOT_PATH)
    with open(SCREENSHOT_PATH, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode("ascii")
    return {"file": SCREENSHOT_PATH.name, "bytes": encoded}

###############################################################################
# ─── Reset‑helper (not part of TouchCut spec) ────────────────────────────────
###############################################################################

@app.post("/rest/v1/reset-totals", dependencies=[Depends(current_user)], include_in_schema=False)
async def reset_totals():
    state.resettable_totals = {
        "distance_in": 0.0,
        "tool_on_s": 0.0,
        "starts": 0,
        "last_reset": datetime.utcnow().isoformat(),
    }
    return {"reset": True}
