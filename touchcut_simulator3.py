# touchcut_simulator.py
"""TouchCut Plasma‑table **simulator**

Run:
    uvicorn touchcut_simulator:app --reload

Changes in *v0.3.0*
────────────────────
* Added realistic plasma‑machine endpoints:
  - **Program management**  – upload / list / delete G‑code
  - **Program control**     – run / pause / resume / stop
  - **Feed‑override**       – get & set feed‑rate multiplier
  - **Consumables**         – nozzle/pierce life tracking
* Background ticker now advances program progress and torch time.
* All previous endpoints remain fully backward‑compatible.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import math

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont  # pillow must be installed
import base64

###############################################################################
# Settings & security helpers
###############################################################################

SECRET_KEY = "super‑secret‑key‑change‑me"  # noqa: S105 – demo only!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/rest/v1/auth")
app = FastAPI(title="TouchCut Simulator", version="0.3.0")

###############################################################################
# Pydantic models
###############################################################################

class AuthRequest(BaseModel):
    username: str = Field(..., example="demo")
    password: str = Field(..., example="demo")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AxisState(BaseModel):
    pos: float = 0.0  # inches
    vel: float = 0.0  # inches/sec


class MachineState(BaseModel):
    # motion
    axes: Dict[str, AxisState] = {a: AxisState() for a in ("X", "Y", "Z")}
    # execution
    torch_on: bool = False
    program_running: bool = False
    program_paused: bool = False
    current_program: Optional[str] = None
    program_progress: float = 0.0  # 0‑100 %
    # overrides & life
    feed_override: float = 1.0
    torch_seconds: int = 0  # accumulates torch‑on time
    nozzle_life_remaining: int = 1000  # arbitrary units
    # --- Simulated sensors ---
    temperature_c: float = 25.0
    oxygen_pct: float = 21.0
    air_pressure_psi: float = 90.0
    arc_voltage_v: float = 120.0
    # --- Machine on/off cycle ---
    machine_on: bool = True
    _cycle_time: float = 0.0  # seconds since sim start

    def step(self, dt: float):
        # Update cycle time
        self._cycle_time += dt
        cycle_period = 30 * 60 + 5 * 60  # 30 min on, 5 min off
        cycle_pos = self._cycle_time % cycle_period
        self.machine_on = cycle_pos < (30 * 60)

        # Simulate sensors
        if self.machine_on:
            # Temperature rises slowly when on
            self.temperature_c += 0.01 * dt
            self.temperature_c = min(self.temperature_c, 60.0)
            # Oxygen drops slightly when on
            self.oxygen_pct -= 0.0001 * dt
            self.oxygen_pct = max(self.oxygen_pct, 19.0)
            # Air pressure fluctuates
            self.air_pressure_psi = 90 + 5 * math.sin(self._cycle_time / 60)
            # Arc voltage fluctuates when torch is on
            self.arc_voltage_v = 120 + 10 * math.sin(self._cycle_time / 10)
        else:
            # Temperature cools when off
            self.temperature_c -= 0.02 * dt
            self.temperature_c = max(self.temperature_c, 20.0)
            # Oxygen recovers
            self.oxygen_pct += 0.0002 * dt
            self.oxygen_pct = min(self.oxygen_pct, 21.0)
            # Air pressure stabilizes
            self.air_pressure_psi = 90
            # Arc voltage drops
            self.arc_voltage_v = 0.0

        # Move axes per velocity only if machine is on
        if self.machine_on:
            for axis in self.axes.values():
                axis.pos += axis.vel * dt
        # Advance program simulation
        if self.program_running and not self.program_paused and self.machine_on:
            self.program_progress += 2.0 * self.feed_override  # 50 sec program by default
            if self.program_progress >= 100.0:
                self.finish_program()
        # Torch & consumables accounting
        if self.torch_on and self.machine_on:
            self.torch_seconds += int(dt)
            self.nozzle_life_remaining = max(0, 1000 - self.torch_seconds)

    def finish_program(self):
        self.program_running = False
        self.program_paused = False
        self.torch_on = False
        self.current_program = None
        self.program_progress = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# In‑memory stores
# ──────────────────────────────────────────────────────────────────────────────
_fake_users = {"demo": "demo"}
_programs: Dict[str, str] = {}
state = MachineState()

###############################################################################
# Auth helpers
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
# Background ticker – 1 Hz simulation
###############################################################################

async def _ticker():
    while True:
        state.step(1.0)
        await asyncio.sleep(1)


@app.on_event("startup")
async def _bg():
    asyncio.create_task(_ticker())

###############################################################################
# REST API ─────────────
###############################################################################
# ── Authentication ──

@app.post("/rest/v1/auth", response_model=Token, summary="Login to obtain JWT")
async def login(body: AuthRequest):
    if not _authenticate_user(body.username, body.password):
        raise HTTPException(401, "Incorrect username or password")
    tok = _create_token(body.username, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return Token(access_token=tok)

# ── Info & Status ──

@app.get("/rest/v1/machine-info", dependencies=[Depends(current_user)])
async def machine_info():
    return {"model": "K2500‑Sim", "axes": list(state.axes.keys()), "units": "imperial", "sw_version": app.version}


@app.get("/rest/v1/status", dependencies=[Depends(current_user)])
async def status():
    d = state.dict()
    # Add sensors and machine_on to status output
    d["machine_on"] = state.machine_on
    d["temperature_c"] = round(state.temperature_c, 2)
    d["oxygen_pct"] = round(state.oxygen_pct, 2)
    d["air_pressure_psi"] = round(state.air_pressure_psi, 2)
    d["arc_voltage_v"] = round(state.arc_voltage_v, 2)
    return d

# ── Axis jog ──

@app.post("/rest/v1/jog/{axis}", dependencies=[Depends(current_user)])
async def jog(axis: str, velocity: float):
    ax = axis.upper()
    if ax not in state.axes:
        raise HTTPException(404, f"Axis {axis} not found")
    state.axes[ax].vel = velocity
    return {"axis": ax, "vel": velocity}

# ── Torch control ──

@app.post("/rest/v1/torch/{mode}", dependencies=[Depends(current_user)])
async def torch(mode: str):
    if mode not in ("on", "off"):
        raise HTTPException(400, "mode must be 'on' or 'off'")
    state.torch_on = mode == "on"
    return {"torch_on": state.torch_on}

###############################################################################
# New: Program management
###############################################################################

class ProgramUpload(BaseModel):
    name: str
    content: str


@app.get("/rest/v1/programs", dependencies=[Depends(current_user)])
async def list_programs():
    return {"programs": list(_programs.keys())}


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

###############################################################################
# New: Program control
###############################################################################

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
# New: Feed override & consumables
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
# Screenshot endpoint (unchanged)
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
    return {
        "file": SCREENSHOT_PATH.name,
        "bytes": encoded
    }
