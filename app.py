"""
FastAPI server exposing the Strategic Argument Red-Teaming Environment.
Endpoints:
    GET  /              — Command Center UI (HTML)
    GET  /docs          — Interactive OpenAPI (Swagger UI)
    GET  /health        — Liveness probe (JSON)
    POST /reset         — Start a new debate
    POST /step          — Apply an argument action
    GET  /state         — Read current state without advancing
"""

from __future__ import annotations
import io
from typing import Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

# Import your actual environment and schemas
from environment import DebateEnvironment 
from schema.schemas import DebateAction

app = FastAPI(title="Strategic Argument Command Center", version="1.0.0")
_env = DebateEnvironment()

# ── request / response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    topic: str = Field(default="Universal Basic Income is necessary")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")

class StepRequest(BaseModel):
    action: DebateAction

class EnvResponse(BaseModel):
    observation: dict[str, Any] | None
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


def _format_obs(obs: Any) -> dict[str, Any]:
    """Helper to convert your environment's observation object to a dictionary for JSON."""
    if obs is None:
        return None
    # Assuming these are properties in your environment's observation
    return {
        "opponent_challenge": getattr(obs, "opponent_challenge", "Episode Terminated."),
        "phase": getattr(obs, "phase", "ENDED"),
        "history": getattr(obs, "history", [])
    }

# ── FRONTEND HTML/CSS/JS PAYLOAD ─────────────────────────────────────────────

_DEBUG_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Debate Command Center</title>
  <style>
    :root {
      --bg: #0f172a; --surface: #1e293b; --border: #334155; --text: #f8fafc;
      --muted: #94a3b8; --accent: #0ea5e9; --exec: #0ea5e9; --exec-hover: #0284c7;
      --card-blue: #2563eb; --card-orange: #ea580c; --card-green: #16a34a; --card-grey: #475569;
      --mono: ui-monospace, "Cascadia Code", monospace;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }
    .wrap { max-width: 60rem; margin: 0 auto; padding: 2rem 1rem; }
    
    .page-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }
    h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
    .sub { color: var(--muted); font-size: 0.9rem; margin: 0; }
    .sub a { color: var(--accent); text-decoration: none; } .sub a:hover { text-decoration: underline; }
    .btn-mini { background: #fdfd96; color: #111; padding: 0.35rem 0.65rem; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 0.75rem; text-transform: uppercase; }
    .btn-mini:hover { filter: brightness(1.1); }

    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .metric { padding: 1rem; border-radius: 8px; color: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border-top: 4px solid; background: #1e293b; }
    .metric .m-label { font-size: 0.7rem; text-transform: uppercase; color: var(--muted); font-weight: bold; }
    .metric .m-val { font-size: 1.5rem; font-weight: bold; margin-top: 0.25rem; }
    
    label { font-size: 0.8rem; color: var(--muted); font-weight: bold; display: block; margin-bottom: 0.4rem; text-transform: uppercase; }
    select, input, textarea { width: 100%; padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border); background: #0f172a; color: var(--text); font-family: inherit; margin-bottom: 1rem;}
    textarea { min-height: 80px; resize: vertical; }
    
    .btn-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.5rem; }
    button { padding: 0.75rem 1.25rem; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; text-transform: uppercase; font-size: 0.8rem; }
    .btn-exec { background: var(--exec); color: #fff; flex: 2; }
    .btn-exec:hover { background: var(--exec-hover); }
    .btn-sec { background: var(--card-grey); color: #fff; flex: 1; }
    .btn-sec:hover { background: #334155; }
    .btn-auto { background: #047857; color: #fff; flex: 2; border: 1px solid #10b981; }
    .btn-auto:hover { background: #065f46; }
    
    .ticket-pills { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    .pill { background: #334155; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-family: var(--mono); border: 1px solid #475569;}
    .opponent-text { background: #0f172a; padding: 1rem; border-radius: 6px; border: 1px solid var(--border); line-height: 1.5; color: #e2e8f0; white-space: pre-wrap;}
    
    .timeline { max-height: 400px; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; padding-right: 0.5rem; }
    .tl-item { background: #0f172a; border: 1px solid var(--border); padding: 1rem; border-radius: 6px; }
    .tl-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem;}
    .tl-reward { font-weight: bold; font-family: var(--mono); }
    .tl-pos { color: #10b981; } .tl-neg { color: #ef4444; }
    .tl-body { font-size: 0.85rem; line-height: 1.4; color: var(--muted); }
    .tl-body strong { color: #e2e8f0; }

    .tag { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: bold; }
    pre { background: #0f172a; padding: 1rem; border-radius: 6px; border: 1px solid var(--border); overflow-x: auto; font-family: var(--mono); font-size: 0.8rem; margin: 0; color: #e2e8f0; }
    .debug-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 1.25rem; }
    
    #toast { position: fixed; top: 1rem; right: 1rem; background: #ef4444; color: white; padding: 1rem; border-radius: 6px; display: none; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    ::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-track { background: #0f172a; } ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  </style>
</head>
<body>
  <div id="toast"></div>
  <div class="wrap">
    
    <header class="page-head">
      <div>
        <h1>Strategic Argument Red-Teaming</h1>
        <p class="sub">OpenEnv Command Center • OPENING → CLOSING · <a href="/health">/health</a></p>
      </div>
      <a class="btn-mini" href="/docs" target="_blank">Docs</a>
    </header>
    
    <div class="metrics">
      <div class="metric" style="border-color: var(--card-blue);">
        <div class="m-label">Step Reward</div><div class="m-val" id="mStep">0.00</div>
      </div>
      <div class="metric" style="border-color: var(--card-orange);">
        <div class="m-label">Total Reward</div><div class="m-val" id="mTotal">0.00</div>
      </div>
      <div class="metric" style="border-color: var(--card-green);">
        <div class="m-label">Current Phase</div><div class="m-val" id="mPhase" style="font-size: 1.1rem; padding-top: 0.4rem;">STANDBY</div>
      </div>
      <div class="metric" style="border-color: var(--card-grey);">
        <div class="m-label">Status</div><div class="m-val" id="mStatus" style="font-size: 1.1rem; padding-top: 0.4rem;">ACTIVE</div>
      </div>
    </div>

    <div class="panel">
      <div class="ticket-pills">
        <div class="pill">DIFFICULTY: <span id="pDiff">medium</span></div>
        <div class="pill">PHASE: <span id="pPhase">STANDBY</span></div>
      </div>
      <label>Latest Opponent Challenge</label>
      <div class="opponent-text" id="oppText">Initialize the environment to begin the debate.</div>
    </div>

    <div class="panel">
      <div style="display: flex; gap: 1rem;">
        <div style="flex: 3;"><label>Topic / Seed</label><input type="text" id="iTopic" value="Universal Basic Income is necessary"></div>
        <div style="flex: 1;"><label>Difficulty</label><select id="iDiff"><option value="easy">Easy</option><option value="medium" selected>Medium</option><option value="hard">Hard</option></select></div>
      </div>
      <label>Phase Tag</label>
      <select id="iTag">
        <option value="OPENING">OPENING</option>
        <option value="CHALLENGE">CHALLENGE</option>
        <option value="REBUTTAL">REBUTTAL</option>
        <option value="CONSOLIDATION">CONSOLIDATION</option>
        <option value="CLOSING">CLOSING</option>
      </select>
      <label>Argument Payload</label>
      <textarea id="iArg" placeholder="Type your logical argument here..."></textarea>
      
      <div class="btn-row">
        <button class="btn-exec" onclick="doStep()">Execute Step</button>
        <button class="btn-sec" onclick="doState()">Get State</button>
        <button class="btn-sec" onclick="doReset()">Reset Env</button>
        <button class="btn-auto" onclick="doAutoFill()">⚡ Auto-Fill Ideal Move</button>
      </div>
    </div>

    <div class="panel">
      <label>Action Timeline</label>
      <div class="timeline" id="timeline">
        <div style="color: var(--muted); text-align: center; padding: 2rem;">No actions yet. Click Reset to begin.</div>
      </div>
    </div>

    <div class="panel">
      <div class="tag">observation</div>
      <pre id="outObs">—</pre>
    </div>
    <div class="debug-grid">
      <div class="panel" style="margin: 0;">
        <div class="tag">reward (response)</div>
        <pre id="outReward">—</pre>
      </div>
      <div class="panel" style="margin: 0;">
        <div class="tag">done</div>
        <pre id="outDone">—</pre>
      </div>
    </div>
    <div class="panel" style="margin-top: 1.25rem;">
      <div class="tag">info</div>
      <pre id="outInfo">—</pre>
    </div>
    <p style="text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 1.5rem;">Built from dropdowns → POST /step. Toggle action type to edit fields.</p>

  </div>

<script>
  let totalReward = 0;
  let currentPhase = "STANDBY";
  let stepCount = 0;

const AUTO_FILL = {
    "STANDBY": { tag: "OPENING", arg: "Universal Basic Income is an essential policy for the future economy. As automation and AI rapidly displace traditional jobs, UBI ensures a fundamental safety net, preventing mass poverty and maintaining consumer demand." },
    
    "OPENING": { tag: "OPENING", arg: "Universal Basic Income is an essential policy for the future economy. As automation and AI rapidly displace traditional jobs, UBI ensures a fundamental safety net, preventing mass poverty and maintaining consumer demand." },
    
    "CHALLENGE": { tag: "CHALLENGE", arg: "While critics argue that UBI is too expensive or causes inflation, economic models show that when funded through progressive taxation—like a wealth tax—it does not trigger hyperinflation. It simply redistributes capital locally." },
    
    "REBUTTAL": { tag: "REBUTTAL", arg: "Furthermore, the assumption that UBI destroys the incentive to work is contradicted by real-world data. Pilot programs demonstrate that freed from survival stress, people invest more time in education and small businesses." },
    
    "CONSOLIDATION": { tag: "CONSOLIDATION", arg: "Looking at the macroeconomic scale, the velocity of money generated by UBI actually creates a robust, bottom-up economic stimulus that far outweighs the initial taxation costs required to implement it." },
    
    "CLOSING": { tag: "CLOSING", arg: "Ultimately, Universal Basic Income is not just a welfare handout; it is a necessary economic stabilizer for the 21st century. It enables human innovation and builds a resilient society against technological disruption." },
    
    "ENDED": { tag: "CLOSING", arg: "The debate is over." }
  };

  const $ = id => document.getElementById(id);
  
  function showToast(msg) {
    const t = $('toast'); t.innerText = msg; t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 4000);
  }

  function doAutoFill() {
    const move = AUTO_FILL[currentPhase] || AUTO_FILL["STANDBY"];
    $('iTag').value = move.tag;
    $('iArg').value = move.arg;
  }

  function updateUI(data, isReset = false) {
    if(isReset) { totalReward = 0; stepCount = 0; $('timeline').innerHTML = ''; }
    
    const obs = data.observation || {};
    const reward = data.reward || 0;
    totalReward += reward;
    currentPhase = obs.phase || "ENDED";
    
    $('mStep').innerText = reward.toFixed(2);
    $('mTotal').innerText = totalReward.toFixed(2);
    $('mPhase').innerText = currentPhase;
    $('mStatus').innerText = data.done ? "TERMINATED" : "ACTIVE";
    
    $('pPhase').innerText = currentPhase;
    $('pDiff').innerText = $('iDiff').value;
    $('oppText').innerText = obs.opponent_challenge || "Episode Terminated.";

    // Update Raw JSON displays
    $('outObs').innerText = data.observation ? JSON.stringify(data.observation, null, 2) : "null";
    $('outReward').innerText = data.reward !== undefined ? data.reward : "—";
    $('outDone').innerText = data.done !== undefined ? data.done : "—";
    $('outInfo').innerText = data.info ? JSON.stringify(data.info, null, 2) : "{}";

    if(data.action_taken) {
      stepCount++;
      const colorCls = reward >= 0 ? "tl-pos" : "tl-neg";
      const sign = reward > 0 ? "+" : "";
      const html = `
        <div class="tl-item">
          <div class="tl-header">
            <strong style="color: #fff;">Step ${stepCount}: ${data.action_taken.phase_tag}</strong>
            <span class="tl-reward ${colorCls}">${sign}${reward.toFixed(2)}</span>
          </div>
          <div class="tl-body">
            <div><strong>Agent:</strong> ${data.action_taken.argument}</div>
            <div style="margin-top: 0.5rem; padding-left: 0.5rem; border-left: 2px solid var(--border);"><strong>Opponent:</strong> ${obs.opponent_challenge || 'None'}</div>
          </div>
        </div>`;
      $('timeline').insertAdjacentHTML('beforeend', html);
      $('timeline').scrollTop = $('timeline').scrollHeight; 
    }
  }

  async function apiCall(endpoint, payload) {
    try {
      const res = await fetch(endpoint, {
        method: payload ? "POST" : "GET",
        headers: { "Content-Type": "application/json" },
        body: payload ? JSON.stringify(payload) : null
      });
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail || "API Error");
      return data;
    } catch (e) { showToast(e.message); throw e; }
  }

  async function doReset() {
    const data = await apiCall("/reset", { topic: $('iTopic').value, difficulty: $('iDiff').value });
    updateUI(data, true);
    $('iArg').value = ""; 
  }

  async function doState() {
    const data = await apiCall("/state");
    updateUI(data);
  }

  async function doStep() {
    if(currentPhase === "STANDBY" || currentPhase === "ENDED") {
      showToast("Please Reset the environment first!"); return;
    }
    const action = { phase_tag: $('iTag').value, argument: $('iArg').value };
    const data = await apiCall("/step", { action });
    data.action_taken = action; 
    updateUI(data);
    $('iArg').value = ""; 
  }
</script>
</body>
</html>
"""

# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Interface"], summary="Web Command Center")
def ui() -> HTMLResponse:
    """Browser debug UI (HTML)."""
    return HTMLResponse(content=_DEBUG_UI_HTML)

@app.get("/health", tags=["System"], summary="Health check")
async def health() -> dict[str, str]:
    """Return 'status: ok' when the server is up."""
    return {"status": "ok"}

@app.post("/reset", response_model=EnvResponse, tags=["Environment"])
async def reset(req: ResetRequest) -> dict[str, Any]:
    """Start a new episode."""
    obs = _env.reset(req.topic)
    return {
        "observation": _format_obs(obs),
        "reward": 0.0,
        "done": False,
        "info": {"difficulty": req.difficulty}
    }

@app.post("/step", response_model=EnvResponse, tags=["Environment"])
async def step(req: StepRequest) -> dict[str, Any]:
    """Apply one agent action."""
    obs = _env.step(req.action)
    return {
        "observation": _format_obs(obs),
        "reward": float(obs.reward),
        "done": obs.done,
        "info": {}
    }

@app.get("/state", response_model=EnvResponse, tags=["Environment"])
async def state() -> dict[str, Any]:
    """Read current state."""
    obs = getattr(_env, "_state", None)
    return {
        "observation": _format_obs(obs),
        "reward": 0.0,
        "done": getattr(obs, "done", False),
        "info": {}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)