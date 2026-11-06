"""
RedBlue — Adversarial Execution Intelligence Platform
FastAPI Backend Server

REST API + WebSocket for real-time simulation streaming.
Serves the frontend dashboard as static files.
Enhanced with: Real market data, alpha leakage cost, market risk panel,
predictability heatmap, shadow mode, and intensity control.
"""
import asyncio
import json
import time
import os
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import copy as _copy

from engine.order_flow import OrderFlowGenerator
from engine.attacker import AttackerAgent
from engine.defender import DefenderAgent, DefenderAction
from engine.adversarial_defender import AdversarialDefender
from engine.metrics import MetricsEngine
from engine.market_data import MarketDataEngine
from engine.ml_models import StrategyClassifier, DefenseDiscriminator
import pandas as pd
import io
import uuid
from fastapi import UploadFile, File


# ============================================================
# Simulation State
# ============================================================
class SimulationState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.running = False
        self.speed = 1.0
        self.tick_count = 0
        self.attacker = AttackerAgent()
        self.defender = DefenderAgent(aggressiveness=0.7)
        self.metrics = MetricsEngine()
        self.market = MarketDataEngine()
        self.undefended_orders = []
        self.defended_orders = []
        # SINGLE generator is the ground truth. The "defended" stream is
        # always a deep copy of the SAME child orders, run through the
        # Defender — never a second, independently-randomized generator.
        # This keeps the before/after comparison apples-to-apples: both streams
        # share the same slice pattern, direction and symbol.
        self.gen = OrderFlowGenerator(slice_pattern="TWAP")
        # Leakage tracking
        self.total_leakage_cost = 0.0
        self.leakage_today = 0.0
        self.leakage_history = []  # (tick, cost) pairs
        # Heatmap data: time_bucket x size_bucket -> accuracy scores
        self.heatmap_data = {}
        # Strategy-level tracking
        self.strategy_scores = {}
        # Trained ML models (see engine/ml_models.py). These load from disk
        # if scripts/train_and_evaluate.py has been run; otherwise `trained`
        # stays False and the API reports that honestly instead of a number.
        self.strategy_classifier = StrategyClassifier()
        self.discriminator = DefenseDiscriminator()
        # Discriminator-guided defense (engine/adversarial_defender.py).
        # Off by default, and can only be turned on once a DefenseDiscriminator
        # has actually been trained (scripts/train_and_evaluate.py) — there is
        # nothing to optimize against otherwise. See configure_simulation().
        self.adversarial_defender: Optional[AdversarialDefender] = None
        self.use_adversarial_defender = False

    def new_parent_orders(self):
        import random
        patterns = ["TWAP", "VWAP", "ICEBERG"]
        symbol = random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"])
        self.gen = OrderFlowGenerator(
            slice_pattern=random.choice(patterns),
            symbol=symbol,
            base_price=self.market.get_price(symbol),
        )
        self.defender.reset_profile()
        if self.adversarial_defender is not None:
            self.adversarial_defender.reset()


sim = SimulationState()
ws_clients: List[WebSocket] = []


# ============================================================
# FastAPI App
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[RedBlue] Server starting up...")
    # Initialize market data
    sim.market.initialize()
    print(f"[RedBlue] Market data initialized. VIX={sim.market.get_vix():.1f}")
    # Use real market prices for the initial generator
    sim.gen = OrderFlowGenerator(
        slice_pattern="TWAP", symbol="AAPL",
        base_price=sim.market.get_price("AAPL")
    )
    # Auto-start simulation on bootup so it "sends signals" immediately
    sim.running = True
    asyncio.create_task(simulation_loop())
    print("[RedBlue] Simulation loop started.")
    yield
    print("[RedBlue] Server shutting down...")

app = FastAPI(
    title="RedBlue — Adversarial Execution Intelligence",
    version="0.2.0-poc",
    lifespan=lifespan,
)

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REST API Endpoints
# ============================================================
class ConfigRequest(BaseModel):
    aggressiveness: Optional[float] = None
    speed: Optional[float] = None
    intensity: Optional[int] = None
    use_adversarial: Optional[bool] = None


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.2.0-poc",
        "engine": "RedBlue Adversarial Execution Intelligence",
        "market_data": "active",
        "vix": sim.market.get_vix(),
        "regime": sim.market.get_market_regime(),
    }


@app.post("/api/simulation/start")
async def start_simulation():
    if not sim.running:
        sim.running = True
        asyncio.create_task(simulation_loop())
    return {"status": "running", "tick": sim.tick_count}


@app.post("/api/simulation/stop")
async def stop_simulation():
    sim.running = False
    return {"status": "stopped", "tick": sim.tick_count}


@app.post("/api/simulation/reset")
async def reset_simulation():
    sim.reset()
    sim.market.initialize()
    return {"status": "reset"}


@app.post("/api/simulation/configure")
async def configure_simulation(config: ConfigRequest):
    if config.aggressiveness is not None:
        sim.defender.set_aggressiveness(config.aggressiveness)
        if sim.adversarial_defender is not None:
            sim.adversarial_defender.aggressiveness = max(0.0, min(1.0, float(config.aggressiveness)))
    if config.speed is not None:
        sim.speed = max(0.5, min(config.speed, 10.0))
    if config.intensity is not None:
        sim.defender.set_intensity(config.intensity)

    adversarial_note = None
    if config.use_adversarial is not None:
        if config.use_adversarial:
            if not sim.discriminator.trained:
                sim.use_adversarial_defender = False
                adversarial_note = ("DefenseDiscriminator has not been trained yet "
                                     "(run scripts/train_and_evaluate.py) — there is nothing "
                                     "for the adversarial defender to optimize against, so the "
                                     "fixed-recipe DefenderAgent stays active instead.")
            else:
                if sim.adversarial_defender is None:
                    sim.adversarial_defender = AdversarialDefender(
                        sim.discriminator, aggressiveness=sim.defender.aggressiveness
                    )
                sim.use_adversarial_defender = True
        else:
            sim.use_adversarial_defender = False

    return {
        "aggressiveness": sim.defender.aggressiveness,
        "speed": sim.speed,
        "intensity": sim.defender.intensity_level,
        "use_adversarial": sim.use_adversarial_defender,
        "adversarial_available": sim.discriminator.trained,
        "adversarial_note": adversarial_note,
    }


@app.get("/api/benchmarks")
async def get_benchmarks():
    report = sim.metrics.generate_benchmark_report(sim.defender.actions)
    return report


@app.get("/api/audit-log")
async def get_audit_log(limit: int = Query(default=50, le=200)):
    return sim.defender.compliance.get_audit_log(limit=limit)


SAMPLES_DIR = os.path.abspath(os.path.join("data", "audit_samples"))


def _resolve_sample_path(filename: str) -> Optional[str]:
    """Resolve a sample filename safely, rejecting path traversal attempts."""
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".csv"):
        return None
    path = os.path.abspath(os.path.join(SAMPLES_DIR, safe_name))
    if os.path.commonpath([path, SAMPLES_DIR]) != SAMPLES_DIR:
        return None
    return path


@app.get("/api/audit/samples/{filename}")
async def download_sample(filename: str):
    path = _resolve_sample_path(filename)
    if path and os.path.exists(path):
        return FileResponse(path, media_type='text/csv', filename=os.path.basename(path))
    return {"error": "File not found"}


@app.post("/api/audit/analyze-sample/{filename}")
async def analyze_sample(filename: str):
    path = _resolve_sample_path(filename)
    if not path or not os.path.exists(path):
        return {"error": "Sample not found"}
    
    df = pd.read_csv(path)
    # Re-use the logic from upload_audit (DRY)
    return await process_audit_df(df)


async def process_audit_df(df: pd.DataFrame):
    # Normalise columns
    df.columns = [c.strip().lower() for c in df.columns]
    required = ['timestamp','side','size','venue']
    missing = [r for r in required if r not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}
    
    from engine.order_flow import ChildOrder
    import uuid
    orders = []
    for _, row in df.head(500).iterrows():
        try:
            orders.append(ChildOrder(
                id=str(uuid.uuid4())[:10],
                timestamp=float(row.get('timestamp', 0))/1000,
                symbol=str(row.get('symbol','AAPL')),
                direction='BUY' if str(row.get('side','')).upper() in ['BUY','B','1'] else 'SELL',
                size=max(1, int(float(row.get('size',100)))),
                price=float(row.get('price',150)),
                venue=str(row.get('venue','NYSE')),
                timing_ms=500.0,
                parent_id='audit',
            ))
        except Exception:
            continue
            
    if len(orders) < 5:
        return {"error": "Need at least 5 valid orders"}
        
    analysis = sim.attacker.analyze(orders)
    leakage = sim.market.compute_leakage_cost(
        analysis['detection_score'], 
        int(sum(o.size for o in orders)/len(orders)),
        orders[0].symbol, 
        orders[0].direction
    ) * len(orders)

    # Run the GroupKFold-validated models against this window too, not just
    # the fast heuristic above. If a model hasn't been trained yet (no
    # trained_models/*.pkl on disk), predict() reports trained=False
    # instead of returning a number — see engine/ml_models.py.
    window = orders[-20:] if len(orders) >= 20 else orders
    strategy_pred = sim.strategy_classifier.predict(window)
    discriminator_pred = sim.discriminator.predict(window)

    return {
        'orders_analysed': len(orders),
        'detection_score': analysis['detection_score'],
        'risk_level': analysis['risk_level'],
        'patterns': analysis['patterns'],
        'top_features': analysis['top_features'],
        'estimated_annual_leakage_usd': round(leakage * 252, 0),
        'strategy_classification': analysis['strategy_classification'],
        'message': f"Your execution stream is {analysis['detection_score']:.0f}% readable to adversarial AI.",
        'validated_models': {
            'strategy_classifier': strategy_pred,
            'defense_discriminator': discriminator_pred,
            'note': "These come from the GroupKFold-validated models (engine/ml_models.py), "
                    "not the heuristic above. If 'trained' is false, no model has been fit yet "
                    "(run scripts/train_and_evaluate.py) and these fields are not meaningful.",
        },
    }


MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@app.post("/api/audit/upload")
async def upload_audit(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return {"error": "File too large (max 5MB)"}
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        return {"error": f"Invalid CSV format: {str(e)}"}
    
    return await process_audit_df(df)


@app.get("/api/attacker/analysis")
async def get_attacker_analysis():
    u_analysis = sim.attacker.analyze(sim.undefended_orders) if len(sim.undefended_orders) >= 5 else {}
    d_analysis = sim.attacker.analyze(sim.defended_orders) if len(sim.defended_orders) >= 5 else {}
    return {"undefended": u_analysis, "defended": d_analysis}


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    if sim.metrics.snapshots:
        return sim.metrics.snapshots[-1]
    return {}


# ---- NEW ENDPOINTS ----

@app.get("/api/intelligence/leakage-cost")
async def get_leakage_cost():
    """Dollar estimate of alpha leakage."""
    return {
        "leakage_today": round(sim.leakage_today, 2),
        "leakage_total": round(sim.total_leakage_cost, 2),
        "leakage_per_order": round(sim.total_leakage_cost / max(sim.tick_count, 1), 2),
        "leakage_history": sim.leakage_history[-60:],
        "currency": "USD",
    }


@app.get("/api/intelligence/market-risk")
async def get_market_risk():
    """Market risk panel data."""
    return sim.market.get_market_risk_snapshot()


@app.get("/api/intelligence/heatmap")
async def get_heatmap():
    """Predictability heatmap: time_bucket × size_bucket × accuracy."""
    return sim.heatmap_data


@app.get("/api/market/live")
async def get_market_live():
    """Live market data snapshot for all tracked symbols."""
    return sim.market.get_market_risk_snapshot()


@app.post("/api/orders/shadow")
async def shadow_assess():
    """Shadow mode: heuristic assessment only, no execution modification.

    Everything returned is either a re-analysis of the current order stream
    or explicitly labeled as a synthetic estimate. No regulatory status is
    asserted.
    """
    if not sim.undefended_orders:
        return {"status": "no_data", "message": "Need active simulation data"}

    analysis = sim.attacker.analyze(sim.undefended_orders)
    score = analysis.get("detection_score", 0)

    # Estimated leakage cost uses the SAME calibrated model as the live
    # dashboard's leakage tracker (engine/market_data.py), not a fresh
    # random draw — see that module for the calibration assumptions and
    # their caveats.
    avg_size = int(sum(o.size for o in sim.undefended_orders) / len(sim.undefended_orders))
    last_order = sim.undefended_orders[-1]
    estimated_leakage_per_order = sim.market.compute_leakage_cost(
        score, avg_size, last_order.symbol, last_order.direction
    )

    rec_intensity = sim.market._recommended_intensity()
    cures = {
        0: "No action needed.",
        1: "Enable Defender at intensity L1 (minimal blend toward ambient flow statistics).",
        2: "Enable Defender at intensity L2.",
        3: "Enable Defender at intensity L3 (default).",
        4: "Enable Defender at intensity L4 — note: measured evidence (docs/experiments.md) shows higher "
           "intensity increases detectability against a trained model, even though it changes "
           "the raw heuristic score more. Don't assume higher = safer.",
        5: "Enable Defender at intensity L5 — same caveat as L4, more pronounced.",
    }

    return {
        "execution_unchanged": True,
        "attacker_assessment": {
            "detection_score": score,
            "confidence": analysis.get("confidence", 0),
            "risk_level": analysis.get("risk_level", "LOW"),
            "strategy_classification": analysis.get("strategy_classification", "UNKNOWN"),
            "top_features": analysis.get("top_features", []),
            "note": "This is the fast heuristic score (engine/attacker.py), not the "
                    "GroupKFold-validated model. For the validated number, see "
                    "ml_metrics.defense_discriminator.held_out_auc on the live feed.",
        },
        "estimated_leakage": {
            "per_order_usd": round(estimated_leakage_per_order, 2),
            "basis": "engine/market_data.py's calibrated cost-vs-detection-score model, "
                     "not a live/observed fill cost. Treat as directional, not exact.",
        },
        "recommended_intensity": rec_intensity,
        "recommended_action": cures.get(rec_intensity, "Enable default defenses."),
    }


# ============================================================
# WebSocket — Real-time simulation stream
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            # Keep connection alive, receive control messages
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "start":
                if not sim.running:
                    sim.running = True
                    asyncio.create_task(simulation_loop())
            elif msg.get("action") == "stop":
                sim.running = False
            elif msg.get("action") == "reset":
                sim.reset()
                sim.market.initialize()
            elif msg.get("action") == "configure":
                if "aggressiveness" in msg:
                    sim.defender.set_aggressiveness(msg["aggressiveness"])
                    if sim.adversarial_defender is not None:
                        sim.adversarial_defender.aggressiveness = max(
                            0.0, min(1.0, float(msg["aggressiveness"]))
                        )
                if "speed" in msg:
                    sim.speed = max(0.5, min(msg["speed"], 10.0))
                if "intensity" in msg:
                    sim.defender.set_intensity(int(msg["intensity"]))
                if "use_adversarial" in msg:
                    if msg["use_adversarial"] and sim.discriminator.trained:
                        if sim.adversarial_defender is None:
                            sim.adversarial_defender = AdversarialDefender(
                                sim.discriminator, aggressiveness=sim.defender.aggressiveness
                            )
                        sim.use_adversarial_defender = True
                    else:
                        sim.use_adversarial_defender = False
    except WebSocketDisconnect:
        ws_clients.remove(websocket)
    except Exception:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


async def broadcast(data: dict):
    """Send data to all connected WebSocket clients."""
    dead = []
    msg = json.dumps(data)
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


def _defend_with_adversarial(order):
    """Defend one order with the discriminator-guided AdversarialDefender
    instead of DefenderAgent's fixed recipe, and wrap the result in the same
    DefenderAction/LPA shape DefenderAgent produces — so /api/benchmarks,
    /api/audit-log, and the WebSocket feed work identically regardless of
    which defense actually ran.

    Returns (action, defended_order). `order` itself is not mutated; the
    AdversarialDefender's candidate search returns a new ChildOrder.
    """
    start_ns = time.perf_counter_ns()
    action = DefenderAction()
    action.original = order.model_dump()
    # -1 signals "not on the DefenderAgent L0-L5 scale" to consumers of
    # intensity_level; the adversarial defender has its own aggressiveness
    # knob instead (see AdversarialDefender.aggressiveness).
    action.intensity_level = -1

    best = sim.adversarial_defender.defend(order)

    modifications = []
    if best.size != order.size:
        modifications.append({
            "type": "SIZE_SEARCH", "from": order.size, "to": best.size,
            "rationale": "Candidate (of N sampled) minimizing the trained "
                         "DefenseDiscriminator's p_defended for this window",
        })
    if abs(best.timing_ms - order.timing_ms) > 5:
        modifications.append({
            "type": "TIMING_SEARCH", "from": round(order.timing_ms, 1), "to": round(best.timing_ms, 1),
            "rationale": "Candidate (of N sampled) minimizing the trained "
                         "DefenseDiscriminator's p_defended for this window",
        })
    if best.venue != order.venue:
        modifications.append({
            "type": "VENUE_SEARCH", "from": order.venue, "to": best.venue,
            "rationale": "Candidate (of N sampled) minimizing the trained "
                         "DefenseDiscriminator's p_defended for this window",
        })

    original_price = action.original["price"]
    price_dev_bps = (
        abs(best.price - original_price) / original_price * 10_000 if original_price > 0 else 0
    )
    latency_ns = time.perf_counter_ns() - start_ns

    # Reuse DefenderAgent's ComplianceEngine so the adversarial defender's
    # actions show up in the SAME audit trail (/api/audit-log), rather than
    # a second, invisible log.
    lpa = sim.defender.compliance.generate_lpa(
        order_id=best.id,
        modifications=[m["type"] for m in modifications],
        vwap_deviation_bps=price_dev_bps,
        max_allowed_bps=sim.adversarial_defender.max_vwap_deviation_bps,
        latency_ns=latency_ns,
        recent_orders=None,
    )

    action.modified = best.model_dump()
    action.modifications = modifications
    action.lpa = lpa
    action.latency_ns = latency_ns
    action.latency_ms = round(latency_ns / 1e6, 4)
    sim.defender.actions.append(action)
    return action, best


# ============================================================
# Simulation Loop
# ============================================================
async def simulation_loop():
    while sim.running:
        try:
            tick_start = time.perf_counter_ns()

            # Tick market data (micro-price updates)
            sim.market.tick_update()

            # Generate ONE order from the single ground-truth stream, then
            # derive the "defended" version as a deep copy of that SAME
            # order run through the Defender. Both streams are now always
            # the same underlying strategy/parent order — the only
            # difference is whether defend() touched it. This is what makes
            # the before/after comparison below actually mean something.
            raw_order = sim.gen.generate_child_order(defended=False)
            last_action = None
            if raw_order:
                raw_order.price = sim.market.get_price(raw_order.symbol)
                sim.undefended_orders.append(raw_order)

                d_order = _copy.deepcopy(raw_order)
                if sim.use_adversarial_defender and sim.adversarial_defender is not None:
                    last_action, d_order = _defend_with_adversarial(d_order)
                else:
                    d_analysis_pre = sim.attacker.analyze(sim.defended_orders + [d_order])
                    last_action = sim.defender.defend(d_order, d_analysis_pre)
                sim.defended_orders.append(d_order)

            # Run attacker on both streams (same underlying parent order now)
            u_analysis = sim.attacker.analyze(sim.undefended_orders)
            d_analysis = sim.attacker.analyze(sim.defended_orders)

            # Compute metrics
            metrics_snapshot = sim.metrics.compute(
                sim.undefended_orders, sim.defended_orders,
                {"undefended": u_analysis, "defended": d_analysis}
            )

            sim.tick_count += 1

            # Compute alpha leakage cost
            if raw_order and u_analysis.get("detection_score", 0) > 50:
                tick_leakage = sim.market.compute_leakage_cost(
                    u_analysis["detection_score"], raw_order.size,
                    raw_order.symbol, raw_order.direction
                )
                sim.leakage_today += tick_leakage
                sim.total_leakage_cost += tick_leakage
                sim.leakage_history.append({"tick": sim.tick_count, "cost": round(tick_leakage, 2)})
                if len(sim.leakage_history) > 200:
                    sim.leakage_history = sim.leakage_history[-200:]

            # Update heatmap data
            if raw_order:
                time_bucket = (sim.tick_count % 13)  # 13 half-hour buckets
                size_bucket = min(4, raw_order.size // 5000)
                key = f"{time_bucket}_{size_bucket}"
                if key not in sim.heatmap_data:
                    sim.heatmap_data[key] = {"count": 0, "total_score": 0}
                sim.heatmap_data[key]["count"] += 1
                sim.heatmap_data[key]["total_score"] += u_analysis.get("detection_score", 0)
                sim.heatmap_data[key]["avg_score"] = round(
                    sim.heatmap_data[key]["total_score"] / sim.heatmap_data[key]["count"], 1
                )
                sim.heatmap_data[key]["time_bucket"] = time_bucket
                sim.heatmap_data[key]["size_bucket"] = size_bucket

            # Update strategy tracking
            strategy = u_analysis.get("strategy_classification", "UNKNOWN")
            if strategy not in sim.strategy_scores:
                sim.strategy_scores[strategy] = {"count": 0, "total_score": 0}
            sim.strategy_scores[strategy]["count"] += 1
            sim.strategy_scores[strategy]["total_score"] += u_analysis.get("detection_score", 0)
            sim.strategy_scores[strategy]["avg_score"] = round(
                sim.strategy_scores[strategy]["total_score"] / sim.strategy_scores[strategy]["count"], 1
            )

            # Check if the parent order is complete — both streams share one
            # generator now, so there is only one progress value to check.
            if sim.gen.progress >= 1.0:
                sim.new_parent_orders()

            # Prevent unbounded memory/CPU growth
            max_history = 200
            if len(sim.undefended_orders) > max_history:
                sim.undefended_orders = sim.undefended_orders[-max_history:]
            if len(sim.defended_orders) > max_history:
                sim.defended_orders = sim.defended_orders[-max_history:]

            # Live ML readout, using models trained OFFLINE with grouped CV
            # (scripts/train_and_evaluate.py) — never trained online here.
            # Training online from a single running stream would reintroduce
            # exactly the leakage that was removed; see engine/ml_models.py.
            ml_metrics = {
                "strategy_classifier": {
                    "trained": sim.strategy_classifier.trained,
                    "held_out_accuracy": (
                        round(sim.strategy_classifier.last_eval.mean_accuracy, 3)
                        if sim.strategy_classifier.last_eval else None
                    ),
                    "live_prediction": (
                        sim.strategy_classifier.predict(sim.undefended_orders[-20:])
                        if len(sim.undefended_orders) >= 20 else None
                    ),
                },
                "defense_discriminator": {
                    "trained": sim.discriminator.trained,
                    "held_out_auc": (
                        round(sim.discriminator.last_eval.mean_auc, 3)
                        if sim.discriminator.last_eval and sim.discriminator.last_eval.mean_auc is not None
                        else None
                    ),
                    "live_p_defended": (
                        sim.discriminator.predict(sim.defended_orders[-20:])
                        if len(sim.defended_orders) >= 20 else None
                    ),
                    "note": (
                        "held_out_auc is from scripts/train_and_evaluate.py's GroupKFold "
                        "evaluation, not measured live. If null, run that script first."
                    ),
                },
            }

            # Broadcast to WebSocket clients
            market_risk = sim.market.get_market_risk_snapshot()
            payload = {
                "type": "tick",
                "tick": sim.tick_count,
                "metrics": metrics_snapshot,
                "undefended_analysis": u_analysis,
                "defended_analysis": d_analysis,
                "last_action": last_action.to_dict() if last_action else None,
                "audit_log": sim.defender.compliance.get_audit_log(limit=50),
                "progress": {
                    "undefended": round(sim.gen.progress, 4),
                    "defended": round(sim.gen.progress, 4),
                },
                # New data
                "market_risk": market_risk,
                "leakage": {
                    "today": round(sim.leakage_today, 2),
                    "total": round(sim.total_leakage_cost, 2),
                    "per_order": round(sim.total_leakage_cost / max(sim.tick_count, 1), 2),
                },
                "strategy_scores": sim.strategy_scores,
                "heatmap_summary": {
                    k: v["avg_score"] for k, v in sim.heatmap_data.items()
                },
                "ml_metrics": ml_metrics,
            }
            await broadcast(payload)

            # Pace the simulation and always yield
            tick_ns = time.perf_counter_ns() - tick_start
            delay = max(0.05, 0.3 / sim.speed) - (tick_ns / 1e9)
            await asyncio.sleep(max(0.01, delay))
        except Exception as e:
            print(f"[RedBlue] Error in simulation loop: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)


# ============================================================
# Serve Frontend
# ============================================================
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.get("/demo")
async def serve_demo():
    return FileResponse("static/demo.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    dev_mode = os.getenv("DEV_RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=dev_mode)
