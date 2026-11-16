"""FastAPI server: REST API + web dashboard.

    uvicorn pmtrader.api:app --reload      # then open http://127.0.0.1:8000
    Interactive API docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .service import Service

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Prediction Market RL Trader", version="1.0.0",
              description="EV pricing + PPO reinforcement-learning agent for binary prediction-market contracts.")
svc = Service()


class PriceRequest(BaseModel):
    yes_bid: float = Field(..., ge=0, description="Best YES bid, in $ (same units as payout)")
    yes_ask: float = Field(..., ge=0, description="Best YES ask")
    payout: float = Field(1.0, gt=0, description="Contract payout if YES (Kalshi/Polymarket = 1.00)")
    signal_prob: Optional[float] = Field(None, gt=0, lt=1, description="Your external forecast of P(YES), e.g. polls/sportsbook/model. Omit to use the market alone.")
    prev_mid: Optional[float] = Field(None, gt=0, description="Mid price ~3 periods ago (for momentum)")
    time_frac: float = Field(0.5, ge=0, le=1, description="Fraction of the market's life remaining (1 = just opened, 0 = expiring)")
    fee_rate: float = Field(0.07, ge=0, le=0.5, description="Fee coefficient: fee = rate * P * (1-P) per contract")
    min_edge: float = Field(0.0, ge=0, description="Minimum EV per $1 payout to recommend a side")
    prob_override: Optional[float] = Field(None, gt=0, lt=1, description="Price with this probability instead of the model's")

    @model_validator(mode="after")
    def check(self):
        if not (0 <= self.yes_bid < self.yes_ask <= self.payout):
            raise ValueError("need 0 <= yes_bid < yes_ask <= payout")
        return self


class RecommendRequest(PriceRequest):
    position: float = Field(0, description="Current position in contracts: +YES, -NO")
    max_position: float = Field(100, gt=0, description="Max contracts per market")


class BacktestRequest(BaseModel):
    n_markets: int = Field(2000, ge=50, le=20000)
    seed: int = 12345


class TrainRequest(BaseModel):
    iterations: int = Field(200, ge=1, le=2000)
    fresh: bool = Field(False, description="Start from scratch instead of fine-tuning the current agent")
    seed: int = 0


@app.get("/api/health")
def health():
    return {"status": "ok", "agent_trained": svc.trained, "training": svc.train_state["running"]}


@app.get("/api/model")
def model_info():
    return svc.info()


@app.post("/api/price")
def price(req: PriceRequest):
    """Model probability, fair value, and after-fee EV for both sides."""
    return svc.price(req.model_dump())


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """RL agent's trade recommendation given current quotes and position."""
    return svc.recommend(req.model_dump())


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    return svc.run_backtest(req.n_markets, req.seed)


@app.get("/api/simulate")
def simulate_market(seed: int = 0):
    """One simulated market with the agent trading it, for visual inspection."""
    return svc.simulate_market(seed)


@app.post("/api/train")
def train(req: TrainRequest):
    try:
        st = svc.start_training(req.iterations, req.fresh, req.seed)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {k: v for k, v in st.items() if k != "history"}


@app.post("/api/train/stop")
def train_stop():
    svc.stop_training()
    return {"stopping": True}


@app.get("/api/train/status")
def train_status():
    return svc.train_state


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
