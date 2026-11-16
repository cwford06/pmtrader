import time
from fastapi.testclient import TestClient
from pmtrader.api import app

c = TestClient(app)


def test_health_and_index():
    assert c.get("/api/health").json()["status"] == "ok"
    assert "PM" in c.get("/").text


def test_price_and_recommend():
    body = {"yes_bid": 0.41, "yes_ask": 0.43, "signal_prob": 0.55, "time_frac": 0.5}
    p = c.post("/api/price", json=body).json()
    assert 0 < p["prob_yes"] < 1 and p["side"] in ("YES", "NO", "PASS")
    r = c.post("/api/recommend", json={**body, "position": 0, "max_position": 100}).json()
    assert abs(sum(r["action_probs"].values()) - 1) < 1e-6
    assert c.post("/api/price", json={"yes_bid": 0.5, "yes_ask": 0.4}).status_code == 422


def test_simulate_and_backtest():
    s = c.get("/api/simulate?seed=3").json()
    assert len(s["mid"]) == len(s["position"]) == 41
    b = c.post("/api/backtest", json={"n_markets": 200, "seed": 1}).json()
    assert set(b["results"]) == {"PPO agent (RL)", "PPO + EV guardrail", "EV threshold", "Half Kelly"}
