import numpy as np
import pytest
from fastapi.testclient import TestClient

from pmtrader.api import app
from pmtrader.pricing import ev_guardrail

c = TestClient(app)
POS, NEG = 0.01, -0.01  # an edge and a loss, per contract per $1


@pytest.mark.parametrize("pos,target,ev_yes,ev_no,expected", [
    (0, 25, NEG, NEG, 0),        # buying YES with no edge: blocked
    (0, 25, POS, NEG, 25),       # buying YES with edge: allowed
    (0, -50, NEG, NEG, 0),       # buying NO with no edge: blocked
    (0, -50, NEG, POS, -50),     # buying NO with edge: allowed
    (40, 100, NEG, NEG, 40),     # adding to YES without edge: stays put
    (40, 0, NEG, NEG, 0),        # selling YES back to flat: always allowed
    (-40, 0, NEG, NEG, 0),       # selling NO back to flat: always allowed
    (40, -50, NEG, NEG, 0),      # flip YES -> NO without NO edge: sell YES only
    (40, -50, NEG, POS, -50),    # flip with NO edge: whole flip allowed
    (40, 20, NEG, NEG, 20),      # trimming YES: allowed
    (-40, -100, POS, NEG, -40),  # adding NO without NO edge (YES edge irrelevant): stays put
])
def test_guardrail_rules(pos, target, ev_yes, ev_no, expected):
    assert ev_guardrail(pos, target, ev_yes, ev_no) == expected


def test_guardrail_min_edge_and_arrays():
    assert ev_guardrail(0, 25, 0.004, NEG, min_edge=0.005) == 0
    assert ev_guardrail(0, 25, 0.006, NEG, min_edge=0.005) == 25
    out = ev_guardrail(np.array([0, 0, 40]), np.array([25, -25, -50]),
                       np.array([NEG, NEG, NEG]), np.array([NEG, POS, NEG]))
    assert out.tolist() == [0, -25, 0]


def test_recommend_blocks_buy_at_one_dollar():
    # The live Oklahoma City case: YES 99c / $1.00 late in the game, where the agent
    # wanted to buy 25 YES although no YES purchase at $1.00 can profit.
    body = {"yes_bid": 0.99, "yes_ask": 1.0, "prev_mid": 0.96, "time_frac": 0.1, "position": 0, "max_position": 100}
    raw = c.post("/api/recommend", json={**body, "ev_guardrail": False}).json()
    guarded = c.post("/api/recommend", json=body).json()
    assert raw["pricing"]["ev_yes"] < 0 and raw["pricing"]["ev_no"] < 0
    assert raw["target_position"] > 0                       # the raw agent buys YES
    assert guarded["target_position"] == 0 and guarded["order"] == "No trade"
    assert guarded["guardrail"]["applied"] and guarded["agent_action"] == raw["action"]
    assert "Blocked buying YES" in guarded["guardrail"]["reason"]


def test_recommend_unchanged_when_trade_has_edge():
    body = {"yes_bid": 0.41, "yes_ask": 0.43, "signal_prob": 0.75, "prob_override": 0.75, "time_frac": 0.5,
            "position": 0, "max_position": 100}
    r = c.post("/api/recommend", json=body).json()
    assert not r["guardrail"]["applied"]
    assert r["target_position"] == r["agent_target_position"]


def test_backtest_guarded_agent_never_buys_negative_ev():
    b = c.post("/api/backtest", json={"n_markets": 300, "seed": 2}).json()["results"]
    assert "PPO + EV guardrail" in b
    assert b["PPO + EV guardrail"]["contracts_traded"] <= b["PPO agent (RL)"]["contracts_traded"]
