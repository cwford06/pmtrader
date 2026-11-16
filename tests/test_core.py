import numpy as np

from pmtrader.nn import MLP
from pmtrader.ppo import PPOAgent, softmax
from pmtrader.pricing import Quote, price_contract, kelly_fraction
from pmtrader.env import TradingEnv
from pmtrader.prob_model import ProbModel
from pmtrader.simulator import SimConfig


def test_mlp_gradient_check():
    rng = np.random.default_rng(0)
    net = MLP([5, 8, 8, 3], rng, out_scale=0.5)
    x = rng.normal(size=(4, 5))
    w = rng.normal(size=(4, 3))
    net.forward(x)
    grads = net.backward(w)
    for pi, p in enumerate(net.params):
        flat = p.reshape(-1)
        for k in rng.choice(flat.size, min(5, flat.size), replace=False):
            old = flat[k]
            flat[k] = old + 1e-6; up = (net.forward(x) * w).sum()
            flat[k] = old - 1e-6; dn = (net.forward(x) * w).sum()
            flat[k] = old
            assert abs((up - dn) / 2e-6 - grads[pi].reshape(-1)[k]) < 1e-5


def test_ppo_actor_gradient_matches_finite_difference():
    """The hand-derived PPO+entropy logit gradient must match numerical diff."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(6, 7)); a = rng.integers(0, 7, 6); A = rng.normal(size=6)
    old_logp = np.log(softmax(z + rng.normal(0, 0.05, z.shape)))[np.arange(6), a]
    clip, ent = 0.2, 0.01

    def loss(z):
        pi = softmax(z); lp = np.log(pi)
        r = np.exp(lp[np.arange(6), a] - old_logp)
        surr = np.minimum(r * A, np.clip(r, 1 - clip, 1 + clip) * A)
        return (-surr - ent * (-(pi * lp).sum(-1))).mean()

    pi = softmax(z); lp = np.log(pi)
    r = np.exp(lp[np.arange(6), a] - old_logp)
    clipped = ((A > 0) & (r > 1 + clip)) | ((A < 0) & (r < 1 - clip))
    dlogp = np.where(clipped, 0, -r * A) / 6
    oh = np.zeros_like(pi); oh[np.arange(6), a] = 1
    H = -(pi * lp).sum(-1)
    g = dlogp[:, None] * (oh - pi) + ent * pi * (lp + H[:, None]) / 6
    num = np.zeros_like(z)
    for i in range(6):
        for j in range(7):
            zp, zm = z.copy(), z.copy(); zp[i, j] += 1e-6; zm[i, j] -= 1e-6
            num[i, j] = (loss(zp) - loss(zm)) / 2e-6
    assert np.allclose(g, num, atol=1e-6)


def test_ev_pricing():
    q = Quote(0.39, 0.41)
    r = price_contract(0.55, q, fee_rate=0.0)
    assert abs(r.ev_yes - (0.55 - 0.41)) < 1e-9
    assert abs(r.ev_no - (0.45 - 0.61)) < 1e-9
    assert r.side == "YES"
    assert price_contract(0.40, q, fee_rate=0.07).side == "PASS"
    assert abs(kelly_fraction(0.6, 0.5) - 0.2) < 1e-9


def test_env_reward_sums_to_settlement_pnl():
    cfg = SimConfig(T=10)
    env = TradingEnv(32, ProbModel(0), cfg, seed=3)
    env.reset()
    total = np.zeros(32)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        _, r, done, info = env.step(rng.integers(0, 8, 32))
        total += info["pnl"]
    # at settlement equity is cash + q*outcome; must equal summed step pnl
    assert np.allclose(total, env.cash + env.q * env.b.outcome)


def test_ev_reward_is_unbiased_decomposition():
    """sum(ev reward) + martingale term == realised pnl, exactly."""
    cfg = SimConfig(T=12)
    env = TradingEnv(64, ProbModel(0), cfg, reward_mode="ev", reward_scale=1.0, max_pos=1.0, seed=5)
    env.reset()
    rng = np.random.default_rng(1)
    ev_sum = np.zeros(64); mart = np.zeros(64); pnl = np.zeros(64)
    done = False
    while not done:
        t = env.t
        _, r, done, info = env.step(rng.integers(0, 8, 64))
        ev_sum += r; pnl += info["pnl"]
        mart += env.q * (env.b.p_true[:, t + 1] - env.b.p_true[:, t])
    assert np.allclose(ev_sum + mart, pnl)
