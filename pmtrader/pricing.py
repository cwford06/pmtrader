"""Expected-value pricing for binary prediction-market contracts.

A binary contract pays `payout` (e.g. $1.00) if the event happens (YES) and 0
otherwise. Buying YES at price P costs P; buying NO costs (payout - P) on
exchanges where NO is quoted as the complement (Kalshi, Polymarket).

Given a probability estimate p for YES:
    fair value (YES)    = p * payout
    EV per YES contract = p * payout - ask_yes - fee
    EV per NO contract  = (1 - p) * payout - ask_no - fee
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def kalshi_style_fee(price: float | np.ndarray, payout: float = 1.0, rate: float = 0.07):
    """Per-contract fee proportional to P*(1-P), the shape Kalshi uses.

    Kalshi rounds the *order* fee up to the cent; we keep it continuous here so
    the RL environment stays smooth. Set rate=0 for fee-free venues.
    """
    p = np.clip(np.asarray(price) / payout, 0.0, 1.0)
    return rate * payout * p * (1.0 - p)


@dataclass
class Quote:
    yes_bid: float
    yes_ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.yes_bid + self.yes_ask)

    @classmethod
    def from_mid(cls, mid: float, spread: float, payout: float = 1.0) -> "Quote":
        half = spread / 2.0
        return cls(max(mid - half, 0.0), min(mid + half, payout))


@dataclass
class PriceResult:
    prob_yes: float
    fair_value_yes: float
    fair_value_no: float
    ev_yes: float            # EV per contract buying YES at the ask, after fees
    ev_no: float             # EV per contract buying NO at its ask, after fees
    roi_yes: float           # ev / cost
    roi_no: float
    kelly_yes: float         # full-Kelly fraction of bankroll on YES (>=0)
    kelly_no: float
    side: str                # "YES", "NO" or "PASS"
    edge: float              # prob_yes - market implied prob (mid)

    def to_dict(self) -> dict:
        return asdict(self)


def kelly_fraction(p_win: float, cost: float, payout: float = 1.0) -> float:
    """Full-Kelly fraction for a contract costing `cost` that pays `payout`.

    Net odds b = (payout - cost) / cost, f* = (b*p - (1-p)) / b = (p*payout - cost)/(payout - cost).
    """
    if cost <= 0 or cost >= payout:
        return 0.0
    return max(0.0, (p_win * payout - cost) / (payout - cost))


def price_contract(
    prob_yes: float,
    quote: Quote,
    payout: float = 1.0,
    fee_rate: float = 0.07,
    min_edge: float = 0.0,
) -> PriceResult:
    """Price a binary contract and decide which side (if any) has positive EV."""
    p = float(np.clip(prob_yes, 1e-6, 1 - 1e-6))
    ask_yes = quote.yes_ask
    ask_no = payout - quote.yes_bid          # buying NO = selling YES at the bid
    fee_yes = float(kalshi_style_fee(ask_yes, payout, fee_rate))
    fee_no = float(kalshi_style_fee(ask_no, payout, fee_rate))

    ev_yes = p * payout - ask_yes - fee_yes
    ev_no = (1 - p) * payout - ask_no - fee_no
    roi_yes = ev_yes / max(ask_yes + fee_yes, 1e-9)
    roi_no = ev_no / max(ask_no + fee_no, 1e-9)

    side = "PASS"
    if max(ev_yes, ev_no) > min_edge * payout:
        side = "YES" if ev_yes >= ev_no else "NO"

    return PriceResult(
        prob_yes=p,
        fair_value_yes=p * payout,
        fair_value_no=(1 - p) * payout,
        ev_yes=ev_yes,
        ev_no=ev_no,
        roi_yes=roi_yes,
        roi_no=roi_no,
        kelly_yes=kelly_fraction(p, ask_yes + fee_yes, payout),
        kelly_no=kelly_fraction(1 - p, ask_no + fee_no, payout),
        side=side,
        edge=p - quote.mid / payout,
    )
