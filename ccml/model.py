"""The ML model: one gradient-boosted regressor per strike choice, each predicting the
next month's call overlay return from the stock's and the market's features.

Decision rule: write the call at the strike with the highest predicted overlay return,
or write nothing when no strike is predicted to beat MIN_EDGE.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES, STRIKES

NO_CALL = "No call"
MIN_EDGE = 0.0


class CoveredCallModel:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict[str, HistGradientBoostingRegressor] = {}

    def _new(self):
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31,
                                             min_samples_leaf=300, l2_regularization=1.0, random_state=self.seed)

    def fit(self, panel: pd.DataFrame):
        X = panel[FEATURES].values
        for name in STRIKES:
            self.models[name] = self._new().fit(X, panel[f"ov_{name}"].values)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].values
        return pd.DataFrame({name: m.predict(X) for name, m in self.models.items()}, index=df.index)

    @staticmethod
    def choose(pred: pd.DataFrame, min_edge: float = MIN_EDGE) -> pd.Series:
        best = pred.idxmax(axis=1)
        return best.where(pred.max(axis=1) > min_edge, NO_CALL)

    def save(self, path: Path):
        joblib.dump({"seed": self.seed, "models": self.models}, path)

    @classmethod
    def load(cls, path: Path):
        s = joblib.load(path)
        m = cls(s["seed"])
        m.models = s["models"]
        return m

    def importances(self, panel: pd.DataFrame, n: int = 20000) -> pd.Series:
        """Permutation importance averaged over strikes (drop in R^2 when a feature is shuffled)."""
        from sklearn.inspection import permutation_importance
        sample = panel.sample(min(n, len(panel)), random_state=self.seed)
        X = sample[FEATURES].values
        imp = np.zeros(len(FEATURES))
        for name, m in self.models.items():
            imp += permutation_importance(m, X, sample[f"ov_{name}"].values, n_repeats=3,
                                          random_state=self.seed).importances_mean
        return pd.Series(imp / len(self.models), index=FEATURES).sort_values(ascending=False)
