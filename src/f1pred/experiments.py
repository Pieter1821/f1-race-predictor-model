"""Experiment log: every experiment, failures included, appended to reports/experiments.csv."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1pred.config import PROJECT_ROOT

LOG = PROJECT_ROOT / "reports" / "experiments.csv"


def bootstrap_ci(diff: pd.Series, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of per-race differences, resampling races with replacement."""
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff.to_numpy(), len(diff)).mean() for _ in range(n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def log(change: str, seasons, metrics: dict, ci: tuple[float, float] | None, decision: str, notes: str = "") -> None:
    row = {"date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "change": change,
           "seasons": " ".join(map(str, seasons)), **{k: round(float(v), 4) for k, v in metrics.items()},
           "ci_low": None if ci is None else round(ci[0], 4), "ci_high": None if ci is None else round(ci[1], 4),
           "decision": decision, "notes": notes}
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)
