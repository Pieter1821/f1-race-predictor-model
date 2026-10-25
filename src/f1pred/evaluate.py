"""Part 2 harness: compare a candidate change with the current model on walk-forward 2023-26.

Candidate (README "Rules"): the bootstrap 95% CI of the per-race MAE change excludes zero (better),
it helps in at least 3 of 4 seasons, and no guardrail (Spearman, winner, podium) is clearly worse.
A simplification is a candidate only when its worst case (the CI's upper end) stays within the noise
floor. Candidates become the model only through a separate adoption check, logged as "adopted".
"""
import pandas as pd

from f1pred import experiments
from f1pred.backtest import walk_forward
from f1pred.models import final_model
from f1pred.validate import score_many

SEASONS = [2023, 2024, 2025, 2026]
NOISE_FLOOR = 0.022  # Part 1: range of overall MAE across 5 seeds
GUARDRAILS = {"spearman": +1, "winner_hit": +1, "podium_hit": +1}  # +1 = higher is better


def order_scores(table: pd.DataFrame, features: list[str], make=final_model, seasons=SEASONS,
                 weighting=None) -> pd.DataFrame:
    preds = walk_forward(table, make, features, seasons, weighting)
    return score_many(preds, "pred").set_index(["season", "round"])


def compare(base: pd.DataFrame, cand: pd.DataFrame, change: str, simplification: bool = False,
            notes: str = "", log: bool = True) -> dict:
    d = (cand["mae"] - base["mae"]).dropna()
    lo, hi = experiments.bootstrap_ci(d)
    per_season = d.groupby(level="season").mean()
    better_seasons = int((per_season < 0).sum())
    clearly_worse = []
    for g, sign in GUARDRAILS.items():
        gd = (cand[g] - base[g]).dropna() * sign  # + = better
        glo, ghi = experiments.bootstrap_ci(gd)
        if ghi < 0:
            clearly_worse.append(g)
    # Decisions are "candidate" or "reject": adopting a change is a separate, logged step (Part 3).
    if simplification:
        # Simpler is only allowed when even the worst case (the CI's upper end) stays within the noise floor;
        # a wide CI means "not measured precisely", not "not worse".
        ok = hi <= NOISE_FLOOR and not clearly_worse
        decision = "candidate (simpler; worst case within noise)" if ok else "reject (worst case above noise)"
    else:
        ok = hi < 0 and better_seasons >= 3 and not clearly_worse
        decision = "candidate (beats noise, 3+ seasons)" if ok else "reject"
    result = {"change": change, "mae_change": d.mean(), "ci": (lo, hi), "better_seasons": better_seasons,
              "per_season": per_season.round(3).to_dict(), "guardrails_worse": clearly_worse,
              "decision": decision, "accepted": ok}
    if log:
        experiments.log(change, SEASONS, {"mae_change": d.mean(), "better_seasons": better_seasons,
                                          **{f"mae_change_{s}": v for s, v in per_season.items()}},
                        (lo, hi), decision, notes or ("guardrails worse: " + ",".join(clearly_worse) if clearly_worse else ""))
    return result


def line(r: dict) -> str:
    ps = " ".join(f"{s}:{v:+.3f}" for s, v in r["per_season"].items())
    g = f" | worse: {','.join(r['guardrails_worse'])}" if r["guardrails_worse"] else ""
    return (f"{r['change']:<42} {r['mae_change']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] "
            f"better in {r['better_seasons']}/4 ({ps}){g} -> {r['decision']}")
