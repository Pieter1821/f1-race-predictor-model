"""Stage 6: Monte Carlo simulation settings, chosen on 2024 dev races only.

    python scripts/stage06_simulation.py dev    # choose target variant, spacing, grid blend and noise
    python scripts/stage06_simulation.py test   # the locked choice vs the naive baseline on 2025-26

Selection rule (fixed before running): lowest brier_total (per-driver Brier: win + podium + points) on dev.
History: v1 (dev 2024, linear spacing) lost to the naive baseline on the official test
(reports/stage06_test_v1.json). v2 adds log spacing and tunes on 2023-24; because v2 was designed after
seeing the v1 test, its 2025-26 numbers are a contaminated second look, not a clean test.
"""
import json
import sys

import numpy as np
import pandas as pd

from f1pred.backtest import DEV_SEASONS, TEST_SEASONS, race_key, walk_forward
from f1pred.config import PROCESSED_DIR, PROJECT_ROOT
from f1pred.models import FINAL_FEATURES, FINISHER_TARGET, TARGET, add_finisher_target, final_model
from f1pred.simulate import SimSettings, add_dnf_probability, grid_slot_baseline, probability_scores, simulate_many

pd.set_option("display.width", 170)
SETTINGS_PATH = PROJECT_ROOT / "reports" / "stage06_sim_settings.json"
phase = sys.argv[1] if sys.argv[1:] in (["dev"], ["test"]) else sys.exit("usage: stage06_simulation.py dev|test")
SIM_DEV_SEASONS = [2023, 2024]  # 2024 alone was unusually open (pole won 46%); two seasons for "how dominant is pole"
seasons = SIM_DEV_SEASONS if phase == "dev" else TEST_SEASONS

table = add_dnf_probability(add_finisher_target(pd.read_parquet(PROCESSED_DIR / "features.parquet")))
history = table[~table["is_target"] & table["position"].notna()]


def scores_for(target: str) -> pd.DataFrame:
    return walk_forward(table, lambda: final_model(target=target), FINAL_FEATURES, seasons, target=target)


def baseline(preds: pd.DataFrame) -> pd.DataFrame:
    keys = race_key(history)
    return pd.concat([grid_slot_baseline(history[keys < k], race)
                      for k, race in preds.groupby(race_key(preds))])


def show(label: str, sim: pd.DataFrame) -> dict:
    s = probability_scores(sim)
    print(f"  {label:<46} brier_total {s['brier_total']:.4f}  winner log loss {s['winner_log_loss']:.3f}  "
          f"P(actual winner) {s['p_actual_winner']:.0%}", flush=True)
    return {"label": label, **s.round(4).to_dict()}


if phase == "dev":
    print(f"DEV: {seasons}\n")
    preds = {"A: official position (DNFs inside the score)": scores_for(TARGET),
             "B: order among finishers (DNFs only in the sim)": scores_for(FINISHER_TARGET)}

    dev = next(iter(preds.values()))
    dnf = dev.assign(dnf=(~dev["finished"].astype(bool) & (dev["status_bucket"] != "dsq")).astype(float))
    print("DNF probability check (sim_dnf_prob):")
    print(f"  predicted average {dnf['sim_dnf_prob'].mean():.3f} vs actual rate {dnf['dnf'].mean():.3f} | "
          f"Brier {((dnf['sim_dnf_prob'] - dnf['dnf']) ** 2).mean():.4f} vs constant-rate Brier "
          f"{((dnf['dnf'].mean() - dnf['dnf']) ** 2).mean():.4f}\n")

    print("Naive baseline:")
    rows = [show("grid-slot history (no model)", baseline(dev))]
    best = None
    for variant, p in preds.items():
        print(f"\n{variant}:")
        for spacing in ("linear", "log"):
            for blend in (0.0, 0.25, 0.5, 0.75, 1.0):
                for sigma in (0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25):
                    st = SimSettings(grid_blend=blend, sigma=sigma, spacing=spacing, n_sims=3000)
                    s = probability_scores(simulate_many(p, st))
                    rows.append({"label": f"{variant[:1]} {spacing} blend={blend} sigma={sigma}", **s.round(4).to_dict()})
                    if best is None or s["brier_total"] < best[0]:
                        best = (s["brier_total"], variant, blend, sigma, spacing)
        sub = pd.DataFrame([r for r in rows if r["label"].startswith(variant[:1] + " ")])
        top = sub.nsmallest(3, "brier_total")[["label", "brier_total", "winner_log_loss", "p_actual_winner"]]
        print(top.to_string(index=False))

    _, variant, blend, sigma, spacing = best
    target = TARGET if variant.startswith("A") else FINISHER_TARGET
    SETTINGS_PATH.write_text(json.dumps({
        "version": 2, "chosen_on": SIM_DEV_SEASONS, "rule": "lowest brier_total", "target": target,
        "grid_blend": blend, "sigma": sigma, "spacing": spacing, "dev_brier_total": round(best[0], 4),
        "test_status": "contaminated: designed after the v1 test, see stage06_test_v1.json"}, indent=2))
    print(f"\nCHOSEN on dev: {variant} | {spacing} grid_blend={blend} sigma={sigma} | saved {SETTINGS_PATH.name}")
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "stage06_dev.csv", index=False)

else:
    chosen = json.loads(SETTINGS_PATH.read_text())
    print(f"SECOND LOOK (contaminated, not a clean test): {seasons} | settings: {chosen}\n")
    preds = scores_for(chosen["target"])
    sim = simulate_many(preds, SimSettings(grid_blend=chosen["grid_blend"], sigma=chosen["sigma"],
                                           spacing=chosen["spacing"]))
    sim.to_parquet(PROCESSED_DIR / "stage06_test_sim.parquet", index=False)
    show("naive grid-slot history", baseline(preds))
    show("v2 simulation (second look)", sim)
    pole = sim[sim["grid"] == 1]
    print(f"\n  pole-sitter P(win) avg {pole['p_win'].mean():.2f} vs actual pole win rate {(pole['position'] == 1).mean():.2f}")
