"""Part 2, experiment 5b: a structural start-position advantage in the simulation, size learned NESTED.

Before each half-season the bonus (pace head start for P1, or for the whole front row) is chosen on
earlier races only (2023 blocks: no earlier races -> 0, i.e. v1). Judged on win Brier, winner log loss
and polesitter predicted vs actual win rate per season, against v1 (no bonus).

    python scripts/part2_pole_effect.py
"""
import numpy as np
import pandas as pd

from f1pred import config, experiments
from f1pred.backtest import race_key, walk_forward
from f1pred.evaluate import SEASONS
from f1pred.models import final_sim_model
from f1pred.pipeline import history_table
from f1pred.simulate import SimSettings, simulate_many, temper_win

BONUSES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
_, _, _, table = history_table()
pace = walk_forward(table, final_sim_model, config.FEATURES, SEASONS, target=config.SIM_TARGET)
print(f"pace scores for {race_key(pace).nunique()} races; simulating {len(BONUSES)} bonus sizes x 2 modes", flush=True)


def per_race(sim: pd.DataFrame) -> pd.DataFrame:
    won = sim["position"] == 1
    key = [sim["season"], sim["round"]]
    pole = sim["grid"] == 1
    return pd.DataFrame({
        "win_brier": ((sim["p_win"] - won) ** 2).groupby(key).mean(),
        "log_loss": (-np.log(sim.loc[won, "p_win"].clip(1e-4))).groupby([sim.loc[won, "season"], sim.loc[won, "round"]]).mean(),
        "pole_p_win": sim.loc[pole, "p_win"].groupby([sim.loc[pole, "season"], sim.loc[pole, "round"]]).mean(),
        "pole_won": (sim.loc[pole, "position"] == 1).astype(float).groupby([sim.loc[pole, "season"], sim.loc[pole, "round"]]).mean(),
    })


for mode, grid_max in (("pole only", 1), ("front row", 2)):
    runs = {}
    for b in BONUSES:
        s = SimSettings(grid_blend=config.SIM_GRID_BLEND, sigma=config.SIM_SIGMA, spacing=config.SIM_SPACING,
                        pole_bonus=b, bonus_grid_max=grid_max, n_sims=3000)
        runs[b] = per_race(temper_win(simulate_many(pace, s), config.SIM_WIN_TEMPERATURE))
    v1 = runs[0.0]
    keys = v1.index.get_level_values(0) * 100 + v1.index.get_level_values(1)
    picked, parts = {}, []
    for season in SEASONS:
        for half, lo, hi in (("H1", 1, 11), ("H2", 12, 30)):
            mask = (v1.index.get_level_values(0) == season) & (v1.index.get_level_values(1) >= lo) & \
                   (v1.index.get_level_values(1) <= hi)
            if not mask.any():
                continue
            earlier = keys < keys[mask].min()
            b = 0.0 if season == 2023 else min(BONUSES, key=lambda b: runs[b].loc[earlier, "win_brier"].mean())
            picked[f"{season}{half}"] = b
            parts.append(runs[b][mask])
    nested = pd.concat(parts)
    d = nested["win_brier"] - v1["win_brier"]
    lo, hi = experiments.bootstrap_ci(d)
    better = int((d.groupby(level=0).mean() < 0).sum())
    dl = nested["log_loss"] - v1["log_loss"]
    llo, lhi = experiments.bootstrap_ci(dl)
    ok = hi < 0 and better >= 3 and llo <= 0
    decision = "ACCEPT" if ok else "reject"
    experiments.log(f"structural start bonus ({mode}), nested", SEASONS,
                    {"win_brier_change": d.mean(), "log_loss_change": dl.mean(), "better_seasons": better},
                    (lo, hi), decision, notes="bonus per half-season: " + str(picked))
    print(f"\n{mode}: bonus chosen per half-season (earlier races only): {picked}")
    print(f"  win Brier {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}] better in {better}/4 | "
          f"winner log loss {dl.mean():+.3f} [{llo:+.3f}, {lhi:+.3f}] -> {decision}")
    cal = pd.DataFrame({"v1 pole P(win)": v1.groupby(level=0)["pole_p_win"].mean(),
                        "nested pole P(win)": nested.groupby(level=0)["pole_p_win"].mean(),
                        "pole actually won": v1.groupby(level=0)["pole_won"].mean()})
    print(cal.round(3).to_string())
