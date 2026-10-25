"""Monte Carlo race simulation: turn one pace score per driver into finishing-position probabilities.

Each simulated race:
  1. pace  = blended score (model + a share of the grid) + random noise of size sigma
  2. DNFs  = each car retires with its own probability; retired cars go behind every finisher
  3. order = finishers by pace, then retirements in random order
Repeated n_sims times; probabilities are the share of simulations in which something happened.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

POINTS_PLACES = 10
# Chosen on 2023-24 DNF outcomes (Brier flat from 40 to 160): team DNF history predicts future DNFs
# only weakly, so a team needs ~40 races (80 car-starts) of history to count as much as the season rate
DNF_PRIOR_WEIGHT = 80


@dataclass(frozen=True)
class SimSettings:
    grid_blend: float = 0.0   # lambda: 0 = pure model score, 1 = pure grid position
    sigma: float = 0.15       # pace noise, in "share of the field" units
    spacing: str = "linear"   # "log": gaps at the front are bigger than in the midfield
    n_sims: int = 10_000
    seed: int = 0


def _spaced(rank: np.ndarray, n: int, spacing: str) -> np.ndarray:
    """Rank 1..n -> pace 0..1. Linear: equal gaps. Log: P1-P2 gap >> P11-P12 gap, as in real races."""
    if spacing == "log":
        return np.log(rank) / np.log(n)
    return (rank - 1) / max(n - 1, 1)


def pace_score(model_score: np.ndarray, grid: np.ndarray, grid_blend: float, spacing: str = "linear") -> np.ndarray:
    """Model score and grid both turned into within-race ranks scaled 0..1, then blended.

    Ties share an average rank (equal pace). Never break them by row order: rows from the
    backtest are stored in finishing order, so that would quietly use the result.
    """
    n = len(model_score)
    model_pct = _spaced(pd.Series(model_score).rank(method="average").to_numpy(), n, spacing)
    grid_pct = _spaced(pd.Series(grid).rank(method="average").to_numpy(), n, spacing)
    return (1 - grid_blend) * model_pct + grid_blend * grid_pct


def simulate_positions(pace: np.ndarray, p_dnf: np.ndarray, sigma: float, n_sims: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Finishing positions, shape (n_sims, n_drivers), 1 = winner."""
    n = len(pace)
    noisy = pace + rng.normal(0.0, sigma, size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < p_dnf
    key = np.where(dnf, 10.0 + rng.random((n_sims, n)), noisy)  # every DNF sorts behind every finisher
    order = np.argsort(key, axis=1)
    positions = np.empty_like(order)
    positions[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)
    return positions


def add_dnf_probability(table: pd.DataFrame, prior_weight: float = DNF_PRIOR_WEIGHT) -> pd.DataFrame:
    """P(DNF) per car for the simulation: team rate this season, shrunk hard toward this season's
    league rate so far (itself shrunk toward the all-time rate). Earlier races only."""
    from f1pred.features import league_prior, prior_mean

    t = table.copy()
    dnf = (~t["finished"].fillna(True).astype(bool) & (t["status_bucket"] != "dsq")).astype(float)
    t["_dnf"] = dnf.where(t["position"].notna())
    t["_season"] = t["season"].astype(str)
    t["_team_season"] = t["team"] + "_" + t["_season"]
    season_rate = prior_mean(t, "_season", "_dnf", prior_weight=44, prior_value=league_prior(t, "_dnf"))
    return table.assign(sim_dnf_prob=prior_mean(t, "_team_season", "_dnf", prior_weight=prior_weight,
                                                 prior_value=season_rate))


def race_probabilities(drivers: pd.DataFrame, settings: SimSettings, score_col: str = "pred",
                       dnf_col: str = "sim_dnf_prob") -> pd.DataFrame:
    """One race: P(win), P(podium), P(points), expected position and the DNF probability used."""
    rng = np.random.default_rng(settings.seed)
    pace = pace_score(drivers[score_col].to_numpy(float), drivers["grid"].to_numpy(float), settings.grid_blend,
                      settings.spacing)
    pos = simulate_positions(pace, drivers[dnf_col].to_numpy(float), settings.sigma, settings.n_sims, rng)
    return drivers.assign(
        p_win=(pos == 1).mean(axis=0),
        p_podium=(pos <= 3).mean(axis=0),
        p_points=(pos <= POINTS_PLACES).mean(axis=0),
        p_dnf=drivers[dnf_col].to_numpy(float),
        expected_position=pos.mean(axis=0),
        # ties in expected position broken by grid, never by row order
        predicted_position=pd.DataFrame({"e": pos.mean(axis=0), "g": drivers["grid"].to_numpy()})
        .sort_values(["e", "g"], kind="stable").assign(r=range(1, len(drivers) + 1)).sort_index()["r"].to_numpy(),
    )


def simulate_many(preds: pd.DataFrame, settings: SimSettings, **kwargs) -> pd.DataFrame:
    return pd.concat([race_probabilities(race, settings, **kwargs)
                      for _, race in preds.groupby(["season", "round"], sort=True)])


def probability_scores(sim: pd.DataFrame, eps: float = 1e-4) -> pd.Series:
    """Lower is better. Winner log loss: -log P(actual winner). Brier: mean squared probability error."""
    winners = sim[sim["position"] == 1]
    win_b = float(((sim["p_win"] - (sim["position"] == 1)) ** 2).mean())
    pod_b = float(((sim["p_podium"] - (sim["position"] <= 3)) ** 2).mean())
    pts_b = float(((sim["p_points"] - (sim["position"] <= POINTS_PLACES)) ** 2).mean())
    return pd.Series({
        "brier_total": win_b + pod_b + pts_b,  # the selection metric: per-driver Brier, win + podium + points
        "winner_log_loss": float(-np.log(winners["p_win"].clip(eps, 1)).mean()),
        "win_brier": win_b,
        "podium_brier": pod_b,
        "points_brier": pts_b,
        "p_actual_winner": float(winners["p_win"].mean()),
    })


def grid_slot_baseline(history: pd.DataFrame, race: pd.DataFrame, prior_races: float = 2.0) -> pd.DataFrame:
    """Naive probabilities: how often each grid slot won / podiumed / scored in earlier races.

    Smoothed toward 'every slot equal' with the weight of `prior_races` races, then renormalised
    so a race's P(win) sums to 1, P(podium) to 3 and P(points) to 10.
    """
    n = len(race)
    out = race.copy()
    for col, places in (("p_win", 1), ("p_podium", 3), ("p_points", POINTS_PLACES)):
        hits = history.assign(hit=history["position"] <= places).groupby("grid")["hit"].agg(["sum", "count"])
        rate = (hits["sum"] + prior_races * places / n) / (hits["count"] + prior_races)
        p = race["grid"].map(rate).fillna(places / n).to_numpy()
        out[col] = p * places / p.sum()
    return out
