import numpy as np
import pandas as pd
import pytest

from f1pred.features import (FEATURE_NAMES, OUTCOME_COLUMNS, build_features, league_prior,
                             prior_mean)

TEAMS = {"ferrari": ("lec", "ham"), "haas": ("oco", "bea"), "mclaren": ("nor", "pia")}


def _season_data(seasons=(2025, 2026), rounds=6, seed=0) -> pd.DataFrame:
    """Small but realistic: 3 teams x 2 cars, several circuits, random results and qualifying."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            drivers = [(t, d) for t, ds in TEAMS.items() for d in ds]
            quali = rng.permutation(len(drivers)) + 1
            finish = rng.permutation(len(drivers)) + 1
            for (team, driver), q, f in zip(drivers, quali, finish):
                finished = rng.random() > 0.2
                rows.append({
                    "season": season, "round": rnd, "driver_id": driver, "team": team,
                    "circuit_id": f"c{rnd % 4}", "field_size": len(drivers),
                    "quali_position": float(q), "grid": int(q), "sprint_sets_grid": False,
                    "q1": 90 + q * 0.1 + rng.random() * 0.05, "q2": np.nan if q > 4 else 89.5 + q * 0.1,
                    "q3": np.nan if q > 2 else 89.0 + q * 0.1,
                    "position": f, "on_track_position": f, "position_pct": f / len(drivers),
                    "finished": finished, "status_bucket": "finished" if finished else "dnf_unknown",
                    "status": "Finished" if finished else "Retired", "laps": 50, "points": max(0, 10 - f),
                })
    return pd.DataFrame(rows)


def _scramble_outcomes(df: pd.DataFrame, from_race: tuple[int, int], seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    later = list(zip(out["season"], out["round"]))
    mask = pd.Series([r >= from_race for r in later], index=out.index)
    n = int(mask.sum())
    out.loc[mask, "position"] = rng.integers(1, 7, n)
    out.loc[mask, "on_track_position"] = rng.integers(1, 7, n)
    out.loc[mask, "points"] = rng.integers(0, 26, n)
    out.loc[mask, "finished"] = rng.random(n) > 0.5
    out.loc[mask, "status_bucket"] = np.where(out.loc[mask, "finished"], "finished", "dnf_unknown")
    return out


@pytest.mark.parametrize("k", [(2025, 1), (2025, 4), (2026, 1), (2026, 3)])
def test_no_feature_uses_its_own_race_or_later(k):
    df = _season_data()
    before = build_features(df)
    after = build_features(_scramble_outcomes(df, k))
    upto_k = (df["season"] * 100 + df["round"]) <= k[0] * 100 + k[1]  # race k and earlier
    pd.testing.assert_frame_equal(before.loc[upto_k, FEATURE_NAMES], after.loc[upto_k, FEATURE_NAMES])


def test_prediction_rows_match_training_rows():
    # The target race at prediction time has no results; its features must equal those built with results
    df = _season_data()
    target = (df["season"] == 2026) & (df["round"] == 6)
    blanked = df.copy()
    outcomes = [c for c in OUTCOME_COLUMNS if c in blanked]
    blanked[outcomes] = blanked[outcomes].astype(object)
    blanked.loc[target, outcomes] = np.nan
    pd.testing.assert_frame_equal(build_features(df).loc[target, FEATURE_NAMES],
                                  build_features(blanked).loc[target, FEATURE_NAMES])


def test_every_feature_is_present_and_filled_after_the_first_race():
    f = build_features(_season_data())
    assert set(FEATURE_NAMES) <= set(f.columns)
    later = ~((f["season"] == 2025) & (f["round"] == 1))
    assert not f.loc[later, FEATURE_NAMES].isna().any().any()


def test_league_prior_ignores_later_races():
    # Regression: the old default shrank toward df[col].mean(), an average that included future races
    df = _season_data()
    changed = _scramble_outcomes(df, (2026, 1))
    early = df["season"] == 2025
    pd.testing.assert_series_equal(league_prior(df.assign(x=df["position"].astype(float)), "x")[early],
                                   league_prior(changed.assign(x=changed["position"].astype(float)), "x")[early])
    assert prior_mean(df.assign(x=1.0), "driver_id", "x", prior_weight=5)[early & (df["round"] == 1)].isna().all()


def test_teammate_in_same_race_does_not_leak():
    df = pd.DataFrame({"season": 2026, "round": [1, 1, 2, 2], "team": "haas",
                       "driver_id": ["a", "b", "a", "b"], "dnf": [0.0, 1.0, 0.0, 0.0]})
    f = prior_mean(df, "team", "dnf", prior_value=0.5)
    assert f.iloc[:2].isna().all()  # race 1: no history, not each other's result
    assert (f.iloc[2:] == 0.5).all()  # race 2: race 1's team average (0 + 1) / 2


def test_shrinkage_pulls_small_samples_toward_prior():
    df = pd.DataFrame({"season": 2026, "round": [1, 2], "team": "x", "driver_id": "a", "dnf": [1.0, 0.0]})
    f = prior_mean(df, "team", "dnf", prior_weight=10, prior_value=0.2)
    assert f.iloc[1] == pytest.approx((1 + 10 * 0.2) / (1 + 10))


def test_failed_session_is_capped_and_missing_time_flagged():
    df = _season_data(seasons=(2026,), rounds=2)
    first = df.index[(df["round"] == 2) & (df["driver_id"] == "lec")][0]
    second = df.index[(df["round"] == 2) & (df["driver_id"] == "ham")][0]
    df.loc[first, ["q1", "q2", "q3"]] = [200.0, np.nan, np.nan]  # aborted lap, 100%+ slow
    df.loc[second, ["q1", "q2", "q3"]] = np.nan                   # no time at all
    f = build_features(df)
    assert f.loc[first, "quali_best_gap_pct"] == 5.0 and f.loc[first, "no_quali_time"] == 0
    assert f.loc[second, "no_quali_time"] == 1 and f.loc[second, "grid_penalty"] == 0
    assert f.loc[second, "teammate_quali_delta"] == 0  # no fair teammate comparison without a time


def test_standings_come_from_earlier_races_only():
    df = _season_data(seasons=(2026,), rounds=3)
    f = build_features(df)
    assert (f.loc[f["round"] == 1, "driver_standing"] == 1).all()  # round 1: everyone level on 0 points
    r1 = df[df["round"] == 1].set_index("driver_id")["points"]
    leader = r1.idxmax()
    assert f[(f["round"] == 2) & (f["driver_id"] == leader)]["driver_standing"].iloc[0] == 1
