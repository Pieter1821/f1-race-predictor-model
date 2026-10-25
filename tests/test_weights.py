import pandas as pd
import pytest

from f1pred.weights import recency_weights


def _train():
    rows = [{"season": s, "round": r, "driver_id": d, "position": p}
            for s, n in ((2025, 24), (2026, 17)) for r in range(1, n + 1)
            for p, d in enumerate(("a", "b"), start=1)]
    return pd.DataFrame(rows)


def test_uniform_is_all_ones():
    assert (recency_weights(_train()) == 1.0).all()


def test_half_life_halves_weight_every_h_races():
    df = _train()
    w = recency_weights(df, half_life_races=12)
    latest = (df["season"] == 2026) & (df["round"] == 17)
    twelve_back = (df["season"] == 2026) & (df["round"] == 5)
    assert (w[latest] == 1.0).all()
    assert w[twelve_back].iloc[0] == pytest.approx(0.5)
    by_race = w.groupby([df["season"], df["round"]]).first()
    assert by_race.is_monotonic_increasing  # older races never outweigh newer ones


def test_weights_ignore_results():
    df = _train()
    flipped = df.assign(position=3 - df["position"])
    pd.testing.assert_series_equal(recency_weights(df, 24), recency_weights(flipped, 24))


def test_regulation_boost_applies_to_current_era_only():
    df = _train()
    w = recency_weights(df, current_regs_boost=3.0)
    assert (w[df["season"] == 2026] == 3.0).all() and (w[df["season"] == 2025] == 1.0).all()


def test_weights_are_relative_to_the_training_cutoff():
    # Training on 2025 only (e.g. a 2025 backtest fold): its last race gets weight 1, not something tiny
    df = _train()
    w = recency_weights(df[df["season"] == 2025], half_life_races=12)
    assert w.max() == 1.0
