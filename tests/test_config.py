"""Guards for the tunable settings: config.py must match the recorded decisions, and the official
Austin prediction must reproduce exactly from its snapshot (so refactors can't silently change it)."""
import json

import pandas as pd
import pytest

from f1pred import config
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.pipeline import predict, race_inputs


def test_config_matches_recorded_simulation_decision():
    s = json.loads((config.PROJECT_ROOT / "reports" / "sim_settings.json").read_text())
    assert (config.SIM_TARGET, config.SIM_SPACING, config.SIM_SIGMA, config.SIM_GRID_BLEND, config.SIM_WIN_TEMPERATURE) == \
           (s["target"], s["spacing"], s["sigma"], s["grid_blend"], s["win_temperature"])
    assert config.SIM_POLE_BONUS == 0.0  # untested on dev: must stay off in the locked model


def test_official_austin_prediction_reproduces_exactly():
    official = sorted(f for f in config.PREDICTIONS_DIR.glob("2026_R18_2026*.csv"))[0]
    saved = pd.read_csv(official).set_index("driver_id").sort_index()
    data = read_snapshot(config.DATA_DIR / "snapshots" / saved["data_source"].iloc[0].removeprefix("snapshot "))
    as_of = race_start(data.schedule, 2026, 18)
    history = load_results(as_of=as_of, data=data)
    inputs, _ = race_inputs(2026, 18, data, history, as_of, backtest=False)
    sim = predict(history, inputs, "americas", as_of, n_sims=20_000).set_index("driver_id").sort_index()
    pd.testing.assert_series_equal(sim["predicted_position"], saved["predicted_position"], check_dtype=False)
    for col in ("p_win", "p_podium", "p_points", "expected_position"):
        assert (sim[col].round(4) - saved[col]).abs().max() == pytest.approx(0, abs=1e-9), col
