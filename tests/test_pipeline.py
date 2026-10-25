"""Integration test: the full prediction pipeline on a past race, offline from the committed snapshot."""
import pytest

from f1pred.config import SNAPSHOT
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.pipeline import prediction_table, predict, race_inputs


@pytest.fixture(scope="module")
def prediction():
    data = read_snapshot(SNAPSHOT)
    as_of = race_start(data.schedule, 2026, 17)
    history = load_results(as_of=as_of, data=data)
    inputs, source = race_inputs(2026, 17, data, history, as_of, backtest=True)
    sim = predict(history, inputs, "marina_bay", as_of, n_sims=2000)
    return prediction_table(sim, "Singapore Grand Prix", as_of, source, data.source), history


def test_one_row_per_starter_with_consistent_probabilities(prediction):
    out, _ = prediction
    assert len(out) == 22 and out["driver_id"].is_unique
    assert sorted(out["predicted_position"]) == list(range(1, 23))
    assert out["p_win"].sum() == pytest.approx(1.0, abs=1e-3)
    assert out["p_podium"].sum() == pytest.approx(3.0, abs=1e-2)
    assert (out["p_win"] <= out["p_podium"] + 1e-9).all() and (out["p_podium"] <= out["p_points"] + 1e-9).all()


def test_provenance_is_recorded(prediction):
    out, _ = prediction
    for col in ("generated_at_utc", "as_of_utc", "inputs_source", "data_source", "model", "git_commit", "train_races"):
        assert out[col].notna().all()
    assert out["as_of_utc"].iloc[0] == "2026-10-11T12:00:00Z"


def test_trained_only_on_races_before_the_target(prediction):
    out, history = prediction
    assert history["season"].mul(100).add(history["round"]).max() == 202616
    assert out["train_races"].iloc[0] == history.groupby(["season", "round"]).ngroups
