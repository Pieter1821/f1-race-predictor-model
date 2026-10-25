import pytest

from f1pred.strategy import best_plans, plan_time, plans

CAPS = {"SOFT": 30, "MEDIUM": 40, "HARD": 50}
OFF = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.4}


def test_plan_time_adds_wear_and_pit_loss():
    # 10 laps on medium: 10 * 0 + 0.1 * (0+1+...+9) = 4.5; 10 on hard: 10 * 0.4 + 0.05 * 45 = 6.25; + 20 s stop
    assert plan_time(("MEDIUM", "HARD"), (10, 10), OFF, {"MEDIUM": 0.1, "HARD": 0.05, "SOFT": 0.2}, 20) == pytest.approx(30.75)


def test_every_plan_uses_two_compounds_and_respects_caps():
    for comps, lengths in plans(56, CAPS):
        assert len(set(comps)) >= 2 and sum(lengths) == 56
        assert all(6 <= L <= CAPS[c] for c, L in zip(comps, lengths))


def test_high_wear_or_cheap_stops_favour_two_stops():
    low = best_plans(56, OFF, {"SOFT": 0.05, "MEDIUM": 0.04, "HARD": 0.03}, 22, CAPS, top=1)
    high = best_plans(56, OFF, {"SOFT": 0.30, "MEDIUM": 0.25, "HARD": 0.20}, 22, CAPS, top=1)
    assert low["stops"].iloc[0] == 1 and high["stops"].iloc[0] == 2
