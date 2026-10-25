import pandas as pd

from f1pred.practice import add_practice_features, clean_laps, longrun_features, map_drivers


def _session(n_drivers=4, laps=12):
    """One practice session: each driver does a 2-lap run, pits, then a long run on MEDIUM.
    Driver d is d*0.2s slower per lap and wears tyres d*0.02 s/lap faster."""
    rows, stints = [], []
    for d in range(1, n_drivers + 1):
        for lap in range(1, laps + 1):
            out_lap = lap in (1, 4)
            t = 90 + d * 0.2 + (lap - 4) * 0.05 * d / 2
            rows.append({"season": 2026, "round": 18, "session_key": 1, "session_name": "Practice 2",
                         "driver_number": d, "lap_number": lap, "is_pit_out_lap": out_lap,
                         "lap_duration": t + (8 if out_lap else 0)})
        stints += [{"session_key": 1, "driver_number": d, "stint_number": 1, "lap_start": 1, "lap_end": 3,
                    "compound": "SOFT", "tyre_age_at_start": 0},
                   {"session_key": 1, "driver_number": d, "stint_number": 2, "lap_start": 4, "lap_end": laps,
                    "compound": "MEDIUM", "tyre_age_at_start": 0}]
    return pd.DataFrame(rows), pd.DataFrame(stints)


def test_cleaning_drops_out_laps_in_laps_and_slow_laps():
    laps, stints = _session()
    laps.loc[(laps["driver_number"] == 2) & (laps["lap_number"] == 8), "lap_duration"] += 10  # traffic lap
    clean, report = clean_laps(laps, stints)
    assert report["pit-out and in-laps"] == 4 * 3  # laps 1 and 4 are pit-out laps; lap 3 is an in-lap
    assert report["slow laps (> 4.0% over stint median)"] == 1
    assert not ((clean["driver_number"] == 2) & (clean["lap_number"] == 8)).any()
    assert set(clean["lap_number"]) == {2} | set(range(5, 13))


def test_tyre_age_comes_from_the_stint():
    laps, stints = _session()
    stints.loc[stints["stint_number"] == 2, "tyre_age_at_start"] = 3  # used set
    clean, _ = clean_laps(laps, stints)
    row = clean[(clean["driver_number"] == 1) & (clean["lap_number"] == 6)].iloc[0]
    assert row["compound"] == "MEDIUM" and row["tyre_age"] == 3 + 6 - 4


def test_longrun_features_rank_pace_and_wear():
    laps, stints = _session()
    f = longrun_features(clean_laps(laps, stints)[0]).set_index("driver_number")
    assert f["fp_longrun_gap_pct"].is_monotonic_increasing      # driver 1 fastest
    assert f["fp_deg"].is_monotonic_increasing                  # driver 1 easiest on tyres
    assert (f["fp_longrun_laps"] == 8).all()                    # laps 5-12 survive in the long run


def test_reserve_drivers_are_not_mapped_and_missing_rows_are_flagged():
    drivers = pd.DataFrame({"season": 2026, "round": 18, "driver_number": [1, 2, 99],
                            "name_acronym": ["VER", "HAD", "RES"]})
    entries = pd.DataFrame({"season": 2026, "round": 18, "driver_id": ["max_verstappen", "hadjar", "lawson"],
                            "code": ["VER", "HAD", "LAW"]})
    mapping = map_drivers(drivers, entries)
    assert set(mapping["driver_id"]) == {"max_verstappen", "hadjar"}  # reserve (RES) dropped
    feats = pd.DataFrame({"season": 2026, "round": 18, "driver_number": [1], "fp_longrun_gap_pct": [-0.5],
                          "fp_deg": [0.01], "fp_longrun_laps": [10]})
    table = pd.DataFrame({"season": 2026, "round": 18, "driver_id": ["max_verstappen", "lawson"]})
    out = add_practice_features(table, feats, mapping).set_index("driver_id")
    assert out.loc["max_verstappen", "fp_has_longrun"] == 1 and out.loc["max_verstappen", "fp_longrun_gap_pct"] == -0.5
    assert out.loc["lawson", "fp_has_longrun"] == 0 and out.loc["lawson", "fp_longrun_gap_pct"] == 0.0


def test_implausible_long_runs_are_dropped():
    laps, stints = _session(n_drivers=5)
    laps.loc[laps["driver_number"] == 5, "lap_duration"] *= 1.30  # a red-flag "long run", 30% slow
    f = longrun_features(clean_laps(laps, stints)[0])
    assert 5 not in set(f["driver_number"])


def test_regular_who_skipped_fp1_is_still_mapped():
    drivers = pd.DataFrame({"season": 2026, "round": [1, 2, 2], "driver_number": [1, 99, 6],
                            "name_acronym": ["VER", "RES", "HAD"]})  # VER skipped FP1 in round 2
    entries = pd.DataFrame({"season": 2026, "round": [1, 2, 2], "driver_id": ["max_verstappen", "max_verstappen", "hadjar"],
                            "code": ["VER", "VER", "HAD"]})
    m = map_drivers(drivers, entries)
    assert ((m["round"] == 2) & (m["driver_number"] == 1) & (m["driver_id"] == "max_verstappen")).any()


def test_driver_with_two_numbers_in_a_season_is_not_guessed():
    drivers = pd.DataFrame({"season": 2024, "round": [2, 17, 20], "driver_number": [38, 50, 99],
                            "name_acronym": ["BEA", "BEA", "RES"]})
    entries = pd.DataFrame({"season": 2024, "round": [2, 17, 20], "driver_id": "bearman", "code": "BEA"})
    m = map_drivers(drivers, entries)
    assert set(zip(m["round"], m["driver_number"])) == {(2, 38), (17, 50)}  # round 20: ambiguous, not guessed
