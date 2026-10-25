"""Data errors (fail loudly) and real-but-unusual cases (report and flag, never drop).

    python scripts/check_outliers.py --season 2026 --round 18      # from the committed snapshot
"""
import argparse

import pandas as pd

from f1pred.config import SNAPSHOT
from f1pred.data import load_raw, load_results, race_start, restrict_to_as_of
from f1pred.race_inputs import RaceInputError, load_race_inputs
from f1pred.validate import score_many

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
errors: list[str] = []


def header(title: str) -> None:
    print(f"\n{'=' * 8} {title}")


parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, required=True, help="check everything before this race's start")
parser.add_argument("--round", type=int, required=True)
args = parser.parse_args()
TARGET_SEASON, TARGET_ROUND = args.season, args.round

data = load_raw(from_snapshot=SNAPSHOT)
as_of = race_start(data.schedule, TARGET_SEASON, TARGET_ROUND)
raw = restrict_to_as_of(data.results, data.schedule, as_of)
results = load_results(as_of=as_of, data=data)
print(f"source: {data.source} | as of {as_of} | {raw.groupby(['season', 'round']).ngroups} races")

# ---------------------------------------------------------------- data errors

header("4. Points match position (fastest-lap bonus point allowed through 2024)")
expected = raw["position"].map(lambda p: RACE_POINTS[p - 1] if p <= 10 else 0)
expected = expected.where(raw["status"] != "Disqualified", 0)
bonus = raw["points"] - expected
fastest_lap_ok = (bonus == 1) & (raw["season"] <= 2024) & (raw["position"] <= 10)
bad = raw[(bonus != 0) & ~fastest_lap_ok]
print(f"fastest-lap bonus points: {int(fastest_lap_ok.sum())} | mismatches: {len(bad)}")
if len(bad):
    print(bad[["season", "round", "driver_code", "position", "status", "points"]].assign(expected=expected[bad.index]).to_string(index=False))
    errors.append(f"{len(bad)} points mismatches")

header("5. Status agrees with laps")
leader_laps = raw.groupby(["season", "round"])["laps"].transform("max")
n_down = raw["status"].str.extract(r"^\+(\d+) Laps?$")[0].astype(float)
bad_finished = raw[(raw["status"] == "Finished") & (raw["laps"] != leader_laps)]
bad_lapped = raw[raw["status"].str.match(r"^(\+\d+ Laps?|Lapped)$") & (raw["laps"] >= leader_laps)]
bad_count = raw[n_down.notna() & (leader_laps - raw["laps"] != n_down)]
print(f"'Finished' not on leader's lap: {len(bad_finished)} | lapped with leader's laps: {len(bad_lapped)} | '+N Laps' off by count: {len(bad_count)}")
for name, frame in [("finished", bad_finished), ("lapped", bad_lapped), ("+N", bad_count)]:
    if len(frame):
        print(frame[["season", "round", "driver_code", "status", "laps"]].assign(leader=leader_laps[frame.index]).to_string(index=False))
        errors.append(f"{len(frame)} status/laps mismatches ({name})")

header("6. Grid slots unique per race (except pit-lane starters) and within 1..entrants")
entrants = raw.groupby(["season", "round"])["driver_id"].transform("count")
on_grid = raw[raw["grid"] > 0]
dupes = on_grid[on_grid.duplicated(["season", "round", "grid"], keep=False)]
out_of_range = raw[(raw["grid"] < 0) | (raw["grid"] > entrants)]
print(f"duplicate grid slots: {len(dupes)} | out of range: {len(out_of_range)}")
for name, frame in [("duplicate grid", dupes), ("grid out of range", out_of_range)]:
    if len(frame):
        print(frame[["season", "round", "driver_code", "grid", "status"]].to_string(index=False))
        errors.append(f"{len(frame)} {name}")

header("7. One team per driver per race, at most 2 cars per team")
cars = raw.groupby(["season", "round", "constructor_id"]).size()
too_many = cars[cars > 2]
print(f"teams with more than 2 cars in a race: {len(too_many)}")
if len(too_many):
    print(too_many.to_string())
    errors.append(f"{len(too_many)} teams with >2 cars")
seq = raw.sort_values(["season", "round"])
seq["prev_team"] = seq.groupby(["season", "driver_id"])["constructor_id"].shift()
changes = seq[seq["prev_team"].notna() & (seq["prev_team"] != seq["constructor_id"])]
print("mid-season team changes:")
print(changes[["season", "round", "race_name", "driver_code", "prev_team", "constructor_id"]].to_string(index=False))
newcomers = seq[seq["prev_team"].isna() & (seq["round"] > 1)]
print("drivers joining mid-season:")
print(newcomers[["season", "round", "race_name", "driver_code", "constructor_id"]].to_string(index=False))

# ---------------------------------------------------------------- real but unusual

header("8. DNF rate per car (starters who didn't finish; DSQ excluded)")
r = results[results["status_bucket"] != "dsq"]
r = r.assign(dnf=~r["finished"])
by_season = r.groupby("season")["dnf"].agg(["mean", "sum", "count"]).rename(columns={"mean": "dnf_rate", "sum": "dnfs", "count": "starts"})
print(by_season.round(3).to_string())
team_rates = r[r["season"] == 2026].groupby("team")["dnf"].agg(["mean", "count"]).rename(columns={"mean": "dnf_rate_2026", "count": "starts"})
print("\n2026 by team:")
print(team_rates.sort_values("dnf_rate_2026").round(3).to_string())

header("9. Races where the grid predicted the result worst / best")
scores = score_many(results, pred_col="grid").merge(
    results.groupby(["season", "round"]).agg(
        race=("race_name", "first"),
        dnfs=("finished", lambda s: int((~s).sum())),
        pit_lane=("pit_lane_start", "sum"),
        winner_grid=("grid", lambda g: int(g[results.loc[g.index, "position"] == 1].iloc[0])),
    ).reset_index(),
    on=["season", "round"],
)
gains = results.assign(gain=results["grid"] - results["position"]).sort_values("gain", ascending=False)
best_gain = gains.groupby(["season", "round"]).first()


def reason(row) -> str:
    g = best_gain.loc[(row["season"], row["round"])]
    return (f"{row['dnfs']} DNFs, winner from P{row['winner_grid']}, "
            f"biggest gain {g['driver_code']} P{int(g['grid'])}->P{int(g['position'])}"
            + (f", {int(row['pit_lane'])} pit-lane starts" if row["pit_lane"] else ""))


scores["reason"] = scores.apply(reason, axis=1)
cols = ["season", "round", "race", "mae", "spearman", "reason"]
print("worst 10:")
print(scores.nlargest(10, "mae")[cols].round(2).to_string(index=False))
print("best 5:")
print(scores.nsmallest(5, "mae")[cols].round(2).to_string(index=False))
print(f"\nDNFs per race vs baseline MAE, correlation: {scores['dnfs'].corr(scores['mae']):.2f}")

header("10. Grid penalties: starting 5+ places behind qualifying (2022 sprint weekends excluded)")
pen = results[results["grid_penalty"] >= 5]
print(f"{len(pen)} cases | per season: {pen.groupby('season').size().to_dict()}")
print(pen[pen["season"] == 2026][["round", "race_name", "driver_code", "quali_position", "grid", "grid_penalty"]].to_string(index=False))
moved_up = results[results["grid_penalty"] < 0]
print(f"started ahead of quali position (others' penalties): {len(moved_up)} cases, max {int(-moved_up['grid_penalty'].min())} places")

header("11. 2022 sprint weekends (race grid set by the sprint)")
print(results[results["sprint_sets_grid"]].groupby(["season", "round", "race_name"]).size().rename("drivers").to_string())

header("12. Rookies (first 3 career races, from full F1 history) and new teams")
rk = results[results["rookie"]].groupby(["driver_code", "season"])["round"].apply(list)
print(rk.to_string())
nt = results[results["new_team"]].groupby(["team", "season"])["round"].apply(lambda s: sorted(set(s)))
print("new teams (fewer than 3 prior races after lineage mapping):")
print(nt.to_string() if len(nt) else "none")

header("13. Join coverage: race results with no qualifying row")
no_quali = results[results["quali_position"].isna()]
print(f"{len(no_quali)} of {len(results)} results have no qualifying row")
if len(no_quali):
    print(no_quali[["season", "round", "race_name", "driver_code", "grid", "status"]].to_string(index=False))

header(f"14. Pre-race inputs for {TARGET_SEASON} R{TARGET_ROUND} (hand-entered CSVs vs real history)")
quali_start = data.schedule.set_index(["season", "round"]).loc[(TARGET_SEASON, TARGET_ROUND), "qualifying_start_utc"]
try:
    inputs = load_race_inputs(TARGET_SEASON, TARGET_ROUND, results, quali_start, as_of)
    print(f"{len(inputs)} drivers OK | qualifying started {quali_start} (before cutoff {as_of})")
    moved = inputs[inputs["grid_penalty"] != 0].sort_values("grid_penalty", ascending=False)
    print(moved[["driver_id", "quali_position", "grid", "grid_penalty"]].to_string(index=False))
except RaceInputError as e:
    print(e)
    errors.append("pre-race inputs invalid")

print("\n" + ("DATA ERRORS: " + "; ".join(errors) if errors else "No data errors found."))
raise SystemExit(1 if errors else 0)
