"""Score a saved strategy estimate against what drivers actually did (OpenF1 race stints, fetched live).

    python scripts/score_strategy.py --season 2026 --round 18
Appends to predictions/strategy_scorecard.csv: per-driver stop-count accuracy vs 'always one stop',
and the pit-lap error for drivers whose stop count was right. Wet races are reported but flagged.
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

from f1pred.config import PREDICTIONS_DIR
from f1pred.openf1 import get

CARD = PREDICTIONS_DIR / "strategy_scorecard.csv"
parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, required=True)
parser.add_argument("--round", type=int, required=True)
args = parser.parse_args()

path = sorted(PREDICTIONS_DIR.glob(f"{args.season}_R{args.round}_strategy_*.csv"))[-1]
pred = pd.read_csv(path)

meetings = get("sessions", year=args.season, session_name="Race")
start = pd.Timestamp(pred["as_of_utc"].iloc[0])
hours_apart = (pd.to_datetime(meetings["date_start"], utc=True) - start).dt.total_seconds().abs() / 3600
race = meetings[hours_apart < 30]
if race.empty:
    raise SystemExit("Race session not found on OpenF1 yet.")
sk = int(race["session_key"].iloc[0])
stints, drivers = get("stints", session_key=sk), get("drivers", session_key=sk)
if stints.empty:
    raise SystemExit("No stint data yet; try again later.")

actual = stints.groupby("driver_number").agg(stints=("stint_number", "max"), last_lap=("lap_end", "max"))
actual["actual_stops"] = actual["stints"] - 1
pits = stints[stints["stint_number"] > 1].groupby("driver_number")["lap_start"].apply(lambda s: [int(x) - 1 for x in s])
actual["actual_pit_laps"] = pits
actual = actual.join(drivers.set_index("driver_number")["name_acronym"])
wet = stints["compound"].isin(["INTERMEDIATE", "WET"]).any()

df = pred.merge(actual.reset_index(), left_on="driver", right_on="name_acronym", how="inner")
df["stops_right"] = df["stops"] == df["actual_stops"]
df["baseline_right"] = df["actual_stops"] == 1


def pit_error(row):
    pred_laps = [int(x) for x in str(row["pit window (best plan)"]).split(" & ")]
    got = row["actual_pit_laps"] if isinstance(row["actual_pit_laps"], list) else []
    return sum(abs(p - a) for p, a in zip(pred_laps, got)) / len(pred_laps) if row["stops_right"] and got else None


df["pit_lap_error"] = df.apply(pit_error, axis=1)
row = {"season": args.season, "round": args.round, "prediction_file": path.name, "wet_race": wet,
       "drivers_scored": len(df), "stops_right": round(df["stops_right"].mean(), 3),
       "always_one_stop_right": round(df["baseline_right"].mean(), 3),
       "mean_pit_lap_error_when_stops_right": round(df["pit_lap_error"].dropna().mean(), 2),
       "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
card = pd.read_csv(CARD) if CARD.exists() else pd.DataFrame()
card = pd.concat([card[card.get("prediction_file", pd.Series(dtype=str)) != path.name], pd.DataFrame([row])],
                 ignore_index=True)
card.to_csv(CARD, index=False)
print(f"{path.name} | wet race: {wet}")
print(df[["driver", "stops", "actual_stops", "pit window (best plan)", "actual_pit_laps", "pit_lap_error"]].to_string(index=False))
print(f"\nstop count right {row['stops_right']:.0%} vs 'always one stop' {row['always_one_stop_right']:.0%} | "
      f"mean pit-lap error {row['mean_pit_lap_error_when_stops_right']} laps")
