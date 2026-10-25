"""Part 5: pit-strategy estimates. Walk-forward validation on 2024-26 dry races, then Austin 2026 per driver.

    python scripts/part5_strategy.py
For each test race, only earlier races and that weekend's practice are used: compound offsets, fuel
effect and stint caps from earlier race laps; tyre wear from this weekend's practice long runs (fuel
corrected); pit loss from earlier races at the same circuit. Validation: does the best plan's stop
count match the most common stop count among finishers? Baseline: "always one stop".
"""
import numpy as np
import pandas as pd

from f1pred import config, experiments
from f1pred.data import read_snapshot
from f1pred.practice import clean_laps
from f1pred.strategy import DRY, best_plans, clean_race_laps, fit_race, pit_loss

pd.set_option("display.width", 170)
O = sorted((config.DATA_DIR / "snapshots").glob("openf1_*"))[-1]
sessions = pd.read_csv(O / "sessions.csv")
r_laps, r_stints = pd.read_parquet(O / "race_laps.parquet"), pd.read_parquet(O / "race_stints.parquet")
p_clean, _ = clean_laps(pd.read_parquet(O / "practice_laps.parquet"), pd.read_parquet(O / "practice_stints.parquet"))
jolpica = read_snapshot(sorted((config.DATA_DIR / "snapshots").glob("jolpica_2022-2026_*T*"))[-1])
race_laps_n = jolpica.results.groupby(["season", "round"])["laps"].max()
circuit = sessions.drop_duplicates(["season", "round"]).set_index(["season", "round"])["circuit_short_name"]

wet = set(r_stints.loc[r_stints["compound"].isin(["INTERMEDIATE", "WET"]), "session_key"])
clean = clean_race_laps(r_laps, r_stints)
fits = {}
for (s, r), g in clean.groupby(["season", "round"]):
    if g["session_key"].iloc[0] not in wet and (f := fit_race(g)):
        fits[(s, r)] = f
fits = pd.DataFrame(fits).T.sort_index()
losses = pd.Series({k: pit_loss(g) for k, g in r_laps.groupby(["season", "round"])})
stint_len = r_stints.assign(L=r_stints["lap_end"] - r_stints["lap_start"] + 1)
print(f"race fits (dry, enough laps): {len(fits)} | median fuel effect {fits['fuel'].median():+.3f} s/lap | "
      f"median pit loss {losses.median():.1f} s", flush=True)


def practice_wear(season, rnd, fuel):
    """Field-median practice long-run wear per compound (fuel corrected), with the number of runs."""
    fp = p_clean[(p_clean["season"] == season) & (p_clean["round"] == rnd)]
    size = fp.groupby(["session_key", "driver_number", "stint_number"])["lap_duration"].transform("size")
    lr = fp[size >= 5]
    out = {}
    for c in DRY:
        runs = [np.polyfit(g["tyre_age"], g["lap_duration"], 1)[0]
                for _, g in lr[lr["compound"] == c].groupby(["session_key", "driver_number", "stint_number"])
                if g["tyre_age"].nunique() > 2]
        if len(runs) >= 3:
            out[c] = (float(np.median(runs)) - fuel, len(runs))
    return out


# Calibration: race wear / practice wear at the same weekend (practice runs overstate or understate race wear)
calib = []
for (s, r), f in fits.iterrows():
    for c, (pw, _) in practice_wear(s, r, f["fuel"]).items():
        if pw > 0.01 and pd.notna(f.get(f"wear_{c}")):
            calib.append({"key": s * 100 + r, "ratio": f[f"wear_{c}"] / pw})
calib = pd.DataFrame(calib)
print(f"practice -> race wear calibration: median ratio {calib['ratio'].median():.2f} over {len(calib)} compound-weekends")


def inputs_for(season, rnd, driver_wear=None):
    """Offsets, wear, pit loss and stint caps for one race, from earlier races + this weekend's practice."""
    key = season * 100 + rnd
    earlier = fits[[s * 100 + r < key for s, r in fits.index]]
    offset = {c: float(earlier[f"offset_{c}"].median()) for c in DRY}
    fuel = float(earlier["fuel"].median())
    wear = {c: float(earlier[f"wear_{c}"].median()) for c in DRY}  # fallback: typical race wear
    ratio = float(calib.loc[calib["key"] < key, "ratio"].median())  # learned from earlier weekends only
    source = {c: "earlier races" for c in DRY}
    for c, (pw, n) in practice_wear(season, rnd, fuel).items():
        wear[c] = pw * ratio
        source[c] = f"practice ({n} long runs) x {ratio:.2f} calibration"
    if driver_wear is not None:
        wear = {c: max(0.0, w + driver_wear) for c, w in wear.items()}
    else:
        wear = {c: max(0.0, w) for c, w in wear.items()}
    here = [k for k in losses.index if k[0] * 100 + k[1] < key and circuit.get(k) == circuit.get((season, rnd))]
    loss = float(losses[here].median()) if here else float(losses[[k[0] * 100 + k[1] < key for k in losses.index]].median())
    prev = stint_len[stint_len["season"] * 100 + stint_len["round"] < key]
    caps = {c: int(prev.loc[prev["compound"] == c, "L"].quantile(0.98)) for c in DRY}
    return offset, wear, loss, caps, source


# ---------------------------------------------------------------- walk-forward validation
rows = []
for (s, r) in [k for k in fits.index if k[0] >= 2024]:
    offset, wear, loss, caps, _ = inputs_for(s, r)
    best = best_plans(int(race_laps_n[(s, r)]), offset, wear, loss, caps, top=1).iloc[0]
    fin = jolpica.results[(jolpica.results["season"] == s) & (jolpica.results["round"] == r) &
                          jolpica.results["status"].str.match(r"^(Finished|\+\d+ Laps?|Lapped)$")]
    sk = sessions[(sessions["season"] == s) & (sessions["round"] == r) & (sessions["session_name"] == "Race")]["session_key"]
    st = r_stints[r_stints["session_key"].isin(sk)]
    stops = st.groupby("driver_number")["stint_number"].max() - 1
    first = st[st["stint_number"] == 1].groupby("driver_number")["compound"].first()
    rows.append({"season": s, "round": r, "circuit": circuit.get((s, r)), "pred_stops": best["stops"],
                 "actual_stops": int(stops.mode().iloc[0]), "pred_plan": best["plan"],
                 "actual_first": first.mode().iloc[0][0] if len(first) else "?"})
v = pd.DataFrame(rows)
v["stops_ok"] = v["pred_stops"] == v["actual_stops"]
v["first_ok"] = v["pred_plan"].str[0] == v["actual_first"]
base = (v["actual_stops"] == 1).mean()
print(f"\nWALK-FORWARD, {len(v)} dry races 2024-26: stop count right {v['stops_ok'].mean():.0%} "
      f"(baseline 'always one stop' {base:.0%}) | first compound right {v['first_ok'].mean():.0%}")
print(v.groupby("season")[["stops_ok", "first_ok"]].mean().round(2).to_string())
experiments.log("strategy: best-plan stop count, walk-forward", [2024, 2025, 2026],
                {"stops_accuracy": v["stops_ok"].mean(), "baseline_one_stop": base,
                 "first_compound_accuracy": v["first_ok"].mean()}, None,
                "useful" if v["stops_ok"].mean() > base else "no better than 'always one stop'")
cota = v[v["circuit"] == "Austin"]
if len(cota):
    print("COTA:", cota[["season", "pred_plan", "pred_stops", "actual_stops", "actual_first"]].to_string(index=False))

# ---------------------------------------------------------------- Austin 2026
offset, wear, loss, caps, source = inputs_for(2026, 18)
print(f"\nAUSTIN 2026 inputs | pit loss {loss:.1f} s | stint caps {caps}")
print("  wear (s/lap): " + ", ".join(f"{c} {wear[c]:.3f} [{source[c]}]" for c in DRY))
print("  offsets vs MEDIUM (s/lap): " + ", ".join(f"{c} {offset[c]:+.2f}" for c in DRY))
field = best_plans(56, offset, wear, loss, caps, top=4)
print("\nfield-typical best plans (56 laps):")
print(field[["plan", "stops", "pit_laps", "vs_best_s"]].round(1).to_string(index=False))
field.to_csv(config.PROCESSED_DIR / "austin_2026_strategy_field.csv", index=False)

# ---------------------------------------------------------------- is 66% vs 55% more than luck? (paired, per race)
from scipy.stats import binomtest  # noqa: E402

base_ok = v["actual_stops"] == 1
only_model = int((v["stops_ok"] & ~base_ok).sum())
only_base = int((~v["stops_ok"] & base_ok).sum())
p = binomtest(only_model, only_model + only_base, 0.5).pvalue if only_model + only_base else 1.0
print(f"\nraces only the model got right: {only_model} | only 'always one stop' got right: {only_base} | "
      f"two-sided sign test p = {p:.2f}")

# ---------------------------------------------------------------- per driver, Austin 2026
from f1pred.practice import longrun_features  # noqa: E402

fp = longrun_features(p_clean[(p_clean["season"] == 2026) & (p_clean["round"] == 18)])
drivers = pd.read_parquet(O / "drivers.parquet")
drivers = drivers[(drivers["season"] == 2026) & (drivers["round"] == 18)][["driver_number", "name_acronym"]]
grid = pd.read_csv(config.MANUAL_DIR / "2026_R18_grid.csv")[["grid", "code"]]
ratio_now = float(calib["ratio"].median())
per = []
for _, g in grid.iterrows():
    num = drivers.loc[drivers["name_acronym"] == g["code"], "driver_number"]
    rel = fp.loc[fp["driver_number"].isin(num), "fp_deg"]
    rel_wear = float(rel.iloc[0]) * ratio_now if len(rel) else 0.0
    best = best_plans(56, offset, {c: max(0.0, w + rel_wear) for c, w in wear.items()}, loss, caps, top=2)
    per.append({"grid": g["grid"], "driver": g["code"], "own wear vs field (s/lap)": round(rel_wear, 3),
                "data": "own long run" if len(rel) else "field average", "stops": best["stops"].iloc[0],
                "pit window (best plan)": best["pit_laps"].iloc[0],
                "2-stop is slower by (s)": round(
                    best_plans(56, offset, {c: max(0.0, w + rel_wear) for c, w in wear.items()}, loss, caps, top=50)
                    .query("stops == 2")["vs_best_s"].min(), 1)})
per = pd.DataFrame(per)
print("\nPER DRIVER (stop count and pit window are the validated part; compound order is not):")
print(per.to_string(index=False))
per.to_csv(config.PROCESSED_DIR / "austin_2026_strategy_per_driver.csv", index=False)

# Saved before lights out (with --save), with provenance, so it can be scored honestly after the race
from datetime import datetime, timezone  # noqa: E402

from f1pred.pipeline import git_commit  # noqa: E402

now = datetime.now(timezone.utc)
out = per.assign(field_best_plan=field["plan"].iloc[0], pit_loss_s=round(loss, 1),
                 wear_source="; ".join(f"{c}: {s}" for c, s in source.items()),
                 validation=f"stop count right {v['stops_ok'].mean():.0%} vs {base:.0%} 'always one stop' "
                            f"(56 dry races, sign test p={p:.2f}); first compound right {v['first_ok'].mean():.0%} (not reliable)",
                 generated_at_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"), as_of_utc="2026-10-25T20:00:00Z",
                 data_source=f"snapshot {O.name}", git_commit=git_commit(), status="EXPERIMENTAL, not official")
if "--save" in __import__("sys").argv:
    path = config.PREDICTIONS_DIR / f"2026_R18_strategy_{now:%Y%m%dT%H%M%SZ}.csv"
    out.to_csv(path, index=False)
    print(f"saved {path.name}")
