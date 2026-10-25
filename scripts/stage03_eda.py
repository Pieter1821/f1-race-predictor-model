"""Stage 3: which signals explain what the starting grid doesn't?

    python scripts/stage03_eda.py        # prints the tables, writes reports/figures/stage03_*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from f1pred.config import PROJECT_ROOT, SNAPSHOT, TARGET_ROUND, TARGET_SEASON, WEATHER_SNAPSHOT
from f1pred.data import load_raw, load_results, race_start
from f1pred.features import add_dnf_rate, prior_mean
from f1pred.validate import score_many
from f1pred.weather import forecasts_as_of, read_forecasts

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
BLUE, GREY, INK, MUTED, GRIDLINE = "#2a78d6", "#b9b8b1", "#0b0b0b", "#898781", "#e1e0d9"
pd.set_option("display.width", 160)
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDLINE,
    "grid.linewidth": 0.6, "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9,
})

# ---------------------------------------------------------------- data, everything as of Austin's lights out
data = load_raw(from_snapshot=SNAPSHOT)
as_of = race_start(data.schedule, TARGET_SEASON, TARGET_ROUND)
df = load_results(as_of=as_of, data=data)
weather = forecasts_as_of(read_forecasts(WEATHER_SNAPSHOT), data.schedule, as_of)
df = df.merge(weather[["season", "round", "forecast_rain_mm", "forecast_temp_c"]],
              on=["season", "round"], how="left", validate="many_to_one")

df["places_lost"] = df["position"] - df["grid"]           # + = went backwards
df["on_track_pct"] = df["on_track_position"] / df["field_size"]  # 0 = front, 1 = back

# Qualifying gap: % slower than the fastest lap in the deepest session the driver reached
session = np.select([df["q3"].notna(), df["q2"].notna(), df["q1"].notna()], [3, 2, 1], 0)
best = np.select([session == 3, session == 2, session == 1], [df["q3"], df["q2"], df["q1"]], np.nan)
df["quali_session"], df["quali_best"] = session, best
session_best = df.groupby(["season", "round", "quali_session"])["quali_best"].transform("min")
df["quali_gap_pct"] = (df["quali_best"] / session_best - 1) * 100
# Alternative: the driver's smallest gap in ANY session they ran. A failed final session
# (deleted laps, a crash, a wet Q3) shouldn't hide the pace shown in an earlier one.
gaps = [(df[q] / df.groupby(["season", "round"])[q].transform("min") - 1) * 100 for q in ("q1", "q2", "q3")]
df["quali_best_gap_pct"] = pd.concat(gaps, axis=1).min(axis=1)

# Candidate history signals: all from strictly earlier races (leak-free helper), shrunk when history is short
df["driver_level"] = prior_mean(df, "driver_id", "on_track_pct", prior_weight=5)
df["driver_racecraft"] = prior_mean(df, "driver_id", "places_lost", prior_weight=5, prior_value=0)
df["team_season"] = df["team"] + "_" + df["season"].astype(str)
df["team_form"] = prior_mean(df, "team_season", "on_track_pct", prior_weight=3)
df["driver_circuit"] = df["driver_id"] + "@" + df["circuit_id"]
df["track_places_gained"] = -prior_mean(df, "driver_circuit", "places_lost", prior_weight=2, prior_value=0)
df["track_level"] = prior_mean(df, "driver_circuit", "on_track_pct", prior_weight=2)
df = add_dnf_rate(df)
df["rain_risk"] = df["forecast_rain_mm"]  # rain *probability* is mostly missing in the archive; amount is complete for 2024+


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Rank correlation of x and y after removing what z (the grid) explains about each."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    resid = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


SIGNALS = {
    "grid_penalty": "Started behind qualifying spot (places)",
    "quali_gap_pct": "Qualifying gap to session best, deepest session (%)",
    "quali_best_gap_pct": "Qualifying gap, best of any session (%)",
    "team_form": "Team's avg finish this season, before this race",
    "driver_level": "Driver's avg finish, all earlier races",
    "track_level": "Driver's avg finish at this track before",
    "driver_racecraft": "Driver's avg places lost per race before",
    "track_places_gained": "Driver's avg places gained at this track before",
    "team_dnf_rate": "Team's DNF rate before this race",
    "rain_risk": "Forecast rain over race window, mm (2024+ only)",
    "rookie": "Rookie (first 3 career races)",
}
rows = []
finishers = df["finished"]
for col, desc in SIGNALS.items():
    m = df[col].notna()
    sub, fin = df[m], df[m & finishers]
    rows.append({
        "signal": col, "what": desc, "rows": int(m.sum()),
        "vs_finish": sub[col].astype(float).corr(sub["position"], method="spearman"),
        "beyond_grid": partial_spearman(sub[col].astype(float), sub["position"], sub["grid"]),
        "beyond_grid_finishers": partial_spearman(fin[col].astype(float), fin["position"], fin["grid"]),
    })
signals = pd.DataFrame(rows).sort_values("beyond_grid", key=abs, ascending=False)
print("Signal strength (Spearman; 'beyond_grid' = after removing what the grid already explains)")
print(signals.round(3).to_string(index=False), "\n")

# ---------------------------------------------------------------- 1. grid vs finish
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
groups = [df.loc[df["grid"] == g, "position"] for g in range(1, 21)]
a1.boxplot(groups, positions=range(1, 21), widths=0.6, showfliers=False,
           medianprops={"color": BLUE, "linewidth": 2}, boxprops={"color": MUTED},
           whiskerprops={"color": MUTED}, capprops={"color": MUTED})
a1.plot([1, 20], [1, 20], color=GREY, linewidth=1, linestyle="--")
a1.set(title="Finishing position by grid slot (2022-26)", xlabel="Grid position", ylabel="Finishing position")
a1.invert_yaxis()
win = df[df["grid"] <= 10].groupby("grid")["position"].apply(lambda p: (p == 1).mean() * 100)
a2.bar(win.index, win.values, color=BLUE, width=0.7)
a2.set(title="How often each grid slot wins", xlabel="Grid position", ylabel="Win rate (%)")
for g, v in win.items():
    if v >= 1:
        a2.text(g, v + 1, f"{v:.0f}%", ha="center", color=INK, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "stage03_grid_vs_finish.png", dpi=130)

# ---------------------------------------------------------------- 2. qualifying gap vs places lost
fig, ax = plt.subplots(figsize=(7, 4))
q = df[df["quali_best_gap_pct"].notna()]
shown = q[q["quali_best_gap_pct"] <= 3]
ax.scatter(shown["quali_best_gap_pct"], shown["places_lost"], s=10, color=GREY, alpha=0.5, linewidths=0)
bins = pd.cut(q["quali_best_gap_pct"], [-0.01, 0.2, 0.4, 0.7, 1.0, 1.5, 3])
binned = q.groupby(bins, observed=True).agg(gap=("quali_best_gap_pct", "mean"), lost=("places_lost", "mean"))
ax.plot(binned["gap"], binned["lost"], color=BLUE, linewidth=2, marker="o", markersize=6)
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.text(2.95, ax.get_ylim()[1] * 0.92, f"{(q['quali_best_gap_pct'] > 3).sum()} drivers > 3% not shown", ha="right", color=MUTED, fontsize=8)
ax.set(title="Bigger qualifying gap, more places lost?", xlabel="Smallest gap to the session's fastest lap, any session (%)",
       ylabel="Places lost in the race (+ = went backwards)")
fig.tight_layout()
fig.savefig(FIG_DIR / "stage03_quali_gap.png", dpi=130)

# ---------------------------------------------------------------- 3. DNF rate by season and 2026 team
r = df[df["status_bucket"] != "dsq"].assign(dnf=lambda x: ~x["finished"])
by_season = r.groupby("season")["dnf"].mean() * 100
by_team = r[r["season"] == 2026].groupby("team")["dnf"].mean().sort_values() * 100
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.bar(by_season.index.astype(str), by_season.values, color=[GREY] * 4 + [BLUE], width=0.6)
a1.set(title="DNF rate per car, by season", ylabel="DNF rate (%)")
for i, v in enumerate(by_season.values):
    a1.text(i, v + 0.4, f"{v:.1f}%", ha="center", color=INK, fontsize=8)
a2.barh(by_team.index, by_team.values, color=BLUE, height=0.6)
a2.axvline(by_season[2026], color=MUTED, linewidth=1, linestyle="--")
a2.text(by_season[2026] + 0.5, -0.6, "2026 average", color=MUTED, fontsize=8)
a2.set(title="2026 DNF rate by team (about 34 starts each)", xlabel="DNF rate (%)")
fig.tight_layout()
fig.savefig(FIG_DIR / "stage03_dnf_rates.png", dpi=130)

# ---------------------------------------------------------------- 4. Austin vs other tracks
scores = score_many(df, "grid").merge(df.groupby(["season", "round"]).agg(
    circuit=("circuit_id", "first"), dnfs=("finished", lambda s: int((~s).sum())),
    moved=("places_lost", lambda s: s.abs().mean())).reset_index(), on=["season", "round"])
by_circuit = scores.groupby("circuit").agg(races=("mae", "size"), mae=("mae", "mean"),
                                            dnfs=("dnfs", "mean"), pole_won=("winner_hit", "mean"))
by_circuit = by_circuit[by_circuit["races"] >= 3].sort_values("mae")
fig, ax = plt.subplots(figsize=(7, 6))
colors = [BLUE if c == "americas" else GREY for c in by_circuit.index]
ax.barh(by_circuit.index, by_circuit["mae"], color=colors, height=0.65)
ax.axvline(scores["mae"].mean(), color=MUTED, linewidth=1, linestyle="--")
ax.text(scores["mae"].mean() + 0.05, -0.9, "all-track average", color=MUTED, fontsize=8)
ax.set(title="How wrong 'finish = grid' is, by track (3+ races)", xlabel="Average error (places per driver)")
fig.tight_layout()
fig.savefig(FIG_DIR / "stage03_tracks.png", dpi=130)

print("Austin vs other tracks (3+ races, sorted by grid-baseline error):")
rank = list(by_circuit.index).index("americas") + 1
print(f"Austin ranks {rank} of {len(by_circuit)} (1 = most predictable)")
print(by_circuit.loc[["americas"]].round(2).to_string())
print("all tracks:", scores[["mae", "dnfs", "winner_hit"]].mean().round(2).to_dict(), "\n")

# ---------------------------------------------------------------- weather: wet-risk races vs dry (2024+)
w = scores.merge(weather[["season", "round", "forecast_rain_mm"]], on=["season", "round"])
w = w[w["forecast_rain_mm"].notna()]
w["forecast"] = np.where(w["forecast_rain_mm"] >= 0.5, "wet forecast (>= 0.5 mm)", "dry forecast")
print("Grid baseline by day-ahead rain forecast (2024+):")
print(w.groupby("forecast")[["mae", "dnfs", "winner_hit"]].agg(["mean", "size"]).round(2).to_string())
print(f"\nfigures written to {FIG_DIR}")
