"""Pit-strategy estimates: which 1- or 2-stop plan minimises race time, given tyre wear and pit loss.

race time = sum over laps of (compound offset + wear_per_lap(compound) * tyre age) + stops * pit loss
Fuel burn speeds every car up equally, so it doesn't change which plan is fastest; it only matters for
measuring wear (a lighter car hides tyre wear), which is why race laps are fitted with a fuel term.
"""
from itertools import product

import numpy as np
import pandas as pd

DRY = ("SOFT", "MEDIUM", "HARD")
MIN_STINT = 6


# ---------------------------------------------------------------- measuring from race laps (history)

def clean_race_laps(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Race laps with compound and tyre age; drops out-laps, in-laps and slow laps (safety car, traffic)."""
    x = laps[["season", "round", "session_key", "driver_number", "lap_number", "lap_duration", "is_pit_out_lap"]]
    x = x[x["lap_duration"].notna() & (x["lap_number"] > 1)].copy()
    outs = x[x["is_pit_out_lap"] == True][["session_key", "driver_number", "lap_number"]]  # noqa: E712
    in_laps = outs.assign(lap_number=outs["lap_number"] - 1, _in=True)
    x = x.merge(in_laps, on=["session_key", "driver_number", "lap_number"], how="left")
    x = x[(x["is_pit_out_lap"] != True) & x["_in"].isna()]  # noqa: E712
    st = stints[["session_key", "driver_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]]
    x = x.merge(st, on=["session_key", "driver_number"])
    x = x[(x["lap_number"] >= x["lap_start"]) & (x["lap_number"] <= x["lap_end"]) & x["compound"].isin(DRY)]
    x["tyre_age"] = x["tyre_age_at_start"].fillna(0) + x["lap_number"] - x["lap_start"]
    med = x.groupby("session_key")["lap_duration"].transform("median")
    return x[x["lap_duration"] <= med * 1.05].reset_index(drop=True)


def fit_race(r: pd.DataFrame) -> dict | None:
    """lap = driver effect + fuel * lap_number + offset(compound) + wear(compound) * tyre_age, least squares."""
    comps = [c for c in DRY if (r["compound"] == c).sum() >= 30]
    if "MEDIUM" not in comps or len(comps) < 2:
        return None
    r = r[r["compound"].isin(comps)]
    drivers = sorted(r["driver_number"].unique())
    X = [(r["driver_number"] == d).astype(float) for d in drivers]
    X += [r["lap_number"].astype(float)]
    X += [(r["compound"] == c).astype(float) for c in comps if c != "MEDIUM"]
    X += [(r["compound"] == c) * r["tyre_age"] for c in comps]
    coef = np.linalg.lstsq(np.column_stack(X), r["lap_duration"].to_numpy(), rcond=None)[0]
    k = len(drivers)
    out = {"fuel": coef[k], "offset_MEDIUM": 0.0}
    others = [c for c in comps if c != "MEDIUM"]
    for i, c in enumerate(others):
        out[f"offset_{c}"] = coef[k + 1 + i]
    for i, c in enumerate(comps):
        out[f"wear_{c}"] = coef[k + 1 + len(others) + i]
    return out


def pit_loss(laps: pd.DataFrame) -> float:
    """Median over pit stops of (in-lap + out-lap) minus two of that driver's median laps."""
    x = laps[laps["lap_duration"].notna()]
    med = x.groupby(["session_key", "driver_number"])["lap_duration"].median().rename("med")
    outs = x[x["is_pit_out_lap"] == True][["session_key", "driver_number", "lap_number", "lap_duration"]]  # noqa: E712
    ins = x[["session_key", "driver_number", "lap_number", "lap_duration"]].assign(lap_number=lambda d: d["lap_number"] + 1)
    both = outs.merge(ins, on=["session_key", "driver_number", "lap_number"], suffixes=("_out", "_in"))
    both = both.join(med, on=["session_key", "driver_number"])
    loss = both["lap_duration_out"] + both["lap_duration_in"] - 2 * both["med"]
    return float(loss[(loss > 5) & (loss < 60)].median())


# ---------------------------------------------------------------- choosing a plan

def plans(n_laps: int, max_stint: dict) -> list[tuple]:
    """Every 1- and 2-stop plan using at least two dry compounds: (compounds, stint lengths)."""
    out = []
    for stops in (1, 2):
        for comps in product(DRY, repeat=stops + 1):
            if len(set(comps)) < 2:
                continue
            for cuts in product(range(MIN_STINT, n_laps - MIN_STINT + 1), repeat=stops):
                if list(cuts) != sorted(set(cuts)):
                    continue
                edges = [0, *cuts, n_laps]
                lengths = [b - a for a, b in zip(edges, edges[1:])]
                if all(MIN_STINT <= L <= max_stint[c] for c, L in zip(comps, lengths)):
                    out.append((comps, tuple(lengths)))
    return out


def plan_time(comps, lengths, offset: dict, wear: dict, loss: float) -> float:
    t = sum(L * offset[c] + wear[c] * L * (L - 1) / 2 for c, L in zip(comps, lengths))
    return t + (len(comps) - 1) * loss


def best_plans(n_laps: int, offset: dict, wear: dict, loss: float, max_stint: dict, top: int = 3) -> pd.DataFrame:
    rows = [{"plan": "-".join(c[0] for c in comps), "stops": len(comps) - 1, "stints": lengths,
             "pit_laps": " & ".join(str(int(x)) for x in np.cumsum(lengths)[:-1]),
             "time": plan_time(comps, lengths, offset, wear, loss)}
            for comps, lengths in plans(n_laps, max_stint)]
    df = pd.DataFrame(rows).sort_values("time")
    df["vs_best_s"] = df["time"] - df["time"].iloc[0]
    # best version of each compound sequence, then the overall top few
    return df.drop_duplicates("plan").head(top).reset_index(drop=True)
