"""One-off (25 Oct 2026): re-label reports/experiments.csv after the review.

The old simplification rule ("mean within noise") passed changes whose CI reached +0.06 places, and the
word "ACCEPT" read as adopted. New rule: a simplification is a candidate only if its worst case (CI upper
end) is within the noise floor; candidate and adopted are separate. Adds `status` and `latest` (re-runs
of the same change appended duplicate rows; only the latest run counts).

    python scripts/experiments/relabel_experiments.py
"""
import pandas as pd

from f1pred.config import PROJECT_ROOT
from f1pred.evaluate import NOISE_FLOOR

LOG = PROJECT_ROOT / "reports" / "experiments.csv"
log = pd.read_csv(LOG)


def relabel(row) -> tuple[str, str]:
    d = str(row["decision"])
    if d.startswith("ACCEPT (simpler") or d.startswith("candidate (simpler") or d.startswith("reject (simpl") \
            or d.startswith("reject (worst case"):
        within = pd.notna(row.get("ci_high")) and row["ci_high"] <= NOISE_FLOOR
        return ("candidate (simpler; worst case within noise)", "candidate") if within \
            else ("reject (worst case above noise)", "rejected")
    if d.startswith("ADOPT") or d.startswith("adopted"):
        return "adopted: v2 from the next race", "adopted"
    if d == "ACCEPT" or d.startswith("candidate"):
        return "candidate", "candidate"
    if d == "useful":
        return "candidate: published as experimental", "candidate"
    if d.startswith("reject") or d.startswith("keep v1") or d.startswith("no better"):
        return d, "rejected"
    return d, "check"  # diagnostics, the leak check


log[["decision", "status"]] = log.apply(lambda r: pd.Series(relabel(r)), axis=1)
log["latest"] = ~log.duplicated("change", keep="last")
log.to_csv(LOG, index=False)
latest = log[log["latest"]]
print(f"{len(log)} rows, {len(latest)} distinct changes (latest run each)")
print(latest["status"].value_counts().to_string())
print("candidates:", latest.loc[latest["status"] == "candidate", "change"].tolist())
