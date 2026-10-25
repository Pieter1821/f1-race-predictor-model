import numpy as np
import pandas as pd
import pytest

from f1pred.variant import decayed_prior_mean


def _df():
    return pd.DataFrame({"season": [2024, 2025, 2026, 2026], "round": [1, 1, 1, 2], "driver_id": "v",
                         "x": [0.0, 0.0, 1.0, np.nan]})


def test_older_seasons_count_less():
    df = _df()
    f = decayed_prior_mean(df, df["driver_id"], "x", 0, pd.Series(0.5, index=df.index))
    # at 2026 R2: 2024 weight 0.25, 2025 weight 0.5, 2026 R1 weight 1 -> (0 + 0 + 1) / 1.75
    assert f.iloc[3] == pytest.approx(1 / 1.75)


def test_uses_strictly_earlier_races_only():
    df = _df()
    a = decayed_prior_mean(df, df["driver_id"], "x", 2, pd.Series(0.5, index=df.index))
    changed = df.assign(x=[0.0, 0.0, 0.0, 1.0])  # rewrite the 2026 R1 result and later
    b = decayed_prior_mean(changed, changed["driver_id"], "x", 2, pd.Series(0.5, index=df.index))
    assert a.iloc[:3].equals(b.iloc[:3])  # 2026 R1 and earlier unchanged
