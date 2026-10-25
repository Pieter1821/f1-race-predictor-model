"""Load race results from Jolpica (through FastF1's Ergast client) into one tidy table."""
import pandas as pd
from fastf1.ergast import Ergast

from f1pred.config import SEASONS, enable_cache

RESULT_COLUMNS = {
    "driverId": "driver_id",
    "driverCode": "driver_code",
    "constructorId": "constructor_id",
    "grid": "grid",
    "position": "position",
    "status": "status",
    "laps": "laps",
    "points": "points",
}


def _ergast() -> Ergast:
    enable_cache()
    return Ergast(result_type="pandas", auto_cast=True, limit=100)


def _flatten(response, races: list[pd.DataFrame]) -> None:
    """Each page holds a description table (one row per race) and a results table per race."""
    for (_, race), results in zip(response.description.iterrows(), response.content):
        df = results[list(RESULT_COLUMNS)].rename(columns=RESULT_COLUMNS)
        df.insert(0, "season", race["season"])
        df.insert(1, "round", race["round"])
        df.insert(2, "race_name", race["raceName"])
        df.insert(3, "circuit_id", race["circuitId"])
        df.insert(4, "date", race["raceDate"])
        races.append(df)


def load_season_results(season: int) -> pd.DataFrame:
    """All race results for one season. Jolpica returns 100 rows per page, so we page through."""
    response = _ergast().get_race_results(season=season)
    races: list[pd.DataFrame] = []
    _flatten(response, races)
    while True:
        try:
            response = response.get_next_result_page()
        except ValueError:  # raised when there is no next page
            break
        _flatten(response, races)
    df = pd.concat(races, ignore_index=True)
    # A race split across two pages arrives in two parts; the drivers don't overlap, but be safe
    return df.drop_duplicates(subset=["season", "round", "driver_id"]).reset_index(drop=True)


def load_results(seasons=SEASONS) -> pd.DataFrame:
    df = pd.concat([load_season_results(s) for s in seasons], ignore_index=True)
    return clean_results(df)


NON_STARTERS = {"Did not start", "Withdrew"}


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    # Drivers who never took the start have no race result to learn from
    df = df[~df["status"].isin(NON_STARTERS)].copy()
    df["position"] = df.groupby(["season", "round"])["position"].rank(method="first").astype(int)
    field_size = df.groupby(["season", "round"])["driver_id"].transform("count")
    # grid 0 means a pit-lane start: that's behind the whole field, not ahead of pole
    df["pit_lane_start"] = df["grid"] == 0
    df.loc[df["pit_lane_start"], "grid"] = field_size[df["pit_lane_start"]]
    df["field_size"] = field_size
    df["finished"] = df["status"].str.match(r"^(Finished|\+\d+ Laps?|Lapped)$")
    return df.sort_values(["season", "round", "position"]).reset_index(drop=True)
