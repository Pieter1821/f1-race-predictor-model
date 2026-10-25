/* Shapes of the JSON written by scripts/export_web_data.py. The frontend displays these numbers;
   it never recomputes a metric the pipeline already scored. */

export type Driver = {
  driver_id: string;
  code: string;
  name: string;
  team: string;
  team_name: string;
  grid: number;
  quali_position: number;
  grid_penalty: number;
  predicted: number;
  expected_position: number;
  p_win: number;
  p_podium: number;
  p_points: number;
  p_dnf: number;
  train_races: number;
  variant_predicted?: number;
  variant_p_win?: number;
  stops?: number;
  pit_laps?: number[];
  two_stop_cost_s?: number;
};

export type ActualRow = { driver_id: string; driver_code: string; position: number; status: string; laps: number; grid: number };

export type RaceData = {
  exported_at_utc: string;
  race: { season: number; round: number; name: string; circuit_id: string; laps: number; start_utc: string; qualifying_start_utc: string };
  official: { file: string; generated_at_utc: string; git_commit: string; model: string; data_source: string; inputs_source: string };
  variant_file: string | null;
  strategy: { pit_loss_s: number; wear_source: string; validation: string; generated_at_utc: string; git_commit: string; status: string } | null;
  drivers: Driver[];
  track_history: { season: number; code: string; grid: number; team: string }[];
  actual: ActualRow[] | null;
  scores: Record<string, string | number | boolean | null>[];
  strategy_scores: Record<string, string | number | boolean | null>[];
};

export type Metrics = { mae: number; spearman: number; winner_hit: number; podium_hit: number };
export type BacktestRow = { code: string; team: string; grid: number; grid_penalty: number; predicted: number; position: number; status: string };
export type BacktestRace = {
  season: number; round: number; name: string; circuit_id: string;
  model: Metrics; grid: Metrics; dnfs: number; rows: BacktestRow[];
};
export type Backtests = {
  exported_at_utc: string;
  model: string;
  seasons: number[];
  summary: { races: number; mae_model: number; mae_grid: number; gain: number; ci: [number, number]; better: number; worse: number };
  /* The honest headline: test seasons only, never used for any choice */
  clean: { seasons: number[]; races: number; mae_model: number; mae_grid: number; gain: number; ci: [number, number]; better: number; worse: number };
  dev_seasons: number[];
  by_season: { season: number; model: number; grid: number; model_spearman: number; grid_spearman: number; model_winner: number; grid_winner: number; races: number }[];
  races: BacktestRace[];
};

export type ModelData = {
  exported_at_utc: string;
  calibration: { kind: string; bins: { predicted: number; actual: number; n: number }[] }[];
  slices: { slice: string; rows: number; model: number; grid: number }[];
  experiments: { date_utc: string; change: string; seasons: string; mae_change: number | null; ci_low: number | null; ci_high: number | null; better_seasons: number | null; decision: string; status: "adopted" | "candidate" | "rejected" | "check"; latest: boolean }[];
  versions: Record<string, Record<string, unknown>>;
  simulation: Record<string, unknown>;
};
