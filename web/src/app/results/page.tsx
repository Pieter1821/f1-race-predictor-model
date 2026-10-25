import { TopBar } from "@/components/TopBar";
import { ResultsView, type RaceOption } from "@/components/ResultsView";
import { loadBacktests, loadRace } from "@/lib/data";

export const metadata = { title: "Results · Grid to Flag" };

export default async function ResultsPage() {
  const [race, bt] = await Promise.all([loadRace(), loadBacktests()]);

  /* The live race joins the backtests as one more option; its scores come from score_race.py once the result is in. */
  const actualByDriver = new Map((race.actual ?? []).map((a) => [a.driver_id, a]));
  const live: RaceOption = {
    id: `${race.race.season}-${race.race.round}`,
    season: race.race.season,
    round: race.race.round,
    name: race.race.name,
    live: true,
    scores: race.scores[0] ?? null,
    rows: race.drivers.map((d) => {
      const a = actualByDriver.get(d.driver_id);
      return { code: d.code, name: d.name, team: d.team_name, grid: d.grid, grid_penalty: d.grid_penalty,
               predicted: d.predicted, position: a ? a.position : null, status: a ? a.status : null };
    }),
  };
  const past: RaceOption[] = [...bt.races].reverse().map((r) => ({
    id: `${r.season}-${r.round}`,
    season: r.season,
    round: r.round,
    name: r.name,
    live: false,
    model: r.model,
    grid: r.grid,
    dnfs: r.dnfs,
    rows: r.rows.map((x) => ({ ...x, name: x.code })),
  }));

  return (
    <>
      <TopBar active="/results" startUtc={race.race.start_utc} scored={!!race.actual} />
      <ResultsView options={[live, ...past]} summary={bt.summary} clean={bt.clean} devSeasons={bt.dev_seasons} />
    </>
  );
}
