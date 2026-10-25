"use client";

import { useState } from "react";
import { fixed, places, signed } from "@/lib/format";
import type { Backtests, Metrics } from "@/lib/types";

export type ResultRow = {
  code: string; name: string; team: string; grid: number; grid_penalty: number;
  predicted: number; position: number | null; status: string | null;
};
export type RaceOption = {
  id: string; season: number; round: number; name: string; live: boolean;
  model?: Metrics; grid?: Metrics; dnfs?: number;
  scores?: Record<string, string | number | boolean | null> | null;
  rows: ResultRow[];
};

const short = (name: string) => name.replace(" Grand Prix", "");
const DNF = /^dnf|dsq/;

export function ResultsView({ options, summary, clean, devSeasons }: { options: RaceOption[]; summary: Backtests["summary"]; clean: Backtests["clean"]; devSeasons: number[] }) {
  const [id, setId] = useState(options[0].id);
  const race = options.find((o) => o.id === id)!;
  const scored = race.rows.some((r) => r.position != null);
  const metrics = race.live
    ? race.scores
      ? { model: pick(race.scores, "model"), grid: pick(race.scores, "grid") }
      : null
    : { model: race.model!, grid: race.grid! };

  const misses = scored
    ? [...race.rows].filter((r) => r.position != null)
        .sort((a, b) => Math.abs(b.predicted - b.position!) - Math.abs(a.predicted - a.position!)).slice(0, 5)
    : [];
  const finding = !scored
    ? "The prediction is saved and pushed; the comparison appears here once the official result is published."
    : metrics
      ? `Off by ${places(metrics.model.mae)} places per driver against ${places(metrics.grid.mae)} for the grid: ${
          metrics.model.mae < metrics.grid.mae ? "the model beat the grid" : metrics.model.mae > metrics.grid.mae ? "the grid did better" : "level with the grid"}${
          metrics.model.winner_hit ? ", and it called the winner." : ", and it missed the winner."}`
      : "Result in; run score_race.py to add the official scores.";

  return (
    <main className="mx-auto max-w-[1240px] px-4 pt-8 pb-24 sm:px-8">
      {/* -------------------------------------------------------------- title row: the race and its finding */}
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div className="min-w-0 max-w-[720px]">
          <h1 className="text-[24px] font-semibold leading-tight">
            {race.season} {race.name}
            {race.live && <span className="ml-3 align-middle text-[13px] font-normal text-smoke">Live prediction</span>}
          </h1>
          <p className="mt-2 text-[15px] leading-[1.5] text-smoke">{finding}</p>
        </div>
        <RacePicker options={options} value={id} onChange={setId} />
      </div>

      {/* -------------------------------------------------------------- the scores, one divided strip */}
      <dl className="mt-8 grid grid-cols-2 bg-night sm:grid-cols-4">
        <Kpi label="Places off per driver" value={metrics ? places(metrics.model.mae) : "—"} sub={metrics ? `grid ${places(metrics.grid.mae)} · ${signed(metrics.model.mae - metrics.grid.mae)}` : "awaiting result"} lead />
        <Kpi label="Order correlation" value={metrics ? fixed(metrics.model.spearman, 3) : "—"} sub={metrics ? `grid ${fixed(metrics.grid.spearman, 3)}` : "Spearman, 1 is perfect"} />
        <Kpi label="Winner" value={metrics ? (metrics.model.winner_hit ? "Called" : "Missed") : "—"} sub={metrics ? `grid ${metrics.grid.winner_hit ? "called it" : "missed"}` : "predicted P1 won?"} />
        <Kpi label="Podium places right" value={metrics ? `${Math.round(metrics.model.podium_hit * 3)} /3` : "—"} sub={metrics ? `grid ${Math.round(metrics.grid.podium_hit * 3)} /3` : "top three overlap"} />
      </dl>

      {/* -------------------------------------------------------------- predicted vs finished, every driver */}
      <div className="mt-6 grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="bg-night p-5 sm:p-6" aria-labelledby="vs">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2 id="vs" className="text-[16px] font-semibold">Predicted against finished</h2>
            <Legend />
          </div>
          {scored ? <Dumbbell rows={race.rows} /> : <Pending rows={race.rows} />}
        </section>

        <section className="bg-night p-5 sm:p-6" aria-labelledby="misses">
          <h2 id="misses" className="text-[16px] font-semibold">Biggest misses</h2>
          {scored ? (
            <ol className="mt-4">
              {misses.map((r) => {
                const off = r.position! - r.predicted;
                return (
                  <li key={r.code} className="flex items-baseline justify-between gap-3 border-t border-graphite/70 py-3 first:border-0">
                    <div className="min-w-0">
                      <div className="text-[15px] font-medium">{r.name === r.code ? r.code : r.name}</div>
                      <div className="text-[13px] text-ash">
                        Predicted P{r.predicted}, finished P{r.position}
                        {r.status && DNF.test(r.status) ? " · retired" : ""}
                        {r.grid_penalty >= 5 ? " · penalty" : ""}
                      </div>
                    </div>
                    <span className="num text-[15px] text-ink">{signed(off, 0)}</span>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="mt-4 text-[14px] leading-[1.5] text-smoke">
              Nothing to compare yet. Pick any past race on the right to see how a scored race reads; each was predicted
              using only the races before it.
            </p>
          )}
        </section>
      </div>

      {/* -------------------------------------------------------------- every race: the model against the grid */}
      <section className="mt-6 bg-night p-5 sm:p-6" aria-labelledby="board">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 id="board" className="text-[16px] font-semibold">Every race since 2023</h2>
          <p className="text-[13px] text-ash">
            Places per driver saved against the grid · {clean.seasons.join("–")}: {clean.better} better, {clean.worse} worse,
            average <span className="num text-smoke">{signed(clean.gain)}</span> · {devSeasons.join("–")} were used for choices
            (all {summary.races}: {signed(summary.gain)}, optimistic)
          </p>
        </div>
        <Scoreboard options={options.filter((o) => !o.live)} selected={id} onSelect={setId} />
      </section>
    </main>
  );
}

function pick(s: Record<string, string | number | boolean | null>, prefix: string): Metrics {
  return {
    mae: Number(s[`${prefix}_mae`]), spearman: Number(s[`${prefix}_spearman`]),
    winner_hit: Number(s[`${prefix}_winner_hit`]), podium_hit: Number(s[`${prefix}_podium_hit`]),
  };
}

function Kpi({ label, value, sub, lead }: { label: string; value: string; sub: string; lead?: boolean }) {
  return (
    <div className="border-graphite/70 px-5 py-4 [&:not(:first-child)]:border-l max-sm:[&:nth-child(3)]:border-l-0 max-sm:[&:nth-child(n+3)]:border-t">
      <dt className="text-[13px] text-smoke">{label}</dt>
      <dd className={`num mt-1 font-semibold ${lead ? "text-[32px]" : "text-[24px]"}`}>
        {value === "—" ? <span className="text-[15px] font-normal text-ash">—</span> : value}
      </dd>
      <dd className="num mt-1 text-[13px] text-ash">{sub}</dd>
    </div>
  );
}

function RacePicker({ options, value, onChange }: { options: RaceOption[]; value: string; onChange: (v: string) => void }) {
  const i = options.findIndex((o) => o.id === value);
  const seasons = [...new Set(options.map((o) => o.season))];
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => onChange(options[Math.min(options.length - 1, i + 1)].id)}
        disabled={i === options.length - 1}
        className="h-9 w-9 border border-graphite text-[15px] transition-colors hover:border-ink disabled:opacity-40"
        aria-label="Earlier race"
      >
        ←
      </button>
      <label className="sr-only" htmlFor="race">Race</label>
      <select
        id="race"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-9 min-w-[220px] border border-graphite bg-void px-3 text-[14px] text-ink transition-colors hover:border-ink"
      >
        {seasons.map((s) => (
          <optgroup key={s} label={String(s)}>
            {options.filter((o) => o.season === s).map((o) => (
              <option key={o.id} value={o.id}>
                R{o.round} {short(o.name)}{o.live ? " (live)" : ""}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      <button
        type="button"
        onClick={() => onChange(options[Math.max(0, i - 1)].id)}
        disabled={i === 0}
        className="h-9 w-9 border border-graphite text-[15px] transition-colors hover:border-ink disabled:opacity-40"
        aria-label="Later race"
      >
        →
      </button>
    </div>
  );
}

function Legend() {
  return (
    <ul className="flex flex-wrap items-center gap-4 text-[12px] text-ash">
      <li className="flex items-center gap-2"><span className="inline-block size-2 rounded-full bg-rosso" />Predicted</li>
      <li className="flex items-center gap-2"><span className="inline-block size-2 rounded-full bg-ink" />Finished</li>
      <li className="flex items-center gap-2"><span className="inline-block h-3 w-[2px] bg-ash" />Started</li>
    </ul>
  );
}

/* One row per driver in finishing order; each row is an axis from P1 to the last place. */
function Dumbbell({ rows }: { rows: ResultRow[] }) {
  const n = rows.length;
  const sorted = [...rows].sort((a, b) => (a.position ?? 99) - (b.position ?? 99));
  const x = (p: number) => `${((p - 1) / (n - 1)) * 100}%`;
  return (
    <div className="mt-5">
      <div className="relative ml-[64px] mr-[44px] h-5 text-[12px] text-ash">
        {[1, 5, 10, 15, 20, n].filter((v, i, a) => a.indexOf(v) === i && v <= n).map((t) => (
          <span key={t} className="num absolute -translate-x-1/2" style={{ left: x(t) }}>P{t}</span>
        ))}
      </div>
      <ol>
        {sorted.map((r) => {
          const lo = Math.min(r.predicted, r.position!);
          const hi = Math.max(r.predicted, r.position!);
          const retired = !!r.status && DNF.test(r.status);
          return (
            <li key={r.code} className="group flex h-[28px] items-center" title={`${r.name}: started P${r.grid}, predicted P${r.predicted}, finished P${r.position}`}>
              <span className={`w-[64px] pr-3 text-right text-[13px] font-medium tracking-[0.03em] ${retired ? "text-ash" : "text-ink"}`}>{r.code}</span>
              <div className="relative h-full flex-1">
                <span className="absolute inset-x-0 top-1/2 h-px bg-graphite/50" />
                <span className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-[#6a6a6a]" style={{ left: x(lo), width: `calc(${x(hi)} - ${x(lo)})` }} />
                <span className="absolute top-1/2 h-3 w-[2px] -translate-x-1/2 -translate-y-1/2 bg-ash" style={{ left: x(r.grid) }} />
                <span className="absolute top-1/2 size-[9px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-rosso" style={{ left: x(r.predicted) }} />
                <span className={`absolute top-1/2 size-[9px] -translate-x-1/2 -translate-y-1/2 rounded-full ${retired ? "border border-ink bg-void" : "bg-ink"}`} style={{ left: x(r.position!) }} />
              </div>
              <span className="num w-[44px] text-right text-[13px] text-smoke">{signed(r.position! - r.predicted, 0).replace("±0", "0")}</span>
            </li>
          );
        })}
      </ol>
      <p className="mt-3 text-[12px] text-ash">Rows in finishing order. A hollow dot is a retirement, classified by laps completed.</p>
    </div>
  );
}

function Pending({ rows }: { rows: ResultRow[] }) {
  const sorted = [...rows].sort((a, b) => a.predicted - b.predicted);
  return (
    <div className="mt-5">
      <p className="text-[14px] text-smoke">The predicted order, waiting for the chequered flag and the stewards.</p>
      <ol className="mt-4 grid grid-cols-2 gap-x-8 sm:grid-cols-3">
        {sorted.map((r) => (
          <li key={r.code} className="flex items-baseline gap-3 border-t border-graphite/70 py-2">
            <span className="num w-8 text-[13px] text-ash">P{r.predicted}</span>
            <span className="text-[14px] font-medium">{r.name}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/* Every past race as a column above (model better) or below (grid better) one zero line. */
function Scoreboard({ options, selected, onSelect }: { options: RaceOption[]; selected: string; onSelect: (id: string) => void }) {
  const races = [...options].reverse();
  const gains = races.map((r) => r.grid!.mae - r.model!.mae);
  const max = Math.max(1.5, ...gains.map(Math.abs));
  const H = 84;
  const seasons = [...new Set(races.map((r) => r.season))];
  return (
    <div className="mt-6">
      <div className="flex">
        <div className="relative w-[44px] shrink-0 text-[12px] text-ash" style={{ height: H * 2 }}>
          <span className="num absolute right-2 top-0">+{fixed(max, 1)}</span>
          <span className="num absolute right-2" style={{ top: H - 8 }}>0</span>
          <span className="num absolute bottom-0 right-2">−{fixed(max, 1)}</span>
        </div>
        <div className="relative flex flex-1 items-stretch gap-[2px]" style={{ height: H * 2 }}>
          <span className="pointer-events-none absolute inset-x-0 h-px bg-[#5a5a5a]" style={{ top: H }} />
          {races.map((r, i) => {
            const g = gains[i];
            const h = (Math.abs(g) / max) * H;
            const sel = r.id === selected;
            return (
              <button
                key={r.id}
                type="button"
                onClick={() => onSelect(r.id)}
                className="group relative flex-1 focus-visible:outline-offset-0"
                aria-label={`${r.season} ${r.name}: model ${signed(g)} places per driver against the grid`}
                aria-pressed={sel}
              >
                <span
                  className={`absolute inset-x-0 transition-colors ${g >= 0 ? (sel ? "bg-ink" : "bg-rosso group-hover:bg-ink") : sel ? "bg-ink" : "bg-[#5a5a5a] group-hover:bg-ink"}`}
                  style={g >= 0 ? { bottom: H, height: h } : { top: H, height: h }}
                />
              </button>
            );
          })}
        </div>
      </div>
      <div className="ml-[44px] mt-2 flex text-[12px] text-ash">
        {seasons.map((s) => (
          <span key={s} className="num" style={{ flex: races.filter((r) => r.season === s).length }}>{s}</span>
        ))}
      </div>
      <p className="mt-3 text-[12px] text-ash">Red above the line: the model beat the grid that race. Click a race to open it.</p>
    </div>
  );
}
