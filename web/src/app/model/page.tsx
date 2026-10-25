import { TopBar } from "@/components/TopBar";
import { ExperimentsTable } from "@/components/ExperimentsTable";
import { fixed, loadBacktests, loadModel, loadRace, pct, places, signed } from "@/lib/data";

export const metadata = { title: "Model · Grid to Flag" };

export default async function ModelPage() {
  const [race, bt, model] = await Promise.all([loadRace(), loadBacktests(), loadModel()]);
  const s = bt.summary;
  const c = bt.clean;
  const dev = new Set(bt.dev_seasons);
  const devGain = bt.by_season.filter((r) => dev.has(r.season)).reduce((a, r) => a + (r.grid - r.model), 0) / bt.dev_seasons.length;

  return (
    <>
      <TopBar active="/model" startUtc={race.race.start_utc} scored={!!race.actual} />
      <main className="mx-auto max-w-[1240px] px-4 pt-8 pb-24 sm:px-8">
        <h1 className="text-[24px] font-semibold leading-tight">Model scorecard</h1>
        <p className="mt-2 max-w-[760px] text-[15px] leading-[1.5] text-smoke">
          On the {c.races} races of {c.seasons.join("–")}, never used for any choice, the model beats copying the grid by{" "}
          {places(c.gain)} places per driver (95% interval {signed(c.ci[0], 3)} to {signed(c.ci[1], 3)}): likely real, not
          proven. Its edge is in the midfield and on penalised drivers; at the front it is no better than the grid.
        </p>

        {/* -------------------------------------------------------------- the headline, one divided strip */}
        <dl className="mt-8 grid grid-cols-2 bg-night sm:grid-cols-4">
          <Cell label={`Places saved per driver, ${c.seasons.join("–")}`} value={signed(c.gain)} sub={`95% interval ${signed(c.ci[0], 3)} to ${signed(c.ci[1], 3)}`} lead />
          <Cell label="Model, places off" value={places(c.mae_model)} sub={`grid ${places(c.mae_grid)}`} />
          <Cell label="Races the model won" value={`${c.better}`} sub={`lost ${c.worse} · level ${c.races - c.better - c.worse}`} />
          <Cell label="Model in use" value="v1" sub="v2 from the next race" />
        </dl>

        <div className="mt-6 grid items-start gap-6 lg:grid-cols-2">
          {/* -------------------------------------------------------------- by season */}
          <section className="bg-night p-5 sm:p-6" aria-labelledby="seasons">
            <h2 id="seasons" className="text-[16px] font-semibold">Season by season</h2>
            <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse text-[14px]">
              <thead>
                <tr className="text-[12px] text-ash">
                  <th className="py-2 text-left font-normal">Season</th>
                  <th className="num py-2 text-right font-normal">Races</th>
                  <th className="num py-2 text-right font-normal">Model</th>
                  <th className="num py-2 text-right font-normal">Grid</th>
                  <th className="num py-2 text-right font-normal">Saved</th>
                  <th className="num py-2 text-right font-normal">Winner, model / grid</th>
                </tr>
              </thead>
              <tbody>
                {bt.by_season.map((r) => (
                  <tr key={r.season} className="border-t border-graphite/70">
                    <td className="py-3">
                      {r.season}
                      {dev.has(r.season) && <span className="ml-2 text-[12px] text-ash">used for choices</span>}
                    </td>
                    <td className="num py-3 text-right text-smoke">{r.races}</td>
                    <td className="num py-3 text-right">{places(r.model)}</td>
                    <td className="num py-3 text-right text-smoke">{places(r.grid)}</td>
                    <td className={`num py-3 text-right ${r.grid - r.model > 0 ? "text-ink" : "text-smoke"}`}>{signed(r.grid - r.model)}</td>
                    <td className="num py-3 text-right text-smoke">{pct(r.model_winner, 0)} / {pct(r.grid_winner, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
            <p className="mt-4 text-[12px] leading-[1.5] text-ash">
              {bt.dev_seasons.join(" and ")} were used to choose features, tuning and the simulation, so their gain
              ({signed(devGain)} on average) is flattered, about {fixed(devGain / c.gain, 0)}× the clean {c.seasons.join("–")} gain:
              keeping whichever option happened to do best on them bakes their luck in. All {s.races} races together:{" "}
              {signed(s.gain)} (optimistic).
            </p>
          </section>

          {/* -------------------------------------------------------------- where it helps */}
          <section className="bg-night p-5 sm:p-6" aria-labelledby="slices">
            <h2 id="slices" className="text-[16px] font-semibold">Where it beats the grid</h2>
            <ul className="mt-4">
              {model.slices.map((r) => {
                const gain = r.grid - r.model;
                const w = Math.min(100, (Math.abs(gain) / 4.5) * 100);
                return (
                  <li key={r.slice} className="border-t border-graphite/70 py-3 first:border-0">
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="text-[14px]">{r.slice} <span className="text-ash">· {r.rows} drivers</span></span>
                      <span className="num text-[14px]">{signed(gain)}</span>
                    </div>
                    <div className="mt-2 h-[3px] bg-graphite/60">
                      <div className={`h-full ${gain > 0 ? "bg-rosso" : "bg-[#5a5a5a]"}`} style={{ width: `${w}%` }} />
                    </div>
                  </li>
                );
              })}
            </ul>
            <p className="mt-4 text-[12px] text-ash">Places per driver saved against the grid; grey where the grid did better.</p>
          </section>
        </div>

        {/* -------------------------------------------------------------- calibration */}
        <section className="mt-6 bg-night p-5 sm:p-6" aria-labelledby="calibration">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2 id="calibration" className="text-[16px] font-semibold">Do the probabilities hold?</h2>
            <p className="text-[13px] text-ash">2025–26 races · on the diagonal means a 30% call came true 30% of the time</p>
          </div>
          <div className="mt-5 grid gap-6 sm:grid-cols-3">
            {model.calibration.map((c) => <Calibration key={c.kind} kind={c.kind} bins={c.bins} />)}
          </div>
        </section>

        {/* -------------------------------------------------------------- experiments */}
        <section className="mt-6 bg-night p-5 sm:p-6" aria-labelledby="experiments">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2 id="experiments" className="text-[16px] font-semibold">Every experiment, failures included</h2>
            <p className="text-[13px] text-ash">A candidate beats the noise in 3 of 4 seasons; only an adoption check changes the model</p>
          </div>
          <ExperimentsTable rows={model.experiments} />
        </section>
      </main>
    </>
  );
}

function Cell({ label, value, sub, lead }: { label: string; value: string; sub: string; lead?: boolean }) {
  return (
    <div className="border-graphite/70 px-5 py-4 [&:not(:first-child)]:border-l max-sm:[&:nth-child(3)]:border-l-0 max-sm:[&:nth-child(n+3)]:border-t">
      <dt className="text-[13px] text-smoke">{label}</dt>
      <dd className={`num mt-1 font-semibold ${lead ? "text-[32px] text-rosso" : "text-[24px]"}`}>{value}</dd>
      <dd className="num mt-1 text-[13px] text-ash">{sub}</dd>
    </div>
  );
}

function Calibration({ kind, bins }: { kind: string; bins: { predicted: number; actual: number; n: number }[] }) {
  const S = 220, P = 40, R = 16;
  const sx = (v: number) => P + v * (S - P - R);
  const sy = (v: number) => S - 28 - v * (S - 28 - 10);
  const pts = bins.map((b) => `${sx(b.predicted)},${sy(b.actual)}`).join(" ");
  return (
    <figure>
      <figcaption className="mb-2 text-[14px] font-medium">{kind}</figcaption>
      <svg viewBox={`0 0 ${S} ${S}`} className="w-full max-w-[320px]" role="img" aria-label={`${kind} calibration`}>
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={sx(0)} x2={sx(1)} y1={sy(t)} y2={sy(t)} stroke="var(--color-grid)" />
            <text x={P - 6} y={sy(t) + 4} textAnchor="end" fontSize="12" fill="var(--color-ash)">{fixed(t * 100, 0)}%</text>
            <text x={sx(t)} y={S - 8} textAnchor="middle" fontSize="12" fill="var(--color-ash)">{fixed(t * 100, 0)}%</text>
          </g>
        ))}
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#5a5a5a" strokeDasharray="3 3" />
        <polyline points={pts} fill="none" stroke="var(--color-rosso)" strokeWidth="2" />
        {bins.map((b, i) => <circle key={i} cx={sx(b.predicted)} cy={sy(b.actual)} r={2.5} fill="var(--color-rosso)" />)}
      </svg>
    </figure>
  );
}
