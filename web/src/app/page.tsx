import Link from "next/link";
import { MotionRoot } from "@/components/MotionRoot";
import { SlopeChart } from "@/components/SlopeChart";
import { TopBar } from "@/components/TopBar";
import { fixed, loadBacktests, loadModel, loadRace, pct, signed, surname, utc, type Driver } from "@/lib/data";

export default async function RacePage() {
  const [data, bt, model] = await Promise.all([loadRace(), loadBacktests(), loadModel()]);
  const penalised = model.slices.find((s) => s.slice.startsWith("Penalised"));
  const { race, drivers } = data;
  const byPred = [...drivers].sort((a, b) => a.predicted - b.predicted);
  const byGrid = [...drivers].sort((a, b) => a.grid - b.grid);
  const [p1, p2, p3] = byPred;
  const scored = !!data.actual;

  return (
    <>
      <MotionRoot />
      <TopBar active="/" startUtc={race.start_utc} scored={scored} />
      <main className="overflow-x-clip">
        {/* ------------------------------------------------------------ hero: the race and the call */}
        <section className="mx-auto grid max-w-[1440px] gap-12 px-4 pt-10 pb-16 sm:px-8 lg:grid-cols-[1fr_380px] lg:gap-16 lg:pt-16">
          <div className="min-w-0">
            <h1 className="display text-[clamp(64px,12.6vw,196px)]">
              <span className="hero-line"><span style={{ ["--i" as string]: 0 }}>United States</span></span>
              <span className="hero-line"><span style={{ ["--i" as string]: 1 }}>Grand Prix</span></span>
            </h1>
            <p className="mt-8 max-w-[640px] text-[18px] leading-[1.5] text-ink sm:text-[20px]">
              {p1.name} is predicted to win from pole, taking {pct(p1.p_win)} of 20,000 simulated races to{" "}
              {surname(p2.name)}&apos;s {pct(p2.p_win)}. Circuit of the Americas, {race.laps} laps, lights out {utc(race.start_utc)}.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-6">
              <Link
                href="/results"
                className="group inline-flex h-12 items-center gap-3 bg-ink px-6 text-[13px] font-semibold uppercase tracking-[0.083em] text-void transition-colors hover:bg-rosso hover:text-ink active:bg-rosso-deep"
              >
                Check the results
                <span aria-hidden="true" className="transition-transform group-hover:translate-x-1">→</span>
              </Link>
              <a href="#every-driver" className="caps text-smoke transition-colors hover:text-rosso">All 22 drivers ↓</a>
            </div>

            <dl className="mt-14 grid grid-cols-3 gap-4 sm:gap-10">
              {[p1, p2, p3].map((d, i) => (
                <div key={d.code} data-reveal style={{ ["--i" as string]: i }}>
                  <dt className="text-[14px] text-smoke">
                    <span className="text-ink">{surname(d.name)}</span> to win
                  </dt>
                  <dd className="display num mt-2 text-[clamp(44px,7vw,104px)]" style={{ letterSpacing: "-0.02em" }}>
                    <span className={i === 0 ? "text-rosso" : "text-ink"}>{fixed(d.p_win * 100, 1)}</span>
                    <span className="text-[0.38em] text-ash">%</span>
                  </dd>
                  <dd className="mt-2 text-[13px] text-ash">From P{d.grid} · {d.team_name}</dd>
                </div>
              ))}
            </dl>
          </div>

          <StartingGrid drivers={byGrid} predictedWinner={p1.code} />
        </section>

        {/* ------------------------------------------------------------ the predicted order as a running ticker */}
        <div className="border-y border-graphite/70 py-4" aria-label="Predicted finishing order">
          <div className="marquee">
            {[0, 1].map((k) => (
              <div key={k} className="flex shrink-0 items-baseline gap-10 pr-10" aria-hidden={k === 1}>
                {byPred.map((d) => (
                  <span key={d.code} className="display flex items-baseline gap-3 text-[40px] sm:text-[56px]" style={{ letterSpacing: "-0.01em" }}>
                    <span className="num text-[0.5em] text-ash">P{d.predicted}</span>
                    <span className={d.predicted === 1 ? "text-rosso" : "text-ink"}>{surname(d.name)}</span>
                  </span>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* ------------------------------------------------------------ grid to flag */}
        <section className="mx-auto grid max-w-[1440px] gap-12 px-4 py-28 sm:px-8 lg:grid-cols-[minmax(0,420px)_1fr] lg:gap-20">
          <div data-reveal>
            <h2 className="display text-[clamp(48px,6vw,88px)]">Grid to flag</h2>
            <p className="mt-6 text-[17px] leading-[1.55] text-ink">
              Every line is one driver, from where they start to where the model has them finishing. Red lines move
              three places or more.
            </p>
            <p className="mt-4 text-[17px] leading-[1.55] text-smoke">
              The biggest climbs are the grid penalties: {movers(drivers).slice(0, 3).map((d) => `${surname(d.name)} ${d.grid}→${d.predicted}`).join(", ")}.
              Penalised drivers are where the model beats the grid most, by {penalised ? fixed(penalised.grid - penalised.model, 1) : "—"} places per driver since 2023.
            </p>
          </div>
          <div data-reveal style={{ ["--i" as string]: 1 }}>
            <SlopeChart
              fromLabel="Starting grid"
              toLabel="Predicted finish"
              rows={drivers.map((d) => ({
                code: d.code,
                name: d.name,
                from: d.grid,
                to: d.predicted,
                note: d.grid_penalty > 0 ? `${d.grid_penalty}-place grid penalty` : undefined,
              }))}
            />
          </div>
        </section>

        {/* ------------------------------------------------------------ every driver */}
        <section id="every-driver" className="bg-night">
          <div className="mx-auto max-w-[1440px] px-4 py-24 sm:px-8">
            <div className="flex flex-wrap items-end justify-between gap-6" data-reveal>
              <h2 className="display text-[clamp(48px,6vw,88px)]">Every driver</h2>
              <p className="max-w-[460px] text-[15px] leading-[1.55] text-smoke">
                Chances from 20,000 simulated races. Points means finishing in the top ten; DNF is the chance of not
                finishing, set by the team&apos;s 2026 reliability.
              </p>
            </div>
            <DriverTable drivers={byPred} />
          </div>
        </section>

        {/* ------------------------------------------------------------ strategy */}
        <section className="mx-auto max-w-[1440px] px-4 py-28 sm:px-8">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,420px)_1fr] lg:gap-20">
            <div data-reveal>
              <h2 className="display text-[clamp(48px,6vw,88px)]">Pit strategy</h2>
              <p className="mt-6 text-[17px] leading-[1.55] text-ink">
                One stop around half distance for most of the field. Tyre wear comes from this weekend&apos;s practice long
                runs, corrected for fuel and scaled to race conditions.
              </p>
              <p className="mt-4 text-[15px] leading-[1.55] text-smoke">
                Experimental: the stop count was right in 66% of 56 dry races against 55% for always calling one stop.
                Compound order isn&apos;t predicted reliably, so it isn&apos;t shown. Pit loss {data.strategy?.pit_loss_s} s.
              </p>
            </div>
            <StintStrip drivers={byPred} laps={race.laps} />
          </div>
        </section>

        {/* ------------------------------------------------------------ track history and honesty */}
        <section className="border-t border-graphite/70">
          <div className="mx-auto grid max-w-[1440px] gap-16 px-4 py-24 sm:px-8 lg:grid-cols-2">
            <div data-reveal>
              <h2 className="display text-[clamp(40px,4.6vw,68px)]">Austin before</h2>
              <ol className="mt-8">
                {data.track_history.map((w) => (
                  <li key={w.season} className="flex items-baseline justify-between border-b border-graphite/60 py-4">
                    <span className="num text-[15px] text-smoke">{w.season}</span>
                    <span className="display text-[40px]" style={{ letterSpacing: "0" }}>{w.code}</span>
                    <span className="num text-[15px] text-smoke">won from P{w.grid}</span>
                  </li>
                ))}
              </ol>
              <p className="mt-6 text-[15px] leading-[1.55] text-smoke">
                Pole won one of the last four here; the long run to turn one gives the second row a real chance.
              </p>
            </div>
            <div data-reveal style={{ ["--i" as string]: 1 }}>
              <h2 className="display text-[clamp(40px,4.6vw,68px)]">How sure</h2>
              <p className="mt-8 text-[17px] leading-[1.55] text-ink">
                On the {bt.clean.races} races of {bt.clean.seasons.join("–")}, never used for any choice and each predicted
                from earlier races only, the model is off by <span className="num">{fixed(bt.clean.mae_model, 2)}</span> places
                per driver against <span className="num">{fixed(bt.clean.mae_grid, 2)}</span> for simply copying the grid.
              </p>
              <p className="mt-4 text-[15px] leading-[1.55] text-smoke">
                A gain of {fixed(bt.clean.gain, 2)} places (95% interval {signed(bt.clean.ci[0], 3)} to {signed(bt.clean.ci[1], 3)}),
                better in {bt.clean.better} races and worse in {bt.clean.worse}: likely real, not proven. One race settles
                nothing; across reasonable settings Antonelli&apos;s win chance stays between 42% and 53%.
              </p>
              <Link href="/model" className="caps mt-8 inline-flex items-center gap-2 text-ink transition-colors hover:text-rosso">
                The full scorecard <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-night">
        <div className="mx-auto max-w-[1440px] px-4 pt-20 pb-10 sm:px-8">
          <p className="display flex items-center gap-[0.12em] text-[clamp(56px,14.5vw,224px)]" aria-label="Grid to Flag">
            <span>Grid</span>
            <span className="text-rosso">to</span>
            <span>Flag</span>
          </p>
          <div className="mt-10 grid gap-6 text-[13px] leading-[1.6] text-ash sm:grid-cols-3">
            <p>
              Official prediction <span className="text-smoke">{data.official.file}</span>, generated{" "}
              {utc(data.official.generated_at_utc)}, before lights out.
            </p>
            <p>
              Model {data.official.model.split(";")[0]} · commit <span className="text-smoke">{data.official.git_commit}</span> ·{" "}
              {data.official.data_source.replace("snapshot ", "data ")}.
            </p>
            <p>Results and grids from Jolpica; practice laps and stints from OpenF1. Not affiliated with Formula 1.</p>
          </div>
        </div>
      </footer>
    </>
  );
}

function movers(drivers: Driver[]) {
  return [...drivers].sort((a, b) => b.grid - b.predicted - (a.grid - a.predicted));
}

/* The starting grid drawn from the data: two staggered columns, pole at the top. */
function StartingGrid({ drivers, predictedWinner }: { drivers: Driver[]; predictedWinner: string }) {
  return (
    <figure className="self-start lg:pt-4" data-reveal style={{ ["--i" as string]: 2 }}>
      <figcaption className="mb-5 flex items-baseline justify-between text-[13px] text-ash">
        <span>Starting grid, after penalties</span>
        <span>Turn 1 ↑</span>
      </figcaption>
      <ol className="grid grid-cols-2 gap-x-5">
        {drivers.map((d) => {
          const left = d.grid % 2 === 1;
          const win = d.code === predictedWinner;
          return (
            <li
              key={d.code}
              className={`relative flex h-[46px] items-end justify-between border-x-2 border-t-2 px-3 pb-2 ${left ? "" : "translate-y-[23px]"} ${win ? "border-rosso" : "border-ink/80"}`}
              style={{ marginBottom: 8 }}
            >
              <span className="num text-[12px] text-ash">{d.grid}</span>
              <span className={`text-[15px] font-semibold tracking-[0.03em] ${win ? "text-rosso" : "text-ink"}`}>{d.code}</span>
              {d.grid_penalty > 0 && (
                <span className="absolute -top-[9px] right-2 bg-void px-1 text-[11px] text-smoke">+{d.grid_penalty}</span>
              )}
            </li>
          );
        })}
      </ol>
    </figure>
  );
}

function DriverTable({ drivers }: { drivers: Driver[] }) {
  return (
    <div className="mt-12 overflow-x-auto">
      <table className="w-full min-w-[760px] border-collapse text-left">
        <thead>
          <tr className="text-[12px] text-ash">
            <th className="w-[64px] py-3 pr-3 font-normal">Finish</th>
            <th className="py-3 pr-3 font-normal">Driver</th>
            <th className="num py-3 pr-6 text-right font-normal">Grid</th>
            <th className="py-3 pr-6 font-normal">Win</th>
            <th className="num py-3 pr-6 text-right font-normal">Podium</th>
            <th className="num py-3 pr-6 text-right font-normal">Points</th>
            <th className="num py-3 pr-6 text-right font-normal">DNF</th>
            <th className="py-3 font-normal">Strategy</th>
          </tr>
        </thead>
        <tbody>
          {drivers.map((d, i) => {
            const delta = d.grid - d.predicted;
            return (
              <tr key={d.code} className="group border-t border-graphite/70 transition-colors hover:bg-graphite/40" data-reveal style={{ ["--i" as string]: Math.min(i, 8) }}>
                <td className="display num py-3 pr-3 text-[32px]" style={{ letterSpacing: 0 }}>{d.predicted}</td>
                <td className="py-3 pr-3">
                  <div className="text-[16px] font-medium text-ink">{d.name}</div>
                  <div className="text-[13px] text-ash">{d.team_name}</div>
                </td>
                <td className="num py-3 pr-6 text-right text-[15px]">
                  <span className="text-ink">P{d.grid}</span>
                  <span className={`ml-2 text-[13px] ${delta >= 3 ? "text-ink" : "text-ash"}`}>{delta === 0 ? "" : signed(delta, 0)}</span>
                </td>
                <td className="py-3 pr-6">
                  <div className="flex items-center gap-3">
                    <div className="h-[6px] w-[140px] bg-graphite/60" aria-hidden="true">
                      <div className="bar-grow h-full bg-rosso" style={{ width: `${d.p_win * 100}%`, ["--i" as string]: i }} />
                    </div>
                    <span className="num w-[52px] text-right text-[15px] text-ink">{d.p_win >= 0.0005 ? pct(d.p_win) : "—"}</span>
                  </div>
                </td>
                <td className="num py-3 pr-6 text-right text-[15px] text-ink">{d.p_podium >= 0.0005 ? pct(d.p_podium) : "—"}</td>
                <td className="num py-3 pr-6 text-right text-[15px] text-ink">{pct(d.p_points)}</td>
                <td className="num py-3 pr-6 text-right text-[15px] text-smoke">{pct(d.p_dnf, 0)}</td>
                <td className="py-3 text-[14px] text-smoke">
                  {d.stops ? `${d.stops} stop${d.stops > 1 ? "s" : ""}, lap ${d.pit_laps?.join(" & ")}` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-6 text-[13px] text-ash">A dash means under 0.05%: possible, but not once in a meaningful share of simulations.</p>
    </div>
  );
}

function StintStrip({ drivers, laps }: { drivers: Driver[]; laps: number }) {
  const ticks = [1, 10, 20, 30, 40, 50, laps];
  return (
    <figure data-reveal style={{ ["--i" as string]: 1 }}>
      <div className="relative ml-[72px] h-6 text-[12px] text-ash">
        {ticks.map((t) => (
          <span key={t} className="num absolute -translate-x-1/2" style={{ left: `${((t - 1) / (laps - 1)) * 100}%` }}>
            {t === 1 ? "Lap 1" : t}
          </span>
        ))}
      </div>
      <ol className="space-y-[6px]">
        {drivers.map((d, i) => {
          const pits = d.pit_laps ?? [];
          const edges = [0, ...pits, laps];
          return (
            <li key={d.code} className="group flex items-center gap-4">
              <span className={`w-[56px] text-right text-[13px] font-medium tracking-[0.03em] ${d.stops === 2 ? "text-ink" : "text-smoke"}`}>{d.code}</span>
              <div className="relative flex h-[18px] flex-1 gap-[3px]" title={`${d.name}: ${d.stops} stop(s), lap ${pits.join(" & ")}`}>
                {edges.slice(1).map((end, k) => (
                  <div
                    key={k}
                    className={`bar-grow h-full ${k % 2 === 0 ? "bg-graphite" : "bg-[#454545]"} group-hover:bg-[#5c5c5c]`}
                    style={{ width: `${((end - edges[k]) / laps) * 100}%`, ["--i" as string]: i }}
                  />
                ))}
                {pits.map((p) => (
                  <span key={p} className="absolute top-[-3px] h-[24px] w-[3px] bg-rosso" style={{ left: `calc(${(p / laps) * 100}% - 1.5px)` }} />
                ))}
              </div>
              <span className="num w-[86px] text-[12px] text-ash">
                {d.stops === 2 ? `2 stops${d.two_stop_cost_s === 0 ? ", close" : ""}` : `1 stop`}
              </span>
            </li>
          );
        })}
      </ol>
      <figcaption className="mt-5 text-[13px] text-ash">Red marks are the predicted pit stops; each grey block is a stint.</figcaption>
    </figure>
  );
}
