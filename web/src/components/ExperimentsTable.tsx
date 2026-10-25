"use client";

import { useState } from "react";
import { signed } from "@/lib/format";
import type { ModelData } from "@/lib/types";

type Row = ModelData["experiments"][number];
type Status = Row["status"];

/* Adopted is the only state that changed the model; a candidate passed the rules but was never adopted. */
const PILL: Record<Status, string> = {
  adopted: "bg-[#172a1d] text-[#3fc07a]",
  candidate: "bg-graphite/70 text-ink",
  rejected: "bg-graphite/40 text-smoke",
  check: "bg-graphite/20 text-ash",
};
const FILTERS: { id: "all" | Status; label: string }[] = [
  { id: "all", label: "All" },
  { id: "adopted", label: "Adopted" },
  { id: "candidate", label: "Candidates" },
  { id: "rejected", label: "Rejected" },
  { id: "check", label: "Checks" },
];

export function ExperimentsTable({ rows }: { rows: Row[] }) {
  const [filter, setFilter] = useState<"all" | Status>("all");
  const [reruns, setReruns] = useState(false);
  const base = reruns ? rows : rows.filter((r) => r.latest);
  const shown = base.filter((r) => filter === "all" || r.status === filter);
  const count = (id: "all" | Status) => (id === "all" ? base.length : base.filter((r) => r.status === id).length);
  const hidden = rows.length - rows.filter((r) => r.latest).length;

  return (
    <>
      <div className="mt-4 flex flex-wrap items-end justify-between gap-4 border-b border-graphite/70">
        <div className="flex flex-wrap gap-5" role="tablist" aria-label="Filter experiments">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              role="tab"
              aria-selected={filter === f.id}
              onClick={() => setFilter(f.id)}
              className={`-mb-px border-b-2 pb-2 text-[14px] transition-colors ${filter === f.id ? "border-rosso text-ink" : "border-transparent text-smoke hover:text-ink"}`}
            >
              {f.label} <span className="num text-ash">{count(f.id)}</span>
            </button>
          ))}
        </div>
        {hidden > 0 && (
          <label className="mb-2 flex cursor-pointer items-center gap-2 text-[13px] text-smoke">
            <input type="checkbox" checked={reruns} onChange={(e) => setReruns(e.target.checked)} className="accent-rosso" />
            Include {hidden} earlier re-runs
          </label>
        )}
      </div>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse text-[14px]">
          <thead>
            <tr className="text-[12px] text-ash">
              <th className="py-2 text-left font-normal">Change</th>
              <th className="num py-2 text-right font-normal">Effect on places off</th>
              <th className="num py-2 text-right font-normal">95% interval</th>
              <th className="num py-2 text-right font-normal">Seasons better</th>
              <th className="py-2 pl-6 text-left font-normal">Decision</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr key={`${r.date_utc}-${i}`} className="border-t border-graphite/70 align-top">
                <td className="max-w-[440px] py-3 pr-4">{r.change}</td>
                <td className="num py-3 text-right">{r.mae_change == null ? <span className="text-ash">—</span> : signed(r.mae_change, 3)}</td>
                <td className="num py-3 text-right text-smoke">
                  {r.ci_low == null || r.ci_high == null ? <span className="text-ash">—</span> : `${signed(r.ci_low, 3)} to ${signed(r.ci_high, 3)}`}
                </td>
                <td className="num py-3 text-right text-smoke">{r.better_seasons == null ? <span className="text-ash">—</span> : `${r.better_seasons} /4`}</td>
                <td className="py-3 pl-6">
                  <span className={`inline-block whitespace-nowrap px-2 py-[2px] text-[12px] ${PILL[r.status]}`}>{r.decision}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[12px] leading-[1.5] text-ash">
        Negative effect = fewer places off, better. A simplification is a candidate only if its worst case (the
        interval&apos;s upper end) stays within the +0.022 noise floor. A dash means the run measured something else.
      </p>
    </>
  );
}
