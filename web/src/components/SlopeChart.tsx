"use client";

import { useEffect, useRef, useState } from "react";

export type SlopeRow = { code: string; name: string; from: number; to: number; note?: string };

/* The signature element: every driver as a line from where they start to where they finish.
   Movers of 3+ places carry the data hue; the rest stay quiet. Labels live in HTML so they never scale. */
const PITCH = 30;

export function SlopeChart({
  rows,
  fromLabel,
  toLabel,
  highlight = 3,
}: {
  rows: SlopeRow[];
  fromLabel: string;
  toLabel: string;
  highlight?: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  const [hover, setHover] = useState<string | null>(null);
  const n = rows.length;
  const h = n * PITCH;

  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(120, e.contentRect.width)));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);

  const y = (p: number) => (p - 0.5) * PITCH;
  const byFrom = [...rows].sort((a, b) => a.from - b.from);
  const byTo = [...rows].sort((a, b) => a.to - b.to);
  const moved = (r: SlopeRow) => Math.abs(r.from - r.to) >= highlight;

  const label = (r: SlopeRow, side: "from" | "to") => {
    const pos = side === "from" ? r.from : r.to;
    const active = hover === r.code;
    const loud = moved(r) || active;
    return (
      <li
        key={`${side}-${r.code}`}
        style={{ height: PITCH }}
        className={`flex items-center gap-3 ${side === "to" ? "flex-row" : "flex-row-reverse"} cursor-default`}
        onMouseEnter={() => setHover(r.code)}
        onMouseLeave={() => setHover(null)}
      >
        <span className="num w-7 text-[13px] text-ash text-center">{pos}</span>
        <span className={`text-[14px] font-medium tracking-[0.02em] transition-colors ${loud ? "text-ink" : "text-smoke"}`}>
          {r.code}
        </span>
      </li>
    );
  };

  return (
    <figure className="w-full">
      <div className="mb-3 grid grid-cols-[64px_1fr_64px] text-[12px] text-ash sm:grid-cols-[88px_1fr_88px]">
        <span className="text-right">{fromLabel}</span>
        <span />
        <span>{toLabel}</span>
      </div>
      <div className="grid grid-cols-[64px_1fr_64px] sm:grid-cols-[88px_1fr_88px]">
        <ol aria-label={fromLabel}>{byFrom.map((r) => label(r, "from"))}</ol>
        <div ref={box} className="relative" style={{ height: h }}>
          <svg width={w} height={h} className="absolute inset-0 overflow-visible" role="img"
               aria-label={`Lines from ${fromLabel.toLowerCase()} to ${toLabel.toLowerCase()} for ${n} drivers`}>
            {[...Array(n)].map((_, i) => (
              <line key={i} x1={0} x2={w} y1={y(i + 1)} y2={y(i + 1)} stroke="var(--color-grid)" strokeWidth={1} opacity={0.35} />
            ))}
            {[...rows]
              .sort((a, b) => Number(moved(a) || hover === a.code) - Number(moved(b) || hover === b.code))
              .map((r, i) => {
                const active = hover === r.code;
                const loud = moved(r);
                const len = Math.hypot(w, y(r.to) - y(r.from));
                return (
                  <g key={r.code} onMouseEnter={() => setHover(r.code)} onMouseLeave={() => setHover(null)}>
                    <line x1={0} y1={y(r.from)} x2={w} y2={y(r.to)} stroke="transparent" strokeWidth={14} />
                    <line
                      className="draw-line"
                      style={{ ["--len" as string]: len, ["--i" as string]: i }}
                      x1={0} y1={y(r.from)} x2={w} y2={y(r.to)}
                      stroke={active ? "#fff" : loud ? "var(--color-rosso)" : "#5a5a5a"}
                      strokeWidth={active || loud ? 2.5 : 1.5}
                    />
                    <circle cx={0} cy={y(r.from)} r={3} fill={loud || active ? "#fff" : "#5a5a5a"} />
                    <circle cx={w} cy={y(r.to)} r={3.5} fill={active ? "#fff" : loud ? "var(--color-rosso)" : "#8a8a8a"} />
                  </g>
                );
              })}
          </svg>
        </div>
        <ol aria-label={toLabel}>{byTo.map((r) => label(r, "to"))}</ol>
      </div>
      <figcaption className="mt-4 min-h-6 text-[14px] text-smoke">
        {(() => {
          const r = rows.find((x) => x.code === hover);
          if (!r) return "Hover a line or a driver to follow them.";
          const d = r.from - r.to;
          return `${r.name}: P${r.from} to P${r.to} (${d > 0 ? `${d} places gained` : d < 0 ? `${-d} places lost` : "no change"})${r.note ? ` · ${r.note}` : ""}`;
        })()}
      </figcaption>
    </figure>
  );
}
