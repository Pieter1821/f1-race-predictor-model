"use client";

import { useEffect, useState } from "react";

/* The live element: counts down to lights out, then says what the scoring is waiting for. */
function format(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export function RaceStatus({ startUtc, scored }: { startUtc: string; scored: boolean }) {
  const start = new Date(startUtc).getTime();
  const [now, setNow] = useState<number | null>(null);

  useEffect(() => {
    // The clock starts after hydration (the server can't know the visitor's time), via a scheduled first tick.
    const tick = () => setNow(Date.now());
    const first = setTimeout(tick, 0);
    const t = setInterval(tick, 1000);
    return () => {
      clearTimeout(first);
      clearInterval(t);
    };
  }, []);

  if (scored) return <span className="caps text-smoke">Result scored</span>;
  if (now == null) return <span className="caps text-smoke">Lights out 22:00 SAST</span>;
  const left = start - now;
  if (left > 0) {
    return (
      <span className="caps flex items-baseline gap-2 text-smoke">
        Lights out in <span className="num text-[13px] tracking-[0.04em] text-ink">{format(left)}</span>
      </span>
    );
  }
  const running = left > -2 * 3600 * 1000;
  return (
    <span className="caps flex items-center gap-2 text-ink">
      <span className={`inline-block size-2 rounded-full bg-rosso ${running ? "animate-pulse" : ""}`} aria-hidden="true" />
      {running ? "Race running" : "Awaiting the official result"}
    </span>
  );
}
