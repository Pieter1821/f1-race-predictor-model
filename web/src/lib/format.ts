/* Pure formatting helpers: safe in browser and server components alike. */
/* Decimal half-up rounding: 0.4855 shows as 48.6%, matching the pipeline, where toFixed would say 48.5. */
export const round = (v: number, digits: number) => Number(Math.round(Number(`${v}e${digits}`)) + `e-${digits}`);
export const fixed = (v: number, digits: number) => round(v, digits).toFixed(digits);
export const pct = (v: number | null | undefined, digits = 1) =>
  v == null ? "—" : `${fixed(v * 100, digits)}%`;
export const places = (v: number | null | undefined, digits = 2) => (v == null ? "—" : fixed(v, digits));
export const signed = (v: number | null | undefined, digits = 2) =>
  v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : "±"}${fixed(Math.abs(v), digits)}`;
export const surname = (name: string) => name.split(" ").slice(1).join(" ") || name;
export const utc = (iso: string) =>
  new Date(iso).toLocaleString("en-ZA", { timeZone: "Africa/Johannesburg", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) + " SAST";
