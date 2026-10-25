import Link from "next/link";
import { RaceStatus } from "./RaceStatus";

const LINKS = [
  { href: "/", label: "Race" },
  { href: "/results", label: "Results" },
  { href: "/model", label: "Model" },
];

/* The product's mark: a line rising from the grid to a flag. */
export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" aria-hidden="true">
      <path d="M2 19 L13 7" stroke="var(--color-rosso)" strokeWidth="2.4" strokeLinecap="square" />
      <rect x="13" y="2" width="4" height="4" fill="#fff" />
      <rect x="17" y="6" width="4" height="4" fill="#fff" />
      <rect x="17" y="2" width="4" height="4" fill="none" stroke="#fff" strokeWidth="1" />
      <rect x="13" y="6" width="4" height="4" fill="none" stroke="#fff" strokeWidth="1" />
    </svg>
  );
}

export function TopBar({ active, startUtc, scored }: { active: string; startUtc: string; scored: boolean }) {
  return (
    <header className="relative z-20 border-b border-graphite/70">
      <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-6 px-4 sm:px-8">
        <Link href="/" className="flex items-center gap-3 caps text-ink hover:text-rosso transition-colors">
          <Mark />
          <span className="font-semibold tracking-[0.12em]">Grid to Flag</span>
        </Link>
        <nav aria-label="Main" className="ml-2 hidden items-center gap-5 sm:flex">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              aria-current={active === l.href ? "page" : undefined}
              className={`caps py-1 transition-colors hover:text-ink ${active === l.href ? "text-ink" : "text-smoke"}`}
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-4 border-l border-graphite/70 pl-4 sm:pl-6">
          <RaceStatus startUtc={startUtc} scored={scored} />
        </div>
      </div>
      <nav aria-label="Main" className="flex border-t border-graphite/70 sm:hidden">
        {LINKS.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            aria-current={active === l.href ? "page" : undefined}
            className={`caps flex-1 py-3 text-center ${active === l.href ? "text-ink" : "text-smoke"}`}
          >
            {l.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
