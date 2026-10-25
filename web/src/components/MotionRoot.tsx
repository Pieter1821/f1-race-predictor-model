"use client";

import { useEffect } from "react";

/* Reveals [data-reveal] blocks once as they enter. Without JS (or with reduced motion) nothing is hidden. */
export function MotionRoot() {
  useEffect(() => {
    const root = document.documentElement;
    root.classList.add("js-motion");
    const els = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add("is-in");
            io.unobserve(e.target);
          }
        }
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.08 },
    );
    els.forEach((el) => io.observe(el));
    return () => {
      io.disconnect();
      root.classList.remove("js-motion");
    };
  }, []);
  return null;
}
