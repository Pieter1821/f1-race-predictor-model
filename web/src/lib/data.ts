import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import type { Backtests, ModelData, RaceData } from "./types";

export * from "./types";
export * from "./format";

/* Server-only: reads the exported JSON. Client components import ./format and ./types instead. */
async function load<T>(name: string): Promise<T> {
  "use cache"; // static per export: the JSON only changes when scripts/export_web_data.py runs
  const file = path.join(process.cwd(), "public", "data", `${name}.json`);
  return JSON.parse(await readFile(file, "utf8")) as T;
}
export const loadRace = () => load<RaceData>("race");
export const loadBacktests = () => load<Backtests>("backtests");
export const loadModel = () => load<ModelData>("model");
