import type { DeviceProbe, BenchCell } from "../schema";
import type { ProfCounters } from "./profiler";

// Un campione dei contatori: snapshot cumulativo + fase corrente (ultimo progress del
// BenchServer). L'analisi offline ricava le finestre (load/warmup/decode) dalle fasi e
// i rapporti dai Δ fra campioni — stesso metodo del tool Playwright.
export interface ProfSample extends ProfCounters {
  tMs: number;
  phase: string;
}

// Stessa forma del JSON del tool (results/dispatch-profile/), con due differenze
// dichiarate: `source` distingue il produttore, e al posto del testo grezzo della
// tabella c'è il BenchCell completo (tok/s inclusi — serve a correlare i contatori
// con il rate osservato sul device).
export interface ProfRunFile {
  schemaVersion: 1;
  kind: "dispatch-profile";
  source: "prof-page";
  deviceLabel: string;
  ts: string;
  probe: DeviceProbe;
  status: string;
  sampleMs: number;
  clockMinDeltaMs: number | null;
  missingApis: string[];
  totals: ProfCounters;
  samples: ProfSample[];
  cell: BenchCell | null;
}
