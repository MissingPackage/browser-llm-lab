// Tipi di `pairedstats.mjs`. Il modulo resta JS perche' vive in `scripts/`, che
// gira con `node` senza passare da tsc; ma i suoi casi di test sono TypeScript e
// senza queste firme sarebbero `any`, cioe' un test che non verifica nemmeno di
// star chiamando le funzioni giuste.
export interface BootstrapCI {
  mean: number; lo: number; hi: number;
  blockLen: number; resamples: number; seed: number; alpha: number;
}
export interface WinRate {
  better: number; worse: number; tie: number; n: number; fracBetter: number;
}
export interface PairedNll {
  n: number; bitsA: number; bitsB: number;
  deltaBits: { mean: number; lo: number; hi: number };
  blockLen: number; resamples: number; seed: number;
  win: WinRate;
  deltaTail: Record<string, number>;
}
export interface BootstrapOpts {
  blockLen?: number; resamples?: number; seed?: number; alpha?: number;
}

export function rng(seed: number): () => number;
export function mean(xs: ArrayLike<number> & Iterable<number>): number;
export const NATS_PER_BIT: number;
export function toBits(nats: number): number;
export function blockBootstrapCI(xs: ArrayLike<number> & Iterable<number>, opts?: BootstrapOpts): BootstrapCI;
export function winRate(deltas: ArrayLike<number> & Iterable<number>, eps?: number): WinRate;
export function quantiles(xs: ArrayLike<number>, ps?: number[]): Record<string, number>;
export function pairedNll(
  nllA: ArrayLike<number> & Iterable<number>,
  nllB: ArrayLike<number> & Iterable<number>,
  opts?: BootstrapOpts,
): PairedNll;
