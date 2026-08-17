// La statistica del confronto fra quant, sotto test — SENZA modello e senza GPU.
//
// PERCHE' ESISTE. La decisione «il Q2 e' ancora abbastanza intelligente?» poggia
// su questi numeri, e un intervallo di confidenza sbagliato non si vede a
// occhio: si vede solo quando qualcuno ci costruisce sopra un ruling. I casi qui
// pinnano le proprieta' che rendono leggibile il risultato — appaiamento,
// determinismo, e il rifiuto di confrontare cio' che non e' confrontabile.
import { describe, expect, it } from "vitest";
import {
  blockBootstrapCI, mean, pairedNll, quantiles, rng, toBits, winRate,
} from "../scripts/lib/pairedstats.mjs";

describe("pairedstats: il confronto appaiato fra due quantizzazioni", () => {
  it("rifiuta due vettori di lunghezza diversa invece di allinearli in silenzio", () => {
    // se i due modelli non hanno visto gli stessi token, ogni statistica qui
    // dentro direbbe una cosa diversa da quella che sembra
    expect(() => pairedNll([1, 2, 3], [1, 2])).toThrow(/confronto appaiato/);
  });

  it("un delta costante viene ritrovato esattamente, e in BIT", () => {
    const a = Array.from({ length: 4096 }, (_, i) => 1 + Math.sin(i) * 0.5);
    const b = a.map((x) => x + Math.LN2); // esattamente +1 bit per token
    const r = pairedNll(a, b, { blockLen: 128, resamples: 200 });
    expect(r.deltaBits.mean).toBeCloseTo(1, 10);
    // delta costante ⇒ ogni ricampionamento da' la stessa media: intervallo nullo
    expect(r.deltaBits.hi - r.deltaBits.lo).toBeCloseTo(0, 10);
    expect(r.win.worse).toBe(4096);
  });

  it("l'intervallo e' DETERMINISTICO a parita' di seme, e cambia col seme", () => {
    const rnd = rng(7);
    const xs = Array.from({ length: 2000 }, () => rnd() - 0.5);
    const a = blockBootstrapCI(xs, { seed: 42, resamples: 300 });
    const b = blockBootstrapCI(xs, { seed: 42, resamples: 300 });
    const c = blockBootstrapCI(xs, { seed: 43, resamples: 300 });
    expect(a).toEqual(b);
    expect(c.lo).not.toBe(a.lo);
    // e la media osservata non dipende dal ricampionamento
    expect(a.mean).toBeCloseTo(mean(xs), 12);
  });

  it("l'intervallo CONTIENE la media osservata e si stringe coi dati", () => {
    const rnd = rng(11);
    const gauss = () => { let s = 0; for (let i = 0; i < 12; i++) s += rnd(); return s - 6; };
    const corto = Array.from({ length: 512 }, gauss);
    const lungo = Array.from({ length: 8192 }, gauss);
    const a = blockBootstrapCI(corto, { blockLen: 64, resamples: 400 });
    const b = blockBootstrapCI(lungo, { blockLen: 64, resamples: 400 });
    expect(a.lo).toBeLessThanOrEqual(a.mean);
    expect(a.hi).toBeGreaterThanOrEqual(a.mean);
    expect(b.hi - b.lo).toBeLessThan(a.hi - a.lo);
  });

  it("il blocco conserva l'autocorrelazione: su dati correlati l'intervallo NON si restringe come su dati indipendenti", () => {
    // serie molto correlata: un blocco da 1 (= bootstrap classico) crederebbe di
    // avere 4096 osservazioni indipendenti mentre ne ha molte meno
    const rnd = rng(3);
    const xs = [];
    let v = 0;
    for (let i = 0; i < 4096; i++) { v = 0.99 * v + (rnd() - 0.5); xs.push(v); }
    const indip = blockBootstrapCI(xs, { blockLen: 1, resamples: 500, seed: 5 });
    const blocchi = blockBootstrapCI(xs, { blockLen: 512, resamples: 500, seed: 5 });
    expect(blocchi.hi - blocchi.lo).toBeGreaterThan(2 * (indip.hi - indip.lo));
  });

  it("il tasso di vittorie e' un test di SEGNO: immune alla coda che trascina la media", () => {
    // B e' meglio su 999 token e catastrofico su 1: la media dice «peggio»,
    // il segno dice «quasi sempre meglio». Servono entrambi, ed e' il punto.
    const d = Array.from({ length: 1000 }, () => -0.01);
    d[0] = 500;
    expect(mean(d)).toBeGreaterThan(0);
    expect(winRate(d).fracBetter).toBeCloseTo(0.999, 6);
    expect(quantiles(d, [1])["1"]).toBe(500);
  });

  it("toBits converte i nat in bit e la media vuota LANCIA invece di dare NaN", () => {
    expect(toBits(Math.LN2)).toBeCloseTo(1, 12);
    expect(() => mean([])).toThrow(/vuoto/);
    expect(() => blockBootstrapCI([])).toThrow(/vuoto/);
  });
});
