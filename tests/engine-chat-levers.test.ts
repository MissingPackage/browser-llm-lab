// La chat accende le leve misurate — e le dichiara nell'artefatto (it.39).
//
// PERCHE' ESISTE, ed e' il difetto piu' caro trovato in questo goal.
// `kfan` (riga 2c) e la rotta split-K (riga 2d) nascono SPENTE, perche' i loro
// A/B devono accenderle a caldo su un braccio solo dentro lo stesso processo.
// Nessuno le riaccendeva nel path di PRODUZIONE: per giorni la chat — l'unico
// path che un utente tocca — ha girato senza entrambe, mentre i digest
// riportavano 40,06 tok/s.
//
//     senza leve   22,58 tok/s
//     + kfan       28,90
//     + rotta      40,06   (a caldo, zero miss)
//
// Il difetto non era in un kernel: era che «misurato» e «consegnato» non
// coincidevano, e niente lo diceva. Questi casi rendono la differenza
// meccanica invece che affidata a chi si ricorda.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

const src = readFileSync(new URL("../src/engine/chat/chat.worker.ts", import.meta.url), "utf8");

describe("chat: le leve misurate sono accese in produzione", () => {
  it("accende il kfan, e solo dove esiste", () => {
    expect(src).toMatch(/model\.setKfan\(true\)/);
    // `setKfan(true)` LANCIA su un modello senza expert MoE: 4B e 9B sono
    // densi, e chiederlo li' farebbe morire il load invece di degradare
    expect(src).toMatch(/const kfanOn = isMoe && cfg\.select === "optimistic"/);
    expect(src).toMatch(/if \(kfanOn\) model\.setKfan\(true\)/);
  });

  it("accende la rotta split-K, e solo se il piano ha instradato qualcosa", () => {
    expect(src).toMatch(/model\.splitkAvail\(\)/);
    expect(src).toMatch(/if \(splitkOn\) model\.setSplitk\(true\)/);
    // stessa ragione: `setSplitk(true)` lancia se nessun tensore e' instradato,
    // quindi la disponibilita' va CHIESTA prima. Si confrontano le due RIGHE DI
    // CODICE e non le prime occorrenze nel file: i commenti qui sopra nominano
    // `setSplitk(true)` prima della chiamata, e un indexOf nudo leggerebbe
    // quelli — e' esattamente il modo in cui un test sul sorgente dice il falso.
    const iAvail = src.indexOf("const splitkOn = model.splitkAvail()");
    const iSet = src.indexOf("if (splitkOn) model.setSplitk(true)");
    expect(iAvail).toBeGreaterThan(0);
    expect(iAvail).toBeLessThan(iSet);
  });

  it("dichiara nell'artefatto QUALI leve sono attive", () => {
    // senza questo campo un JSON di chat non dice se il suo numero viene dal
    // motore con le leve o senza — e i due differiscono di 1,8x
    expect(src).toMatch(/levers:\s*\{\s*kfan:\s*kfanOn,\s*splitk:\s*splitkOn\s*\}/);
  });

  it("le leve si accendono PRIMA del primo token, non al primo turno", () => {
    // se si accendessero dentro il ciclo di generazione, il turno freddo — che
    // e' quello che l'utente aspetta di piu' — girerebbe senza
    const iSet = src.indexOf("model.setKfan(true)");
    const iLoaded = src.indexOf('type: "loaded"');
    expect(iSet).toBeGreaterThan(0);
    expect(iSet).toBeLessThan(iLoaded);
  });
});
