// Test di `src/engine/gemvcaps.ts` (T1-gemv-caps).
//
// Il gate esiste per una ragione MISURATA in fase 0 (gate della cella
// `vec4-rows2-sg` in src/microbench/kdRunner.ts): la forma vincente mette 2
// righe in un workgroup da 64 thread assumendo UNA riga per subgroup di 32
// lane. Con subgroupSize 16 o 64 la mappatura riga->subgroup si rompe in
// SILENZIO: nessun errore, solo risultati sbagliati. Quindi non basta che la
// feature "subgroups" ci sia — serve la dimensione FISSA a 32, cioè
// subgroupMinSize === subgroupMaxSize === 32.
//
// Le due condizioni sulla dimensione vanno esercitate UNA PER VOLTA: un test
// che guarda solo 32/32 contro 16/64 non distingue `min===32 && max===32` da
// `min===32` da solo, e un'implementazione che controlla una condizione sola
// passerebbe. Da qui i casi 32/64 e 16/32, e il caso `adapterInfo: {}` (campi
// assenti), che smaschera un default implicito tipo `subgroupMinSize ?? 32`.
//
// Device FINTI: nessuna GPU, il modulo vede solo `features.has` e `adapterInfo`.
import { describe, expect, it } from "vitest";
import { gemvCapsFor, GEMV_SG_FEATURE, type GemvCaps, type GemvCapsSource } from "../src/engine/gemvcaps";

/** Device finto: un set di feature + un adapterInfo opzionale, niente altro. */
function fakeDevice(features: string[], adapterInfo?: GemvCapsSource["adapterInfo"]): GemvCapsSource {
  const set = new Set(features);
  return { features: { has: (f: string) => set.has(f) }, adapterInfo };
}

const SG = GEMV_SG_FEATURE as string;

describe("gemvCapsFor — gate del percorso subgroup di gemv", () => {
  it("(1) feature 'subgroups' assente -> sg false, anche se l'adapterInfo dichiara 32/32", () => {
    const caps = gemvCapsFor(fakeDevice(["shader-f16"], { subgroupMinSize: 32, subgroupMaxSize: 32 }));
    expect(caps.sg).toBe(false);
    expect(caps.why.length).toBeGreaterThan(0);
  });

  it("(2) subgroups + min 32 + max 32 -> sg true", () => {
    const caps = gemvCapsFor(fakeDevice([SG], { subgroupMinSize: 32, subgroupMaxSize: 32 }));
    expect(caps.sg).toBe(true);
    expect(caps.why.length).toBeGreaterThan(0);
  });

  it("(3) subgroups ma dimensione non fissa (min 16 / max 64) -> sg false", () => {
    const caps = gemvCapsFor(fakeDevice([SG], { subgroupMinSize: 16, subgroupMaxSize: 64 }));
    expect(caps.sg).toBe(false);
    expect(caps.why.length).toBeGreaterThan(0);
  });

  it("(4) subgroups ma adapterInfo non esposto -> sg false", () => {
    const caps = gemvCapsFor(fakeDevice([SG]));
    expect(caps.sg).toBe(false);
    expect(caps.why.length).toBeGreaterThan(0);
  });

  it("(5) `why` è sempre una stringa non vuota, nei quattro casi", () => {
    const cases: GemvCapsSource[] = [
      fakeDevice(["shader-f16"], { subgroupMinSize: 32, subgroupMaxSize: 32 }),
      fakeDevice([SG], { subgroupMinSize: 32, subgroupMaxSize: 32 }),
      fakeDevice([SG], { subgroupMinSize: 16, subgroupMaxSize: 64 }),
      fakeDevice([SG]),
    ];
    for (const d of cases) {
      const why = gemvCapsFor(d).why;
      expect(typeof why).toBe("string");
      expect(why.trim().length).toBeGreaterThan(0);
    }
  });

  it("min 32 ma max 64 (profilo wave32/wave64) -> sg false: il max da solo basta a rompere la mappatura", () => {
    // Se il gate guardasse solo subgroupMinSize, questo device accenderebbe
    // vec4-rows2-sg su hardware che può schedulare subgroup da 64 lane.
    const caps = gemvCapsFor(fakeDevice([SG], { subgroupMinSize: 32, subgroupMaxSize: 64 }));
    expect(caps.sg).toBe(false);
    expect(caps.why.length).toBeGreaterThan(0);
  });

  it("min 16 ma max 32 -> sg false: anche il min da solo basta a romperla", () => {
    // Simmetrico del precedente: smaschera un gate che guarda solo il max.
    const caps = gemvCapsFor(fakeDevice([SG], { subgroupMinSize: 16, subgroupMaxSize: 32 }));
    expect(caps.sg).toBe(false);
    expect(caps.why.length).toBeGreaterThan(0);
  });

  it("adapterInfo esposto ma senza i campi subgroup* -> sg false, e `why` non finge una misura", () => {
    // Un default implicito (`subgroupMinSize ?? 32`) darebbe sg=true su una
    // dimensione mai letta: campi assenti === dimensione ignota === false.
    const caps = gemvCapsFor(fakeDevice([SG], {}));
    expect(caps.sg).toBe(false);
    expect(caps.why.trim().length).toBeGreaterThan(0);
    // La diagnosi va in telemetria: non deve dichiarare una dimensione
    // "misurata" che in realtà non è stata letta.
    expect(caps.why).not.toContain("undefined");
    expect(caps.why).toContain("subgroupMinSize");
  });

  it("un solo campo subgroup esposto -> sg false: mezza lettura non è una lettura", () => {
    const soloMin = gemvCapsFor(fakeDevice([SG], { subgroupMinSize: 32 }));
    const soloMax = gemvCapsFor(fakeDevice([SG], { subgroupMaxSize: 32 }));
    expect(soloMin.sg).toBe(false);
    expect(soloMax.sg).toBe(false);
    expect(soloMin.why).not.toContain("undefined");
    expect(soloMax.why).not.toContain("undefined");
  });

  it("il `why` dice QUALE delle tre condizioni è mancata", () => {
    const noFeat = gemvCapsFor(fakeDevice([])).why;
    const noInfo = gemvCapsFor(fakeDevice([SG])).why;
    const badSize = gemvCapsFor(fakeDevice([SG], { subgroupMinSize: 16, subgroupMaxSize: 64 })).why;
    // tre diagnosi distinte: un `why` unico per tutti e tre non spiega niente
    expect(new Set([noFeat, noInfo, badSize]).size).toBe(3);
    expect(noFeat).toContain("subgroups");
    expect(noInfo).toContain("adapterInfo");
    expect(badSize).toContain("32");
    // il caso "campi assenti" appartiene alla terza condizione (dimensione non
    // leggibile), non alla seconda ("misurata e diversa da 32")
    const noFields = gemvCapsFor(fakeDevice([SG], {})).why;
    expect(noFields).toContain("adapterInfo");
    expect(noFields).not.toBe(badSize);
  });

  it("uniforme a 16 o a 64 (min === max ma != 32) -> sg false: è la rottura silenziosa", () => {
    for (const n of [16, 64]) {
      const caps = gemvCapsFor(fakeDevice([SG], { subgroupMinSize: n, subgroupMaxSize: n }));
      expect(caps.sg).toBe(false);
      expect(caps.why.length).toBeGreaterThan(0);
    }
  });

  it("GEMV_SG_FEATURE è la stringa 'subgroups' del gate di fase 0", () => {
    expect(GEMV_SG_FEATURE).toBe("subgroups");
  });

  it("un GPUDevice nudo è accettato per struttura (controllo di TIPO, non di runtime)", () => {
    // I chiamanti (T4/T5) passano `device` senza adattatori. Se GPUDevice
    // smettesse di essere assegnabile a GemvCapsSource — p.es. perché
    // @webgpu/types cambia `adapterInfo` o i campi subgroup* — questa riga non
    // compila, e `npx tsc --noEmit` (parte del gate) diventa rosso.
    const passaIlDeviceNudo = (dev: GPUDevice): GemvCaps => gemvCapsFor(dev);
    const finto = fakeDevice([SG], { subgroupMinSize: 32, subgroupMaxSize: 32 }) as unknown as GPUDevice;
    expect(passaIlDeviceNudo(finto).sg).toBe(true);
  });
});
