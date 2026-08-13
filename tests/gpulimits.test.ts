// Test di `src/engine/gpulimits.ts` (C3a fase 3/it.6).
//
// Il gruppo "derivazione vs parco kernel" è il cuore: ri-deriva le costanti
// SCANSIONANDO il WGSL vero invece di fidarsi di un numero scritto a mano.
// È l'unica cosa che tiene insieme un limite e il codice che lo consuma — il
// motore ha già sbagliato due volte proprio perché i due vivevano in file
// diversi senza niente in mezzo (prima costanti difensive inventate, poi il
// massimo dell'adapter chiesto senza consumatore).
import { attnDecodeWorkgroupStorageBytes } from "../src/engine/kernels/wgsl";
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  engineNeeds, limitsFor, negotiateLimits, grantedLimits, UnmetLimitError,
  MAX_WORKGROUP_SIZE, MAX_STATIC_STORAGE_BINDINGS, QWEN_WORKGROUP_STORAGE_BYTES,
  ARENA_BUFFERS_MAX, expertArenaBindings,
  mlaWorkgroupStorageBytes, q6kHeadBytes, glmKvBytesPerLayer,
} from "../src/engine/gpulimits";
import { pairGemvSiluFastWgsl, gemvAccumFastWgsl, type ArenaOpts } from "../src/engine/kernels/wgsl";
import { SLAB_DOWN_Q4_1 as SLAB } from "../src/engine/moe";
import { GLM47_FLASH as G } from "../src/engine/shape";

// Limiti misurati sul 4090 Laptop (results/engine/webgpu-limits-4090laptop-2026-08-02.json)
const ADAPTER_4090 = {
  maxBufferSize: 4294967292,
  maxStorageBufferBindingSize: 2147483644,
  maxStorageBuffersPerShaderStage: 16,
  maxComputeWorkgroupStorageSize: 49152,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
};
// I default di spec WebGPU, per verificare quali requisiti li sfondano davvero
const SPEC_DEFAULTS = {
  maxBufferSize: 268435456,
  maxStorageBufferBindingSize: 134217728,
  maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
};
const fakeAdapter = (limits: Record<string, number>): GPUAdapter =>
  ({ limits }) as unknown as GPUAdapter;

const GLM_NEEDS = { ctxMax: 525, head: { vocab: G.vocab, dModel: G.dModel } };

describe("derivazione vs parco kernel (scansione del WGSL vero)", () => {
  const src = readFileSync(join(__dirname, "../src/engine/kernels/wgsl.ts"), "utf8");

  it("MAX_WORKGROUP_SIZE copre ogni @workgroup_size del repo", () => {
    const sizes = [...src.matchAll(/@compute\s+@workgroup_size\((\d+)/g)].map((m) => Number(m[1]));
    expect(sizes.length).toBeGreaterThan(20); // sanity: la scansione ha trovato i kernel
    expect(Math.max(...sizes)).toBe(MAX_WORKGROUP_SIZE);
  });

  it("MAX_STATIC_STORAGE_BINDINGS copre il bind group più affollato", () => {
    // conta i binding `var<storage>` dentro ogni generatore, perché gli indici
    // ripartono da 0 a ogni kernel. I blocchi si separano su `function`
    // ESPORTATA O NO: un template privato (es. `attnDecodeLegacyWgsl`) emette
    // il proprio bind group, e attribuire i suoi binding al blocco che lo
    // precede gonfiava il conteggio a 10 su kernel che ne dichiarano 5
    // (goal engine-kernel-decode it.4).
    const blocks = src.split(/^(?:export )?function /m).slice(1);
    const perKernel = blocks.map((b) => (b.match(/@group\(0\)\s*@binding\(\d+\)\s*var<storage/g) ?? []).length);
    expect(blocks.length).toBeGreaterThan(20);
    expect(Math.max(...perKernel)).toBe(MAX_STATIC_STORAGE_BINDINGS);
  });

  it("nessun kernel dichiara più binding LETTERALI di quanti il limite ne conceda", () => {
    const blocks = src.split(/^(?:export )?function /m).slice(1);
    for (const b of blocks) {
      const n = (b.match(/@group\(0\)\s*@binding\(\d+\)\s*var<storage/g) ?? []).length;
      expect(n).toBeLessThanOrEqual(MAX_STATIC_STORAGE_BINDINGS);
    }
  });

  // La scansione qui sopra vede solo i binding scritti a mano. I kernel d'arena
  // (fase 4, strato 1) ne generano nBuf+3, e quel numero non è nel sorgente: si
  // conta sul WGSL PRODOTTO, al tetto di progetto.
  it("i kernel d'arena emettono esattamente expertArenaBindings(nBuf) storage", () => {
    const arena = (nBuf: number): ArenaOpts => ({
      nBuf, slabWords: SLAB.bytes / 4, slabsPerBuf: 390,
      qsWords: SLAB.downQs / 4, scalesWords: SLAB.downScales / 4,
      gateQsWords: SLAB.gateQs / 4, gateScWords: SLAB.gateScales / 4,
      upQsWords: SLAB.upQs / 4, upScWords: SLAB.upScales / 4,
    });
    for (const nBuf of [1, 3, ARENA_BUFFERS_MAX]) {
      const codes = [
        pairGemvSiluFastWgsl({ K: G.dModel, N: G.dFfnExpert, arena: arena(nBuf) }),
        gemvAccumFastWgsl({ kind: "q4_1", K: G.dFfnExpert, N: G.dModel, arena: arena(nBuf) }),
      ];
      for (const code of codes) {
        const storages = (code.match(/@group\(0\) @binding\(\d+\) var<storage/g) ?? []).length;
        const uniforms = (code.match(/@group\(0\) @binding\(\d+\) var<uniform/g) ?? []).length;
        expect(storages, `nBuf=${nBuf}`).toBe(expertArenaBindings(nBuf));
        expect(uniforms).toBe(1); // MoeIdx: non conta nel limite sugli storage
        // Un arco di `ld4` per buffer, e ogni arco deve leggere IL SUO buffer:
        // la backreference lega l'indice del case a quello del binding. Senza,
        // uno switch con tutti i case su `arena0` — cioè un'arena che ignora il
        // buffer scelto — passerebbe il conteggio.
        const cases = [...code.matchAll(/^ {4}case (\d+)u: \{ return arena\1\[i\]; \}$/gm)].map((m) => Number(m[1]));
        expect(cases, `nBuf=${nBuf}: archi di ld4`).toEqual(Array.from({ length: nBuf }, (_, j) => j));
        // e il default esiste (la grammatica WGSL lo impone) senza aggiungere archi
        expect((code.match(/^ {4}default: \{ return arena0\[i\]; \}$/gm) ?? []).length).toBe(1);
        // ogni buffer dichiarato è raggiungibile da un arco, e viceversa
        const declared = [...code.matchAll(/@binding\(\d+\) var<storage, read> arena(\d+):/g)].map((m) => Number(m[1]));
        expect(declared).toEqual(cases);
      }
    }
    // il tetto di progetto sta nel limite che l'adapter di riferimento offre
    expect(expertArenaBindings(ARENA_BUFFERS_MAX)).toBeLessThanOrEqual(ADAPTER_4090.maxStorageBuffersPerShaderStage);
  });

  it("senza `arena` i due generatori emettono il testo a binding fisso di prima", () => {
    // Vincolo duro del design: il regime a sotto-range non cambia di un byte.
    // Qui si asserisce la forma (binding letterali, nessuna traccia d'arena);
    // l'identità byte-per-byte è stata verificata sul diff dei due dump in it.15.
    const pair = pairGemvSiluFastWgsl({ K: G.dModel, N: G.dFfnExpert });
    const down = gemvAccumFastWgsl({ kind: "q4_1", K: G.dFfnExpert, N: G.dModel });
    expect((pair.match(/@group\(0\) @binding\(\d+\) var<storage/g) ?? []).length).toBe(6);
    expect((down.match(/@group\(0\) @binding\(\d+\) var<storage/g) ?? []).length).toBe(5);
    for (const code of [pair, down]) {
      for (const token of ["ld4", "ldw", "selBuf", "moeIdx", "SLAB_W", "arena0"]) {
        expect(code.includes(token), token).toBe(false);
      }
    }
    expect(down.includes("accScale[0]")).toBe(true); // il peso è ancora un binding
  });
});

describe("requisiti derivati", () => {
  it("la testa Q6_K sfonda il default di spec sul binding size", () => {
    const head = q6kHeadBytes(G.vocab, G.dModel);
    expect(head).toBe(262_676_480); // 250,5 MiB
    expect(head).toBeGreaterThan(SPEC_DEFAULTS.maxStorageBufferBindingSize);
    // ...e sta appena SOTTO il default sul buffer size (margine 2,1%)
    expect(head).toBeLessThan(SPEC_DEFAULTS.maxBufferSize);
  });

  it("il workgroup storage e' il max fra path Qwen fuso e mlaAttnDecode(ctxMax)", () => {
    // a ctx corto vince Qwen (costante), a ctx lungo vince MLA
    const short = engineNeeds({ ctxMax: 525 }).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
    expect(short.value).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    const long = engineNeeds({ ctxMax: 8192 }).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
    expect(long.value).toBe(mlaWorkgroupStorageBytes(8192));
    expect(long.value).toBe(33_024);
    // il vecchio cap a mano di 32768 tagliava il contesto a 8128 senza dirlo
    expect(mlaWorkgroupStorageBytes(8128)).toBeLessThanOrEqual(32768);
    expect(mlaWorkgroupStorageBytes(8129)).toBeGreaterThan(32768);
  });

  // goal engine-kernel-decode, docket item 2. Il difetto NON era un limite
  // sbagliato in produzione — il path q35 otteneva il valore giusto perche'
  // nessuno passava `mlaAttention: false`. Era una TRAPPOLA: il commento
  // invitava a passarlo "per un consumatore che quel modello non ha", mentre
  // `attnDecodeWgsl` ha lo stesso `scores[ctxMax]`. Questo test la chiude:
  // spegnere l'MLA non deve poter far sparire il fabbisogno di Qwen.
  it("spegnere mlaAttention NON toglie il fabbisogno dell'attenzione Qwen", () => {
    for (const ctxMax of [525, 4096, 8192, 16384]) {
      const off = engineNeeds({ ctxMax, mlaAttention: false }).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
      expect(off.value).toBeGreaterThanOrEqual(attnDecodeWorkgroupStorageBytes(ctxMax));
      expect(off.value).toBeGreaterThanOrEqual(QWEN_WORKGROUP_STORAGE_BYTES);
      expect(off.consumer).toContain("attnDecode");
    }
    // FINO A it.3 QUI SI ASSERIVA IL CONTRARIO: che il valore CRESCESSE col
    // contesto. Era vero del kernel di ieri (`scores[ctxMax]` in memoria di
    // gruppo) ed era il modo di provare che la frase "path Qwen indipendente
    // dal contesto" fosse falsa. La riscrittura in streaming (riga 1 del goal)
    // l'ha resa VERA: il fabbisogno ora e' costante, ed e' un done-when del
    // contratto. Il test non si cancella — si rovescia, e resta a guardia del
    // fatto nuovo: nessun ritorno a una forma che lega la memoria al contesto.
    const a = engineNeeds({ ctxMax: 8192, mlaAttention: false }).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
    const b = engineNeeds({ ctxMax: 16384, mlaAttention: false }).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
    expect(b.value).toBe(a.value);
  });

  // Done-when (e) della riga 1 — e qui il contratto chiedeva piu' di quanto la
  // riga 1 possa dare, scoperto ESEGUENDO (docket item 4).
  //
  // Cio' che la riga 1 ha ottenuto: il TETTO DI CONTESTO non esiste piu'. Il
  // fabbisogno dell'attenzione e' 1.536 B COSTANTI (16·ceil(headDim/4)+512),
  // contro 4·ctxMax+256 di ieri, e sta larghissimo sotto i 16 KB garantiti da
  // WebGPU. Nessun contesto, per quanto lungo, puo' piu' impedire la creazione
  // della pipeline.
  //
  // Cio' che NON dipende dalla riga 1: il totale richiesto dal motore resta
  // 30.848 B, perche' un ALTRO consumatore — `rmsPairGemmSiluChunkFast`, il
  // kernel fuso del prefill — sta sopra la garanzia da solo e non dipende dal
  // contesto. E' un limite di portabilita' vero, ma appartiene al prefill
  // (goal TTFT), non a questo.
  it("l'attenzione non lega piu' la memoria di gruppo al contesto", () => {
    const WEBGPU_GUARANTEED = 16384;
    // la formula del kernel non guarda piu' il contesto...
    expect(attnDecodeWorkgroupStorageBytes(64)).toBe(attnDecodeWorkgroupStorageBytes(1_000_000));
    // ...e sta sotto la garanzia con tre ordini di grandezza di margine
    expect(attnDecodeWorkgroupStorageBytes(1_000_000)).toBeLessThan(WEBGPU_GUARANTEED);
    // il totale richiesto non cresce piu' col contesto a MLA spenta
    const vals = [64, 4096, 65536, 1_000_000].map((ctxMax) =>
      engineNeeds({ ctxMax, mlaAttention: false }).find((x) => x.limit === "maxComputeWorkgroupStorageSize")!.value);
    expect(new Set(vals).size).toBe(1);
    // e cio' che resta sopra i 16 KB e' UN consumatore solo, nominato: se un
    // domani sparisce anche quello, questo test va aggiornato di proposito
    expect(vals[0]).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(QWEN_WORKGROUP_STORAGE_BYTES).toBeGreaterThan(WEBGPU_GUARANTEED);
  });

  it("l'arena alza binding size e storage per stage, col suo consumatore", () => {
    const window = 390 * SLAB.bytes; // finestra da 2 GiB, classe q4_1
    const needs = engineNeeds({ ...GLM_NEEDS, arenaBuffers: 7, arenaWindowBytes: window });
    const bind = needs.find((n) => n.limit === "maxStorageBufferBindingSize")!;
    expect(bind.value).toBe(window);            // batte la testa Q6_K (250 MiB)
    expect(bind.consumer).toMatch(/arena/);
    const stage = needs.filter((n) => n.limit === "maxStorageBuffersPerShaderStage");
    expect(stage.map((n) => n.value)).toEqual([MAX_STATIC_STORAGE_BINDINGS, expertArenaBindings(7)]);
    expect(limitsFor(fakeAdapter(ADAPTER_4090), needs).maxStorageBuffersPerShaderStage).toBe(10);
    // il buffer segue il binding: un binding di N byte vive in un buffer >= N
    expect(needs.find((n) => n.limit === "maxBufferSize")!.value).toBe(window);
  });

  it("oltre ARENA_BUFFERS_MAX si ferma qui, non al createBindGroupLayout", () => {
    expect(() => engineNeeds({ ...GLM_NEEDS, arenaBuffers: ARENA_BUFFERS_MAX + 1 }))
      .toThrow(/ARENA_BUFFERS_MAX/);
  });

  it("ogni requisito porta il suo consumatore", () => {
    for (const n of engineNeeds(GLM_NEEDS)) {
      expect(n.consumer.length).toBeGreaterThan(10);
      expect(n.value).toBeGreaterThan(0);
    }
  });
});

describe("limitsFor", () => {
  it("chiede il REQUISITO, non il massimo dell'adapter", () => {
    const got = negotiateLimits(fakeAdapter(ADAPTER_4090), GLM_NEEDS);
    // l'adapter concede 1024, ma il parco kernel ne usa 256: si chiede 256
    expect(got.maxComputeInvocationsPerWorkgroup).toBe(256);
    expect(got.maxComputeWorkgroupSizeX).toBe(256);
    // l'adapter concede 16 storage buffer, ne servono 7
    expect(got.maxStorageBuffersPerShaderStage).toBe(7);
    // qui invece il requisito e' reale e sopra il default
    expect(got.maxStorageBufferBindingSize).toBe(262_676_480);
  });

  it("non chiede MAI piu' del disponibile", () => {
    const lim = ADAPTER_4090 as Record<string, number>;
    for (const [k, v] of Object.entries(negotiateLimits(fakeAdapter(lim), GLM_NEEDS))) {
      expect(v).toBeLessThanOrEqual(lim[k]);
    }
  });

  it("un requisito HARD non servibile fallisce subito, col consumatore nel messaggio", () => {
    const povero = { ...ADAPTER_4090, maxStorageBufferBindingSize: SPEC_DEFAULTS.maxStorageBufferBindingSize };
    let err: unknown;
    try { negotiateLimits(fakeAdapter(povero), GLM_NEEDS); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(UnmetLimitError);
    expect((err as UnmetLimitError).unmet[0].limit).toBe("maxStorageBufferBindingSize");
    expect((err as Error).message).toMatch(/output\.weight Q6_K/);
  });

  it("un requisito SOFT (packing) viene troncato, non fa fallire", () => {
    const needs = engineNeeds({ ...GLM_NEEDS, slabClassBytes: 99e9 });
    const got = limitsFor(fakeAdapter(ADAPTER_4090), needs);
    expect(got.maxBufferSize).toBe(ADAPTER_4090.maxBufferSize);
  });

  it("senza slabClassBytes il buffer resta al requisito hard, non al massimo", () => {
    const got = negotiateLimits(fakeAdapter(ADAPTER_4090), GLM_NEEDS);
    expect(got.maxBufferSize).toBe(262_676_480);
    expect(got.maxBufferSize).toBeLessThan(ADAPTER_4090.maxBufferSize);
  });

  it("un limite non esposto dall'adapter non viene chiesto a 0", () => {
    let err: unknown;
    try { limitsFor(fakeAdapter({ maxBufferSize: 1024 }), engineNeeds(GLM_NEEDS)); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(UnmetLimitError); // tutti i hard non serviti
  });

  it("la KV cache diventa il vincolo solo a contesti irraggiungibili", () => {
    const head = q6kHeadBytes(G.vocab, G.dModel);
    // il workgroup storage taglia molto prima: 49152 B => ctxMax <= 12224
    expect(glmKvBytesPerLayer(12224)).toBeLessThan(head);
    expect(glmKvBytesPerLayer(114009)).toBeGreaterThan(head);
  });
});

describe("grantedLimits", () => {
  it("riporta i valori del DEVICE, non dell'adapter", () => {
    const dev = { limits: { ...SPEC_DEFAULTS, maxComputeWorkgroupsPerDimension: 65535 } } as unknown as GPUDevice;
    const g = grantedLimits(dev);
    expect(g.maxComputeInvocationsPerWorkgroup).toBe(256);
    expect(g.maxStorageBufferBindingSize).toBe(SPEC_DEFAULTS.maxStorageBufferBindingSize);
  });
});
