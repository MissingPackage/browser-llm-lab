// Test di `src/engine/gpulimits.ts` (C3a fase 3/it.6).
//
// Il gruppo "derivazione vs parco kernel" è il cuore: ri-deriva le costanti
// SCANSIONANDO il WGSL vero invece di fidarsi di un numero scritto a mano.
// È l'unica cosa che tiene insieme un limite e il codice che lo consuma — il
// motore ha già sbagliato due volte proprio perché i due vivevano in file
// diversi senza niente in mezzo (prima costanti difensive inventate, poi il
// massimo dell'adapter chiesto senza consumatore).
import {
  attnDecodeWgsl, attnDecodeLegacyBatchWgsl, attnDecodeWorkgroupStorageBytes, attnPrefillChunkWgsl,
  prefillGemmWorkgroupStorageBytes,
  qwenFusedChunkWorkgroupStorageBytes, qwenGemvResidualWorkgroupStorageBytes,
} from "../src/engine/kernels/wgsl";
import { PREFILL_M, PREFILL_M_DENSE05B } from "../src/engine/prefillplan";
import { QWEN25_05B } from "../src/engine/shape";
import { GLM_PREFILL_M } from "../src/engine/moeprefillplan";
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
    // cio' che resta sopra i 16 KB e' il path 0.5B. NON e' "un consumatore
    // solo": quella frase stava qui ed era FALSA (it.24). Misurando il WGSL
    // generato, i kernel del 0.5B sopra la garanzia sono QUATTRO — i tre di
    // chunk a 30.848/30.720 B e il down-proj del DECODE a 19.712 — e i 30.848
    // erano il massimo dei quattro, non l'unico. Ora la costante e' un
    // `Math.max` calcolato dalle formule accanto ai kernel, quindi non puo'
    // piu' scendere sotto un consumatore vivo.
    expect(vals[0]).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(QWEN_WORKGROUP_STORAGE_BYTES).toBeGreaterThan(WEBGPU_GUARANTEED);
    // il massimo e' DAVVERO il massimo: il down-proj del decode sta sotto i
    // kernel di chunk, e se un domani lo superasse la costante deve seguirlo
    expect(QWEN_WORKGROUP_STORAGE_BYTES).toBeGreaterThanOrEqual(
      qwenGemvResidualWorkgroupStorageBytes(QWEN25_05B.dFfn));
    expect(QWEN_WORKGROUP_STORAGE_BYTES).toBeGreaterThanOrEqual(
      qwenFusedChunkWorkgroupStorageBytes({ K: QWEN25_05B.dModel, mMax: PREFILL_M_DENSE05B }));
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

// ---------------------------------------------------------------------------
// Il GEMM multi-riga del prefill (goal engine-ttft, riga 2) e il debito che
// resta scoperto accanto a lui.
//
// La riga 2 del goal porta in produzione `prefillGemmQ4SplitK*`: un kernel che
// tiene le attivazioni in memoria di gruppo, quindi un consumatore NUOVO di
// `maxComputeWorkgroupStorageSize`. Qui si asserisce che venga DICHIARATO — col
// suo valore, DENTRO il `value` che si negozia col device e non solo nel testo
// del `consumer` — come ogni altro requisito di questo file.
//
// Il numero non si ricopia: viene da `prefillGemmWorkgroupStorageBytes`,
// importata dal file del kernel. Se cambia la tilatura cambia il numero
// dichiarato — e non il test.
// ---------------------------------------------------------------------------

// I kernel dell'attenzione col `scores[ctxMax]` in memoria di gruppo. Le loro
// formule non si ricopiano nemmeno qui: si LEGGONO dal WGSL che i generatori
// producono, sommando le `array<f32, N>` di memoria di gruppo. Se domani uno
// dei due guadagna o perde un `var<workgroup>`, il pareggio e il tetto si
// spostano QUI invece di restare asseriti su numeri vecchi.
//
// DEI DUE, UNO SOLO E' ANCORA IN PRODUZIONE. Il ramo batch dell'attenzione non
// instrada piu' al legacy (task T1-kernel-batch-streaming): la sua forma e'
// quella in streaming, costante in ctxMax, e questo modulo la DICHIARA (vedi il
// describe in fondo, che era il sensore del debito ed e' diventato la garanzia).
// `attnDecodeLegacyBatchWgsl` resta esportata come fallback dichiarato e come
// termine di paragone: e' cio' che si misura qui sotto per dire quanto valeva
// il debito. Quello ancora scoperto e' `attnPrefillChunkWgsl` (path di
// conformita' 0.5B), che la clausola e2a della riga 4 tiene fuori scope.
const ATTN = { nHead: 16, nKvHead: 2, headDim: 256 };
const workgroupF32Bytes = (code: string, decls: number): number => {
  const found = [...code.matchAll(/var<workgroup>\s+\w+\s*:\s*array<f32,\s*(\d+)>/g)];
  // il conteggio e' parte dell'asserzione: se una dichiarazione cambia tipo
  // (vec4<f32>, u32, ...) la somma non deve poter calare in silenzio
  expect(found.length, "var<workgroup> array<f32> nel WGSL generato").toBe(decls);
  return found.reduce((s, m) => s + Number(m[1]) * 4, 0);
};
/** ramo `batch` LEGACY (fallback dichiarato, non piu' instradato): scores[ctxMax] + red[64]. */
const legacyBatchBytes = (ctxMax: number): number =>
  workgroupF32Bytes(attnDecodeLegacyBatchWgsl({ ...ATTN, ctxMax }), 2);
/** attnPrefillChunkWgsl a headDim 256: qh[headDim] + scores[ctxMax] + red[64]. */
const prefillChunkAttnBytes = (ctxMax: number): number =>
  workgroupF32Bytes(attnPrefillChunkWgsl({ ...ATTN, ctxMax, mMax: PREFILL_M }), 3);

describe("workgroup storage: il GEMM multi-riga del prefill", () => {
  const storageNeed = (o: Parameters<typeof engineNeeds>[0]) =>
    engineNeeds(o).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
  // La shape che la riga 1 del goal ha misurato (K2560xN9216, 4 fette). NON e'
  // la shape a decidere il fabbisogno — solo M — e il primo test lo dimostra:
  // per questo il test se la puo' scrivere qui senza condividerla con
  // `gpulimits.ts`.
  const GEMM = (M: number) => ({ kind: "q4_0" as const, K: 2560, N: 9216, M, splits: 4 });
  // il termine come lo dichiara il consumer: LETTO dal testo, non dedotto
  const gemmTerm = (consumer: string): { M: number; via: string; bytes: number } => {
    const m = /prefillGemm splitK M=(\d+) \(([^)]+)\) = (\d+) B/.exec(consumer);
    if (!m) throw new Error(`il consumer non nomina il GEMM del prefill: ${consumer}`);
    return { M: Number(m[1]), via: m[2], bytes: Number(m[3]) };
  };

  it("il fabbisogno dipende SOLO da M: la shape scelta da gpulimits non conta", () => {
    // e' cio' che autorizza gpulimits a passare una shape valida qualunque
    for (const via of ["idot", "f32"] as const) {
      expect(prefillGemmWorkgroupStorageBytes(GEMM(16), via), via)
        .toBe(prefillGemmWorkgroupStorageBytes({ kind: "q4_0", K: 64, N: 1, M: 16, splits: 1 }, via));
    }
  });

  it("il need NOMINA il GEMM del prefill col valore della formula, e segue M", () => {
    const need = storageNeed({ ctxMax: 6400 });
    expect(need.consumer).toContain("prefillGemm");
    expect(gemmTerm(need.consumer)).toEqual({
      M: PREFILL_M, via: "peggiore fra idot e f32",
      bytes: prefillGemmWorkgroupStorageBytes(GEMM(PREFILL_M)),
    });
    // se si cambia M il numero dichiarato cambia DI CONSEGUENZA: viene dalla
    // formula importata, non da una costante ricopiata qui accanto
    for (const M of [8, 16, 32, 64]) {
      const t = gemmTerm(storageNeed({ ctxMax: 6400, prefillM: M }).consumer);
      expect(t.M, `M=${M}`).toBe(M);
      expect(t.bytes, `M=${M}`).toBe(prefillGemmWorkgroupStorageBytes(GEMM(M)));
    }
    // ...e il legame e' quello lineare del kernel, non una coincidenza a M=16
    expect(gemmTerm(storageNeed({ ctxMax: 6400, prefillM: 32 }).consumer).bytes)
      .toBe(2 * gemmTerm(storageNeed({ ctxMax: 6400, prefillM: 16 }).consumer).bytes);
  });

  it("a M=16: 1.152 B via idot, 4.096 B via f32, e senza dirlo si dichiara il peggiore", () => {
    expect(prefillGemmWorkgroupStorageBytes(GEMM(16), "idot")).toBe(1_152);
    expect(prefillGemmWorkgroupStorageBytes(GEMM(16), "f32")).toBe(4_096);
    // etichette asserite per UGUAGLIANZA: `toContain("idot")` non
    // discriminerebbe niente, perche' l'etichetta di default ("peggiore fra
    // idot e f32") contiene entrambi i nomi.
    const idot = gemmTerm(storageNeed({ ctxMax: 6400, prefillM: 16, prefillGemmIdot: true }).consumer);
    expect(idot).toEqual({ M: 16, via: "via idot", bytes: 1_152 });
    const f32 = gemmTerm(storageNeed({ ctxMax: 6400, prefillM: 16, prefillGemmIdot: false }).consumer);
    expect(f32).toEqual({ M: 16, via: "via f32", bytes: 4_096 });
    // Senza `prefillGemmIdot` si dichiara il PEGGIORE: quale via giri lo decide
    // la language feature a runtime, e chi negozia i limiti lo fa PRIMA di
    // saperlo. Chiedere 1.152 e ritrovarsi sulla via f32 = pipeline invalida su
    // un device che concede esattamente il richiesto.
    expect(gemmTerm(storageNeed({ ctxMax: 6400, prefillM: 16 }).consumer))
      .toEqual({ M: 16, via: "peggiore fra idot e f32", bytes: 4_096 });
  });

  it("il termine e' COSTANTE in ctxMax", () => {
    for (const idot of [undefined, true, false]) {
      const vals = [525, 4096, 8192, 16384].map((ctxMax) =>
        gemmTerm(storageNeed({ ctxMax, prefillGemmIdot: idot }).consumer).bytes);
      expect(new Set(vals).size, `prefillGemmIdot=${idot}`).toBe(1);
    }
  });

  // IL TERMINE STA NEL `value`, NON SOLO NEL TESTO. `limitsFor` e
  // `negotiateLimits` leggono `need.value` e non guardano MAI il `consumer`: un
  // termine che comparisse solo nella stringa sarebbe una dichiarazione che il
  // device non vede. A M=16 le due forme non si distinguono (4.096 < 30.848: il
  // max non cambia), quindi si SONDA l'aritmetica a un M in cui il GEMM domina.
  // Se il termine uscisse dal `Math.max`, questo test cadrebbe da solo.
  //
  // LA SONDA GIRA SULLA VIA f32, e da oggi non e' un dettaglio (task
  // limits-q5k). Accanto a questo termine ne vive un secondo, il GEMM q5_K di
  // `ssm_out`, e sul peggiore-delle-due-vie quello e' PIU' RIPIDO (320·M contro
  // 256·M): a M=256 vincerebbe lui, e la sonda misurerebbe il vicino invece del
  // termine che vuole isolare. Sulla via f32 il rapporto si inverte (256·M
  // contro 128·M) ed e' di nuovo QUESTO termine a decidere il `value` — quindi
  // la sonda torna a cadere da sola se il termine q4_0 uscisse dal `Math.max`.
  // Il numero pinnato non cambia: `prefillGemmWorkgroupStorageBytes(GEMM(M))`
  // senza via risponde gia' il peggiore, che per il q4_0 E' la f32.
  it("il termine entra nel value negoziato, non solo nel consumer", () => {
    const M = 256; // sonda: 256·16·16 = 65.536 B > 30.848 del path fuso
    expect(prefillGemmWorkgroupStorageBytes(GEMM(M))).toBe(65_536);
    expect(prefillGemmWorkgroupStorageBytes(GEMM(M), "f32"))
      .toBe(prefillGemmWorkgroupStorageBytes(GEMM(M)));
    expect(storageNeed({ ctxMax: 6400, prefillM: M, prefillGemmIdot: false }).value)
      .toBe(prefillGemmWorkgroupStorageBytes(GEMM(M)));
    // ...e a ogni M il value e' esattamente il maggiore fra il path fuso e il
    // termine dichiarato: ne' sotto (dichiarazione muta) ne' sopra (requisito
    // gonfiato). 121 e' il primo M in cui il GEMM supera i 30.848.
    for (const M2 of [8, 16, 32, 121]) {
      const need = storageNeed({ ctxMax: 6400, prefillM: M2, prefillGemmIdot: false });
      expect(need.value, `M=${M2}`).toBe(
        Math.max(QWEN_WORKGROUP_STORAGE_BYTES, gemmTerm(need.consumer).bytes));
    }
    expect(prefillGemmWorkgroupStorageBytes(GEMM(121))).toBeGreaterThan(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(prefillGemmWorkgroupStorageBytes(GEMM(120))).toBeLessThan(QWEN_WORKGROUP_STORAGE_BYTES);
  });

  // NON-REGRESSIONE sul percorso di prodotto: dichiarare un consumatore in piu'
  // non deve alzare di un byte cio' che si chiede al device. Uguaglianza, non
  // `<=`: se un domani il totale scende, questo test va aggiornato di
  // proposito, come ogni altro numero pinnato di questo file.
  it("a ctxMax 6400 il valore complessivo resta 30.848 B", () => {
    for (const idot of [undefined, true, false]) {
      expect(storageNeed({ ctxMax: 6400, prefillGemmIdot: idot }).value, `prefillGemmIdot=${idot}`)
        .toBe(30_848);
    }
    expect(storageNeed({ ctxMax: 6400 }).value).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    // il GEMM del prefill sta DENTRO cio' che si chiedeva gia': e' dichiarato
    // perche' esiste, non perche' vince
    expect(gemmTerm(storageNeed({ ctxMax: 6400 }).consumer).bytes)
      .toBeLessThan(QWEN_WORKGROUP_STORAGE_BYTES);
  });

  // L'M di DEFAULT e' `PREFILL_M` (lo impone il contratto di questo task), ma
  // nessun percorso di prodotto legge oggi quella costante: il prefill denso
  // gira a `PREFILL_M_DENSE05B` e quello GLM a `GLM_PREFILL_M`. Finche' il
  // default coincide col piu' grande dei due, dichiara abbastanza per entrambi;
  // se un domani uno dei due lo supera, gpulimits dichiarerebbe un M che
  // nessuno esegue — e si scopre qui, non in validazione di pipeline.
  it("il default PREFILL_M copre l'M dei percorsi di prodotto", () => {
    expect(Math.max(PREFILL_M_DENSE05B, GLM_PREFILL_M)).toBe(PREFILL_M);
  });

  // Il rapporto col legacy scritto come ASSERZIONE e non come commento: il
  // termine che il ramo batch chiedeva FINO A IERI sta un ordine di grandezza
  // sopra quello del GEMM. E' la misura di quanto valeva il debito — oggi
  // chiuso, perche' quel ramo non instrada piu' li'.
  it("il termine del ramo legacy vale >= 6x quello del GEMM (>= 22x via idot)", () => {
    const legacyBatch = legacyBatchBytes(6400);
    expect(legacyBatch).toBe(25_856); // 4·6400 + 256
    expect(legacyBatch / 4_096).toBeGreaterThanOrEqual(6);
    expect(legacyBatch / 1_152).toBeGreaterThanOrEqual(22);
  });
});

// IL DEBITO, CHIUSO. Era un sensore su cio' che non si dichiarava; ora e' una
// GARANZIA su cio' che si dichiara.
//
// Fino a ieri il ramo `batch` di `attnDecodeWgsl` (q35gpumodel) chiedeva
// `4·ctxMax + 256` B di memoria di gruppo e questo file NON lo dichiarava:
// dichiararlo avrebbe alzato il requisito HARD sopra i 49.152 B del device di
// riferimento, e `limitsFor` sarebbe fallita dove passava. Il task
// T1-kernel-batch-streaming toglie quel ramo dal legacy: la sua forma e' in
// streaming e il fabbisogno e' `attnDecodeWorkgroupStorageBytes` — 1.536 B
// COSTANTI. Ora si dichiara, ed e' il requisito HARD a NON muoversi.
//
// Il test cade in entrambe le direzioni: se il ramo batch torna a una forma che
// lega la memoria al contesto (i numeri sono LETTI dal WGSL generato, non
// ricopiati), e se il declarare ricomincia ad alzare il requisito.
describe("garanzia: il ramo batch dell'attenzione e' contato e non alza il requisito", () => {
  const CHIUSO_DA = "T1-kernel-batch-streaming: il ramo batch di `attnDecodeWgsl` non instrada " +
    "piu' al legacy — softmax in streaming, workgroup storage costante in ctxMax";
  const storageNeed = (o: Parameters<typeof engineNeeds>[0]) =>
    engineNeeds(o).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
  /** il termine del ramo batch come lo dichiara il consumer: LETTO, non dedotto */
  const batchTerm = (consumer: string): number => {
    const m = /attnDecode batch \(prefill a chunk, streaming, costante in ctxMax\) = (\d+) B/.exec(consumer);
    if (!m) throw new Error(`il consumer non nomina il ramo batch dell'attenzione: ${consumer}`);
    return Number(m[1]);
  };

  it("il ctxMax di pareggio del LEGACY era 7.648, e a 12.224 sfondava i 49.152", () => {
    // I numeri del debito restano asseriti — sul fallback, che e' ancora
    // generabile. Sono la misura di cio' che la riscrittura ha tolto.
    expect(prefillChunkAttnBytes(7_392)).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(prefillChunkAttnBytes(7_393)).toBeGreaterThan(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(legacyBatchBytes(7_648)).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(legacyBatchBytes(12_224)).toBe(ADAPTER_4090.maxComputeWorkgroupStorageSize);
    expect(legacyBatchBytes(12_225)).toBeGreaterThan(ADAPTER_4090.maxComputeWorkgroupStorageSize);
    // ...e la forma di oggi, allo STESSO contesto, non ci si avvicina nemmeno
    expect(attnDecodeWorkgroupStorageBytes(12_225)).toBeLessThan(16_384);
  });

  it("[f] a ctxMax 12.225 il ramo batch e' DICHIARATO e limitsFor non lancia", () => {
    const ctxMax = 12_225; // il contesto in cui il debito, dichiarato, sfondava
    const legacy = legacyBatchBytes(ctxMax);
    expect(legacy, CHIUSO_DA).toBeGreaterThan(ADAPTER_4090.maxComputeWorkgroupStorageSize);
    const needs = engineNeeds({ ctxMax, mlaAttention: false });
    const storage = needs.find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
    // dichiarare non alza il requisito HARD: il value resta il termine fuso
    expect(storage.value, CHIUSO_DA).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    // il consumer NOMINA il ramo batch, col suo valore costante
    expect(batchTerm(storage.consumer), CHIUSO_DA).toBe(attnDecodeWorkgroupStorageBytes(ctxMax));
    // ...e quel valore descrive il kernel VERO: nel suo testo non c'e' piu'
    // nessun array di memoria di gruppo dimensionato sul contesto
    expect(attnDecodeWgsl({ ...ATTN, ctxMax, batch: true }), CHIUSO_DA)
      .not.toContain(`array<f32, ${ctxMax}>`);
    // e il motore negozia dove il debito, dichiarato, avrebbe fatto fallire
    expect(() => limitsFor(fakeAdapter(ADAPTER_4090), needs), CHIUSO_DA).not.toThrow();
    // ...mentre lo STESSO requisito nella forma legacy resterebbe non servibile
    expect(() => limitsFor(fakeAdapter(ADAPTER_4090), [{
      limit: "maxComputeWorkgroupStorageSize", value: legacy, hard: true,
      consumer: "ramo batch legacy di attnDecodeWgsl (fallback, non instradato)",
    }]), CHIUSO_DA).toThrow(UnmetLimitError);
  });

  it("[f] il termine del ramo batch e' costante in ctxMax e non muove il value", () => {
    const vals = [525, 6_400, 12_225, 65_536, 1_000_000].map((ctxMax) => {
      const need = storageNeed({ ctxMax, mlaAttention: false });
      return [need.value, batchTerm(need.consumer)] as const;
    });
    expect(new Set(vals.map((v) => v[1])).size, "termine batch costante in ctxMax").toBe(1);
    expect(new Set(vals.map((v) => v[0])).size, "requisito HARD costante in ctxMax").toBe(1);
    expect(vals[0][0]).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(vals[0][1]).toBe(attnDecodeWorkgroupStorageBytes(6_400));
  });
});

// ---------------------------------------------------------------------------
// IL GEMM Q5_K DEL PREFILL (task limits-q5k).
//
// La riga 5 del goal engine-ttft ha misurato che il collo del prefill non era
// la ricorrenza ma un Q5_K rimasto sul percorso vecchio: `ssm_out`. Il kernel
// nuovo che lo copre e' un GEMM multi-riga come quello q4_0, e come quello
// tiene le attivazioni della tile in memoria di gruppo — quindi e' un
// consumatore di `maxComputeWorkgroupStorageSize`, e qui si asserisce che venga
// DICHIARATO col suo valore DENTRO il `value` negoziato, non solo nel testo.
//
// Il fabbisogno non si ricopia: viene da `prefillGemmWorkgroupStorageBytes`,
// la STESSA formula del termine q4_0, interrogata sulla shape q5_K. Le due vie
// (intera e f32) hanno costi diversi e — al contrario del q4_0 — qui e' la via
// INTERA la piu' cara: 320·M contro 128·M. Chi negozia i limiti prima di sapere
// quale via girera' deve percio' dichiarare 320·M, ed e' quello che il default
// (via omessa) fa.
//
// La shape e' scritta qui come in gpulimits.ts perche' il fabbisogno dipende
// dal solo M — la shape serve solo a farsi validare dalla formula. E' la shape
// VERA di `ssm_out` (K4096xN2560, 4 fette), non una qualsiasi.
// ---------------------------------------------------------------------------
describe("workgroup storage: il GEMM q5_K del prefill (ssm_out)", () => {
  const storageNeed = (o: Parameters<typeof engineNeeds>[0]) =>
    engineNeeds(o).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
  const Q5K = (M: number) => ({ kind: "q5_K" as const, K: 4096, N: 2560, M, splits: 4 });
  /** il termine q5_K come lo dichiara il consumer: LETTO dal testo, non dedotto */
  const q5kTerm = (consumer: string): { M: number; via: string; bytes: number } => {
    const m = /prefillGemmQ5K \(ssm_out\) splitK M=(\d+) \(([^)]+)\) = (\d+) B/.exec(consumer);
    if (!m) throw new Error(`il consumer non nomina il GEMM q5_K del prefill: ${consumer}`);
    return { M: Number(m[1]), via: m[2], bytes: Number(m[3]) };
  };
  // il termine q4_0, per verificare che i due convivano senza confondersi
  const gemmQ4Term = (consumer: string): number => {
    const m = /prefillGemm splitK M=(\d+) \(([^)]+)\) = (\d+) B/.exec(consumer);
    if (!m) throw new Error(`il consumer non nomina il GEMM q4_0 del prefill: ${consumer}`);
    return Number(m[3]);
  };

  // (a) i due valori, pinnati UNA volta sola; ovunque altrove si confronta con
  // la formula, mai con un letterale riscritto.
  it("a M=16: 5.120 B via idot, 2.048 B via f32 — qui la via intera e' la piu' cara", () => {
    expect(prefillGemmWorkgroupStorageBytes(Q5K(16), "idot")).toBe(5_120);
    expect(prefillGemmWorkgroupStorageBytes(Q5K(16), "f32")).toBe(2_048);
    // senza dire la via si dichiara il PEGGIORE, che qui e' l'intera
    expect(prefillGemmWorkgroupStorageBytes(Q5K(16)))
      .toBe(prefillGemmWorkgroupStorageBytes(Q5K(16), "idot"));
  });

  // (b) il consumer nomina il kernel col suo numero di byte, e quel numero e'
  // esattamente quello che la formula produce — per ogni via e per ogni M.
  it("il need NOMINA il GEMM q5_K col valore della formula, e segue M e la via", () => {
    const need = storageNeed({ ctxMax: 6400 });
    expect(need.consumer).toContain("prefillGemmQ5K");
    expect(q5kTerm(need.consumer)).toEqual({
      M: PREFILL_M, via: "peggiore fra idot e f32",
      bytes: prefillGemmWorkgroupStorageBytes(Q5K(PREFILL_M)),
    });
    for (const [idot, via] of [[true, "via idot"], [false, "via f32"]] as const) {
      for (const M of [8, 16, 32, 64]) {
        const t = q5kTerm(storageNeed({ ctxMax: 6400, prefillM: M, prefillGemmIdot: idot }).consumer);
        expect(t, `M=${M} idot=${idot}`).toEqual({
          M, via, bytes: prefillGemmWorkgroupStorageBytes(Q5K(M), idot ? "idot" : "f32"),
        });
      }
    }
    // il legame e' quello lineare del kernel, non una coincidenza a M=16
    expect(q5kTerm(storageNeed({ ctxMax: 6400, prefillM: 32 }).consumer).bytes)
      .toBe(2 * q5kTerm(storageNeed({ ctxMax: 6400, prefillM: 16 }).consumer).bytes);
    // e i due termini di prefill restano DISTINTI: stesso M, cifre diverse
    const c = storageNeed({ ctxMax: 6400, prefillM: 16 }).consumer;
    expect(q5kTerm(c).bytes).not.toBe(gemmQ4Term(c));
  });

  it("il termine q5_K e' COSTANTE in ctxMax", () => {
    for (const idot of [undefined, true, false]) {
      const vals = [525, 4096, 8192, 16384].map((ctxMax) =>
        q5kTerm(storageNeed({ ctxMax, prefillGemmIdot: idot }).consumer).bytes);
      expect(new Set(vals).size, `prefillGemmIdot=${idot}`).toBe(1);
    }
  });

  // (c) DICHIARARE non e' ALZARE: sul percorso di prodotto il tetto non muove
  // di un byte. Uguaglianza, non `<=`.
  it("a ctxMax 6400 il valore complessivo resta 30.848 B", () => {
    for (const idot of [undefined, true, false]) {
      expect(storageNeed({ ctxMax: 6400, prefillGemmIdot: idot }).value, `prefillGemmIdot=${idot}`)
        .toBe(30_848);
    }
    expect(storageNeed({ ctxMax: 6400 }).value).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(q5kTerm(storageNeed({ ctxMax: 6400 }).consumer).bytes)
      .toBeLessThan(QWEN_WORKGROUP_STORAGE_BYTES);
  });

  // (d) la soglia in cui lo ALZEREBBE, pinnata come per il q4_0
  // (GEMM(121)/GEMM(120)): il termine q5_K e' piu' ripido, quindi sfonda prima.
  it("il termine entra nel value negoziato: soglia a M=97 (a 96 no)", () => {
    expect(prefillGemmWorkgroupStorageBytes(Q5K(97))).toBeGreaterThan(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(prefillGemmWorkgroupStorageBytes(Q5K(96))).toBeLessThan(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(storageNeed({ ctxMax: 6400, prefillM: 97 }).value)
      .toBeGreaterThan(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(storageNeed({ ctxMax: 6400, prefillM: 96 }).value).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    // ...ed e' il termine q5_K a farlo salire, non quello q4_0: a M=97 il q4_0
    // sta ancora sotto il tetto, e il value coincide col solo termine q5_K.
    const need = storageNeed({ ctxMax: 6400, prefillM: 97 });
    expect(gemmQ4Term(need.consumer)).toBeLessThan(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(need.value).toBe(q5kTerm(need.consumer).bytes);
    expect(need.value).toBe(prefillGemmWorkgroupStorageBytes(Q5K(97)));
    // a ogni M il value e' il maggiore fra il path fuso e i termini dichiarati:
    // ne' sotto (dichiarazione muta) ne' sopra (requisito gonfiato)
    for (const M of [8, 16, 32, 96, 97, 121, 256]) {
      const n = storageNeed({ ctxMax: 6400, prefillM: M });
      expect(n.value, `M=${M}`).toBe(Math.max(
        QWEN_WORKGROUP_STORAGE_BYTES, gemmQ4Term(n.consumer), q5kTerm(n.consumer).bytes));
    }
  });
});
