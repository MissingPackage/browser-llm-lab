// Contratto di conformance condiviso da TUTTI gli adapter (stesso set di check, stesso
// ordine, stessa severità). Esiste per chiudere esattamente la classe di bug che i test
// unitari con engine finti non possono vedere: i test unitari iniettano un motore fasullo
// e quindi non possono verificare come l'adapter è davvero cablato alla libreria reale
// (streamer giusto/sbagliato, callback di default che scrivono su stdout, ecc.). Questo
// contratto va eseguito in un browser vero (vedi scripts/conformance.mjs), MAI da `npm test`.
//
// Non lancia mai eccezioni: ogni check è isolato in un try/catch e riportato come
// pass/fail, così il fallimento di un adapter non impedisce all'altro di girare e
// riportare (requisito esplicito).
import type { InferenceAdapter } from "../adapters/types";
import type { GenTimeline } from "../metrics";

export interface CheckResult {
  name: string;
  pass: boolean;
  detail: string;
}

export interface ContractResult {
  adapterId: string;
  modelId: string;
  checks: CheckResult[];
}

// Prompt scelto per rendere improbabile uno stop anticipato per end-of-sequence: chiediamo
// di contare/elencare, non di rispondere a una domanda breve (che potrebbe terminare in
// pochissimi token e far apparire un bug di conteggio come "stop legittimo").
const COUNTING_PROMPT = "Count from one to one hundred, writing exactly one number per line and nothing else.";

async function attempt(name: string, fn: () => Promise<Omit<CheckResult, "name">>): Promise<CheckResult> {
  try {
    const r = await fn();
    return { name, ...r };
  } catch (e) {
    return { name, pass: false, detail: `threw: ${e instanceof Error ? e.message : String(e)}` };
  }
}

/**
 * Esegue il contratto di conformance contro un adapter REALE (motore reale, non un fake
 * iniettato) e un modello REALE. `tokenBudget` è il numero di token generati nel check
 * cardine (timestamp-per-token) e nel check di determinismo — tenerlo piccolo (16) per
 * mantenere il run veloce.
 */
export async function runConformanceContract(
  adapter: InferenceAdapter,
  modelId: string,
  tokenBudget: number,
): Promise<ContractResult> {
  const checks: CheckResult[] = [];

  // 1. capabilities(): forma (tre booleani).
  checks.push(
    await attempt("capabilities() shape", async () => {
      const caps = adapter.capabilities();
      const fields: Array<[string, unknown]> = [
        ["logprobs", caps.logprobs],
        ["streaming", caps.streaming],
        ["seed", caps.seed],
      ];
      const bad = fields.filter(([, v]) => typeof v !== "boolean").map(([k]) => k);
      return {
        pass: bad.length === 0,
        detail:
          bad.length === 0
            ? `logprobs=${caps.logprobs} streaming=${caps.streaming} seed=${caps.seed}`
            : `non-boolean field(s): ${bad.join(", ")}`,
      };
    }),
  );
  // Serve più avanti (streaming claim); se capabilities() è rotto usiamo un default che
  // non sopprime i check successivi (il check "streaming claim" fallirà comunque se serve).
  const streamingClaim = (() => {
    try {
      return adapter.capabilities().streaming;
    } catch {
      return true; // conservativo: verifica comunque che i timestamp siano distribuiti
    }
  })();

  // 2. load(): LoadReport valido.
  checks.push(
    await attempt("load() resolves to a valid LoadReport", async () => {
      const report = await adapter.load(modelId, () => {});
      const finitePositive = Number.isFinite(report.loadMs) && report.loadMs > 0;
      const validCacheState = report.cacheState === "cold" || report.cacheState === "warm" || report.cacheState === "unknown";
      return {
        pass: finitePositive && validCacheState,
        detail: `loadMs=${report.loadMs} cacheState=${report.cacheState}`,
      };
    }),
  );

  // Le check seguenti richiedono un adapter caricato con successo. Se load() è fallito, le
  // eseguiamo comunque (per riportare il loro fallimento reale invece di ometterle in
  // silenzio) — generate() dovrebbe rigettare in modo pulito e verrà riportato come FAIL con
  // il motivo, non come eccezione che interrompe il contratto.

  let timeline1: GenTimeline | null = null;
  checks.push(
    await attempt(`one timestamp per generated token (N=${tokenBudget})`, async () => {
      timeline1 = await adapter.generate({ prompt: COUNTING_PROMPT, maxTokens: tokenBudget });
      const n = timeline1.chunkTimestamps.length;
      // Uguaglianza stretta: un conteggio più basso può essere uno stop anticipato legittimo
      // (EOS), ma è comunque un FAIL — riportiamo il conteggio reale per distinguerlo da un
      // bug di wiring (es. callback_function al posto di token_callback_function, che in
      // certi casi sottoconta silenziosamente invece di fermarsi prima).
      return {
        pass: n === tokenBudget,
        detail: `expected ${tokenBudget} timestamps, got ${n}${n < tokenBudget ? " (possible early EOS or under-counting bug)" : ""}`,
      };
    }),
  );

  checks.push(
    await attempt("timestamps are monotonic and start at/after tRequestStart", async () => {
      if (!timeline1) throw new Error("no timeline captured (generate() failed above)");
      const t = timeline1;
      if (t.chunkTimestamps.length === 0) {
        return { pass: false, detail: "no chunks to check (empty timeline)" };
      }
      const startsAfterRequest = t.tRequestStart <= t.chunkTimestamps[0];
      let nonDecreasing = true;
      for (let i = 1; i < t.chunkTimestamps.length; i++) {
        if (t.chunkTimestamps[i] < t.chunkTimestamps[i - 1]) {
          nonDecreasing = false;
          break;
        }
      }
      return {
        pass: startsAfterRequest && nonDecreasing,
        detail: `tRequestStart=${t.tRequestStart} first=${t.chunkTimestamps[0]} startsAfterRequest=${startsAfterRequest} nonDecreasing=${nonDecreasing}`,
      };
    }),
  );

  checks.push(
    await attempt("streaming claim matches reality (timestamps spread over time)", async () => {
      if (!timeline1) throw new Error("no timeline captured (generate() failed above)");
      const t = timeline1;
      if (!streamingClaim) {
        return { pass: true, detail: "capabilities().streaming=false — nothing to verify" };
      }
      const distinct = new Set(t.chunkTimestamps).size;
      return {
        pass: distinct > 1,
        detail: `capabilities().streaming=true, distinct timestamp values=${distinct} (need >1 to prove genuine streaming, not one emit-everything-at-the-end burst)`,
      };
    }),
  );

  // Determinismo: solo a livello di conteggio token, l'unica granularità che GenTimeline
  // espone. GenTimeline non porta il testo generato, quindi il determinismo TESTUALE non è
  // verificabile attraverso l'interfaccia InferenceAdapter così com'è oggi: richiederebbe un
  // cambio a src/adapters/types.ts, che è docket-gated e fuori scope qui. Non impostiamo
  // quell'illusione: verifichiamo solo ciò che l'interfaccia espone davvero.
  checks.push(
    await attempt("determinism (token count) across two identical generate() calls", async () => {
      if (!timeline1) throw new Error("no first timeline captured (generate() failed above)");
      const t1 = timeline1;
      const timeline2 = await adapter.generate({ prompt: COUNTING_PROMPT, maxTokens: tokenBudget });
      const n1 = t1.chunkTimestamps.length;
      const n2 = timeline2.chunkTimestamps.length;
      return {
        pass: n1 === n2,
        detail: `run1=${n1} tokens, run2=${n2} tokens (text-level determinism not checkable: GenTimeline carries no generated text)`,
      };
    }),
  );

  // dispose(): resolves, e generate() dopo dispose() rigetta.
  checks.push(
    await attempt("dispose() resolves", async () => {
      await adapter.dispose();
      return { pass: true, detail: "dispose() resolved without throwing" };
    }),
  );
  checks.push(
    await attempt("generate() after dispose() rejects", async () => {
      try {
        await adapter.generate({ prompt: COUNTING_PROMPT, maxTokens: 1 });
        return { pass: false, detail: "generate() resolved after dispose() — expected rejection" };
      } catch (e) {
        return { pass: true, detail: `rejected as expected: ${e instanceof Error ? e.message : String(e)}` };
      }
    }),
  );

  return { adapterId: adapter.id, modelId, checks };
}
