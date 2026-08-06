// hostState per i report di bench (C3a fase 4d, em.6): lo stato dell'host
// DICHIARATO dal runner + CAMPIONATO da nvidia-smi. Vive nel runner e non nel
// worker perche' il browser non vede clock/temperatura/throttling — e' la
// firma che discrimina "regressione" da "host contaminato" (norma PI
// 2026-08-01: bench a macchina scarica; lezione it.11: baseline Qwen
// contaminata da un processo host).
//
// declared: la dichiarazione del runner ("quiescent" quando la run e' di gate,
// altrimenti cio' che l'operatore passa con --host-state). Il default e'
// "undeclared": un report senza dichiarazione dice che nessuno ha controllato
// — che e' un'informazione, non un errore.
import { execFileSync } from "node:child_process";

const QUERY = "temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu,clocks_throttle_reasons.active";

/** Un campione nvidia-smi, o null se l'host non lo espone (dichiarato, non errore). */
export function sampleSmi() {
  try {
    const raw = execFileSync("nvidia-smi", [`--query-gpu=${QUERY}`, "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim();
    const [t, sm, pw, mem, util, thr] = raw.split(", ");
    return {
      temperatureC: Number(t), smClockMHz: Number(sm), powerW: Number(pw),
      memUsedMiB: Number(mem), utilizationPct: Number(util),
      throttleReasons: thr, // bitmask esadecimale di nvidia-smi, tenuta com'e'
    };
  } catch {
    return null;
  }
}

/**
 * hostState completo: campione all'avvio; chiamare `close()` a run finita per
 * il campione di chiusura (il delta before/after mostra clock che salgono col
 * carico o thermal cap che si accumula — la firma della landmine §5).
 */
export function hostState(declared = "undeclared") {
  const state = { declared, before: sampleSmi(), after: null };
  return {
    state,
    close() { state.after = sampleSmi(); return state; },
  };
}
