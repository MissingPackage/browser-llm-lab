// TEST DI TIPO (goal engine-fase-d it.6) — non gira in vitest: lo verifica
// `npx tsc --noEmit`, che è già un gate permanente del progetto.
//
// COSA GARANTISCE, ESATTAMENTE (formulazione corretta dal verifier di it.6,
// che ha bocciato la prima versione per sovravendita): il marchio impedisce
// di **contraffare** uno `SlotRef` — nessuno può spacciare la propria arena
// per la residenza unica. NON impedisce di **ignorarlo**: un'arena che non
// usa `SlotRef` non viene nemmeno sfiorata dal compilatore, ed è esattamente
// il caso di `src/engine/q35gpumodel.ts` finché la fase 1 non chiude (it.7).
// Il marchio diventa PORTANTE solo quando il binding degli expert passa da
// `SlotRef`/`slotBindRanges`: da lì in poi, una seconda arena non può più
// arrivare ai kernel.
//
// Limiti NOTI (verificati con sonde ostili, non congetturati): passano
// `as any as SlotRef` (cast dichiarato, rumoroso in review), l'inflow
// any-tipizzato (`JSON.parse`, `Object.create`) e lo **spread di uno
// `SlotRef` genuino** (`{...vero, idx: 99}`), che ne conserva il marchio.
// Chi ha già uno slot legittimo può derivarne altri senza cast: il marchio
// sorveglia il CONIO, non la circolazione.
//
// Se qualcuno togliesse il marchio, l'errore atteso qui sotto sparirebbe e
// `@ts-expect-error` diventerebbe una direttiva inutilizzata ⇒ tsc ROSSO.
import type { SlotRef } from "../../src/engine/residency";

// @ts-expect-error — un'arena parallela non può CONIARE uno SlotRef valido
export const slotRefContraffatto: SlotRef = {
  cls: "q4_K",
  layout: null as never,
  buffer: null as never,
  offset: 0,
  idx: 0,
};

// Consumare uno SlotRef ottenuto dalla meccanica resta lecito (nessun errore).
export const consuma = (s: SlotRef): number => s.offset + s.idx;
