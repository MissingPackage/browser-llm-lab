// TEST DI TIPO (goal engine-fase-d it.6) — non gira in vitest: lo verifica
// `npx tsc --noEmit`, che è già un gate permanente del progetto.
//
// È l'invariante VERO della meccanica unica, dopo che tre gate a scansione
// del sorgente sono stati bocciati (it.4-5, cinque evasioni eseguite dal
// verifier): `SlotRef` è ciò che i kernel esigono per bindare un expert, e
// il suo marchio di conio fa sì che possa nascere SOLO in `residency.ts`.
// Un'arena parallela non può fabbricarne uno: non è una questione di stile,
// è il compilatore che la ferma.
//
// Se qualcuno togliesse il marchio, l'errore atteso qui sotto sparirebbe e
// `@ts-expect-error` diventerebbe una direttiva inutilizzata ⇒ tsc ROSSO.
import type { SlotRef } from "../../src/engine/residency";

// @ts-expect-error — un'arena parallela NON può coniare uno SlotRef valido
export const slotRefContraffatto: SlotRef = {
  cls: "q4_K",
  layout: null as never,
  buffer: null as never,
  offset: 0,
  idx: 0,
};

// Consumare uno SlotRef ottenuto dalla meccanica resta lecito (nessun errore).
export const consuma = (s: SlotRef): number => s.offset + s.idx;
