// Sorgente dei byte GREZZI degli expert per la famiglia Qwen 3.5/3.6, e
// costruzione della `MoeModelConfig` dai metadata del GGUF (goal fase-D
// fase 1, it.7). È il GEMELLO di `expertstore.ts`, che fa la stessa cosa per
// GLM: nomina i tensori del file, ne legge le fette per-expert, e NON tocca
// la GPU — slab, arena, LRU e binding stanno tutti in `residency.ts`, uguali
// per le due famiglie (ruling direction §7-ter: una meccanica, una
// implementazione).
//
// Prima di questo modulo l'orchestratore q35 nominava i tensori, calcolava a
// mano gli offset degli slab e teneva la propria arena: era il "path di
// seconda classe" che la fase D esiste per eliminare.
import { tensorByteSize, GGML_TYPE, type GgufTensorInfo } from "./gguf";
import { mkSlabLayout, type QuantKind, type SlabLayout } from "./moe";
import type { ExpertRawBytes, MoeModelConfig } from "./residency";
import type { Q35Shape } from "./q35shape";
import { slabDescOf, type SlabModelDesc } from "./slabgeom";

/** Il nome GGUF di un tensore expert: l'unico posto che lo scrive per q35. */
export const q35ExpertTensor = (layer: number, which: "gate" | "up" | "down"): string =>
  `blk.${layer}.ffn_${which}_exps.weight`;

const KIND_OF_GGML: Record<number, QuantKind> = {
  [GGML_TYPE.Q4_0]: "q4_0", [GGML_TYPE.Q4_1]: "q4_1", [GGML_TYPE.Q8_0]: "q8_0",
  [GGML_TYPE.Q4_K]: "q4_K", [GGML_TYPE.Q5_K]: "q5_K", [GGML_TYPE.Q6_K]: "q6_K",
};

/** Nome corto della classe di slab, dal formato del tensore `down`. */
const SHORT: Record<QuantKind, string> = {
  q4_0: "q4_0", q4_1: "q4_1", q8_0: "q8_0", q4_K: "q4k", q5_K: "q5k", q6_K: "q6k",
};

const kindOf = (t: GgufTensorInfo): QuantKind => {
  const k = KIND_OF_GGML[t.type];
  if (!k) throw new Error(`q35 expert: formato ggml ${t.type} non gestito`);
  return k;
};

/** elementi di UN expert (il tensore è lo stack dei `nExpert`). */
const elemsOf = (t: GgufTensorInfo): number => t.dims[0] * t.dims[1];

/**
 * La config di residenza del modello, DEDOTTA dal file: quante classi di
 * slab ci sono, quale layer sta in quale classe, e con che layout. La classe
 * è il formato del `down` — l'unico grado di libertà osservato nella famiglia
 * (mix UD: Q4_K sulla maggioranza dei layer, Q6_K su pochi). Se un giorno
 * anche gate/up variassero, la funzione lo dice subito invece di far
 * collidere due layout diversi nella stessa classe.
 */
export function q35MoeConfig(shape: Q35Shape, info: (name: string) => GgufTensorInfo): MoeModelConfig {
  const nE = shape.nExpert as number;
  const topK = shape.nExpertUsed as number;
  if (!nE || !topK) throw new Error("q35 expert: shape senza nExpert/nExpertUsed (modello denso?)");
  const at = (l: number, w: "gate" | "up" | "down"): GgufTensorInfo => info(q35ExpertTensor(l, w));
  const g0 = at(0, "gate"), u0 = at(0, "up");
  const layouts = new Map<string, SlabLayout>();
  const clsByLayer: string[] = [];
  for (let l = 0; l < shape.nLayer; l++) {
    const gi = at(l, "gate"), ui = at(l, "up"), di = at(l, "down");
    if (kindOf(gi) !== kindOf(g0) || elemsOf(gi) !== elemsOf(g0)
      || kindOf(ui) !== kindOf(u0) || elemsOf(ui) !== elemsOf(u0)) {
      throw new Error(`q35 expert: blk.${l} ha gate/up diversi da blk.0 — la classe non può essere il solo formato del down`);
    }
    const cls = SHORT[kindOf(di)];
    if (!layouts.has(cls)) {
      layouts.set(cls, mkSlabLayout(cls,
        { kind: kindOf(gi), elems: elemsOf(gi) },
        { kind: kindOf(ui), elems: elemsOf(ui) },
        { kind: kindOf(di), elems: elemsOf(di) }));
    }
    clsByLayer[l] = cls;
  }
  return {
    id: shape.arch,
    nLayer: shape.nLayer,
    denseLead: 0, // nei MoE della famiglia gli expert ci sono da blk.0
    nExpert: nE,
    nExpertUsed: topK,
    classes: [...layouts.keys()],
    classOf: (l) => clsByLayer[l],
    layout: (c) => {
      const L = layouts.get(c);
      if (!L) throw new Error(`q35 expert: classe "${c}" non nel modello (${[...layouts.keys()].join(", ")})`);
      return L;
    },
  };
}

/**
 * Il descrittore del file slab per un modello della famiglia (it.46).
 *
 * E' un adattatore di due righe sopra `q35MoeConfig`, ed e' voluto: quella
 * funzione **legge dall'header** quale layer ha il `down` in q4_K e quale in
 * q6_K, layer per layer. Scrivere qui una lista come «i q6_K sono 34, 38, 39»
 * sarebbe vero per QUESTO file e falso per il prossimo — e la lista sbagliata
 * non darebbe un errore, darebbe slab letti all'offset di un'altra classe.
 *
 * Il nome del file porta lo SHA del GGUF sorgente: due quantizzazioni dello
 * stesso modello — UD-Q4_K_S e, poniamo, Q5_K_M — hanno geometrie diverse e non
 * devono sovrascriversi il file a vicenda. L'header porta lo SHA e lo
 * rifiuterebbe comunque, ma un rifiuto che costa una rigenerazione da 17 GiB e'
 * peggio di un nome distinto.
 */
/**
 * Dove il server serve i file slab: la STESSA cartella dei GGUF, con un symlink
 * per file — la convenzione che `public/models/` usa gia' per i modelli. Sta
 * accanto al nome del file perche' i due si leggono insieme: chi cambia l'uno
 * deve vedere l'altro.
 */
export const Q35_SLAB_BASE_URL = "/models/";

export function q35SlabDesc(
  shape: Q35Shape, info: (name: string) => GgufTensorInfo, sourceSha256: string,
): SlabModelDesc {
  if (!/^[0-9a-f]{64}$/.test(sourceSha256)) {
    throw new Error("q35SlabDesc: SHA-256 sorgente non valido");
  }
  return slabDescOf(q35MoeConfig(shape, info), `q35-${sourceSha256.slice(0, 16)}.slabs.bin`);
}

/**
 * Il reader dei byte grezzi di un expert. Il tensore contiene i `nExpert`
 * impilati: la fetta è (byte totali / nExpert), e il repack nel layout dello
 * slab lo fa `packExpertSlab` dentro la cache — qui si legge e basta.
 *
 * ASINCRONO by design: il 35B non sta in RAM, i byte arrivano da fetch Range.
 * `ExpertCache.ensure` è sincrona, quindi il forward usa `isResident` per
 * `await`tare SOLO i miss e poi consegnare i byte già in mano.
 */
export function q35ExpertReader(
  shape: Q35Shape,
  info: (name: string) => GgufTensorInfo,
  readRange: (name: string, off: number, len: number) => Promise<Uint8Array>,
): (layer: number, expert: number) => Promise<ExpertRawBytes> {
  const nE = shape.nExpert as number;
  const per = (l: number, w: "gate" | "up" | "down"): number => tensorByteSize(info(q35ExpertTensor(l, w))) / nE;
  return async (layer, expert) => {
    const [gate, up, down] = await Promise.all(
      (["gate", "up", "down"] as const).map((w) => {
        const n = per(layer, w);
        return readRange(q35ExpertTensor(layer, w), expert * n, n);
      }));
    return { gate, up, down };
  };
}
