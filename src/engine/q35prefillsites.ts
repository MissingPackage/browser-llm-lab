// INVENTARIO DEI SITI DI GEMM DEL PREFILL del Qwen3.5-4B, e la sua attribuzione
// ai SEGMENTI cronometrati (`pbCat`). Modulo PURO come `prefillbytes.ts` e
// `prefillgemmplan.ts`: nessun `GPUDevice`, nessuna allocazione, gira in vitest
// e sotto `node` senza GPU.
//
// PERCHE' ESISTE, cioe' il difetto che toglie. Questa lista viveva DENTRO
// `tests/engine-prefillgemmplan.test.ts`. Finche' a leggerla era solo quel test
// andava bene; ma i byte per segmento del checkpoint TTFT
// (`scripts/build-ttft-checkpoint.mjs`) sono gli STESSI numeri, e uno script non
// puo' importare un file di test: li si erano ricopiati a mano. Il risultato e'
// il difetto che il goal chiama «un secondo posto che decide lo stesso numero»,
// e nel checkpoint del 2026-08-14 aveva gia' pubblicato due valori falsi
// (`gemm:deltanet-out` dichiarato `legacy` DOPO che la riga 2 di engine-kquant
// l'aveva portato a multi-riga, e i byte di `gemm:qkv` presi dall'INTERA
// famiglia attn Q4_0 — 80 tensori — quando quel segmento ne cronometra 24).
// Un numero di produzione che vive solo in un test e' esattamente cio' che
// rende possibile ricopiarlo: qui e' in un modulo, e lo importano entrambi.
//
// PROVENIENZA DELLA LISTA (non e' inventata e non e' dedotta dalla shape):
// header GGUF di `/home/neuromancer/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf`
// (426 tensori), riletto tensore per tensore il 2026-08-13; i totali per
// famiglia sono quelli pinnati in `results/engine/q35-header-dump-2026-08-10.json`
// (`typeHistogram`), e `tests/engine-prefillgemmplan.test.ts` [6b] li riverifica
// — n E byte — CONTRO questa lista, cosi' la lista non puo' scivolare in
// silenzio. La struttura (32 layer, full quando l%4==3) e' quella che l'header
// dichiara: `attn_q/k/v/output` compaiono esattamente sui layer 3,7,...,31 e
// `attn_qkv/gate` + `ssm_*` sugli altri 24.
//
// Dimensioni GGUF = [K, N] (ne[0] = ingresso). Nessun sito per-layer e' q6_K —
// il q6_K del 4B sta solo in `token_embd`, che non e' un GEMM di prefill —
// quindi su tutti i kind di questa lista i byte device coincidono con quelli del
// file e il confronto con l'istogramma e' lecito.
import { dispatchWeightBytes, type PrefillQuantKind } from "./prefillbytes";
import { planPrefillGemm, type PrefillSite } from "./prefillgemmplan";

/** Geometria del 4B: 32 layer, full-attention quando `l % 4 == 3`. */
export const Q35_4B = { nLayer: 32, fullInterval: 4 } as const;

/**
 * Un sito con addosso il SEGMENTO che lo cronometra.
 *
 * `cat` e' la stessa stringa che `q35gpumodel.ts` scrive in `pbCat` prima di
 * spingere quel dispatch, e la ATTRIBUZIONE e' una dichiarazione di questo
 * file: il timbro vero lo mette il motore, che non e' importabile senza device.
 * Non e' pero' una dichiarazione non verificata — `tests/engine-ttft-checkpoint-banda.test.ts`
 * ricontrolla ogni `cat` contro le assegnazioni `pbCat = "..."` del sorgente del
 * motore, e il checkpoint confronta i dispatch DERIVATI con quelli MISURATI
 * nell'artefatto prima di pubblicare una banda.
 */
export interface Q35PrefillSite extends PrefillSite {
  /** categoria `pbCat` del segmento che cronometra questo dispatch */
  cat: string;
}

type SiteSpec = { site: string; cat: string; kind: PrefillQuantKind; K: number; N: number };

/** Siti FULL-attention (8 layer: l%4==3) — tutti Q4_0. */
const SITES_FULL: readonly SiteSpec[] = [
  { site: "attn_q", cat: "gemm:qkv", kind: "q4_0", K: 2560, N: 8192 },   // gate fuso: 2*nHead*headDim
  { site: "attn_k", cat: "gemm:qkv", kind: "q4_0", K: 2560, N: 1024 },   // nKvHead*headDim
  { site: "attn_v", cat: "gemm:qkv", kind: "q4_0", K: 2560, N: 1024 },
  { site: "attn_output", cat: "gemm:attn-out", kind: "q4_0", K: 4096, N: 2560 },
];

/** Siti LINEAR-attention / Gated DeltaNet (24 layer). */
const SITES_LINEAR: readonly SiteSpec[] = [
  { site: "attn_qkv", cat: "deltanet:gemm", kind: "q4_0", K: 2560, N: 8192 },  // (2*nK+nV)*hd
  { site: "attn_gate", cat: "deltanet:gemm", kind: "q4_0", K: 2560, N: 4096 }, // nV*hd
  { site: "ssm_alpha", cat: "deltanet:gemm", kind: "q8_0", K: 2560, N: 32 },
  { site: "ssm_beta", cat: "deltanet:gemm", kind: "q8_0", K: 2560, N: 32 },
  { site: "ssm_out", cat: "gemm:deltanet-out", kind: "q5_K", K: 4096, N: 2560 },   // Q5_K nel file, non Q4_0
];

/** FFN densa su OGNI layer; `ffn_down` e' Q4_1 sui layer 0..3, Q4_0 altrove. */
export const FFN_Q41_LAYERS = [0, 1, 2, 3] as const;

/**
 * I 248 siti di GEMM per-layer del 4B, nell'ordine in cui il motore li spinge.
 * `token_embd`/`output` non ci sono: non sono GEMM di prefill a chunk.
 */
export function q35PrefillSites4B(): Q35PrefillSite[] {
  const out: Q35PrefillSite[] = [];
  for (let l = 0; l < Q35_4B.nLayer; l++) {
    const full = l % Q35_4B.fullInterval === Q35_4B.fullInterval - 1;
    for (const s of full ? SITES_FULL : SITES_LINEAR) out.push({ ...s, site: `blk.${l}.${s.site}` });
    out.push({ site: `blk.${l}.ffn_gate`, cat: "gemm:ffn", kind: "q4_0", K: 2560, N: 9216 });
    out.push({ site: `blk.${l}.ffn_up`, cat: "gemm:ffn", kind: "q4_0", K: 2560, N: 9216 });
    // Q4_1 sui primi 4 layer: categoria PROPRIA (riga 4 di engine-ttft), perche'
    // dentro `gemm:ffn-down` la loro quota era dedotta invece che misurata.
    const q41 = (FFN_Q41_LAYERS as readonly number[]).includes(l);
    out.push({
      site: `blk.${l}.ffn_down`,
      cat: q41 ? "gemm:ffn-down-q41" : "gemm:ffn-down",
      kind: q41 ? "q4_1" : "q4_0",
      K: 9216, N: 2560,
    });
  }
  return out;
}

/** Byte di peso di UN segmento, derivati — mai ricopiati. */
export interface Q35PrefillBandaSeg {
  cat: string;
  /** quanti siti (= quanti tensori) compongono il segmento */
  siti: number;
  /** descrizione leggibile: «24 ssm_out Q5_K» — costruita, non scritta */
  tensori: string;
  /** `multirow`, `legacy`, o `mista` quando il segmento contiene entrambe */
  forma: "multirow" | "legacy" | "mista";
  /** ripartizione dei siti per forma: e' quella che decide il moltiplicatore M */
  formaPerSito: { multirow: number; legacy: number };
  /** byte di UNA passata sui pesi del segmento (M non entra): il "peso" dei tensori */
  bytePerPassata: number;
  /**
   * byte che il segmento fa attraversare la memoria IN UN CHUNK.
   * Il moltiplicatore M SEGUE LA FORMA e non e' scritto qui: lo mette
   * `dispatchWeightBytes` (legacy = M·N·bytesPerRow, multirow = N·bytesPerRow).
   * E' questa la differenza fra le due forme, ed e' il motivo per cui un `* M`
   * scritto a mano accanto a una `forma` scritta a mano puo' mentire due volte.
   */
  bytePerChunk: number;
}

/**
 * I BYTE E LE FORME DEL SEGMENTO, DERIVATI.
 *
 * Due sorgenti, nessuna terza: la FORMA la decide `planPrefillGemm` (`via`), i
 * BYTE li conta `dispatchWeightBytes` di `prefillbytes.ts`. Questa funzione non
 * ha aritmetica propria oltre alla somma per categoria.
 *
 * `idot` non cambia la forma per M>1 (le vie `idot` e `f32` sono la STESSA forma
 * multi-riga, cambia solo il ciclo interno) — il test lo verifica invece di
 * darlo per scontato, cosi' un checkpoint prodotto su un device senza la
 * language feature non pubblica byte diversi.
 */
export function q35PrefillBandaByCat(o: { M: number; idot: boolean }): Q35PrefillBandaSeg[] {
  const acc = new Map<string, {
    siti: Q35PrefillSite[]; multirow: number; legacy: number;
    bytePerPassata: number; bytePerChunk: number;
  }>();
  for (const s of q35PrefillSites4B()) {
    const via = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M: o.M, idot: o.idot }).via;
    const form = via === "legacy" ? "legacy" as const : "multirow" as const;
    let e = acc.get(s.cat);
    if (e === undefined) {
      e = { siti: [], multirow: 0, legacy: 0, bytePerPassata: 0, bytePerChunk: 0 };
      acc.set(s.cat, e);
    }
    e.siti.push(s);
    if (form === "legacy") e.legacy++; else e.multirow++;
    // UNA passata = la stessa shape contata a M=1: e' il "peso del tensore"
    // visto dal kernel, e non dipende dalla forma.
    e.bytePerPassata += dispatchWeightBytes({ form: "multirow", kind: s.kind, K: s.K, N: s.N, M: 1 });
    e.bytePerChunk += dispatchWeightBytes({ form, kind: s.kind, K: s.K, N: s.N, M: o.M });
  }
  return [...acc.entries()].map(([cat, e]) => ({
    cat,
    siti: e.siti.length,
    tensori: descrivi(e.siti),
    forma: e.legacy === 0 ? "multirow" : e.multirow === 0 ? "legacy" : "mista",
    formaPerSito: { multirow: e.multirow, legacy: e.legacy },
    bytePerPassata: e.bytePerPassata,
    bytePerChunk: e.bytePerChunk,
  }));
}

/**
 * «24 ssm_out Q5_K», «48 attn_qkv+attn_gate Q4_0 · 48 ssm_alpha+ssm_beta Q8_0».
 * Costruita dai siti: la stringa che nel builder era scritta a mano diceva «80
 * attn Q4_0» per un segmento che ne cronometra 24.
 */
function descrivi(siti: Q35PrefillSite[]): string {
  const perKind = new Map<PrefillQuantKind, Set<string>>();
  const nPerKind = new Map<PrefillQuantKind, number>();
  for (const s of siti) {
    const base = s.site.replace(/^blk\.\d+\./, "");
    if (!perKind.has(s.kind)) perKind.set(s.kind, new Set());
    perKind.get(s.kind)!.add(base);
    nPerKind.set(s.kind, (nPerKind.get(s.kind) ?? 0) + 1);
  }
  return [...perKind.entries()]
    .map(([kind, names]) => `${nPerKind.get(kind)} ${[...names].sort().join("+")} ${kind.toUpperCase()}`)
    .join(" · ");
}
