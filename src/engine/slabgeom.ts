// GEOMETRIA DEL FILE SLAB, parametrica sul modello (goal engine-velocita-decode,
// it.44). Modulo PURO: nessun device, nessun I/O, nessuna allocazione grande.
//
// PERCHE' ESISTE. `slabfile.ts` calcola offset e taglie del file slab con
// un'aritmetica chiusa che assume **due size-class CONTIGUE**: prima tutti gli
// slab della classe A (i primi layer), poi tutti quelli della classe B. Sul
// GLM e' vero — down Q4_1 sui blk.1-4, Q4_0 sui blk.5-46 — e la costante
// `GLM47_DOWN_EXPS_Q4_1_LAST` e' letteralmente il confine fra le due corse.
//
// SUL 35B NON E' VERO, e l'ho verificato sull'header prima di generalizzare:
//
//     gate  0-39 q4_K
//     up    0-39 q4_K
//     down  0-33 q4_K · 34 q6_K · 35-37 q4_K · 38-39 q6_K
//
// Le classi si ALTERNANO. Un `layer <= confine` non le descrive, e portare
// quell'aritmetica sul 35B avrebbe dato offset validi e sbagliati — la classe
// di difetto che questo progetto paga piu' cara, perche' produce pesi
// plausibili invece di un errore.
//
// LA FORMA CHE GENERALIZZA. Il file resta raggruppato PER CLASSE (tutti gli
// slab della classe 0, poi quelli della classe 1, …) — cosi' ogni classe resta
// contigua e l'offset resta un'aritmetica, non una tabella per slab. Cambia
// solo come si trova il posto di un layer dentro la sua classe: invece di
// dedurlo da un confine, si precalcola **una volta** un rango per layer.
// Sono `nLayer` interi, non `nLayer × nExpert`.
import type { SlabLayout } from "./moe";

/** Cosa serve sapere di un modello per disporre i suoi slab in un file. */
export interface SlabModelDesc {
  /** nome del file slab, uno per modello: due modelli non si sovrascrivono */
  fileName: string;
  /** primo layer MoE (i layer densi non hanno expert e non stanno nel file) */
  denseLead: number;
  nLayer: number;
  nExpert: number;
  /**
   * Il layout dello slab di un layer. E' la SOLA cosa che cambia fra modelli, e
   * arriva come funzione perche' la regola («q6_K sui layer 34, 38, 39») e'
   * dato del modello, non aritmetica.
   */
  layoutOf: (layer: number) => SlabLayout;
}

export interface SlabClass {
  /** identita' del layout, per il confronto nell'header */
  id: string;
  bytes: number;
  /** quanti slab di questa classe ci sono nel file */
  nSlabs: number;
  /** offset del primo slab della classe, header ESCLUSO */
  base: number;
}

export interface SlabGeometry {
  desc: SlabModelDesc;
  /** le classi nell'ordine in cui compaiono nel file */
  classes: SlabClass[];
  /** per layer assoluto: indice di classe, oppure -1 se il layer non e' MoE */
  classOfLayer: Int32Array;
  /** per layer assoluto: quanti layer della STESSA classe lo precedono */
  rankOfLayer: Int32Array;
  nSlabs: number;
  dataBytes: number;
}

/**
 * Deriva la geometria dal descrittore. Una passata sui layer, e da qui in poi
 * ogni offset e' O(1) senza tabelle per slab.
 *
 * L'ORDINE DELLE CLASSI e' quello di PRIMA APPARIZIONE scendendo dai layer, non
 * l'ordine alfabetico dei loro id: e' deterministico, dipende solo dal
 * descrittore, e conserva la disposizione del GLM (Q4_1 prima, Q4_0 dopo)
 * senza che nessuno debba dichiararla.
 */
export function slabGeometry(desc: SlabModelDesc): SlabGeometry {
  if (!(desc.nExpert > 0)) throw new Error("slabgeom: nExpert deve essere > 0");
  if (!(desc.denseLead >= 0 && desc.denseLead < desc.nLayer)) {
    throw new Error(`slabgeom: denseLead ${desc.denseLead} fuori da [0, ${desc.nLayer})`);
  }
  const classes: SlabClass[] = [];
  const byId = new Map<string, number>();
  const classOfLayer = new Int32Array(desc.nLayer).fill(-1);
  const rankOfLayer = new Int32Array(desc.nLayer).fill(-1);
  const seenPerClass: number[] = [];

  for (let l = desc.denseLead; l < desc.nLayer; l++) {
    const lay = desc.layoutOf(l);
    let ci = byId.get(lay.id);
    if (ci === undefined) {
      ci = classes.length;
      byId.set(lay.id, ci);
      classes.push({ id: lay.id, bytes: lay.bytes, nSlabs: 0, base: 0 });
      seenPerClass.push(0);
    } else if (classes[ci].bytes !== lay.bytes) {
      // due layout con lo STESSO id e taglie diverse renderebbero l'offset una
      // bugia: l'id e' la chiave del confronto anche nell'header del file
      throw new Error(`slabgeom: layout "${lay.id}" con due taglie (${classes[ci].bytes} e ${lay.bytes})`);
    }
    classOfLayer[l] = ci;
    rankOfLayer[l] = seenPerClass[ci];
    seenPerClass[ci]++;
    classes[ci].nSlabs += desc.nExpert;
  }

  let off = 0;
  for (const c of classes) { c.base = off; off += c.nSlabs * c.bytes; }
  const nSlabs = classes.reduce((a, c) => a + c.nSlabs, 0);
  return { desc, classes, classOfLayer, rankOfLayer, nSlabs, dataBytes: off };
}

/**
 * Offset dello slab dentro l'AREA DATI (header escluso) e sua taglia.
 *
 * Il chiamante somma l'header: tenerlo fuori rende questa funzione verificabile
 * contro l'aritmetica del GLM senza sapere quanto e' grande il suo header.
 */
export function slabRangeOf(
  g: SlabGeometry, layer: number, expert: number,
): { offset: number; bytes: number } {
  if (layer < g.desc.denseLead || layer >= g.desc.nLayer) {
    throw new Error(`slabgeom: layer ${layer} non e' MoE (MoE da ${g.desc.denseLead} a ${g.desc.nLayer - 1})`);
  }
  if (expert < 0 || expert >= g.desc.nExpert) {
    throw new Error(`slabgeom: expert ${expert} fuori da [0, ${g.desc.nExpert})`);
  }
  const ci = g.classOfLayer[layer];
  const c = g.classes[ci];
  const within = g.rankOfLayer[layer] * g.desc.nExpert + expert;
  return { offset: c.base + within * c.bytes, bytes: c.bytes };
}
