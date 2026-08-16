// La selezione del router top-k gira su TUTTO il workgroup, e le due somme che
// dipendono dall'ordine restano seriali (goal engine-velocita-decode it.18).
//
// PERCHE' ESISTE, e perche' e' un test sul TESTO invece che sui valori. I casi
// ktest `router-top4-*` girano su GPU vera e confrontano i VALORI col cpuref:
// se qualcuno riscrivesse questo kernel in forma seriale, quei casi
// resterebbero verdi cifra per cifra — la forma seriale da' gli stessi numeri,
// costa solo 72 us a layer invece di 21 (`q35-kfan-gputime-2026-08-15b.json`
// contro `q35-router-par-gputime-2026-08-16.json`). Nessun gate di correttezza
// puo' vedere la differenza: e' visibile solo nel testo emesso o in un bench.
//
// E il ritorno indietro non e' ipotetico. Il kernel E' STATO seriale fino a
// it.18, difeso da un commento che dichiarava la serializzazione «la specifica»
// perche' una riduzione «cambierebbe il tie-break». Non lo cambia: lo scan con
// `>` stretto e' il massimo sull'ordine TOTALE (punteggio, -indice), e il
// massimo su un ordine totale non dipende da come si associa. Quel commento e'
// costato 2,05 ms/token sul 35B ed e' rimasto in albero per goal interi. Questo
// test pinna la conclusione, cosi' che riaprirla costi cambiare un'asserzione
// con la sua ragione invece di riscrivere un commento.
import { describe, expect, it } from "vitest";
import { routerTopKWgsl } from "../src/engine/kernels/wgsl";
import { WEIGHTS_SUM_CLAMP_MIN } from "../src/engine/moe";

const qwen = () => routerTopKWgsl({
  nExpert: 256, nUsed: 8, weightsScale: 1, clampMin: WEIGHTS_SUM_CLAMP_MIN,
  gating: "softmax", resolve: { nExpert: 256, nUsed: 8, dirty: true },
});
const glm = () => routerTopKWgsl({
  nExpert: 64, nUsed: 4, weightsScale: 1.8, clampMin: WEIGHTS_SUM_CLAMP_MIN,
});

/** L'indice della riga in cui i thread diversi da 0 escono di scena. */
const earlyReturnAt = (src: string) => {
  const lines = src.split("\n");
  const i = lines.findIndex((l) => /if \(t != 0u\) \{ return; \}/.test(l));
  expect(i, "il kernel deve avere UNA uscita anticipata per i thread != 0").toBeGreaterThan(0);
  return i;
};
const lineOf = (src: string, re: RegExp) => {
  const i = src.split("\n").findIndex((l) => re.test(l));
  expect(i, `riga non trovata: ${re}`).toBeGreaterThan(0);
  return i;
};

describe("routerTopKWgsl: la selezione e' parallela", () => {
  for (const [nome, gen] of [["qwen35moe 256x8", qwen], ["glm47 64x4", glm]] as const) {
    describe(nome, () => {
      it("il ciclo di selezione sta PRIMA dell'uscita dei thread != 0", () => {
        const src = gen();
        // il ciclo sui k: e' quello che marca `taken` e scrive `ids`
        expect(lineOf(src, /taken\[best\] = 1u;/)).toBeLessThan(earlyReturnAt(src));
      });

      it("`taken` e' in memoria di workgroup, non un array privato di funzione", () => {
        const src = gen();
        expect(src).toMatch(/var<workgroup> taken: array<u32, \d+>;/);
        // la forma vecchia: `var taken: array<bool, N>;` dentro main
        expect(src).not.toMatch(/var taken: array<bool/);
      });

      it("il comparatore della riduzione porta il tie-break sull'indice minore", () => {
        // senza `oi < ci` la riduzione sceglierebbe un indice qualsiasi fra i
        // pari merito, e il tie-break diventerebbe dipendente dall'associazione
        expect(gen()).toMatch(/ov == redV\[t\] && oi < ci/);
      });

      it("il massimo locale usa `>` stretto: a pari punteggio resta l'indice minore", () => {
        expect(gen()).toMatch(/bi == NE \|\| sel\[i\] > bv/);
      });

      it("la somma dei probs selezionati resta seriale e nell'ordine di selezione", () => {
        const src = gen();
        // NU addendi, k crescente: l'addizione f32 non e' associativa, quindi
        // qui l'ordine E' la specifica e non va parallelizzato
        expect(src).toMatch(/for \(var k = 0u; k < NU; k = k \+ 1u\) \{ sum = sum \+ probs\[ids\[k\]\]; \}/);
        expect(lineOf(src, /sum = sum \+ probs\[ids\[k\]\]/)).toBeGreaterThan(earlyReturnAt(src));
      });
    });
  }

  it("softmax: il denominatore z resta seriale, il resto del blocco no", () => {
    const src = qwen();
    // z: NE addendi in ordine di indice, sul thread 0
    expect(src).toMatch(/zacc = zacc \+ probs\[i\]/);
    expect(src).toMatch(/var<workgroup> zw: f32;/);
    // il massimo invece e' una riduzione (esatta: max su un ordine totale)
    expect(src).toMatch(/if \(t < mstride && redV\[t \+ mstride\] > redV\[t\]\)/);
    // gli exp sono per-elemento, a passo WG e non su un thread solo
    expect(src).toMatch(/for \(var i = t; i < NE; i = i \+ 256u\) \{ probs\[i\] = exp\(probs\[i\] - mx\); \}/);
  });

  it("sigmoid non paga il blocco softmax: niente z, niente zw", () => {
    const src = glm();
    expect(src).not.toMatch(/zacc/);
    expect(src).not.toMatch(/var<workgroup> zw/);
  });

  it("la larghezza del workgroup segue nExpert e resta una potenza di due <= 256", () => {
    const wg = (src: string) => Number(/@workgroup_size\((\d+)\)/.exec(src)![1]);
    expect(wg(qwen())).toBe(256);
    expect(wg(glm())).toBe(64);
    // il buffer della riduzione ha la larghezza del workgroup, altrimenti
    // l'albero leggerebbe fuori (o lascerebbe indietro dei candidati)
    expect(qwen()).toMatch(/var<workgroup> redV: array<f32, 256>;/);
    expect(glm()).toMatch(/var<workgroup> redV: array<f32, 64>;/);
    expect(qwen()).toMatch(/var stride = 128u;/);
    expect(glm()).toMatch(/var stride = 32u;/);
  });
});
