import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  gemvQuantWgsl, prefillGemmQ4SplitKIdotWgsl, prefillGemmQ4SplitKWgsl,
} from "../src/engine/kernels/wgsl";

// ---------------------------------------------------------------------------
// IL DEBITO DELLA BIT-IDENTITA' — un promemoria che SUONA DA SOLO.
//
// RULING DEL PI, 2026-08-14 (docket item 22): «capisco che non sia fattibile in
// questa fase di transizione, ma dovra' esserci in futuro. Quando avremo migrato
// tutto su int.»
//
// Cioe': la bit-identita' fra il prefill a chunk e il `step()` sequenziale e'
// SOSPESA, non abolita. E' caduta in it.15, quando la via intera ha iniziato a
// quantizzare le attivazioni a int8 nel SOLO ramo a chunk: da quel momento i due
// bracci fanno aritmetica diversa e divergono per costruzione.
//
// LA CONDIZIONE CHE LA FA TORNARE non e' «riportare il chunk in virgola
// mobile» — sarebbe buttare via 1,745x. E' l'OPPOSTO: quando anche il percorso
// sequenziale sara' su intero, i due bracci torneranno a fare la stessa
// aritmetica, e la bit-identita' tornera' a essere esigibile.
//
// PERCHE' QUESTO FILE ESISTE. Un debito scritto in un docket lo paga solo chi
// si ricorda di rileggere il docket. Questo test invece FALLISCE DA SOLO nel
// momento esatto in cui la condizione del ruling si avvera: chi migra il
// percorso sequenziale su intero trova un rosso che gli dice, con la citazione
// del ruling in mano, che adesso il criterio va rimesso a bit. E' la stessa
// forma del test [6d] di engine-prefillgemmplan: «e' il promemoria, non un
// lasciapassare».
//
// COME FALLISCE, meccanicamente: il segnale della via intera e' l'intrinseco
// WGSL `dot4I8Packed`. Oggi compare nei kernel del PREFILL a chunk e non nel
// GEMV che `step()` usa. Il giorno in cui comparira' anche li', il primo `it`
// qui sotto diventa rosso.
// ---------------------------------------------------------------------------
describe("debito: la bit-identita' del prefill a chunk torna esigibile quando il sequenziale passa su intero", () => {
  /** Le shape che `step()` usa davvero sul 4B (attn e ffn, q4_0, K%64==0). */
  const SEQ_SHAPES = [
    { kind: "q4_0" as const, K: 2560, N: 9216 },
    { kind: "q4_0" as const, K: 9216, N: 2560 },
  ];

  it("il segnale che usiamo e' DAVVERO quello della via intera, non una stringa a caso", () => {
    // Senza questa riga il test qui sotto passerebbe anche se `dot4I8Packed`
    // smettesse di essere il marcatore della via intera — e un test che passa
    // comunque non sorveglia niente. Qui si prova che il marcatore discrimina:
    // il kernel intero ce l'ha, quello in virgola mobile no.
    const idot = prefillGemmQ4SplitKIdotWgsl({ kind: "q4_0", K: 2560, N: 9216, M: 16, splits: 4 });
    const f32 = prefillGemmQ4SplitKWgsl({ kind: "q4_0", K: 2560, N: 9216, M: 16, splits: 4 });
    expect(idot, "la via intera usa dot4I8Packed").toContain("dot4I8Packed");
    expect(f32, "la via in virgola mobile no").not.toContain("dot4I8Packed");
  });

  it("FINCHE' il GEMV sequenziale e' in virgola mobile, la sospensione regge", () => {
    for (const s of SEQ_SHAPES) {
      const code = gemvQuantWgsl({ ...s, hasBias: false });
      expect(code, `K${s.K}xN${s.N}: se questo kernel usa dot4I8Packed, il percorso sequenziale ` +
        `E' MIGRATO SU INTERO e il ruling del 2026-08-14 (docket item 22) chiede di rimettere ` +
        `il gate q35-prefillchunk-4b sulla BIT-IDENTITA'. Non silenziare questo test: ` +
        `cambia il criterio in q35conf.worker.ts e aggiorna item 22.`)
        .not.toContain("dot4I8Packed");
    }
  });

  it("il gate DICHIARA di essere in sospensione, e dice cosa la fa finire", () => {
    // Se qualcuno cambia il criterio senza toccare il testo, il report mente a
    // chi lo legge fra sei mesi — ed e' la landmine «i JSON possono mentire in
    // silenzio», che qui costerebbe la fiducia in un gate di correttezza.
    const W = readFileSync(join(process.cwd(), "src/engine/q35conf/q35conf.worker.ts"), "utf8");
    expect(W, "il criterio di oggi").toContain('criterion: "argmax"');
    expect(W, "la causa della sospensione").toContain("bitIdentitySuspendedBy");
    expect(W, "la condizione che la fa tornare").toContain("bitIdentityReturnsWhen");
    expect(W, "la bit-identita' resta MISURATA anche se non e' piu' il criterio")
      .toMatch(/bitIdentical:\s*bitEqual === cmp/);
  });

  it("l'argmax e' un criterio piu' debole, e il report deve permettere di accorgersene", () => {
    // L'argmax puo' coincidere anche con logit visibilmente diversi. Il ruling
    // lo accetta come criterio di transizione, ma solo «col numero accanto»:
    // maxAbs e maxRel devono restare nel report, o la prossima persona non ha
    // modo di vedere che la divergenza sta crescendo.
    const W = readFileSync(join(process.cwd(), "src/engine/q35conf/q35conf.worker.ts"), "utf8");
    expect(W).toMatch(/maxAbs,\s*maxRel/);
    expect(W, "argmax per chunk, non aggregato").toContain("argmaxIdentical");
  });
});
