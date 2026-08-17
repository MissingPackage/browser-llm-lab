// IL PREDICATO SULLA SHAPE, e i due formati nuovi (q2_K / q3_K) che lo rendono
// urgente — spec 2026-08-17-q2k-q3k-kernels §4 T5.
//
// LA TRAPPOLA CHE QUESTO FILE SORVEGLIA, scritta per esteso perche' e' gia'
// costata una volta (goal engine-velocita-decode, riga 2d, e il commento di
// tests/engine-prefillgemmplan-notwired.test.ts la racconta): il flag `wired` di
// `PREFILL_GEMM_SPEC` e' per FORMATO, non per SHAPE. Accendere un formato senza
// un predicato sulla geometria fa entrare NELLO STESSO ISTANTE tutti i siti di
// quel formato — compresi i 48 `ssm_alpha`/`ssm_beta` del 4B, che hanno N=32 e
// userebbero mezzo workgroup su una forma che ne produce 64 righe di uscita.
// Il q8_0 e' stato cablato solo dopo che il predicato esisteva; q2_K e q3_K
// restano `wired: false` finche' la misura su GPU non c'e', ma il predicato
// deve essere gia' li' e deve essere gia' ESERCITATO su di loro — un guard che
// nessun test attraversa e' un guard di cui non sai se funziona.
//
// COSA E' CAMBIATO RISPETTO A PRIMA: il controllo su N esisteva gia', scritto
// in linea dentro `kernelVerdict`. Ora ha un NOME (`prefillGemmShapeOk`) e una
// firma, cosi' che (i) la ragione del rifiuto sia una funzione pura
// interrogabile da un test senza passare per tutto il piano, e (ii) il gate
// strutturale [s4] qui sotto possa pretendere che di predicati sulla shape ce
// ne sia UNO SOLO — la stessa postura del gate [w4] su `prefillGemmWiring`.
//
// L'ORDINE DELLA COLPA, e perche' il caso [s3] chiede quel che chiede. q2_K e
// q3_K sono kind VERI (t2 li ha portati) e sono `wired: false`: quindi un sito
// q2_K a N=32 e' rifiutato per DUE ragioni indipendenti, e la prima a parlare
// e' il cablaggio — lo pre-registra il caso [6] di
// tests/engine-prefillgemm-nmin.test.ts, e resta l'ordine giusto perche' con N
// buono quel sito resterebbe legacy comunque. Ma «formato non cablato» DA SOLO
// mentirebbe per omissione: suggerirebbe che cablare il formato manderebbe
// quel sito sulla via veloce. Per questo la ragione del cablaggio PORTA CON
// SE' quella della shape, e [s3] pretende esattamente questo: la ragione del
// predicato dentro la ragione del piano, parola per parola, su una rotta
// legacy.
import { describe, expect, it } from "vitest";
import { readFileSync, globSync } from "node:fs";
import { join, relative } from "node:path";
import { planPrefillGemm, prefillGemmShapeOk } from "../src/engine/prefillgemmplan";
import {
  PREFILL_GEMM_KINDS, PREFILL_GEMM_ROWS_PER_WG, isPrefillGemmKind, prefillGemmWiring,
} from "../src/engine/kernels/wgsl";

/** M del prefill in produzione (`PREFILL_M` = 16). */
const M = 16;

/**
 * LE SHAPE VERE, non numeri inventati.
 *
 * `EXPERT_*` sono le due geometrie degli expert del 35B — le stesse due su cui
 * la fase 0 ha misurato q4_K e q6_K (`[2048,512]` gate/up, `[512,2048]` down):
 * sono le shape che q2_K e q3_K servirebbero, e devono passare il predicato.
 *
 * `SSM_4B` e' la shape dei 48 siti `ssm_alpha`/`ssm_beta` del 4B: N=32, cioe'
 * mezzo workgroup. E' la shape che NON deve mai passare, su nessun formato.
 */
const EXPERT_GATE_UP = { K: 2048, N: 512 } as const;
const EXPERT_DOWN = { K: 512, N: 2048 } as const;
const SSM_4B = { K: 2048, N: 32 } as const;

// ---------------------------------------------------------------------------
// [s0] IL PREREQUISITO — i due formati sono kind veri, e sono spenti.
// ---------------------------------------------------------------------------
describe("[s0] q2_K/q3_K: dove sono, oggi", () => {
  it("sono nell'elenco dei kernel di prefill (t2 e' atterrato)", () => {
    // Questo caso non e' cerimonia: se t2 tornasse indietro, i casi qui sotto
    // continuerebbero a girare sul predicato — che non guarda la famiglia — e
    // sarebbero verdi misurando meno di quel che dichiarano. Qui si dice a voce
    // alta su quale base sono verdi.
    for (const k of ["q2_K", "q3_K"]) {
      expect(isPrefillGemmKind(k), `${k} non e' in PREFILL_GEMM_KINDS`).toBe(true);
    }
  });

  it("il predicato risponde comunque, e sempre con una ragione", () => {
    // E' un predicato sulla GEOMETRIA: non ha bisogno di sapere se il formato
    // ha gia' un kernel, e la sua ragione non e' mai vuota (postura di
    // `GemvCaps.why` — un booleano nudo non e' diagnosticabile a posteriori).
    for (const k of ["q2_K", "q3_K"] as const) {
      expect(prefillGemmShapeOk({ kind: k, ...EXPERT_GATE_UP }).why.length,
        `${k}: il predicato deve dire sempre perche'`).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// [s1] (a) LE SHAPE DEGLI EXPERT DEL 35B PASSANO.
// ---------------------------------------------------------------------------
describe("[s1] le shape degli expert del 35B sono ammesse dal predicato", () => {
  it("q2_K K=2048 N=512 e q3_K K=512 N=2048: ok", () => {
    const a = prefillGemmShapeOk({ kind: "q2_K", ...EXPERT_GATE_UP });
    const b = prefillGemmShapeOk({ kind: "q3_K", ...EXPERT_DOWN });
    expect(a.ok, `q2_K ${EXPERT_GATE_UP.K}x${EXPERT_GATE_UP.N}: ${a.why}`).toBe(true);
    expect(b.ok, `q3_K ${EXPERT_DOWN.K}x${EXPERT_DOWN.N}: ${b.why}`).toBe(true);
    expect(a.why.length).toBeGreaterThan(0);
    expect(b.why.length).toBeGreaterThan(0);
  });

  it("il confine e' la soglia del kernel, non un numero scritto qui", () => {
    // Se questa costante e il divisore della griglia divergessero, il predicato
    // proteggerebbe una soglia diversa da quella vera.
    expect(PREFILL_GEMM_ROWS_PER_WG).toBe(64);
    for (const k of ["q2_K", "q3_K"] as const) {
      expect(prefillGemmShapeOk({ kind: k, K: 2048, N: PREFILL_GEMM_ROWS_PER_WG - 1 }).ok)
        .toBe(false);
      expect(prefillGemmShapeOk({ kind: k, K: 2048, N: PREFILL_GEMM_ROWS_PER_WG }).ok).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// [s2] (b) LA SHAPE ssm_alpha/ssm_beta DEL 4B NON PASSA, E LA RAGIONE NOMINA N.
// ---------------------------------------------------------------------------
describe("[s2] N=32 (ssm_alpha/ssm_beta del 4B) non e' ammessa", () => {
  it("q2_K K=2048 N=32: ok=false e la ragione NOMINA 32", () => {
    const r = prefillGemmShapeOk({ kind: "q2_K", ...SSM_4B });
    expect(r.ok).toBe(false);
    // «nomina N» e' il requisito che rende la ragione utile: senza il numero,
    // in telemetria si leggerebbe «shape non ammessa» su 48 siti senza sapere
    // QUALE dimensione li esclude.
    expect(r.why, "la ragione non nomina N").toContain("32");
    expect(r.why).toContain("workgroup");
  });

  it("...e vale su OGNI formato dell'elenco: e' un predicato sulla SHAPE", () => {
    // Il punto per cui il predicato esiste: se fosse per famiglia, accendere un
    // formato accenderebbe anche i 48 siti. Qui si pretende il contrario su
    // tutto l'elenco vero — cablati compresi — cosi' il caso non marcisce al
    // prossimo cablaggio.
    for (const k of PREFILL_GEMM_KINDS) {
      const r = prefillGemmShapeOk({ kind: k, ...SSM_4B });
      expect(r.ok, `${k}: N=32 ammessa`).toBe(false);
      expect(r.why, `${k}: la ragione non nomina N`).toContain("32");
    }
  });
});

// ---------------------------------------------------------------------------
// [s3] (c) IL PIANO USA IL PREDICATO, non una seconda copia della soglia.
//
// q2_K e' un kind vero e NON cablato: a N=32 il piano ha due ragioni per
// rifiutare, e le riporta ENTRAMBE — il cablaggio per primo (l'ordine
// pre-registrato da [6] in engine-prefillgemm-nmin.test.ts), la shape subito
// dopo, perche' cablare il formato non salverebbe questo sito.
// ---------------------------------------------------------------------------
describe("[s3] planPrefillGemm su N=32 rende legacy con la ragione del predicato", () => {
  it("q2_K K=2048 N=32: legacy, e la ragione contiene quella di `prefillGemmShapeOk`", () => {
    const why = prefillGemmShapeOk({ kind: "q2_K", ...SSM_4B }).why;
    for (const idot of [true, false]) {
      const r = planPrefillGemm({ kind: "q2_K", ...SSM_4B, M, idot });
      expect(r.via, `idot=${idot}: ${r.reason}`).toBe("legacy");
      // NON «contiene N=32» soltanto: contiene LA RAGIONE DEL PREDICATO, parola
      // per parola. E' cio' che impedisce che il piano riscriva la stessa
      // diagnosi con altre parole e che le due versioni divergano in silenzio.
      expect(r.reason, "il piano non riporta la ragione del predicato").toContain(why);
      // e la causa DOMINANTE resta il cablaggio, davanti alla concorrente
      expect(r.reason, "la ragione non accusa piu' il cablaggio").toContain("cablat");
      expect(r.reason.indexOf("cablat"), "la shape parla prima del cablaggio")
        .toBeLessThan(r.reason.indexOf(why));
      // niente scratch prenotato per un dispatch che nessuno emette
      expect(r.splits).toBe(0);
      expect(r.partialFloats).toBe(0);
      expect(r.wgStorageBytes).toBe(0);
      expect(r.xqU32).toBe(0);
      expect(r.xscF32).toBe(0);
    }
  });

  it("su un formato CABLATO la shape parla da sola, ed e' l'unica colpa", () => {
    // L'altra meta' del ramo: qui il cablaggio non ha niente da dire, quindi il
    // rifiuto viene marcato `shape` e la frase manda a leggere la sede vera
    // (prima diceva `prefillGemmCheck`, che su N non decide niente).
    const why = prefillGemmShapeOk({ kind: "q8_0", ...SSM_4B }).why;
    const r = planPrefillGemm({ kind: "q8_0", ...SSM_4B, M, idot: true });
    expect(r.via, r.reason).toBe("legacy");
    expect(r.reason).toContain(why);
    expect(r.reason, "la sede indicata non e' quella vera").toContain("prefillGemmShapeOk");
    expect(r.reason, "manda ancora a cercare la soglia nel kernel")
      .not.toContain("prefillGemmCheck");
  });

  it("il sensore vale: la STESSA shape con N grande non cade per la shape", () => {
    // Senza questa riga i casi precedenti sarebbero verdi anche se il piano
    // rendesse legacy per un motivo qualunque.
    const r = planPrefillGemm({ kind: "q2_K", K: 2048, N: 4096, M, idot: true });
    expect(r.reason, "N=4096 rifiutata dalla shape").not.toContain("workgroup");
    const wired = planPrefillGemm({ kind: "q8_0", K: 2048, N: 4096, M, idot: true });
    expect(wired.via, `q8_0 a N=4096: ${wired.reason}`).not.toBe("legacy");
  });
});

// ---------------------------------------------------------------------------
// [s4] (d) GATE STRUTTURALE: il predicato sulla shape si consuma in UN POSTO SOLO.
//
// Stessa postura del gate [w4] su `prefillGemmWiring`
// (tests/engine-prefillgemmplan-notwired.test.ts): non si elencano i siti buoni
// — lista che marcisce — si intercetta l'ATTO, cioe' «chiedere se una shape e'
// instradabile», e si pretende che avvenga solo dove e' dichiarato. Due sedi
// che decidono la stessa soglia e la decidono diversa e' esattamente il difetto
// di it.7 (righe-per-workgroup stabilite in due posti).
// ---------------------------------------------------------------------------
describe("[s4] `prefillGemmShapeOk` si legge in un posto solo", () => {
  const ROOT = join(__dirname, "..");
  const SRC = globSync(join(ROOT, "src/**/*.{ts,mts,cts,tsx}"))
    .filter((f) => !f.endsWith(".d.ts"))
    .map((f) => relative(ROOT, f))
    .sort();

  it("la scansione non e' vuota (senza questo guard un gate rotto e' verde)", () => {
    expect(SRC.length).toBeGreaterThan(30);
  });

  it("un solo file di produzione lo nomina: la sua sede, che e' anche il suo unico lettore", () => {
    const hits = SRC.filter((f) => /prefillGemmShapeOk/.test(readFileSync(join(ROOT, f), "utf8")));
    expect(hits, "il predicato sulla shape vive e si consuma dentro il piano")
      .toEqual(["src/engine/prefillgemmplan.ts"]);
  });

  it("dentro la sede si INVOCA due volte: la definizione e l'UNICA chiamata", () => {
    // Il gate [w4] puo' fermarsi ai file perche' li' sede e consumatore sono
    // due file diversi. Qui coincidono, quindi il conteggio per file non
    // distinguerebbe «un lettore» da «cinque»: si contano le INVOCAZIONI.
    //
    // Si contano `nome(` e non `nome`: il nome della sede compare anche dentro
    // la ragione che il piano manda in telemetria («prefillgemmplan.ts,
    // prefillGemmShapeOk»), ed e' giusto che ci compaia — e' cio' che dice a
    // chi legge un rifiuto DOVE sta la soglia. Un'occorrenza in una stringa non
    // e' un secondo predicato; una seconda chiamata si'.
    const src = readFileSync(join(ROOT, "src/engine/prefillgemmplan.ts"), "utf8");
    const code = src.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
    expect(code, "il blanking dei commenti ha mangiato la definizione")
      .toContain("export function prefillGemmShapeOk(");
    const n = (code.match(/prefillGemmShapeOk\s*\(/g) ?? []).length;
    expect(n, "invocazioni di `prefillGemmShapeOk` nel CODICE (commenti esclusi)").toBe(2);
  });

  it("la soglia su N non e' scritta una seconda volta nel piano", () => {
    // Il gate precedente sorveglia il NOME; questo sorveglia il NUMERO. Dopo
    // l'estrazione, `PREFILL_GEMM_ROWS_PER_WG` deve comparire nel CODICE del
    // piano solo dentro il predicato — un secondo confronto altrove sarebbe
    // proprio il difetto di it.7, tornato con un altro nome.
    const src = readFileSync(join(ROOT, "src/engine/prefillgemmplan.ts"), "utf8");
    const code = src.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
    const body = code.slice(code.indexOf("export function prefillGemmShapeOk("));
    const bodyEnd = body.indexOf("\nfunction kernelVerdict");
    const inside = (body.slice(0, bodyEnd).match(/PREFILL_GEMM_ROWS_PER_WG/g) ?? []).length;
    const total = (code.match(/PREFILL_GEMM_ROWS_PER_WG/g) ?? []).length;
    expect(bodyEnd, "il ritaglio del predicato non ha trovato `kernelVerdict`")
      .toBeGreaterThan(0);
    // l'import in testa al file + le occorrenze dentro il predicato, e nient'altro
    expect(total - inside, "la soglia su N e' confrontata anche fuori dal predicato").toBe(1);
  });
});

// ---------------------------------------------------------------------------
// [s5] (e) I DUE FORMATI RESTANO SPENTI.
// ---------------------------------------------------------------------------
describe("[s5] q2_K e q3_K NON sono cablati", () => {
  it("nessuno dei due e' instradabile in produzione", () => {
    // Lo stato vietato e' uno solo: presente e acceso senza la misura su GPU
    // che lo autorizzi. La spec li vuole `wired: false` finche' quella misura
    // non c'e', e questo task non la produce (nessuna esecuzione su GPU).
    for (const k of ["q2_K", "q3_K"] as const) {
      expect(prefillGemmWiring(k).wired,
        `${k} risulta CABLATO: nessuna misura su GPU lo autorizza`).toBe(false);
    }
  });

  it("...e nessun sito del piano ci passa, nemmeno su una shape ammessa", () => {
    // La verifica che conta: shape buona (quella degli expert del 35B), quindi
    // se il piano instradasse, instraderebbe qui.
    for (const shape of [EXPERT_GATE_UP, EXPERT_DOWN]) {
      for (const k of ["q2_K", "q3_K"] as const) {
        const r = planPrefillGemm({ kind: k, ...shape, M, idot: true });
        expect(r.via, `${k} ${shape.K}x${shape.N}: ${r.reason}`).toBe("legacy");
        expect(r.reason, `${k}: la ragione non accusa il cablaggio`).toContain("cablat");
        // e non deve essere la SHAPE ad averli fermati: questa shape e' buona,
        // e il predicato lo dice
        expect(prefillGemmShapeOk({ kind: k, ...shape }).ok).toBe(true);
        expect(r.reason, `${k}: la ragione accusa una shape che e' buona`)
          .not.toContain("workgroup");
      }
    }
  });
});
