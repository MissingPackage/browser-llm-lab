// Il predicato su N del piano di prefill (goal engine-velocita-decode, riga 2d).
//
// PERCHE' ESISTE. La forma split-K produce `PREFILL_GEMM_ROWS_PER_WG` righe di
// uscita per workgroup. Con N piu' piccolo il kernel resta CORRETTO — guarda
// `r < N` — ma il dispatch lavora a meno di mezzo workgroup e la forma perde
// contro la legacy, dove non e' mai stata misurata.
//
// COSA PROTEGGE, e non e' teorico: i 48 siti `ssm_alpha`/`ssm_beta` del 4B
// hanno **N=32**. Erano esclusi dal flag `wired` del q8_0, cioe' **per
// famiglia** — e quel flag e' esattamente cio' che il cablaggio del q8_0 doveva
// girare. **Girato il 2026-08-15**: da quel momento questo predicato e' l'UNICA
// cosa che tiene quei 48 siti sulla legacy. Se cadesse, il 4B cambierebbe cio'
// che esegue, in peggio, senza che nessun errore lo dica. Sul 35B gli stessi
// tensori di attn hanno N=4096 e passano.
//
// NOTA STORICA, e vale come metodo: quando questo file e' stato scritto (it.14)
// il q8_0 NON era ancora cablato, quindi il predicato era **irraggiungibile**
// per lui — il controllo del cablaggio veniva prima. Un guard che nessun test
// attraversa e' un guard di cui non sai se funziona (la classe del contatore
// mai incrementato). Per questo i casi girano su OGNI kind cablato invece che
// sul solo caso d'interesse: cosi' il predicato era esercitato gia' il giorno
// prima di servire davvero.
import { describe, expect, it } from "vitest";
import { planPrefillGemm } from "../src/engine/prefillgemmplan";
import {
  PREFILL_GEMM_ROWS_PER_WG, PREFILL_GEMM_WIRED_KINDS, PREFILL_GEMM_KINDS,
} from "../src/engine/kernels/wgsl";

const M = 16;
// K valido per tutti i kind cablati (multiplo di 64 e di 256)
const K = 2048;

describe("piano di prefill — N sotto le righe per workgroup non si instrada", () => {
  it("[1] la costante viene dalla griglia, non da una copia", () => {
    // `prefillGemmGrid` lancia `ceil(N / PREFILL_GEMM_ROWS_PER_WG)` workgroup:
    // se questo numero e il divisore della griglia divergessero, il predicato
    // proteggerebbe una soglia diversa da quella vera.
    expect(PREFILL_GEMM_ROWS_PER_WG).toBe(64);
  });

  it("[2] su OGNI kind cablato, N sotto la soglia cade sulla legacy con la sua ragione", () => {
    // il caso che oggi il q8_0 non puo' esercitare: qui il controllo del
    // cablaggio passa, quindi il predicato su N e' RAGGIUNTO
    expect(PREFILL_GEMM_WIRED_KINDS.length, "nessun kind cablato: il test non prova niente")
      .toBeGreaterThan(0);
    for (const kind of PREFILL_GEMM_WIRED_KINDS) {
      for (const idot of [true, false]) {
        const r = planPrefillGemm({ kind, K, N: 32, M, idot });
        expect(r.via, `${kind} idot=${idot}: N=32 instradato`).toBe("legacy");
        expect(r.reason, `${kind}: la ragione non nomina N`).toContain("N=32");
        expect(r.reason).toContain("workgroup");
        // niente scratch prenotato per un dispatch che nessuno emette
        expect(r.splits).toBe(0);
        expect(r.partialFloats).toBe(0);
      }
    }
  });

  it("[3] il confine e' esattamente la soglia: 63 no, 64 sì", () => {
    const kind = PREFILL_GEMM_WIRED_KINDS[0];
    const below = planPrefillGemm({ kind, K, N: PREFILL_GEMM_ROWS_PER_WG - 1, M, idot: true });
    const at = planPrefillGemm({ kind, K, N: PREFILL_GEMM_ROWS_PER_WG, M, idot: true });
    expect(below.via, `${kind}: N=63 doveva cadere`).toBe("legacy");
    expect(below.reason).toContain("N=63");
    expect(at.via, `${kind}: N=64 doveva passare`).not.toBe("legacy");
  });

  it("[4] il sensore vale: la STESSA shape con N grande viene instradata", () => {
    // senza questa riga, un piano che rendesse legacy per un motivo qualunque
    // farebbe passare il caso [2] a vuoto
    const kind = PREFILL_GEMM_WIRED_KINDS[0];
    const r = planPrefillGemm({ kind, K, N: 4096, M, idot: true });
    expect(r.via, `${kind} a N=4096: ${r.reason}`).not.toBe("legacy");
  });

  it("[5] le shape VERE del 4B e del 35B cadono dalle parti giuste", () => {
    // il caso concreto per cui il predicato esiste, con le shape dell'header
    // dump invece che con numeri inventati
    const kind = PREFILL_GEMM_WIRED_KINDS[0];
    // 4B `ssm_alpha`/`ssm_beta`: N=32 — mezzo workgroup, resta legacy
    expect(planPrefillGemm({ kind, K: 2560, N: 32, M, idot: true }).via).toBe("legacy");
    // 35B attn: N=4096 — deve passare quando il suo formato sara' cablato
    expect(planPrefillGemm({ kind, K: 2048, N: 4096, M, idot: true }).via).not.toBe("legacy");
  });

  it("[6] l'ordine dei rifiuti: prima il cablaggio, poi la shape", () => {
    // su un kind NON cablato con N piccolo la ragione strutturale e' il
    // cablaggio — la shape sarebbe una risposta vera e inutile, perche' anche
    // con N buono quel sito resterebbe legacy. E' lo stesso ordine che il
    // kernel tiene fra formato e geometria.
    //
    // L'ESEMPIO NON E' PIU' IL q8_0: era il kind non cablato di ieri, ed e'
    // stato cablato oggi (riga 2d). Si prende un non-cablato dall'elenco vero
    // invece di scriverne uno a mano, cosi' il caso non marcisce alla prossima
    // accensione.
    const notWired = PREFILL_GEMM_KINDS.filter((k) => !PREFILL_GEMM_WIRED_KINDS.includes(k));
    expect(notWired.length, "nessun kind non cablato: il caso non prova piu' niente")
      .toBeGreaterThan(0);
    const r = planPrefillGemm({ kind: notWired[0], K: 2048, N: 32, M, idot: true });
    expect(r.via).toBe("legacy");
    expect(r.reason, "la ragione deve accusare il cablaggio, non N").toContain("cablato");
  });
});
