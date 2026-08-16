// La lettura a Range del GGUF sta in UN posto solo
// (goal engine-velocita-decode, riga 2b).
//
// PERCHE' ESISTE. Il contratto della riga dice che il raggruppamento delle
// richieste va messo nel path di I/O **condiviso** e non nei call-site, «altrimenti
// e' la piccolezza specifica che il ruling vieta». Il path condiviso non
// esisteva: la stessa funzione da sette righe stava in CINQUE posti, identica a
// meno del prefisso d'errore. Questo caso impedisce alla sesta di nascere —
// e con lei alla finestra di concorrenza di dover essere scritta sei volte.
//
// La copia numero cinque aveva gia' divergito: `q35-moe-block-real` non
// controllava la lunghezza della risposta, quindi un Range corto sarebbe
// diventato pesi troncati in silenzio. E' la ragione per cui il conteggio qui
// sotto e' un gate e non una nota: le copie non restano identiche.
import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { ggufRangeReader } from "../src/engine/ggufrange";

const SRC = new URL("../src", import.meta.url).pathname;
const HOME = "src/engine/ggufrange.ts";

const walk = (dir: string): string[] =>
  readdirSync(dir).flatMap((e) => {
    const p = join(dir, e);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith(".ts") ? [p] : [];
  });

describe("ggufRange: una sede sola per la lettura a Range", () => {
  it("l'header `Range: bytes=` si costruisce in UN file, e e' il modulo condiviso", () => {
    const hits = walk(SRC)
      .filter((p) => /Range: `bytes=/.test(readFileSync(p, "utf8")))
      .map((p) => p.slice(p.indexOf("src/")));
    expect(hits, `chi costruisce l'header: ${hits.join(", ") || "nessuno"}`).toEqual([HOME]);
  });

  it("un Range corto LANCIA — e' il controllo che una delle cinque copie aveva perso", async () => {
    const g = globalThis as unknown as { fetch: unknown };
    const orig = g.fetch;
    g.fetch = async () => ({ status: 206, arrayBuffer: async () => new ArrayBuffer(8) });
    try {
      const r = ggufRangeReader(() => "/x.gguf", "prova");
      await expect(r(0, 16)).rejects.toThrow(/prova: Range corto 8\/16/);
    } finally { g.fetch = orig; }
  });

  it("uno status diverso da 206 LANCIA col nome del lettore", async () => {
    const g = globalThis as unknown as { fetch: unknown };
    const orig = g.fetch;
    g.fetch = async () => ({ status: 200, arrayBuffer: async () => new ArrayBuffer(16) });
    try {
      const r = ggufRangeReader(() => "/x.gguf", "prova");
      // 200 invece di 206 = il server ha ignorato la Range e sta mandando TUTTO
      // il file: leggerlo come se fosse la fetta chiesta darebbe pesi sbagliati
      await expect(r(0, 16)).rejects.toThrow(/prova: Range non onorato \(200\)/);
    } finally { g.fetch = orig; }
  });

  it("l'URL si rilegge a ogni chiamata: i worker lo riassegnano cambiando modello", async () => {
    const g = globalThis as unknown as { fetch: unknown };
    const orig = g.fetch;
    const visti: string[] = [];
    g.fetch = async (u: string) => {
      visti.push(u);
      return { status: 206, arrayBuffer: async () => new ArrayBuffer(4) };
    };
    try {
      let url = "/a.gguf";
      const r = ggufRangeReader(() => url, "prova");
      await r(0, 4);
      url = "/b.gguf";
      await r(0, 4);
      // se il lettore avesse CATTURATO l'URL alla costruzione, il secondo
      // modello verrebbe letto dal file del primo: byte validi, modello sbagliato
      expect(visti).toEqual(["/a.gguf", "/b.gguf"]);
    } finally { g.fetch = orig; }
  });

  it("l'header chiede l'intervallo INCLUSIVO: off .. off+len-1", async () => {
    const g = globalThis as unknown as { fetch: unknown };
    const orig = g.fetch;
    let hdr = "";
    g.fetch = async (_u: string, o: { headers: Record<string, string> }) => {
      hdr = o.headers.Range;
      return { status: 206, arrayBuffer: async () => new ArrayBuffer(100) };
    };
    try {
      await ggufRangeReader(() => "/x", "prova")(1000, 100);
      // un fuori-di-uno qui leggerebbe un byte in piu' o in meno per OGNI
      // tensore del modello
      expect(hdr).toBe("bytes=1000-1099");
    } finally { g.fetch = orig; }
  });
});
