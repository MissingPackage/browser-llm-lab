// GPUDevice finto per i test CPU-side della residenza.
//
// PERCHE' ESISTE. `ExpertCache` e' logica pura — riparto del budget, slot, LRU,
// slotTable, contatori — ma PARLA con un GPUDevice: senza un finto, l'unico
// posto dove la si puo' esercitare e' il ktest (browser + GPU), cioe' quasi
// mai. Qui `createBuffer` registra solo la taglia chiesta e `writeBuffer` solo
// quanti byte sono passati: abbastanza per verificare quanti buffer la cache
// crea, di che taglia, e a quali offset scrive.
//
// PERCHE' IN `helpers/` E NON DENTRO UN FILE DI TEST. Aveva un cliente solo
// (engine-residency); dal 2026-08-17 ne ha due (anche engine-arena-q2k, che
// costruisce una cache vera sul parco Q2_K/Q3_K). La seconda copia sarebbe
// partita da «tanto sono dieci righe» e sarebbe divergata alla prima estensione
// — ruling «riusa, non duplicare».

/** Una writeBuffer osservata: quale buffer, a che offset, quanti byte. */
export interface MockWrite { buf: object; offset: number; bytes: number }

export function mkMockDevice(): {
  device: GPUDevice; writes: MockWrite[]; buffers: Array<{ size: number }>;
} {
  const writes: MockWrite[] = [];
  const buffers: Array<{ size: number }> = [];
  const device = {
    createBuffer: (d: { size: number }) => {
      const b = { size: d.size, destroy() { /* mock */ } };
      buffers.push(b);
      return b;
    },
    queue: {
      writeBuffer: (buf: object, offset: number, data: ArrayBufferView) =>
        writes.push({ buf, offset, bytes: data.byteLength }),
    },
  } as unknown as GPUDevice;
  return { device, writes, buffers };
}
