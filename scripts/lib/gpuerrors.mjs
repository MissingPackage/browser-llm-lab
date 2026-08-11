// Sentinella degli errori GPU per i runner (docket item 24, it.43).
//
// PERCHÉ ESISTE. I runner registravano `page.on("pageerror")` solo per
// STAMPARE, e la condizione d'uscita guardava lo `#status` della pagina. Ma la
// pagina dice "done" perché il WORKER ha postato il report — e il worker posta
// il report anche se nel frattempo la GPU ha invalidato i buffer. Risultato:
// exit 0, JSON scritto, numeri PLAUSIBILI.
//
// Visto dal vivo in it.43: il 35B a tier 16 è andato in
// VK_ERROR_OUT_OF_DEVICE_MEMORY, ha proseguito con `Invalid BindGroup` a
// cascata, ed è uscito 0 scrivendo decode 4,80 tok/s e 7.657 miss — numeri che
// continuavano PERFETTAMENTE la tendenza delle celle sane (miss 23.612 → 9.908
// → 7.657). Nessuno, leggendo quel JSON, avrebbe sospettato niente. È la
// landmine di it.14 nella sua forma peggiore.
//
// Perché conta più che in astratto: il PRODOTTO di queste fasi sono i JSON di
// riferimento. Uno silenziosamente invalido si propaga in `direction §7-bis` e
// nel paper, e a quel punto non è più recuperabile.
//
// Regola di casa già scritta (it.38, `readTap`): uno strumento che tace quando
// dovrebbe urlare è peggio che non averlo.

// Le firme che Dawn/Chrome emettono quando un buffer nasce o diventa invalido.
// Deliberatamente LARGA: un falso positivo costa una run ripetuta, un falso
// negativo costa un riferimento sbagliato pubblicato.
const GPU_RE =
  /gpu error|vkAllocateMemory|OUT_OF_DEVICE_MEMORY|Invalid BindGroup|Invalid CommandBuffer|Invalid Buffer|is invalid due to|validation error|WGSL/i;

/**
 * Attacca i listener e restituisce il collettore. Sostituisce il
 * `page.on("pageerror", console.log)` che ogni runner si scriveva da sé.
 */
export function watchGpuErrors(page, label) {
  const errors = [];
  const push = (src, text) => {
    if (errors.length < 50) errors.push({ src, text: String(text).slice(0, 400) });
  };
  page.on("pageerror", (e) => {
    const t = e?.message ?? String(e);
    console.log(`[${label}][pageerror]`, t.slice(0, 300));
    if (GPU_RE.test(t)) push("pageerror", t);
  });
  page.on("console", (m) => {
    const t = m.text();
    if (GPU_RE.test(t)) {
      console.log(`[${label}][console]`, t.slice(0, 300));
      push("console", t);
    }
  });
  return {
    errors,
    /** La run è CONTAMINATA: il report non è una misura, qualunque cosa dica. */
    get dirty() {
      return errors.length > 0;
    },
  };
}

/**
 * Dove scrivere il report di una run contaminata: MAI al percorso nominale.
 * Un file che si chiama come un riferimento, dentro `results/engine/`, prima o
 * poi viene raccolto da un glob e diventa un numero pubblicato.
 */
export function invalidPath(out) {
  return `${out}.INVALID`;
}
