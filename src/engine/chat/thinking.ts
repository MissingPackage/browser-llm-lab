// LA MODALITA' DI RAGIONAMENTO, DERIVATA DAL TEMPLATE DEL FILE.
//
// IL DIFETTO CHE QUESTO MODULO CHIUDE. Il motore non ha un interprete Jinja e
// rendeva il prompt fermandosi a `<|im_start|>assistant\n`. Il template del GGUF
// pero' non finisce li': dopo quella marca emette un PREFISSO che decide se il
// modello ragiona o no. Non renderlo significa consegnare al modello una
// modalita' che non e' la sua — e sul 35B la modalita' soppressa era quella di
// DEFAULT, cioe' la piu' capace.
//
// LA POLARITA' E' INVERTITA FRA LE DUE GENERAZIONI, verificato sui
// `tokenizer.chat_template` dei file veri (artefatti di chat del 2026-08-15):
//
//   Qwen3.5 (4B, 9B)   if enable_thinking is defined and enable_thinking is TRUE
//                          -> '<think>\n'                       (ragiona)
//                      else -> '<think>\n\n</think>\n\n'        <- DEFAULT: spento
//
//   Qwen3.6 (35B)      if enable_thinking is defined and enable_thinking is FALSE
//                          -> '<think>\n\n</think>\n\n'
//                      else -> '<think>\n'                      <- DEFAULT: acceso
//
// PERCHE' SI DERIVA E NON SI CABLA. Scrivere «i Qwen3.6 ragionano di default»
// sarebbe vero per i tre file che abbiamo in casa e falso al prossimo: e' la
// stessa forma d'errore gia' pagata due volte in questo repo (il confine dei
// layer q6_K, la lista dei formati expert). La regola generale invece regge:
// entrambi i template guardano con `is defined`, quindi con la variabile NON
// definita — che e' il nostro caso, perche' non passiamo variabili — la
// condizione e' falsa e **il default e' sempre il ramo `else`**.
//
// SE UN TEMPLATE NON RISPETTA QUELLA FORMA, questa funzione LANCIA invece di
// tirare a indovinare: un prefisso sbagliato non da' un errore, da' un modello
// che risponde peggio senza che nessuno sappia perche'.

/** Cosa va emesso dopo `<|im_start|>assistant\n`, e perche'. */
export interface ThinkingRendering {
  /** il prefisso da concatenare; "" se il template non ne prevede uno */
  prefix: string;
  /** `true` = il modello ragiona (blocco `<think>` aperto e da riempire) */
  thinking: boolean;
  /** come ci siamo arrivati, per l'artefatto: "default del template" o "richiesto" */
  source: "template-default" | "richiesto" | "assente";
  /** la condizione trovata nel template, testuale — la prova di cosa abbiamo letto */
  condition: string | null;
}

const OPEN = "<think>\\n";
const CLOSED = "<think>\\n\\n</think>\\n\\n";

/** srotola gli escape di una stringa letterale Jinja (`\n` -> a capo). */
const unescape = (s: string): string => s.replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\'/g, "'");

/**
 * Deriva il rendering del blocco di ragionamento dal `tokenizer.chat_template`.
 *
 * `enableThinking` forza la scelta (come farebbe la variabile Jinja); lasciarlo
 * `undefined` — il caso normale — significa «prendi il default del file».
 */
export function thinkingRendering(
  chatTemplateRaw: string | null, enableThinking?: boolean,
): ThinkingRendering {
  if (!chatTemplateRaw || !chatTemplateRaw.includes("enable_thinking")) {
    // un template senza la variabile non prevede il blocco: non e' un errore,
    // e' un modello che non ragiona (o un file senza template)
    return { prefix: "", thinking: false, source: "assente", condition: null };
  }
  const gen = chatTemplateRaw.slice(chatTemplateRaw.indexOf("add_generation_prompt"));
  const m = /\{%-?\s*if\s+([^%]*?enable_thinking[^%]*?)\s*-?%\}([\s\S]*?)\{%-?\s*else\s*-?%\}([\s\S]*?)\{%-?\s*endif/.exec(gen);
  if (!m) {
    throw new Error(
      "thinking: il chat_template nomina enable_thinking ma non nella forma if/else attesa. "
      + "Il prefisso del blocco <think> decide la modalita' del modello: si dichiara, non si indovina");
  }
  const [, condizione, ramoIf, ramoElse] = m;
  if (!/is\s+defined/.test(condizione)) {
    throw new Error(
      `thinking: la condizione "${condizione.trim()}" non guarda con \`is defined\`, quindi non so `
      + "cosa faccia il template quando la variabile manca — che e' il nostro caso");
  }
  const testo = (ramo: string): string => {
    const s = /\{\{-?\s*'([^']*)'\s*-?\}\}/.exec(ramo);
    return s ? s[1] : "";
  };
  // il default e' l'ELSE: la variabile non e' definita, quindi `is defined` e' falso
  const perDefault = testo(ramoElse);
  const perCondizione = testo(ramoIf);
  const scelto = enableThinking === undefined
    ? perDefault
    // se il chiamante forza, si prende il ramo che corrisponde: la condizione
    // dice `is true` oppure `is false`, e da li' si sa quale ramo e' quale
    : (/is\s+false/.test(condizione) ? (enableThinking ? perDefault : perCondizione)
      : (enableThinking ? perCondizione : perDefault));
  if (scelto !== OPEN && scelto !== CLOSED) {
    throw new Error(`thinking: prefisso inatteso nel template: ${JSON.stringify(scelto)}`);
  }
  return {
    prefix: unescape(scelto),
    thinking: scelto === OPEN,
    source: enableThinking === undefined ? "template-default" : "richiesto",
    condition: condizione.trim(),
  };
}
