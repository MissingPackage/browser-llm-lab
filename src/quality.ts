import { QUALITY_PROMPTS } from "./qualityPrompts";

export type QualityScore =
  | { kind: "perplexity"; value: number }
  | { kind: "exact-match"; value: number; total: number };

const EXACT_MATCH_TOTAL = QUALITY_PROMPTS.length;

// capabilities().logprobs decide il percorso (design: perplexity se disponibile,
// altrimenti fallback ai 12 prompt exact-match).
export function selectQualityMethod(caps: { logprobs: boolean }): "perplexity" | "exact-match" {
  return caps.logprobs ? "perplexity" : "exact-match";
}

// Perplexity su un passaggio di testo fisso: exp della log-probabilità media negata,
// per-token, sui logprobs del percorso greedy generato dal modello.
export function computePerplexity(tokenLogProbs: number[]): QualityScore {
  if (tokenLogProbs.length === 0) throw new Error("computePerplexity: empty logprobs array");
  const meanLogProb = tokenLogProbs.reduce((a, b) => a + b, 0) / tokenLogProbs.length;
  return { kind: "perplexity", value: Math.exp(-meanLogProb) };
}

export interface QualityPromptResponse {
  promptId: string;
  response: string;
}

// Valuta le risposte greedy contro QUALITY_PROMPTS. Nessuna soglia pass/fail: il punteggio
// grezzo (n/12) è riportato così com'è, la lettura comparativa resta a chi guarda i risultati.
export function evaluateExactMatch(responses: QualityPromptResponse[]): QualityScore {
  const correct = QUALITY_PROMPTS.filter((p) => {
    const r = responses.find((x) => x.promptId === p.id);
    return r !== undefined && p.isCorrect(r.response);
  }).length;
  return { kind: "exact-match", value: correct, total: EXACT_MATCH_TOTAL };
}
