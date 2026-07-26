export interface QualityPrompt {
  id: string;
  category: "arithmetic" | "factual" | "format" | "json";
  prompt: string;
  isCorrect: (response: string) => boolean;
}

function exact(pattern: RegExp): (response: string) => boolean {
  return (response) => pattern.test(response.trim());
}

function validJson(check: (value: unknown) => boolean): (response: string) => boolean {
  return (response) => {
    try {
      return check(JSON.parse(response.trim()));
    } catch {
      return false;
    }
  };
}

// 12 prompt deterministici, greedy, valutati via exact-match/regex — fallback quando
// l'adapter non espone logprobs (design: docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md).
export const QUALITY_PROMPTS: QualityPrompt[] = [
  { id: "arith-1", category: "arithmetic", prompt: "What is 12 + 7? Answer with only the number.", isCorrect: exact(/^19\.?$/) },
  { id: "arith-2", category: "arithmetic", prompt: "What is 9 * 6? Answer with only the number.", isCorrect: exact(/^54\.?$/) },
  { id: "arith-3", category: "arithmetic", prompt: "What is 100 - 37? Answer with only the number.", isCorrect: exact(/^63\.?$/) },

  { id: "factual-1", category: "factual", prompt: "What is the capital of France? Answer with only the city name.", isCorrect: exact(/^paris\.?$/i) },
  { id: "factual-2", category: "factual", prompt: "How many continents are there on Earth? Answer with only the number.", isCorrect: exact(/^7\.?$/) },
  { id: "factual-3", category: "factual", prompt: "What color do you get by mixing blue and yellow? Answer with only the color name.", isCorrect: exact(/^green\.?$/i) },

  { id: "format-1", category: "format", prompt: 'Repeat the word "banana" three times, separated by commas, nothing else.', isCorrect: exact(/^banana, banana, banana\.?$/i) },
  { id: "format-2", category: "format", prompt: "Write the numbers 1 to 5 separated by spaces, nothing else.", isCorrect: exact(/^1 2 3 4 5$/) },
  { id: "format-3", category: "format", prompt: 'Answer with exactly the single word "yes" or "no": Is the sky blue on a clear day?', isCorrect: exact(/^yes\.?$/i) },

  { id: "json-1", category: "json", prompt: 'Output valid JSON with exactly one key "answer" whose value is the number 4. Output only JSON, nothing else.', isCorrect: validJson((v) => typeof v === "object" && v !== null && (v as Record<string, unknown>).answer === 4) },
  { id: "json-2", category: "json", prompt: 'Output only this exact JSON, nothing else: {"ok": true}', isCorrect: validJson((v) => typeof v === "object" && v !== null && (v as Record<string, unknown>).ok === true) },
  { id: "json-3", category: "json", prompt: 'Output a JSON array containing the three strings "a", "b", "c", in that order — output only the array, nothing else.', isCorrect: validJson((v) => Array.isArray(v) && v.length === 3 && v[0] === "a" && v[1] === "b" && v[2] === "c") },
];
