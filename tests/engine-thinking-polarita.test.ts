// LA POLARITA' DEL RAGIONAMENTO, sui template VERI dei file.
//
// I due frammenti qui sotto sono copiati dai `tokenizer.chat_template` degli
// artefatti di chat (`results/chat/chat-{4b,9b,35b}-*.json`): non sono inventati
// per far passare il test. La polarita' e' INVERTITA fra le due generazioni, ed
// e' esattamente il tipo di dettaglio che si cabla per sbaglio.
import { describe, expect, it } from "vitest";
import { thinkingRendering } from "../src/engine/chat/thinking";

/** Qwen3.5 (4B, 9B): ragiona solo se richiesto ⇒ default SPENTO. */
const T35 = `{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\\n' }}
    {%- else %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}`;

/** Qwen3.6 (35B): NON ragiona solo se glielo si vieta ⇒ default ACCESO. */
const T36 = `{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}`;

describe("la modalita' di ragionamento si deriva dal template, non si cabla", () => {
  it("Qwen3.5: il default e' SPENTO (blocco gia' chiuso)", () => {
    const r = thinkingRendering(T35);
    expect(r.thinking).toBe(false);
    expect(r.prefix).toBe("<think>\n\n</think>\n\n");
    expect(r.source).toBe("template-default");
  });

  it("Qwen3.6: il default e' ACCESO — ed e' il difetto che il motore aveva", () => {
    const r = thinkingRendering(T36);
    expect(r.thinking).toBe(true);
    expect(r.prefix).toBe("<think>\n");
    expect(r.source).toBe("template-default");
  });

  it("le due polarita' sono OPPOSTE a parita' di default: e' il punto", () => {
    expect(thinkingRendering(T35).thinking).not.toBe(thinkingRendering(T36).thinking);
  });

  it("forzare la scelta funziona su ENTRAMBE, nonostante le condizioni invertite", () => {
    for (const t of [T35, T36]) {
      expect(thinkingRendering(t, true).prefix).toBe("<think>\n");
      expect(thinkingRendering(t, false).prefix).toBe("<think>\n\n</think>\n\n");
      expect(thinkingRendering(t, true).source).toBe("richiesto");
    }
  });

  it("un template senza enable_thinking non prevede il blocco, e non e' un errore", () => {
    const r = thinkingRendering("{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}");
    expect(r).toEqual({ prefix: "", thinking: false, source: "assente", condition: null });
    expect(thinkingRendering(null).source).toBe("assente");
  });

  it("una forma inattesa LANCIA invece di indovinare", () => {
    // nomina enable_thinking ma senza if/else: un prefisso sbagliato non darebbe
    // un errore, darebbe un modello che risponde peggio in silenzio
    expect(() => thinkingRendering("add_generation_prompt … enable_thinking …"))
      .toThrow(/non nella forma if\/else/);
    // guarda la variabile senza `is defined`: non sappiamo cosa faccia se manca
    expect(() => thinkingRendering(`add_generation_prompt
      {%- if enable_thinking %}{{- '<think>\\n' }}{%- else %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}`))
      .toThrow(/is defined/);
  });
});
