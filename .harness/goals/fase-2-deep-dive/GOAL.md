GOAL: Produce the Fase-2 deep-dive of the WebLLM/MLC WebGPU path — six documents under
docs/deep-dive/, a reusable bottleneck-brainstorm skill, and a cross-device matmul
micro-bench with real 4090 numbers — so that the public benchmark page has its explanatory
material and Cristiano has the groundwork for a future custom browser inference engine.

<!-- Contract approved by PI 2026-07-27 (chat, after goal-brief with all ASSUMED resolved).
     Spec: docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md.
     Goal start tag: goal-fase-2-start (main @ c78150d). -->

DONE WHEN (all measurable):
- The four subsystem docs exist (`docs/deep-dive/compute-shader-dispatch.md`,
  `buffer-limit-2gb.md`, `dequant-kernels.md`, `kv-cache-layout.md`) and each contains the
  contract sections from the spec skeleton as literal headings — "Cosa fa", "Perché i
  numeri sono quelli", "Bottleneck & vie d'uscita", and (only where an experiment ran)
  "Esperimenti" — verifiable via `grep` of the section headings per file.
- Every claim in the subsystem docs carries a citation (code line, `results/*.json` run, or
  upstream source) or an explicit `[VERIFY]` marker; the goal cannot close with unresolved
  `[VERIFY]` markers — verifiable via `grep -c "\[VERIFY\]" docs/deep-dive/*.md` returning
  0 at close, plus verifier spot-check of ≥3 citations per doc against their targets.
- Skill `bottleneck-brainstorm` exists at `.claude/skills/bottleneck-brainstorm/SKILL.md`
  (project-level — promotion to the personal harness is an end-of-goal PI decision, see
  must-docket) and its invocation is recorded in the journal for all four subsystems
  (dogfooding requirement).
- Micro-bench matmul: a new route in the existing SPA measuring dequant-q4 vs f16 matmul at
  multiple buffer sizes, using WebGPU `timestamp-query` where available with CPU-side
  fallback; `npm run build` exit 0; unit tests on the metric math pass via `npm test`; at
  least one real (non-SwiftShader) 4090 run exported as versioned-schema JSON under
  `results/microbench/`.
- `docs/deep-dive/micro-bench-matmul.md` exists, documents harness + methodology, and
  embeds the 4090 numbers; M4/S22 slots present and explicitly marked as pending manual
  runs (holes logged, never silent).
- `docs/deep-dive/engine-design-notes.md` exists and links every other deep-dive doc
  (verifiable via grep of the five filenames in its body).
- 1–2 feasibility experiments total, each living under `experiments/<nome>/` with a README
  stating what it demonstrates and what it does not; a third experiment requires a PI
  ruling (docket).
- Full existing suite still green: `npm test`, `tsc --noEmit`, `npm run build` — no
  regression to the bench harness; `src/adapters/webllm.ts` diff-clean vs goal start.

EVIDENCE OF DONE: `ls docs/deep-dive/` (6 files); heading-grep per subsystem doc;
`grep -c "\[VERIFY\]"` = 0; `ls .claude/skills/bottleneck-brainstorm/`; journal entries for
4 skill invocations; `npm test` N/N, `tsc --noEmit` clean, `npm run build` exit 0;
`ls results/microbench/*.json` + schema validation; `git diff goal-fase-2-start --
src/adapters/webllm.ts` empty; `ls experiments/` showing ≤2 entries.

AUTHORITY GRANTED:
- may do autonomously: work on the single feature branch `feat/fase-2-deep-dive`
  (sequential work, per PI ruling); create/edit files under `docs/deep-dive/`,
  `experiments/`, `.claude/skills/bottleneck-brainstorm/`, and the new SPA route + its
  tests; read the web-llm bundle and upstream MLC/TVM sources (context7/repo fetch); run
  `npm test`/`tsc`/`npm run build` and headed micro-bench runs on the local 4090; intercept
  `createShaderModule` in a local instrumented run to dump generated WGSL (local
  experiment, touches no committed adapter code); log open questions to the docket.
- must docket (never do): merge to `main`, push to `origin`; run or simulate M4/S22
  micro-bench (manual, Cristiano's hands); integrate any experiment into the production
  SPA or adapters; modify `src/adapters/webllm.ts` or the `InferenceAdapter` contract;
  start a third experiment; begin actual design of the custom engine (engine-design-notes
  prepares the ground, the design itself is out of phase); promote the skill to the
  personal harness (~/.claude) — end-of-goal PI decision; touch docket #10/#12/#8 work
  (they stay independent of this goal).

CONSTRAINTS: no AI attribution in commits or PRs; deep-dive docs written to be publishable
as-is except `engine-design-notes.md` (explicitly personal, per PI ruling 2026-07-27);
micro-bench JSON follows the results/ conventions — versioned schema, manual device label,
no fingerprinting; `erasableSyntaxOnly` tsconfig landmine still applies to any new SPA
code (HANDOFF.md §5); bundle line numbers cited in docs must carry the package version
(they shift on update); docs cite current upstream docs (context7), never memory.

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle; digest
every cycle; stop-by-design when the remaining work is docket-gated (merge decision, manual
M4/S22 micro-bench runs, third-experiment ruling, skill-promotion decision).

CONTEXT ANCHORS: HANDOFF.md; docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md
(this goal's spec); docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md
(spec madre, §Fasatura); results/*.json (the numbers being explained, esp. the 4090 and
S22 runs); node_modules/@mlc-ai/web-llm/lib/index.js (bundled tvmjs runtime,
WebGPUContext ~line 4359 at current version); .harness/goals/fase-1b-matrice/docket.md
(inherited items #8/#10/#12 — context, not workload).
