GOAL: Extend browser-llm-lab with the "1b — matrice" harness — Transformers.js and wllama inference adapters, a quality-scoring module, and schema v3 — verified end-to-end on the existing 4090 rig, so the codebase is ready for the (separately scheduled) manual multi-device sweep.

**STATUS NOTE (2026-07-27, Fase 4 done)**: every line under DONE WHEN below now checks out
mechanically (see `.harness/goals/fase-1b-matrice/journal.md`, last entry, for the evidence run).
**Correction (same day, caught by `loop-verifier` before merge)**: the first version of this note
claimed this without checking one line literally — "each a valid schema-v3 JSON" for the new-adapter
real runs. The existing `results/*.json` runs for `transformersjs`/`wllama` predated the v3 bump
(`schemaVersion: 2`). Fixed by running the existing e2e driver headed on the 4090 (already-granted
authority, no new ruling needed) for both stacks: `results/4090-linux-2026-07-26T23-56-34-978Z.json`
(transformersjs, schemaVersion 3) and `results/4090-linux-2026-07-26T23-57-10-621Z.json` (wllama,
schemaVersion 3) — both carry `BenchCell.protocol` correctly. The line now genuinely checks out.

Not calling this goal DONE unilaterally — that's a PI call, not something a phase gate decides —
but flagging it here so it isn't missed: the only things left touching this goal are docket-gated
(docket #10: whether/when to wire `qualityScore` into real bench runs; docket #8: self-monitoring,
no action needed until the routine fires) or explicitly out of this goal's scope (the manual
M4/S22 sweep, "must docket" above). Nothing autonomously decidable remains under PHASES.md.

DONE WHEN (all measurable):
- `src/adapters/transformersjs.ts` and `src/adapters/wllama.ts` exist, implement `InferenceAdapter`, and each passes a conformance test (load a model, generate deterministic output, `capabilities()` matches real behavior) — **[EMENDATO 2026-07-26, ruling PI docket #2]** verifiable via `npm run test:conformance`, a Playwright script that exercises the same contract against every stack **in a real browser**. The original wording said `npm test`; that is not realizable for WebLLM, which requires WebGPU and cannot run in Node. `npm test` remains the fast, offline unit suite.
- `src/quality.ts` + `src/qualityPrompts.ts` implemented, with unit tests covering both the perplexity path and the 12-prompt exact-match fallback — verifiable via `npm test`.
- Schema bumped to v3: `BenchCell.stack` union extended to `"webllm" | "transformersjs" | "wllama"`, `BenchCell.qualityScore` field added per the design doc's discriminated union — verifiable via `tsc --noEmit` clean + schema-level tests.
- UI stack selector added, filtering the model dropdown to combinations that actually exist (excludes the documented Large-tier gap) — verified manually via a playwright smoke pass.
- `npm run build` succeeds.
- At least one real (non-SwiftShader) run per new adapter recorded under `results/` on the 4090, each a valid schema-v3 JSON — verifiable via `ls results/*.json` + schema validation, using the existing e2e driver (`scripts/e2e-bench.mjs`).
- README gains a "Fase 1b — matrice" section documenting the Large-tier gap finding (no ONNX/wllama path for 7B/8B).
- Full existing suite still green (39/39 baseline before this work) — no regression to the WebLLM adapter.

EVIDENCE OF DONE: `npm test` output (N/N passing), `tsc --noEmit` clean, `npm run build` exit 0, `ls results/*.json` showing new `stack: "transformersjs"` / `stack: "wllama"` entries, `git log`/README diff showing the new section.

AUTHORITY GRANTED:
- may do autonomously: work on a feature branch `feat/fase-1b-matrice`, install the two new npm deps (`@huggingface/transformers`, `wllama`), write/edit code and tests, run `npm test`/`tsc`/`npm run build`, run the existing e2e driver headed on the local 4090, log open questions to the docket.
- must docket (never do): merge to `main`, push to `origin`, delete existing branches/results, change the public `InferenceAdapter` contract, touch `docs/superpowers/` policy, execute or simulate the M4/S22 sweep (physically out of reach — separate goal/manual step), drop model quant below what's specced to force the Large tier to fit.

CONSTRAINTS: no AI attribution in commits or PRs; schema v3 implemented exactly as specced (no ad hoc field additions beyond `qualityScore` and `protocol` — the latter added by ruling, docket #7, to register warm-up policy/replica count explicitly instead of abusing `anomalies`); Large-tier gap stays documented, never silently "fixed"; existing WebLLM adapter/tests must keep passing throughout; `erasableSyntaxOnly` tsconfig — no parameter properties in classes (known landmine, see HANDOFF.md §5).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle; digest every cycle; stop-by-design when remaining work is docket-gated (merge decision, or the manual device sweep).

CONTEXT ANCHORS: HANDOFF.md; `docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md` (this goal's spec); `docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md` §Fasatura; `src/adapters/types.ts` + `src/adapters/webllm.ts` (reference adapter); `src/schema.ts`; `src/metrics.ts`.
