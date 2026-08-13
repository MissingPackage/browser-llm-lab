import { defineConfig } from "vite";

const coopCoep = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  // I workflow dell'harness (sdd-conductor) lavorano su COPIE del repo dentro
  // `.claude/worktrees/`. Vitest non le esclude di default, quindi ogni test
  // girava DUE volte — una sulla copia, con file vecchi — e una build appena
  // integrata risultava rossa per asserzioni che nell'albero vero erano gia'
  // state aggiornate (visto in it.4: 6 fallimenti, di cui 3 fantasmi).
  // Un gate che misura una copia stantia non e' un gate.
  test: { exclude: ["**/node_modules/**", "**/dist/**", ".claude/**", ".harness/worktrees/**"] },
  server: { headers: coopCoep },
  preview: { headers: coopCoep },
  build: {
    rollupOptions: {
      // microbench.html è una seconda entry di produzione (deep-dive fase 2);
      // prof.html è l'entry del dispatch profiler (docket #11, run manuali M4/S22);
      // chat.html è il banco di prova interattivo del motore q35 (l'unico posto
      // dove il motore gira NON teacher-forced), quindi va nel build;
      // conformance.html resta dev/test-only, fuori dal build come prima.
      input: ["index.html", "microbench.html", "prof.html", "chat.html"],
    },
  },
});
