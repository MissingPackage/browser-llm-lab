import { defineConfig } from "vite";

const coopCoep = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
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
