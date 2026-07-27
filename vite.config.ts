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
      // conformance.html resta dev/test-only, fuori dal build come prima.
      input: ["index.html", "microbench.html"],
    },
  },
});
