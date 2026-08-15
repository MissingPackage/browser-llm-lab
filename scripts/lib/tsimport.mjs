// COME UNO SCRIPT `.mjs` DI QUESTO REPO CONSUMA UN MODULO `.ts`.
//
// Il problema. Gli script vivono in `scripts/*.mjs` e girano con `node`; i
// moduli puri del motore (`prefillbytes.ts`, `prefillgemmplan.ts`, …) sono
// TypeScript e si importano fra loro SENZA estensione — risoluzione "bundler",
// che e' quella di vite/vitest e che tsconfig.json dichiara. Node ha entrambe le
// meta' del problema risolte a meta':
//   - i TIPI li toglie da solo (type stripping, di default da Node 22.18; qui
//     gira 22.23) e tsconfig ha gia' `erasableSyntaxOnly: true`, quindi nessun
//     sorgente di `src/` usa costrutti che il type stripping rifiuta;
//   - l'ESTENSIONE no: `import "./kernels/wgsl"` da un `.ts` non risolve, e il
//     grafo di `src/engine` ha 78 import senza estensione.
//
// COSA FA QUESTO FILE: un hook di RISOLUZIONE (`module.registerHooks`, sincrono,
// API stabile di Node 22) che aggiunge `.ts` agli import relativi senza
// estensione, e solo quando il file `.ts` esiste davvero. Non trasforma niente e
// non tocca la cache dei moduli: la traduzione TS resta quella di Node.
//
// LE ALTERNATIVE, e perche' non sono state prese.
//   - `npx vite-node script.mjs`: e' il modo DICHIARATO da quattro script
//     (`q35-attn-fixture-gen.mjs`, `q35-mtp-fixture-gen.mjs`,
//     `q35-deltanet-fixture-gen.mjs`, `q35-looka-run.mjs`), ma `vite-node` non
//     e' in package.json ne' installato in node_modules: quella riga oggi
//     scarica un pacchetto dalla rete. Per uno script che produce un ARTEFATTO
//     DI RIFERIMENTO e' una dipendenza a runtime che non si vede nel lockfile.
//   - `createServer()` + `ssrLoadModule` di vite (installato): funziona e non
//     aggiunge dipendenze, ma monta un dev server per leggere due funzioni pure.
//   - aggiungere `.ts` a tutti gli import di `src/`: 78 punti in un albero che
//     serve al browser, per un consumatore fuori dal browser.
// Restava questa: ~20 righe, zero dipendenze nuove, e l'invocazione degli script
// non cambia (`node scripts/…`, che e' quello che sta scritto nei `repro` degli
// artefatti gia' pubblicati).
//
// LIMITE DICHIARATO: risolve solo `./` e `../`. Un modulo di `src/` che
// importasse un pacchetto npm con sorgenti TS non ci passerebbe — oggi non
// succede (i moduli puri del prefill non importano nulla di esterno) e se
// succedesse fallirebbe rumorosamente, non in silenzio.
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";

let installato = false;

/** Installa l'hook una volta sola (idempotente: due script possono chiamarlo). */
export function abilitaImportTs() {
  if (installato) return;
  installato = true;
  registerHooks({
    resolve(specifier, context, nextResolve) {
      if ((specifier.startsWith("./") || specifier.startsWith("../")) && !/\.[a-zA-Z0-9]+$/.test(specifier)) {
        const url = new URL(`${specifier}.ts`, context.parentURL);
        if (existsSync(fileURLToPath(url))) return { url: url.href, shortCircuit: true };
      }
      return nextResolve(specifier, context);
    },
  });
}

/**
 * Carica un modulo di `src/`|`tests/` per path RELATIVO ALLA RADICE del repo.
 * Il path parte dalla radice e non dallo script perche' cosi' e' leggibile nel
 * chiamante («src/engine/q35prefillsites.ts») ed e' lo stesso path che compare
 * nei commenti e nei report.
 */
export async function importaTs(pathDallaRadice) {
  abilitaImportTs();
  return import(new URL(`../../${pathDallaRadice}`, import.meta.url).href);
}
