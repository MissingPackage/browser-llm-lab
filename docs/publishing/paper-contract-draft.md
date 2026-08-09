# Contratto paper — BOZZA PARCHEGGIATA (si charterizza alla release)

Ruling PI 2026-08-09 (chat): il paper è di ACCOMPAGNAMENTO al rilascio del
motore — si scrive quando esistono la versione finale, i numeri e i modelli
supportati (= a valle del goal di generalizzazione q1). Target: **Zenodo,
preprint** (niente double-blind ⇒ niente anonimizzazione). Punto aperto,
non bloccante: il BLOG del research preview può uscire prima del paper o
insieme — decisione PI a tempo debito.

Cosa è GIÀ in banca per il paper (non si rifà):
- baseline nativa llama.cpp Vulkan stesso hardware
  (results/engine/native-baseline-llamacpp-vulkan-2026-08-09.json);
- tutti gli artefatti di fase C (journal c3a/b/c = la storia dei numeri);
- il piano di split (docs/publishing/split-plan.md): il repo
  browser-llm-paper nascerà da paper/** via filter-repo.

Il contratto (da aggiornare coi numeri q1 quando si charterizza):

GOAL: engine-writeup — Il preprint Zenodo + blog di accompagnamento al
rilascio esistono come artefatti completi e verificabili (ogni numero
tracciato a un artefatto committato via claims-map), pronti per il ruling
di release; la pubblicazione resta fuori dal goal.

DONE WHEN: paper/ con LaTeX che builda (exit 0); contributi coperti coi
numeri FINALI (fase C + generalizzazione q1: modelli supportati, baseline
consumer); claims-map meccanica (numero → JSON+campo, spot-check verifier);
Limitations esplicite (gap nativo come limite inferiore, hardware validati,
margini); Related Work con sweep aggiornato; statement riproducibilità
(commit/tag/SHA/protocolli); blog draft (humanizer); figure rigenerabili da
script sui JSON (dataviz); NESSUNA azione pubblica (Zenodo upload = ruling).

AUTHORITY: paper/** libero; micro-run GPU cap 2 se una claim scopre un buco;
must docket: upload, authorship, licenza, nuove campagne di misura, claim
senza artefatto, modifiche a src/engine (numeri dai TAG, non da HEAD).

CONSTRAINTS: claims-map o niente; inglese asciutto + humanizer; gap nativo
in evidenza, non in appendice; zero attribution AI; figure da dataviz.

CONTEXT ANCHORS: journal c3a/b/c + q1; direction §2/§7/§8; recon Qwen §6;
native-baseline JSON; split-plan.
