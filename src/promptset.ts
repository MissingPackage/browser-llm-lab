// Prompt deterministico ~512 token: paragrafo fisso ripetuto.
// NON MODIFICARE il testo senza incrementare l'id (i risultati citano promptId).
const PARA =
  "The city library opened its doors at nine in the morning, and the archivist began sorting the day's returns. " +
  "Each volume carried a small paper slip noting the date, the borrower's initials, and the shelf it belonged to. " +
  "Outside, the market square filled slowly with vendors arranging crates of apples, bread, and winter vegetables. " +
  "A tram passed every twelve minutes, and its bell could be heard clearly through the reading room windows. ";

export const PROMPT_512 = {
  id: "bench-512-v1" as const,
  text:
    "Read the following passage carefully, then continue the story in the same style.\n\n" +
    PARA.repeat(5) +
    "\n\nContinue the story:",
};
