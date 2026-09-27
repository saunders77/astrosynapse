import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = await readFile(new URL("../app/manual-hard-ai-match.tsx", import.meta.url), "utf8");
const parsed = ts.createSourceFile("manual.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const names = new Set([
  "effectiveCardId", "effectiveDefinition", "trackedHasFaction",
  "applyAutomaticHardResource", "triggerAutomaticHardAllies", "triggerAutomaticAstroAllies",
]);
const selected = parsed.statements.filter((statement) =>
  (ts.isFunctionDeclaration(statement) && names.has(statement.name?.text)) ||
  (ts.isVariableStatement(statement) && statement.declarationList.declarations.some(
    (declaration) => declaration.name.getText(parsed) === "AUTOMATIC_ALLY_EFFECTS",
  )),
).map((statement) => statement.getText(parsed)).join("\n");
const context = vm.createContext({});
vm.runInContext(ts.transpile(selected, { target: ts.ScriptTarget.ES2022 }), context);

for (const side of ["astro", "hard"]) {
  for (const cardId of [26, 33]) {
    test(`${side} automatically applies discard ally ${cardId} once when yellow arrives`, () => {
      const definitions = new Map([
        [cardId, { card_id: cardId, faction: "star_empire", ally: "opponent_discard", ally_amount: 0 }],
        [36, { card_id: 36, faction: "star_empire", ally: "gain_combat", ally_amount: 4 }],
      ]);
      const other = side === "astro" ? "hard" : "astro";
      const trigger = context[side === "astro" ? "triggerAutomaticAstroAllies" : "triggerAutomaticHardAllies"];
      const player = () => ({ inPlay: [], pendingDiscard: 0, combat: 0, trade: 0, authority: 50 });
      let match = { astro: player(), hard: player() };
      match[side].inPlay.push({ uid: "source", cardId });
      match = trigger(match, definitions);
      assert.equal(match[other].pendingDiscard, 0);
      match[side].inPlay.push({ uid: "ally", cardId: 36 });
      match = trigger(match, definitions);
      assert.equal(match[other].pendingDiscard, 1);
      assert.equal(match[side].pendingDiscard, 0);
      assert.equal(match[side].combat, 4);
      assert.ok(match[side].inPlay.every((item) => item.allyTriggered));
      match = trigger(match, definitions);
      assert.equal(match[other].pendingDiscard, 1);
    });
  }
}
