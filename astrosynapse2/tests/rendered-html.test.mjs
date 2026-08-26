import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const projectRoot = new URL("../", import.meta.url);

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: handler } = await import(workerUrl.href);

  const request = new Request("http://127.0.0.1:3000/", {
    headers: { accept: "text/html" },
  });
  if (typeof handler === "function") return handler(request);

  return handler.fetch(
    request,
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Astrosynapse 2 control center", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Astrosynapse 2 · Self-play control center<\/title>/i);
  assert.match(html, /Training the next champion/i);
  assert.match(html, /Trainer live|Demo preview/i);
  assert.match(html, /Models &amp; Arena|Models & Arena/i);
  assert.match(html, /Play/i);
  assert.match(html, /Diagnostics/i);
  assert.match(html, /<canvas/i);
  assert.match(html, /class="jargon-help\b/i);
  assert.match(html, /Outcome Binary Cross-Entropy/i);
  assert.match(html, /A number used to control randomness/i);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site|react-loading-skeleton/i);
});

test("contains the real local API adapters and no disposable starter shell", async () => {
  const [page, styles, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(packageJson, /"name": "astrosynapse2-control-center"/);
  assert.doesNotMatch(packageJson, /site-creator|react-loading-skeleton|drizzle|tailwind/i);
  assert.match(layout, /Astrosynapse 2 · Self-play control center/);
  assert.doesNotMatch(layout, /Starter Project|codex-preview/i);

  assert.match(page, /127\.0\.0\.1:8765\/api/);
  assert.match(page, /\/runs\/\$\{encodeURIComponent\(runId\)\}\/metrics/);
  assert.match(page, /\/runs\/\$\{encodeURIComponent\(runId\)\}\/events/);
  assert.match(page, /fetchJson\("\/presets"\)/);
  assert.match(page, /deploymentPolicySelfplayFraction/);
  assert.match(page, /seed:\s*config\.seed/);
  assert.match(page, /seed:\s*asNumber\(item\.seed, previous\.seed\)/);
  assert.match(page, /updateConfig\("seed", Number\(event\.target\.value\)\)/);
  assert.match(page, /deployment_policy_selfplay_fraction:\s*config\.trainingGeneration >= 3/);
  assert.match(page, /if \(config\.trainingGeneration < 3\) return "m4_24h"/);
  assert.match(page, /launchPresetForConfig\(loaded, basePresetRef\.current\)/);
  assert.match(page, /Deployment-policy self-play/);
  assert.match(page, /\/models\/\$\{encodeURIComponent\(modelId\)\}/);
  assert.match(page, /fetchJson\("\/card-analysis"/);
  assert.match(page, /Run Scrap Elo/);
  assert.match(page, /Run Acquire Elo/);
  assert.match(page, /1,000 self-play games/);
  assert.match(page, /model_value/);
  assert.match(page, /expected_win_rate/);
  assert.doesNotMatch(page, /<Jargon term="outcomeEstimate">Expected win rate/);
  assert.match(page, /policy_replay_capacity:\s*config\.policyReplayCapacity/);
  assert.match(page, /policy_replay_disk_capacity:\s*config\.trainingGeneration >= 5/);
  assert.match(page, /Hot policy capacity \(RAM\)/);
  assert.match(page, /Disk policy capacity/);
  assert.match(page, /Counterfactual decision fraction/);
  assert.match(page, /Minimum head disagreement/);
  assert.match(page, /Full promotion cadence/);
  assert.match(page, /Provisional opportunity every games/);
  assert.match(page, /Adaptive schedule:/);
  assert.match(page, /maximumSignificantDigits:\s*4/);
  assert.match(page, /GUI_SAMPLE_BUCKET_SECONDS = 30/);
  assert.match(page, /Games \/ 30 sec/);
  assert.match(page, /gamesUntilCanary \/ displayRate\.gamesPerSecond/);
  assert.match(page, /Select run to view and control/);
  assert.match(page, /\?\? activeId\s*\?\? liveId/);
  assert.match(page, /selectedRunOverrideRef/);
  assert.match(page, /safetyBusy\.pause \|\| safetyBusy\.stop/);
  assert.match(page, /\|\| safetyBusy\.stop\}>■ Stop/);
  assert.doesNotMatch(page, /invokeControl\("(?:pause|stop)"\)[\s\S]{0,180}commandBusy !== null/);
  assert.match(page, /baseline:balanced/);
  assert.match(page, /explorersRemaining/);
  assert.match(page, /latest_arena/);
  assert.match(page, /modelEvaluationPoints/);
  assert.match(page, /latestEvaluatedModel/);
  assert.match(page, /evaluation\.quality_gate/);
  assert.match(page, /game_grouped_brier/);
  assert.match(page, /baselineMeanScore/);
  assert.match(page, /head_argmax_disagreement_rate/);
  assert.match(page, /mean_probability_std/);
  assert.match(page, /checkpoint_artifacts_complete/);
  assert.match(page, /degraded_reasons/);
  assert.match(page, /artifact_state/);
  assert.match(page, /actorAvailable/);
  assert.match(page, /availableModels/);
  assert.match(page, /availableArenaModels/);
  assert.match(page, /arenaModelGroups/);
  assert.match(page, /Run · \$\{group\.runName\}/);
  assert.match(page, /Arena model A from any run/);
  assert.match(page, /checkpoint from any run/);
  assert.match(page, /artifacts pruned · history retained/);
  assert.match(page, /ModelDiagnosticStrip/);
  assert.match(styles, /\.checkpoint-diagnostics/);
  assert.match(styles, /\.run-selector/);
  assert.match(page, /size_bytes/);
  assert.match(page, /truncation_rate/);
  assert.match(page, /heuristic_bootstrap_updates_remaining/);
  assert.match(page, /term="lineage"/);
  assert.match(page, /term="champion"/);
  assert.match(page, /term="heldOutStrength"/);
  assert.match(page, /term="actions"/);
  assert.match(page, /term="explainedVariance"/);
  assert.match(page, /term="bootstrapUncertainty"/);
  assert.match(page, /model_a_first_seat_score/);
  assert.match(page, /paired_interval_method/);
  assert.match(page, /method === "GET" \? 3_000 : 10_000/);
  assert.match(page, /<b>\{game\.explorersRemaining\}<\/b>/);
  assert.match(page, /observation\.pending_discard/);
  assert.match(page, /observation\.opponent_pending_discard/);
  assert.match(page, /<DiscardNotice count=\{game\.pendingDiscard\} subject="You"/);
  assert.match(page, /<DiscardNotice count=\{game\.opponentPendingDiscard\} subject="Opponent"/);
  assert.match(page, /Your board · all cards in play/);
  assert.match(page, /Opponent board · click an attackable card to attack it/);
  assert.match(page, /<VisiblePile label="Your discard pile" cards=\{game\.ownDiscard\}/);
  assert.match(page, /<VisiblePile label="Opponent discard pile" cards=\{game\.opponentDiscard\}/);
  assert.match(page, /action\.kind === "play_card" && action\.cardId === card\.catalogId/);
  assert.match(page, /action\.kind === "acquire" && action\.cardId === card\.catalogId/);
  assert.match(page, /onClick=\{\(\) => acquireMarketCard\(card\)\}/);
  assert.match(page, /action\.kind === "attack_player"/);
  assert.match(page, /action\.kind === "attack_base" && action\.targetCardId === card\.catalogId/);
  assert.match(page, /className="authority-display attack-target" onClick=\{attackOpponent\}/);
  assert.match(page, /onClick=\{\(\) => attackOpponentCard\(card\)\}/);
  assert.match(page, /action\.kind === "scrap_for_ability" && action\.cardId === card\.catalogId/);
  assert.match(page, /className="card-scrap-action"/);
  assert.match(page, /onScrap=\{hasScrapAbility \? \(\) => scrapInPlayCard\(card\) : undefined\}/);
  assert.match(styles, /\.card-scrap-action:hover:not\(:disabled\)/);
  assert.match(page, /Model lens · hover to reveal/);
  assert.match(styles, /\.model-hint:hover \.model-hint-details/);
  assert.match(styles, /\.model-hint:focus-visible \.model-hint-details/);
  assert.match(page, /Primary: \$\{describeAbility\(item\.primary, 0\)\}/);
  assert.match(styles, /Play table: prioritize complete, readable card state over viewport packing/);
  assert.match(styles, /\.play-panel \.hand-row[\s\S]*?overflow: visible/);
  assert.match(page, /className="card-title" title=\{card\.name\}/);
  assert.doesNotMatch(page, /Math\.max\(80,/);
  assert.doesNotMatch(page, /paired_seeds:\s*true|SkeletonPreview|_sites-preview/);

  for (const relativePath of [
    "app/chatgpt-auth.ts",
    "app/_sites-preview/SkeletonPreview.tsx",
    "app/_sites-preview/preview.css",
    "public/file.svg",
    "public/globe.svg",
    "public/window.svg",
  ]) {
    await assert.rejects(access(new URL(relativePath, projectRoot)));
  }
});

test("contains the editable Hard AI companion and stateless checkpoint advisor client", async () => {
  const [page, companion, styles, server, advisor] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/manual-hard-ai-match.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../backend/astro2/server.py", import.meta.url), "utf8"),
    readFile(new URL("../backend/astro2/advisor.py", import.meta.url), "utf8"),
  ]);

  assert.match(page, /Hard AI companion/);
  assert.match(page, /Simulated match/);
  assert.match(page, /<ManualHardAiMatch/);
  assert.match(companion, /Play the checkpoint\. Record the iPad\./);
  assert.match(companion, /Start new match/);
  assert.match(companion, /Astro5 first|Astro5/);
  assert.match(companion, /Hard AI first|Hard AI/);
  assert.match(companion, /Undefined/);
  assert.match(companion, /role="combobox"/);
  assert.match(companion, /Decks & hidden cards/);
  assert.match(companion, /Scrambled hand \+ deck/);
  assert.match(companion, /Play from hand/);
  assert.match(companion, /Attack base/);
  assert.match(companion, /Trigger ability/);
  assert.match(companion, /Click the action once/);
  assert.match(companion, /Play all cards in hand/);
  assert.match(companion, /What is the trade-row replacement/);
  assert.match(companion, /relay-card-quick-picks/);
  assert.match(companion, /hardLegalActionKinds/);
  assert.match(companion, /setHardAmount\(match\.hard\.combat\)/);
  assert.match(companion, /astroHandCandidateCatalog/);
  assert.match(companion, /catalogForCard=\{possibleAstroHandCards\}/);
  assert.match(companion, /reservedCardId: item\.cardId/);
  assert.match(companion, /Card-triggered decision/);
  assert.match(companion, /Resolve the card effect/);
  assert.match(companion, /All \{recommendation\.actions\.length\} legal/);
  assert.match(companion, /expected_win_rate/);
  assert.match(companion, /model_value/);
  assert.match(companion, /\/advisor\/evaluate/);
  assert.match(companion, /localStorage/);
  assert.match(styles, /\.relay-shell\s*\{[\s\S]*grid-template-columns:\s*minmax\(0, 1fr\) 355px/);
  assert.match(styles, /\.relay-card-undefined/);
  assert.match(server, /@app\.get\("\/api\/cards"\)/);
  assert.match(server, /@app\.post\("\/api\/advisor\/evaluate"/);
  assert.match(advisor, /class CheckpointAdvisor/);
  assert.match(advisor, /def main_phase_actions/);
});
