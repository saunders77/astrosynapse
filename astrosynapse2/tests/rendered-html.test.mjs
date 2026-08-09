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
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
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
  assert.match(page, /\/models\/\$\{encodeURIComponent\(modelId\)\}/);
  assert.match(page, /model_value/);
  assert.match(page, /baseline:balanced/);
  assert.match(page, /explorersRemaining/);
  assert.match(page, /latest_arena/);
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
