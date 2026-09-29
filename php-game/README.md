# Astrosynapse client-only Champion Arena

The former PHP arena now runs entirely in the browser: rules, shuffling, model encoding, float32 weights, neural inference, and lethal search. Upload the static files to an HTTPS host. **No PHP, API, database, Python, Node, or inference server is needed on the host.** The folder name is retained for continuity.

## Hosting

Upload `index.html`, `assets/`, and `models/`, including `.htaccess` on Apache. The generated `php-game-upload.zip` in the parent directory contains the ready-to-upload distribution, without development sources or tests. Open the site over HTTPS; localhost HTTP also works for development. Opening `index.html` as a `file://` URL is not supported.

For a local preview from the repository root:

```sh
python3 -m http.server 8088 --directory php-game
```

Visit `http://localhost:8088/`. Python only serves static files in this preview.

Serve `.js` as JavaScript, `.json` as JSON, and `.astro.gz` as **application/gzip without a Content-Encoding header**. The browser reads and decompresses these model files itself. On non-Apache hosts configure:

- `index.html` (including the directory index): `Cache-Control: no-cache, must-revalidate`.
- `assets/*` and `models/*`: `Cache-Control: public, max-age=31536000, immutable`.
- Preserve URL query strings in cache keys; they ensure a changed HTML file requests fresh resources.

Do not carry forward the old blanket deny rule for `models/`. The included replacement permits static model downloads. When replacing an existing installation, remove the obsolete PHP endpoints and old `.model.php` files instead of leaving them on the server. Existing source training checkpoints are untouched.

## Levels

Level 10 is selected and downloaded at startup. Levels 1–9 download only when selected. Selection loads the opponent for the **next new game**; it does not change an ongoing game's opponent. Downloaded actors remain in worker memory for the page's lifetime.

| Level | Actual source | Compressed model |
|---|---|---:|
| 1 | First checkpoint in all Astrosynapse2: `df337885a4b74416`, untrained, 0 games | 1.78 MB |
| 2 | Astro2 champion `1b822120f1634e46`, 3,303,168 games | 4.67 MB |
| 3 | Astro3 champion `246e56c917644759`, 2,602,496 games | 4.69 MB |
| 4 | Astro4 champion `05ef55aaf4c548c5`, 98,176 games | 7.44 MB |
| 5 | Astro4 champion `0ecf69b96351463d`, 502,656 games | 7.44 MB |
| 6 | Astro5 champion `08aa018c672847d9`, 5,954,048 games | 7.45 MB |
| 7 | Astro6 generation 1 | 7.45 MB |
| 8 | Astro6 generation 2 | 7.45 MB |
| 9 | Astro6 generation 6 | 7.45 MB |
| 10 | Astro6 generation 10, the existing PHP package's top champion | 7.45 MB |

“Generation” refers to the numbered Astro6 champion promotions in `astrosynapse2/data/progressive/evolution-20260923/state.json`. Levels 2–6 are earlier run champions chosen to span the preceding history; level 9 sits between generations 2 and 10. These are historical milestones, not calibrated, evenly spaced difficulty ratings. Level 1 is intentionally the very first, untrained checkpoint.

The UI descriptions use “trained on X games”, with cumulative training games across the checkpoint’s ancestral runs. Each run is counted once; independently initialized models do not inherit unrelated older runs. Astro6 totals also include the 303,360-game source branch and 49,152-game managed experiment before progressive training, followed by the promotion’s cumulative campaign training/search count (excluding separate verification games). The registry stores this total as `training_games`.

`models/registry.json` records exact source paths, checkpoint/generation IDs, source hashes, packaged hashes, byte sizes, and architecture. Weights preserve float32 values without quantization. The actor supports encoder and objective versions 1 and 2, mean-head scoring, the dominated-end-turn mask, and public-information lethal search. Arithmetic uses JavaScript doubles around float32 weights; near ties can differ from NumPy.

## Resource caching

- An inline bootstrap hashes the HTML document **before modifying it**. That hash is attached to every local asset and model URL as a cache key.
- `index.html` is revalidated on normal navigation. Replacing that one file changes the cache key, so the next page load requests the CSS, JavaScript, card data, model registry, and models under fresh URLs.
- The build script stamps the HTML whenever bundles, card data, or the model registry changes.
- Assets are cached on demand, never all prefetched. Card images load when near the viewport or opened in Details. Card image URLs are generated from card names and load WebP files directly from the official Star Realms gallery. Repeated images use normal browser HTTP caching.
- Models and artwork also live in browser Cache Storage, so subsequent visits can reuse them without network transfer. A model's SHA-256 is verified before it is used.
- If browser storage is unavailable or full, the game falls back to network plus memory caching. Browser eviction can require a later re-download.
- Once initialized, an open game keeps working with downloaded models if the connection drops. A page load requires the static HTML; undownloaded levels need a connection.

**After changing code, models, or artwork, run the build script and deploy the generated files. Upload `index.html` last (or deploy atomically).** Uploading the new HTML is the only refresh mechanism; there is no polling, prompt, reload control, service worker, or server-side logic. A failed or mismatched model download displays an error instead of using unverified weights. A saved in-progress game is discarded when its HTML cache key no longer matches.

## Player stats

Every new game randomly chooses who takes the first turn. The choice is saved with the game and stays the same after refresh. Open **Stats** beside New game to view wins, losses, draws, and win rate for each level (and imported opponents). Results are stored in localStorage on this browser and site, survive file replacements, and are counted once per completed game. Resigning requires confirmation and counts as a loss. Abandoned games do not count; draws are excluded from win rate. Clearing site data removes stats.

## Local state and model management

The browser stores the current game’s deterministic seed and action transcript in localStorage. Refreshing or reopening the site resumes the last saved game without repeating past inference. The saved slot is shared by tabs on this browser and site. No moves or hidden hands are sent to a server. All computation is visible to the device owner; this is a local single-player game, not a trusted ranked-game server.

**Manage models** supports importing Astro2 `.actor.npz`, converted `.astro.gz`, and previous guarded `.model.php` model exports. These files are parsed as data, never executed or uploaded. Imports are validated and stored in IndexedDB. Renames live in local storage and apply to new games. No password is needed because the changes affect only this browser and site. Imported models and aliases survive file replacements; incompatible saved games do not. Model files are capped at 64 MB on import and 128 MB after expansion.

The exporter is available for conversion outside the browser:

```sh
astrosynapse2/.venv/bin/python php-game/tools/export_client_models.py checkpoint.actor.npz opponent.astro.gz
```

Legacy `starrealms_policies/*_policy.json` files use a different architecture and are unsupported.

## Gameplay and performance

The UI runs on the main thread; all game rules and inference run in a module Web Worker. Computer turns advance one decision at a time, with visible progress. Card details, scrolling, and browser rendering do not share the inference thread. Download size is not a guarantee of performance on an older phone.

The table fits the viewport with the trade row always between the opponent’s fleet and your In Play row, followed by your hand. Available game-action buttons have white backgrounds. When a card has exactly one available action, clicking its face performs that action; otherwise the face opens details. Click the opponent’s authority to make a legal attack. A fixed opponent header and player footer keep authority, trade, combat, must-discard counts, card-count buttons, and player actions visible. The table reserves the actual height of both bars, including when they wrap on mobile. Clicking a deck or the opponent’s hand button shows the combined hand and deck in random order without revealing hidden hand membership or draw order; your hand button opens your playable hand. Discard piles, the scrap heap, and the game log open in dialogs. Must-discard counters remain visible, and discard piles are highlighted when they contain legal scrap targets. The top-left ⓘ buttons open full-size card details. Scrap decisions automatically open a large picker grouping all legal targets by hand, discard pile, or trade row; it stays open across selections and closes once no targets remain. Optional scrap choices include their decline action. Completed games show a prominent victory, defeat, or draw banner.

The opponent header also shows the opponent model’s estimated win probability. Objective-v2 models use the mean sigmoid of their state-value heads; earlier outcome models use the mean predicted outcome of their preferred action. The current decision is evaluated for the side to move and complemented on human turns to express the opponent’s probability. This is a model estimate, not a calibrated guarantee.

Player End Turn and discard decisions always require manual input, including a lone End Turn action, a single remaining card, or a hand of identical cards. Play all cannot advance into the opponent’s turn automatically. End Turn remains visible and is disabled while another choice must be resolved or the opponent is playing.

Play all probes a cloned game using the actual rules. It appears only when the current hand has at least two cards that can resolve in visible order without another user choice. Newly drawn cards remain in hand. Each play is recorded separately, and batching is absent from the model's action space. Card choices, bases, Explorer recycling, and the 240-turn/220-action limits are preserved. The port also includes the current Python rule that automatically activates discard-triggering allies.

## Development and verification

Readable source is under `assets/runtime/`. The deployed `assets/game.js` and `assets/worker.js` are dependency-free bundles. The build requires the existing repository's esbuild; the host needs no build tooling.

```sh
# Re-export the documented level selection (NumPy required locally).
astrosynapse2/.venv/bin/python php-game/tools/export_client_models.py --levels
# Bundle JS and stamp index.html; edit tools/index.template.html for HTML changes.
python3 php-game/tools/build_client.py
# Package just the static hosting files.
python3 php-game/tools/package_client.py

# Rules, batching, replay, inference and existing on-card controls.
node php-game/tests/client-stats.mjs
node php-game/tests/client-regression.mjs
node php-game/tests/client-import.mjs
node --test php-game/tests/scrap-ui.test.cjs
# Differential verification against the original Python engine and all ten actors.
astrosynapse2/.venv/bin/python php-game/tests/build_reference.py --games 20 --output /tmp/astro-js-reference.json
node php-game/tests/client-reference.mjs /tmp/astro-js-reference.json
# Browser verification (install Playwright and its Chromium browser locally first).
node php-game/tests/client-browser.mjs
```

Verified during this port: 20 complete games / 5,091 decisions, 40 lethal plans, all eight neural families and win-probability calculations for every level, 37 gameplay/session regressions, and six on-card UI regressions. Maximum sampled generation-10 logit error was approximately 0.000002. Browser checks cover lazy downloads, local AI turns, refresh/resume, dropped-connection play, HTML cache invalidation, mobile layout, and native NPZ import persistence. The desktop and mobile layouts were inspected in Chromium; physical-device performance has not been measured.

Artwork loads directly from the [official Star Realms Card Gallery](https://www.starrealms.com/card-gallery); card image files are excluded from the upload archive. The welcome screen uses the gallery’s Scout card. Star Realms and card artwork belong to Wise Wizard Games LLC.
