# Astrosynapse PHP Champion Arena

A standalone HTML/CSS/JavaScript/PHP version of the simulated human-versus-computer game. Upload this folder to your PHP server. **No Python, Node, database, background worker, Composer package, external API, or model service is required on the server.**

## Install

1. Upload the contents of this folder, including `models/`, `assets/`, and dotfiles, into a directory such as `public_html/arena/`.
2. Open `https://your-domain/arena/`. Pick an opponent and click **New game**.
3. The server needs **64-bit PHP 8.1 or newer**, JSON and sessions (standard PHP features), a writable PHP session directory, and preferably **256 MB** of PHP memory. `.user.ini` includes suggested hosting settings. PHP must actually execute `.php` files; static hosting cannot run this game.
4. The PHP worker needs read access to all model files. Only model management needs write access to `models/` and `models/registry.php`. Do not make the entire application world-writable.

On Apache, the included `.htaccess` files deny direct access to internal directories and disable directory listings. If your host does not allow `Options`, remove the `Options -Indexes` line and disable listings through the hosting panel. On Nginx or another server, deny URL access to `src/`, `models/`, `tools/`, `tests/`, and `config.php` in its site configuration. Example for an `/arena/` installation:

```nginx
location ~ ^/arena/(src|models|tools|tests)/ { deny all; }
location = /arena/config.php { deny all; }
location = /arena/config.local.php { deny all; }
```

Model and registry files also contain a fixed PHP guard that returns 404 before their data, even when per-directory access rules are unavailable. They are read as data, never included or evaluated as uploaded PHP code. Use HTTPS for public hosting.

To preview on a computer that already has PHP:

```sh
php -S 127.0.0.1:8080 -t php-game
```

Run that from the parent directory. Visit `http://127.0.0.1:8080/`. This command is for local testing, not production hosting.

## Included opponents

All ten actual promoted **Astro6 champion 1–10** checkpoints are included. Level 10 is the champion recorded in `evolution-20260923/state.json` at export time (September 26, 2026). These are snapshots: subsequent training does not alter the uploaded package. Each model's SHA-256 is recorded in `models/registry.php`.

Levels represent successive champion promotions. They are not calibrated beginner-to-expert difficulty levels; adjacent champions can be close in strength. The PHP actor runs the learned weights, rather than substituting heuristics. It preserves encoder versions 1/2, objective versions 1/2, mean-head deployment, the dominated-end-turn mask, and the public-information lethal finisher. PHP uses double arithmetic around the original float32 weights; very close ties can differ from NumPy. Exact ties choose the first action for deterministic server replay.

## Add or rename models

Model management is initially disabled. Playing the included champions requires no configuration.

1. Generate a password hash on a computer with PHP:

   ```sh
   php -r 'echo password_hash("REPLACE-WITH-A-LONG-PASSWORD", PASSWORD_DEFAULT), PHP_EOL;'
   ```

2. Paste the resulting hash into `admin_password_hash` in `config.php`. Keep the surrounding single quotes. Alternatively create `config.local.php` returning an array with the same key; that file overrides the bundled defaults and is ignored by Git.
3. Ensure PHP can write to `models/` and `models/registry.php`.
4. Open **Manage models**, sign in, and upload an Astro2 `.actor.npz` plus a display name. Or rename any included opponent and select **Save name**. Existing games retain their original opponent label.

NPZ import requires the optional PHP **ZipArchive** extension. It is not needed for gameplay or for importing preconverted `.model.php` files. Tensor shapes, finite weights, architecture, and expanded size are validated. Legacy `starrealms_policies/*_policy.json` files use a different architecture and are not supported by this Astro2 version.

If ZipArchive is unavailable, convert on your training computer (the server still needs only PHP):

```sh
python tools/export_models.py /path/to/checkpoint.actor.npz /path/to/new-champion.model.php
```

The local exporter requires NumPy. Upload the resulting `.model.php` using the model manager. Do not manually add executable PHP to the models directory. NPZ uploads default to a 16 MB limit in `.user.ini`; the importer caps expanded weight data and rejects unsupported model sizes. Large architectures beyond the supported limits need a separate memory/performance review.

For deployment without a web upload workflow, place a converted model in `models/` and add its entry to the JSON portion of `models/registry.php`, preserving the guard line. Each entry needs a unique `id`, `name`, and `file`; `description`, `level`, and `sha256` are optional metadata.

## Play all

**Play all** appears only at a human main-phase decision with at least two cards in hand when all of those cards can be played, left to right, without an additional user choice during their resolution. The server probes a cloned game using the real rules; it does not rely on a list of card names. Thus a single forced copy target is fine, multiple copy targets block it, and abilities with no legal targets do not block it. Optional choices count as choices even when declining is possible.

It plays only the cards in the hand when clicked. Newly drawn cards stay in hand for the next decision, so a draw never silently authorizes playing a new choice-bearing card. Manual base or ally abilities that can be used later remain separate actions. Each played card is recorded individually. The engine's/model's legal-action list never contains a `play_all` action. The server independently validates availability; changing the button in the browser cannot bypass it.

## Server state and performance

The engine, shuffles, hidden card assignments, checkpoint weights, and computer decisions stay on the server. The browser receives a public observation and the current human's legal choices. Opponent hand/deck contents are pooled into an unordered information set, as in simulated mode; their hidden assignment and draw order are not sent. Own unknown deck and remaining trade deck are shown only as unordered multisets in API data.

A PHP session stores a private seed and a transcript of ordinary legal decisions. Each request reconstructs the deterministic game without rerunning model inference for past decisions. Revision and game IDs reject duplicate or stale move submissions, and CSRF tokens protect mutations. The game can be refreshed and resumed while its PHP session remains available. Clearing cookies, server session expiry, or starting a new game ends that saved session. There is one active game per browser session; separate tabs share it.

Computer turns advance one decision per HTTP request to avoid long requests on shared hosting. The browser shows progress while requests continue. Refresh to resume after a connection interruption. Pure PHP neural inference is slower than NumPy; response speed depends on the host. No client-side computation is trusted to choose or apply model moves. The default game limits match simulated play: 240 turns and 220 decisions per turn; a limit produces a draw.

Shuffling uses a deterministic portable PHP generator with separate player/market streams. It preserves the game rules but does not reproduce Python's shuffle for the same numerical seed. Model ties use the first maximum; this also makes replay deterministic.

## Verification results

Validated with temporary PHP 8.1 and PHP 8.5 runtimes:

- 20 complete games / 5,176 decisions matched the Python engine's legal actions, observations, and final results. These games exercised all eight decision families and 48 card types; separate checks cover Fleet HQ.
- 40 bounded lethal-search cases, including 28 winning plans, matched the original Python finisher.
- All eight neural decision families matched NumPy encodings, selected actions, and scores (maximum sampled logit difference approximately 0.000002).
- 27 regression checks cover Play all, conditional copy/destroy/scrap effects, base timing, card draws, Explorer recycling, replay, illegal actions, and a complete champion turn.
- All ten checkpoint hashes/shapes validate; native NPZ import, preconverted import, renaming, and invalid-model rejection pass.
- HTTP tests cover cookie sessions, refresh/resume, CSRF, stale/illegal requests, computer turns, protected model files, and unauthorized administration.

Neural regression peak PHP memory was approximately 57 MB on the PHP 8.5 runtime and 112 MB on PHP 8.1. Host performance and memory allocation can differ. The UI's JavaScript syntax, DOM references, and all 49 card-art paths were checked; no browser visual test was performed.

## Source and verification

- `index.html`, `assets/style.css`, `assets/game.js`: client interface and artwork.
- `api.php`, `src/Session.php`: session protocol and human-only batching.
- `src/Engine.php`: all 49 Core Set cards, rules version 2.
- `src/Encoder.php`, `src/Actor.php`, `src/Lethal.php`: model inputs, inference and finisher.
- `src/Models.php`: password-protected management support and validated model conversion.
- `tools/export_models.py`: optional local checkpoint exporter.
- `tests/`: regression and Python/PHP comparison tools; not needed on the web server.

Run the PHP regression suite:

```sh
php -d memory_limit=256M tests/regression.php
```

To regenerate comparisons from the parent training repository, use its NumPy environment:

```sh
../astrosynapse2/.venv/bin/python tests/build_reference.py --games 20 --output tests/generated/reference.json
php -d memory_limit=768M tests/reference.php tests/generated/reference.json rules
php -d memory_limit=768M tests/reference.php tests/generated/reference.json neural
```

The larger memory setting is for decoding the complete development fixture, not normal gameplay. `tests/generated/` is not part of the upload package. The port is isolated: it does not change the training engine, running trainer, original simulated mode, or source checkpoint files.

The card artwork is copied from the existing project's official-gallery assets. Star Realms and its artwork are owned by Wise Wizard Games LLC; see `assets/card-art/README.md`.
