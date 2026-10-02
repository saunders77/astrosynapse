// Browser game client. Keep the start export, worker payloads, DOM IDs, and storage keys stable.
// Bundled from assets/runtime; running tools/build_client.py will regenerate this file.

// php-game/assets/runtime/resources.mjs
var resourceBaseURL;
var resourceRelease;
var resourcePromises = new Map();
// Resource URLs are scoped to this installation and versioned for cache reuse.
function configureResources(root, version) {
  resourceBaseURL = new URL(root);
  resourceRelease = version;
}

function resourceURL(path) {
  const url = new URL(path, resourceBaseURL);
  if (url.origin !== resourceBaseURL.origin || !url.pathname.startsWith(resourceBaseURL.pathname)) {
    throw new Error("Invalid resource path");
  }
  url.searchParams.set("v", resourceRelease);
  return url.href;
}

// Reuse in-flight downloads. Persistent caching is optional; failed downloads can be retried.
async function resource(path) {
  const url = resourceURL(path);
  if (!resourcePromises.has(url)) {
    resourcePromises.set(url, (async () => {
      let cache;
      try {
        cache = await caches.open(`astro-assets:${resourceBaseURL.pathname}:${resourceRelease}`);
        const hit = await cache.match(url);
        if (hit) {
          return await hit.arrayBuffer();
        }
      } catch {
        // Cache Storage may be unavailable; fall back to downloading the resource.
      }
      const response = await fetch(url, {
        cache: "force-cache"
      });
      if (!response.ok) {
        throw new Error(`Could not download ${path} (${response.status}). Reconnect and retry.`);
      }
      const bytes = await response.arrayBuffer();
      try {
        await cache?.put(url, new Response(bytes, {
          headers: response.headers
        }));
      } catch {
        // Keep the downloaded bytes in memory even if the persistent cache is full.
      }
      return bytes;
    })().catch(error => {
      resourcePromises.delete(url);
      throw error;
    }));
  }
  return resourcePromises.get(url);
}

// php-game/assets/runtime/audio.mjs
var SOUNDS = ["attack", "authority", "combat", "playerturn", "scrap", "shuffle", "trade"];
var GameAudio = class {
  constructor({
    Context = globalThis.AudioContext || globalThis.webkitAudioContext,
    load = resource,
    later = globalThis.setTimeout.bind(globalThis),
    cancel = globalThis.clearTimeout.bind(globalThis)
  } = {}) {
    Object.assign(this, {
      load,
      later,
      cancel,
      buffers: new Map(),
      queue: [],
      sources: new Set(),
      lastStart: -Infinity,
      timer: null
    });
    try {
      this.context = new Context();
      this.gain = this.context.createGain();
      this.gain.connect(this.context.destination);
      this.context.addEventListener("statechange", () => this.pump());
    } catch {
      // Audio support is optional; the game can still run without a context.
    }
  }
  preload() {
    return this.loading ??= Promise.allSettled(SOUNDS.map(async name => {
      if (!this.context) {
        return;
      }
      const bytes = await this.load(`assets/audio/${name}.mp3`);
      this.buffers.set(name, await this.context.decodeAudioData(bytes.slice(0)));
    })).then(() => {
      this.ready = true;
      this.pump();
    });
  }
  unlock() {
    if (this.context && this.context.state !== "running") {
      this.context.resume().then(() => this.pump()).catch(() => {});
    }
  }
  setVolume(value) {
    if (this.gain) {
      this.gain.gain.setValueAtTime(Math.max(0, Math.min(1, value)), this.context.currentTime);
    }
  }
  enqueue(names) {
    this.queue.push(...names);
    this.pump();
  }
  pump() {
    if (!this.ready || this.context?.state !== "running" || this.timer !== null) {
      return;
    }
    while (this.queue.length && !this.buffers.has(this.queue[0])) {
      this.queue.shift();
    }
    if (!this.queue.length) {
      return;
    }
    // Space sound starts by 250 ms while allowing already-playing clips to overlap.
    const remaining = this.lastStart + 250 - this.context.currentTime * 1e3;
    if (remaining > 0) {
      this.timer = this.later(() => {
        this.timer = null;
        this.pump();
      }, Math.ceil(remaining));
      return;
    }
    const source = this.context.createBufferSource();
    source.buffer = this.buffers.get(this.queue.shift());
    source.connect(this.gain);
    source.onended = () => {
      source.disconnect();
      this.sources.delete(source);
    };
    this.sources.add(source);
    source.start();
    this.lastStart = this.context.currentTime * 1e3;
    this.pump();
  }
  clear() {
    this.cancel(this.timer);
    this.timer = null;
    this.queue.length = 0;
    for (const source of this.sources) {
      source.stop();
    }
    this.sources.clear();
  }
};

// php-game/assets/runtime/turn-summary.mjs
// Summarize the latest completed AI turn, omitting the two default starter cards.
function opponentTurnSummary(gameState, cardCatalog) {
  const entries = gameState.action_log.filter(entry => entry.player_id === 1);
  const completed = entries.filter(entry => entry.kind === "end_turn" ||
      entry.turn < gameState.observation.turn || gameState.status === "complete");
  const turn = completed.at(-1)?.turn;
  if (turn === void 0) {
    return "No AI turn yet · Game log";
  }
  const played = [];
  const acquired = [];
  for (const entry of entries.filter(turnEntry => turnEntry.turn === turn)) {
    const id = entry.kind === "free_acquire" ? entry.target_card_id : entry.card_id;
    if (id < 2 || !cardCatalog[id]) {
      continue;
    }
    if (entry.kind === "play_card") {
      played.push(cardCatalog[id].name);
    }
    if (entry.kind === "acquire" || entry.kind === "free_acquire") {
      acquired.push(cardCatalog[id].name);
    }
  }
  const parts = [];
  if (played.length) {
    parts.push(`played ${played.join(", ")}`);
  }
  if (acquired.length) {
    parts.push(`acquired ${acquired.join(", ")}`);
  }
  return `Last turn: ${parts.join("; ") || "no non-default cards played or acquired"}`;
}

// php-game/assets/runtime/stats.mjs
// Store each completed game once so refreshes do not inflate the win/loss totals.
function recordResult(storage, prefix, gameState) {
  if (gameState?.status !== "complete" || !gameState.result || !gameState.model_id) {
    return;
  }
  const outcome = gameState.result.truncated || gameState.result.winner === null ?
      "draws" : gameState.result.winner === 0 ? "wins" : gameState.result.winner === 1 ?
      "losses" : null;
  if (!outcome) {
    return;
  }
  const key = prefix + gameState.id;
  if (storage.getItem(key) === null) {
    storage.setItem(key, JSON.stringify({
      model: gameState.model_id,
      label: gameState.model_label,
      outcome
    }));
  }
}

// Ignore unrelated storage keys and malformed result records.
function readStats(storage, prefix) {
  const stats = new Map();
  for (let index = 0; index < storage.length; index++) {
    const key = storage.key(index);
    if (!key?.startsWith(prefix)) {
      continue;
    }
    let record;
    try {
      record = JSON.parse(storage.getItem(key));
    } catch {
      continue;
    }
    if (!record || typeof record.model !== "string" || !["wins", "losses", "draws"].includes(record.outcome)) {
      continue;
    }
    if (!stats.has(record.model)) {
      stats.set(record.model, {
        label: record.label || record.model,
        wins: 0,
        losses: 0,
        draws: 0
      });
    }
    stats.get(record.model)[record.outcome]++;
  }
  return stats;
}

// php-game/assets/runtime/local-models.mjs
var databasePromise;
// Open one shared database connection for models imported on this installation.
function openModelDatabase() {
  return databasePromise ??= new Promise((resolve, reject) => {
    const openRequest = indexedDB.open("astro-local-models:" + new URL("../", import.meta.url).pathname, 1);
    openRequest.onupgradeneeded = () => openRequest.result.createObjectStore("models", {
      keyPath: "id"
    });
    openRequest.onsuccess = () => resolve(openRequest.result);
    openRequest.onerror = () => reject(openRequest.error);
  });
}

// Resolve only after the transaction completes, including writes and deletes.
async function localModels(operation, value) {
  const modelDatabase = await openModelDatabase();
  return new Promise((resolve, reject) => {
    const transaction = modelDatabase.transaction("models", operation === "list" ||
        operation === "get" ? "readonly" : "readwrite");
    const modelStore = transaction.objectStore("models");
    const storageRequest = operation === "list" ? modelStore.getAll() : operation === "get" ?
        modelStore.get(value) : operation === "put" ? modelStore.put(value) : modelStore.delete(value);
    transaction.oncomplete = () => resolve(storageRequest.result);
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error || new Error("Local model storage failed"));
  });
}

// php-game/assets/runtime/ui.mjs
var getElement = id => document.getElementById(id);
var audio;
var cards = [];
var models = [];
var game = null;
var busy = false;
var advanceTimer = null;
var openPile = null;
var worker;
var release;
var base;
var savedKey;
var dismissedSelection = null;
var selectionKey = () => game ? `${game.id}:${game.revision}` : null;
var requestId = 0;
var pendingRequests = new Map();
var aliasesKey = "astro-model-names:" + location.pathname.replace(/index\.html$/, "");
var aliases = {};
try {
  aliases = JSON.parse(localStorage.getItem(aliasesKey) || "{}");
} catch {
  // Use default model names when saved aliases cannot be read.
}
var cardArtURL = card => `https://www.starrealms.com/card-gallery/images/content/card-gallery/${card.name.toLowerCase().replaceAll(" ",
    "-")}.webp`;
var createElement = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== void 0) {
    element.textContent = text;
  }
  return element;
};
var showError = (message, id = "error") => {
  getElement(id).textContent = message;
  getElement(id).hidden = !message;
};

function setBusy(value) {
  busy = value;
  document.body.classList.toggle("busy", value);
  document.querySelectorAll("button, select, input").forEach(element =>
      element.disabled = value && !element.matches(".details, .pile, [data-close], [data-inspect], [data-open], #audio-volume") ||
      element.dataset.unavailable === "true");
}

// Match each worker response to the promise for its request ID.
function sendWorkerRequest(payload) {
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    pendingRequests.set(id, {
      resolve,
      reject
    });
    worker.postMessage({
      ...payload,
      requestId: id
    });
  });
}

function saveGame(saved) {
  try {
    if (saved) {
      localStorage.setItem(savedKey, JSON.stringify({
        release: release,
        saved
      }));
    } else {
      localStorage.removeItem(savedKey);
    }
  } catch {
    showError("Browser storage is unavailable. This game works, but cannot resume after leaving this page.");
  }
}
var imageObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      imageObserver.unobserve(entry.target);
      loadImage(entry.target, entry.target.dataset.art);
    }
  }
}, {
  rootMargin: "150px"
});
function loadImage(image, path) {
  image.onerror = () => {
    image.title = "Card image unavailable. Reconnect to try again.";
  };
  image.onload = () => {
    image.title = "";
  };
  image.src = path;
}

function setImage(image, path, immediate = false) {
  image.dataset.art = path;
  if (immediate) {
    loadImage(image, path);
  } else {
    imageObserver.observe(image);
  }
}

// Serialize game commands, persist returned state, and refresh the UI from the worker snapshot.
async function request(payload, errorTarget = "error") {
  if (busy) {
    return;
  }
  clearTimeout(advanceTimer);
  showError("", errorTarget);
  setBusy(true);
  try {
    const data = await sendWorkerRequest(payload);
    if (data.models) {
      models = data.models;
      renderModels();
    }
    if ("saved" in data) {
      saveGame(data.saved);
    }
    if (data.notice) {
      showError(data.notice);
    }
    if ("game" in data) {
      if (data.game) {
        const sameGame = game?.id === data.game.id;
        if (!sameGame || payload.op === "undo") {
          audio.clear();
        }
        // Replayed history is deterministic; only newly applied effects should make sound.
        audio.enqueue(data.game.sounds.slice(sameGame ? game.sounds.length : 0));
      }
      game = data.game;
      if (payload.op === "new" && game) {
        location.hash = "#game";
      }
      if (payload.op === "undo") {
        dismissedSelection = selectionKey();
      }
      render();
      renderStats();
    }
    return data;
  } catch (error) {
    showError(error.message, errorTarget);
    getElement("status").textContent = "Paused. Your last saved game is preserved. Retry or refresh to resume.";
  } finally {
    setBusy(false);
    if (game?.status === "model_thinking" && !getElement("error").textContent) {
      scheduleModelTurn();
    }
  }
}

// Advance the AI in small steps so the browser can display progress between actions.
function scheduleModelTurn() {
  clearTimeout(advanceTimer);
  advanceTimer = setTimeout(() => {
    if (!busy && game?.status === "model_thinking") {
      request({
        op: "advance",
        id: game.id,
        revision: game.revision
      });
    }
  }, 120);
}

function move(id) {
  if (busy) {
    return;
  }
  if (openPile === "hand") {
    getElement("pile-dialog").close();
  }
  request({
    op: "choose",
    id: game.id,
    revision: game.revision,
    action_id: id
  }, getElement("decision-dialog").open ? "decision-error" : "error");
}

// Cancel by undoing the choice; retain restored options for the current revision.
function cancelSelection() {
  if (busy || !game?.can_undo) {
    return;
  }
  const errorTarget = getElement("scrap-dialog").open ? "scrap-error" : "decision-error";
  return request({
    op: "undo",
    id: game.id,
    revision: game.revision
  }, errorTarget);
}

function inspect(card) {
  setImage(getElement("card-large"), cardArtURL(card), true);
  getElement("card-large").alt = card.name;
  getElement("card-name").textContent = card.name;
  getElement("card-description").textContent = `${card.faction.replaceAll("_",
      " ")} · ${card.card_type} · Cost ${card.cost}${card.defense ? ` · Defense ${card.defense}` : ""}`;
  getElement("card-dialog").showModal();
}

function actionName(action) {
  const names = {
    scrap_card: "SCRAP",
    play_card: "Play",
    acquire: `Acquire · ${action.amount} trade`,
    activate_base: "Use ability",
    activate_ally: "Use ally",
    scrap_for_ability: "🗑️ Scrap for ability",
    attack_base: `Attack base · ${action.amount} combat`,
    destroy_base: "DESTROY",
    free_acquire: "ACQUIRE FREE",
    copy_ship: "COPY",
    scrap_trade_row: "SCRAP",
    discard_card: "DISCARD"
  };
  return names[action.kind] || action.label;
}

function renderScrap(actions) {
  const targets = game?.status === "your_turn" ? actions.filter(action =>
      ["scrap_card", "scrap_trade_row"].includes(action.kind)) : [];
  const dialog = getElement("scrap-dialog");
  if (!targets.length || dismissedSelection === selectionKey()) {
    if (dialog.open) {
      dialog.close();
    }
    return;
  }
  getElement("scrap-prompt").textContent = game.decision.prompt || "Choose a card to scrap";
  for (const [zoneName, title] of [["hand", "Your hand"], ["discard",
      "Your discard pile"], ["trade_row", "Trade row"]]) {
    const options = targets.filter(action => action.source_zone === zoneName);
    const section = getElement("scrap-options-" + zoneName);
    section.hidden = !options.length;
    section.replaceChildren(createElement("h3", "", title));
    const list = createElement("div", "cards");
    list.append(...options.map(action => cardView(action.kind === "scrap_trade_row" ?
        action.target_card_id : action.card_id, [action])));
    section.append(list);
  }
  getElement("scrap-decline").replaceChildren(...actions.filter(action => action.kind === "decline").map(action => {
    const button = createElement("button", "move", action.label);
    button.addEventListener("click", () => move(action.id));
    return button;
  }));
  if (!dialog.open) {
    if (getElement("pile-dialog").open) {
      getElement("pile-dialog").close();
    }
    dialog.showModal();
  }
}

// Build a card with its legal actions. Scrapping for an ability always requires its explicit button.
function cardView(id, actions = [], state = "") {
  const card = cards[id];
  if (!card) {
    return createElement("span", "empty", "Empty trade slot");
  }
  const node = createElement("article", `card ${card.card_type !== "ship" ?
      "base" : ""} ${actions.length ? "actionable" : ""}`);
  node.dataset.faction = card.faction;
  const face = createElement("button", "card-face");
  face.type = "button";
  const action = actions.length === 1 && actions[0].kind !== "scrap_for_ability" ? actions[0] : null;
  face.setAttribute("aria-label", action ? action.label || `${actionName(action)} ${card.name}` : `Details for ${card.name}`);
  face.title = face.getAttribute("aria-label");
  face.addEventListener("click", () => action ? move(action.id) : inspect(card));
  const image = createElement("img");
  setImage(image, cardArtURL(card));
  image.alt = card.name;
  image.loading = "lazy";
  face.append(image);
  node.append(face);
  const meta = createElement("div", "card-meta");
  const heading = createElement("div", "card-heading");
  const cost = createElement("span", "card-cost", card.cost);
  cost.setAttribute("aria-label", `Cost ${card.cost}`);
  heading.append(createElement("span", "card-name", card.name), cost);
  meta.append(heading);
  if (state) {
    meta.append(createElement("span", "card-state", state));
  }
  const controls = createElement("div", "card-actions");
  for (const action of actions) {
    const button = createElement("button", "move", actionName(action));
    button.type = "button";
    button.title = action.label;
    button.addEventListener("click", () => move(action.id));
    controls.append(button);
  }
  const detail = createElement("button", "details", "Details");
  detail.type = "button";
  detail.setAttribute("aria-label", `Details for ${card.name}`);
  detail.addEventListener("click", () => inspect(card));
  controls.append(detail);
  meta.append(controls);
  node.append(meta);
  return node;
}

function zone(id, entries) {
  const node = getElement(id);
  node.replaceChildren(...entries);
  if (!entries.length) {
    node.append(createElement("span", "empty", "No cards"));
  }
}

// Render the current worker snapshot. Action IDs and zone names come from the game engine.
function render() {
  const home = location.hash === "#home";
  document.body.classList.toggle("active-game", !!game && game.status !== "complete" && !home);
  recordStats();
  imageObserver.disconnect();
  document.body.classList.toggle("playing", !!game && location.hash !== "#stats" && !home);
  if (!game || home) {
    setImage(getElement("welcome-art"), cardArtURL({
      name: "Scout"
    }));
  }
  getElement("opponent-bar").hidden = !game || home;
  getElement("player-bar").hidden = !game || home;
  getElement("welcome").hidden = !!game && !home;
  getElement("board").hidden = !game || home;
  getElement("resume-game").hidden = !home || !game || game.status === "complete";
  if (!game || home) {
    getElement("result-banner").hidden = true;
    getElement("status").textContent = "";
    return;
  }
  const observation = game.observation;
  const decision = game.decision;
  const actions = decision?.actions || [];
  const main = decision?.family === "main";
  getElement("opponent-name").textContent = game.model_label;
  getElement("opponent-last-turn").textContent = opponentTurnSummary(game, cards);
  for (const [id, value] of Object.entries({
    authority: observation.own_authority,
    trade: observation.trade,
    combat: observation.combat,
    deck: observation.own_deck_count,
    "opponent-authority": observation.opponent_authority,
    "opponent-deck": observation.opponent_hand_count + observation.opponent_deck_count,
    "must-discard": observation.pending_discard || 0,
    "opponent-must-discard": observation.opponent_pending_discard || 0,
    "opponent-trade": game.opponent_trade || 0,
    "opponent-combat": game.opponent_combat || 0
  })) {
    getElement(id).textContent = value;
  }
  getElement("opponent-win").textContent = Number.isFinite(game.opponent_win_probability) ?
      `${(game.opponent_win_probability * 100).toFixed(1)}%` : "—";
  getElement("turn").textContent = `TURN ${observation.turn}`;
  getElement("hand-count").textContent = `${observation.hand.length} cards`;
  getElement("supply-count").textContent = `${observation.trade_deck_count} trade cards · ${observation.explorers_remaining} Explorers`;
  getElement("discard-count").textContent = observation.own_discard.length;
  getElement("opponent-discard-count").textContent = observation.opponent_discard.length;
  const decisionForCard = (id, zoneName) => game.status === "your_turn" ? actions.filter(action => {
    if (action.kind === "choose_mode") {
      return zoneName === "in_play" && action.card_id === id;
    }
    if (["destroy_base", "free_acquire", "copy_ship", "scrap_trade_row"].includes(action.kind)) {
      return action.source_zone === zoneName && action.target_card_id === id;
    }
    return action.kind === "discard_card" && zoneName === "hand" && action.card_id === id;
  }) : [];
  const scrapForCard = (id, zoneName) => game.status === "your_turn"
    ? actions.filter(action =>
        action.kind === "scrap_card" &&
        action.card_id === id &&
        action.source_zone === zoneName
      ).slice(0, 1)
    : [];

  const forCard = (id, kinds, field = "card_id", zoneName) => main && game.status === "your_turn"
    ? actions.filter(action =>
        action[field] === id &&
        kinds.includes(action.kind) &&
        (!zoneName || action.source_zone === zoneName)
      )
    : [];

  const handCards = () => observation.hand.map(id => cardView(id, [
    ...forCard(id, ["play_card"]),
    ...scrapForCard(id, "hand"),
    ...decisionForCard(id, "hand")
  ]));
  zone("hand", handCards());
  zone("hand-inspector", handCards());

  zone("market", [
    ...observation.trade_row.map(id => id === null
      ? createElement("span", "empty", "Empty slot")
      : cardView(id, [
          ...forCard(id, ["acquire"], "card_id", "trade_row"),
          ...decisionForCard(id, "trade_row")
        ])
    ),
    ...(observation.explorers_remaining
      ? [cardView(
          2,
          forCard(2, ["acquire"], "card_id", "explorer_supply"),
          `${observation.explorers_remaining} available`
        )]
      : [])
  ]);

  zone("own-fleet", observation.own_in_play.map(inPlayCard => cardView(
    inPlayCard.card,
    [
      ...forCard(inPlayCard.card, ["activate_base", "activate_ally", "scrap_for_ability"]),
      ...decisionForCard(inPlayCard.card, "in_play")
    ],
    inPlayCard.copied_from_stealth_needle
      ? "Stealth Needle copy"
      : inPlayCard.ally_triggered ? "Ally used" : "In play"
  )));

  zone("opponent-fleet", observation.opponent_in_play.map(inPlayCard => cardView(
    inPlayCard.card,
    [
      ...forCard(inPlayCard.card, ["attack_base"], "target_card_id"),
      ...decisionForCard(inPlayCard.card, "opponent_in_play")
    ],
    cards[inPlayCard.card].defense
      ? `${cards[inPlayCard.card].card_type} · ${cards[inPlayCard.card].defense} defense`
      : "Ship"
  )));
  zone("discard", observation.own_discard.map(id => cardView(id, scrapForCard(id, "discard"))));

  const canScrapDiscard = actions.some(action => action.kind === "scrap_card" &&
      action.source_zone === "discard") && game.status === "your_turn";
  getElement("own-discard-pile").classList.toggle("actionable", canScrapDiscard);
  getElement("own-discard-pile").title = canScrapDiscard ? "Open discard pile to choose a card to scrap" : "Inspect your discard pile";
  zone("opponent-discard", observation.opponent_discard.map(id => cardView(id)));
  zone("scrap", observation.scrap_heap.map(id => cardView(id)));
  getElement("play-all").hidden = !game.can_play_all;
  getElement("play-all").title = "Play the cards currently in your hand, from left to right. Newly drawn cards remain in your hand.";
  const attack = game.status === "your_turn" && actions.find(action => action.kind === "attack_player");
  const end = actions.find(action => action.kind === "end_turn");
  getElement("attack").hidden = !attack;
  getElement("attack").textContent = attack ? `Attack · ${attack.amount} combat` : "Attack";
  getElement("attack").onclick = () => attack && move(attack.id);
  getElement("opponent-attack").dataset.unavailable = String(!attack);
  getElement("opponent-attack").disabled = !attack;
  getElement("opponent-attack").title = attack ? `Attack opponent with ${attack.amount} combat` : "Attack requires combat and no defending outposts; resolve any pending choice first.";
  getElement("opponent-attack").setAttribute("aria-label", attack ?
      `Opponent authority ${observation.opponent_authority}. Attack with ${attack.amount} combat` : `Opponent authority ${observation.opponent_authority}. Attack unavailable`);
  getElement("opponent-attack").onclick = () => attack && move(attack.id);
  getElement("end-turn").hidden = false;
  getElement("end-turn").dataset.unavailable = String(!end);
  getElement("end-turn").disabled = !end;
  getElement("end-turn").onclick = () => {
    if (observation.hand.length && !confirm("End your turn and discard the unplayed cards in your hand?")) {
      return;
    }
    if (end) {
      move(end.id);
    }
  };
  getElement("resign").hidden = game.status === "complete";
  getElement("resign").onclick = () => {
    if (busy || game.status === "complete" || !confirm("Are you sure you want to resign?")) {
      return;
    }
    request({
      op: "resign",
      id: game.id,
      revision: game.revision
    });
  };
  const decisionDialog = getElement("decision-dialog");
  const selectionDismissed = dismissedSelection === selectionKey();
  for (const id of ["scrap-close", "scrap-cancel", "decision-close", "decision-cancel"]) {
    getElement(id).dataset.unavailable = String(!game.can_undo);
    getElement(id).disabled = busy || !game.can_undo;
  }
  const needsDecision = !selectionDismissed && game.status === "your_turn" &&
      decision && !main && actions.length && !actions.some(action => ["scrap_card",
      "scrap_trade_row"].includes(action.kind));
  getElement("choices").replaceChildren();
  getElement("decision-options").replaceChildren();
  if (needsDecision || selectionDismissed && !main) {
    for (const action of actions) {
      const button = createElement("button", "choice");
      const id = action.target_card_id >= 0 ? action.target_card_id : action.card_id;
      if (id >= 0 && action.kind !== "decline") {
        const image = createElement("img");
        setImage(image, cardArtURL(cards[id]));
        image.alt = "";
        button.append(image);
      }
      button.append(createElement("span", "", action.label));
      button.addEventListener("click", () => move(action.id));
      getElement(selectionDismissed ? "choices" : "decision-options").append(button);
    }
  }
  getElement("choice-note").hidden = !(decision && !main);
  getElement("choice-note").textContent = "Resolve this choice to continue. Other actions become available afterward.";
  let status = game.status === "model_thinking" ? `${game.model_label} is playing…` : "Your turn · Play cards, use abilities, buy cards, or attack.";
  let title = decision ? main ? "Your move" : decision.prompt : "Computer’s turn";
  if (game.status === "complete") {
    status = game.result.resigned ? `You resigned. ${game.model_label} wins.` : game.result.truncated ?
        "Draw · the game reached its turn or action limit." : game.result.winner === 0 ?
        "Victory! You defeated the champion." : `${game.model_label} wins. Ready for a rematch?`;
    title = "Game complete";
  }
  getElement("status").textContent = status;
  getElement("decision-title").textContent = title;
  getElement("result-banner").hidden = game.status !== "complete";
  if (game.status === "complete") {
    const outcome = game.result.truncated ? "draw" : game.result.winner === 0 ? "win" : "loss";
    getElement("result-banner").dataset.outcome = outcome;
    getElement("result-title").textContent = {
      win: "Victory!",
      loss: "Defeat",
      draw: "Draw"
    }[outcome];
    getElement("result-message").textContent = status;
  }
  if (!needsDecision && decisionDialog.open) {
    decisionDialog.close();
  }
  renderScrap(actions);
  if (needsDecision) {
    getElement("decision-prompt").textContent = decision.prompt || "Choose an option";
    if (!decisionDialog.open) {
      for (const id of ["pile-dialog", "card-dialog"]) {
        if (getElement(id).open) {
          getElement(id).close();
        }
      }
      decisionDialog.showModal();
    }
  }
  if (openPile) {
    renderPile();
  }
  getElement("log").replaceChildren(...game.action_log.slice().reverse().map(entry => {
    const logItem = createElement("li", entry.player_id ? "computer" : "");
    logItem.append(createElement("b", "", `Turn ${entry.turn} · ${entry.player_id ?
        game.model_label : "You"}`), document.createTextNode(entry.label));
    return logItem;
  }));
}

// Shuffle a copy for display only; never change the game's RNG or hidden draw order.
function shuffled(ids) {
  const result = [...ids];
  for (let index = result.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

// Deck inspectors show composition without revealing the order of hidden cards.
function renderPile() {
  const observation = game.observation;
  const views = {
    "own-deck": ["Your deck", "own-deck-cards", [...(observation.own_deck ||
        []), ...(observation.own_known_top || [])]],
    "opponent-deck": ["Opponent’s hand + deck", "opponent-deck-cards",
        [...(observation.opponent_hidden || []), ...(observation.opponent_known_hand ||
        []), ...(observation.opponent_known_top || [])]],
    hand: ["Your hand", "hand-inspector", observation.hand],
    discard: ["Your discard pile", "discard", observation.own_discard],
    "opponent-discard": ["Opponent’s discard pile", "opponent-discard", observation.opponent_discard],
    scrap: ["Scrap heap", "scrap", observation.scrap_heap],
    log: ["Game log", "log", null]
  };
  const [title, id, ids] = views[openPile];
  const deck = openPile.endsWith("-deck");
  getElement("pile-title").textContent = title + (ids ? ` · ${ids.length} cards` : "");
  getElement("pile-note").textContent = openPile === "own-deck" ? "Your deck, shown in random order without revealing draw order." : deck ?
      "Hand and deck combined, shown in random order. This does not reveal which cards are in the opponent’s hand or the draw order." : openPile === "discard" ?
      "Cards here may be chosen when a scrap ability allows it." : "";
  for (const view of Object.values(views)) {
    getElement(view[1]).hidden = view[1] !== id;
  }
  if (deck) {
    zone(id, shuffled(ids).map(card => cardView(card)));
  }
}

function inspectPile(id) {
  if (!game) {
    return;
  }
  openPile = id;
  renderPile();
  getElement("pile-dialog").showModal();
}

function renderModels() {
  const selected = getElement("opponent").value;
  getElement("opponent").replaceChildren(...models.map(model => {
    const option = createElement("option", "", aliases[model.id] || model.name);
    option.value = model.id;
    return option;
  }));
  getElement("opponent").value = models.some(model => model.id === selected) ? selected : "level-05";
  getElement("start").disabled = !models.length;
  getElement("model-list").replaceChildren(...models.map(model => {
    const form = createElement("form", "rename-form");
    const input = createElement("input");
    input.value = aliases[model.id] || model.name;
    input.maxLength = 100;
    input.required = true;
    input.setAttribute("aria-label", `Name for ${model.name}`);
    const save = createElement("button", "", "Save name");
    form.append(input, save);
    form.addEventListener("submit", event => {
      event.preventDefault();
      try {
        aliases[model.id] = input.value.trim() || model.name;
        localStorage.setItem(aliasesKey, JSON.stringify(aliases));
        renderModels();
      } catch {
        showError("Local storage is unavailable.", "admin-error");
      }
    });
    form.append(createElement("p", "muted", model.description));
    return form;
  }));
}
var statsPrefix = "astro-results-v2:" + location.pathname.replace(/index\.html$/, "") + ":";
function recordStats() {
  if (game?.status !== "complete") {
    return;
  }
  try {
    recordResult(localStorage, statsPrefix, game);
  } catch {
    getElement("stats-error").hidden = false;
    getElement("stats-error").textContent = "Browser storage is unavailable. Results could not be saved.";
  }
}

function renderStats() {
  try {
    const stats = readStats(localStorage, statsPrefix);
    const opponents = new Map(Array.from({
      length: 5
    }, (unusedValue, index) => [`level-${index + 1}`, `Level ${index + 1}`]));
    for (const model of models) {
      opponents.set(model.id, aliases[model.id] || model.name);
    }
    for (const [id, row] of stats) {
      if (!opponents.has(id)) {
        opponents.set(id, row.label);
      }
    }
    let wins = 0;
    let losses = 0;
    let draws = 0;
    getElement("stats-rows").replaceChildren(...[...opponents].map(([id, name]) => {
      const row = stats.get(id) || {
        wins: 0,
        losses: 0,
        draws: 0
      };
      wins += row.wins;
      losses += row.losses;
      draws += row.draws;
      const tableRow = createElement("tr");
      const heading = createElement("th", "", name);
      heading.scope = "row";
      tableRow.append(heading);
      for (const value of [row.wins, row.losses, row.draws, row.wins + row.losses ?
          `${Math.round(100 * row.wins / (row.wins + row.losses))}%` : "—"]) {
        tableRow.append(createElement("td", "", value));
      }
      return tableRow;
    }));
    getElement("stats-summary").textContent = `${wins} wins · ${losses} losses · ${draws} draws`;
  } catch {
    getElement("stats-error").hidden = false;
    getElement("stats-error").textContent = "Browser storage is unavailable. Stats cannot be loaded.";
  }
}

function navigate() {
  const stats = location.hash === "#stats";
  getElement("game-page").hidden = stats;
  getElement("stats-page").hidden = !stats;
  document.body.classList.toggle("viewing-stats", stats);
  document.body.classList.toggle("playing", !!game && !stats && location.hash !== "#home");
  if (location.hash === "#home") {
    for (const id of ["pile-dialog", "card-dialog", "scrap-dialog", "decision-dialog"]) {
      if (getElement(id).open) {
        getElement(id).close();
      }
    }
    openPile = null;
  }
  if (cards.length) {
    render();
  }
  if (stats) {
    renderStats();
    getElement("stats-title").focus();
  }
}

// Fixed bars may wrap on narrow screens or with larger browser text sizes.
var barObserver = new ResizeObserver(entries => {
  for (const {
    target
  } of entries) {
    document.documentElement.style.setProperty(`--${target.id}-height`, `${target.getBoundingClientRect().height}px`);
  }
});
barObserver.observe(getElement("opponent-bar"));
barObserver.observe(getElement("player-bar"));
getElement("new-game").addEventListener("submit", event => {
  event.preventDefault();
  if (game && game.status !== "complete" && !confirm("Start a new game and replace this game?")) {
    return;
  }
  request({
    op: "new",
    model: getElement("opponent").value,
    label: aliases[getElement("opponent").value]
  });
});
getElement("opponent").addEventListener("change", async () => {
  const id = getElement("opponent").value;
  const model = models.find(candidateModel => candidateModel.id === id);
  const result = await request({
    op: "load",
    model: id
  });
  if (result) {
    getElement("status").textContent = `${aliases[id] || model.name} is ready. Start a new game to play this level.`;
  }
});
getElement("play-all").addEventListener("click", () => request({
  op: "play_all",
  id: game.id,
  revision: game.revision
}));
document.querySelectorAll("[data-inspect]").forEach(button => button.addEventListener("click",
    () => inspectPile(button.dataset.inspect)));
getElement("pile-dialog").addEventListener("close", () => {
  if (!getElement("pile-dialog").open) {
    openPile = null;
  }
});
for (const id of ["scrap-dialog", "decision-dialog"]) {
  getElement(id).addEventListener("cancel", event => {
    event.preventDefault();
    cancelSelection();
  });
}
for (const id of ["scrap-close", "scrap-cancel", "decision-close", "decision-cancel"]) {
  getElement(id).addEventListener("click", cancelSelection);
}
document.querySelectorAll("[data-open]").forEach(button => button.addEventListener("click",
    () => getElement(button.dataset.open).showModal()));
getElement("models-open").addEventListener("click", () => getElement("models-dialog").showModal());
document.querySelectorAll("[data-close]").forEach(button => button.addEventListener("click",
    () => getElement(button.dataset.close).close()));
getElement("upload-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const file = form.get("model_file");
  if (file.size > 64 * 1024 * 1024) {
    showError("Choose a model smaller than 64 MB.", "admin-error");
    return;
  }
  const result = await request({
    op: "import",
    name: form.get("name"),
    bytes: await file.arrayBuffer()
  }, "admin-error");
  if (result?.imported) {
    models.push(result.imported);
    renderModels();
    getElement("opponent").value = result.imported.id;
    getElement("upload-form").reset();
  }
});
// Public entry point used by the page bootstrap: start({ base, release }).
async function start(options) {
  window.addEventListener("hashchange", navigate);
  window.addEventListener("storage", renderStats);
  navigate();
  ({
    release: release,
    base: base
  } = options);
  configureResources(base, release);
  audio = new GameAudio();
  audio.preload();
  for (const event of ["pointerdown", "keydown", "click"]) {
    document.addEventListener(event, () => audio.unlock(), {
      capture: true
    });
  }
  getElement("audio-volume").addEventListener("input", event => {
    audio.setVolume(Number(event.target.value) / 100);
    event.target.setAttribute("aria-valuetext", event.target.value + "%");
  });
  savedKey = "astro-game:" + new URL(base).pathname;
  worker = new Worker(resourceURL("assets/worker.js"), {
    type: "module"
  });
  worker.onmessage = ({
    data: message
  }) => {
    if (message.progress) {
      getElement("status").textContent = message.progress;
      return;
    }
    const pendingRequest = pendingRequests.get(message.requestId);
    if (!pendingRequest) {
      return;
    }
    pendingRequests.delete(message.requestId);
    if (message.error) {
      pendingRequest.reject(new Error(message.error));
    } else {
      pendingRequest.resolve(message.data);
    }
  };
  worker.onerror = () => {
    for (const pendingRequest of pendingRequests.values()) {
      pendingRequest.reject(new Error("The game worker stopped. Reload to resume."));
    }
    pendingRequests.clear();
    setBusy(false);
    showError("The game worker stopped. Reload to resume.");
  };
  let saved = null;
  let changed = false;
  try {
    const stored = JSON.parse(localStorage.getItem(savedKey) || sessionStorage.getItem(savedKey) || "null");
    if (stored?.release === release) {
      saved = stored.saved;
    } else if (stored) {
      changed = true;
    }
  } catch {
    // Ignore unreadable saves and initialize a fresh game.
  }
  setBusy(true);
  try {
    const data = await sendWorkerRequest({
      op: "init",
      base: base,
      release: release,
      saved
    });
    cards = data.cards;
    models = data.models;
    try {
      models.push(...(await localModels("list")).map(({
        bytes,
        ...model
      }) => model));
    } catch {
      // Imported models are optional; gameplay does not require IndexedDB.
    }
    game = data.game;
    renderModels();
    render();
    renderStats();
    saveGame(data.saved);
    if (data.notice) {
      showError(data.notice);
    }
    if (changed) {
      getElement("status").textContent = "";
    }
  } catch (error) {
    showError(error.message);
    getElement("status").textContent = "Could not load the game. Reconnect and refresh to retry.";
  } finally {
    setBusy(false);
    getElement("start").disabled = !models.length;
    if (game?.status === "model_thinking") {
      scheduleModelTurn();
    }
  }
}
export { start };
