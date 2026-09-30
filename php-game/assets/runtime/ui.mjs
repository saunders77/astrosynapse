import { GameAudio } from './audio.mjs';
import { opponentTurnSummary } from './turn-summary.mjs';
import { recordResult, readStats } from './stats.mjs';
import { configureResources, resourceURL } from './resources.mjs';
import { localModels } from './local-models.mjs';
const $ = id => document.getElementById(id);
let audio;
let cards = [], models = [], game = null, busy = false, timer = null, openPile = null, worker, release, base, savedKey;
let dismissedSelection = null;
const selectionKey = () => game ? `${game.id}:${game.revision}` : null;
let requestId = 0; const waiting = new Map();
const aliasesKey = 'astro-model-names:' + location.pathname.replace(/index\.html$/, '');
let aliases = {}; try { aliases = JSON.parse(localStorage.getItem(aliasesKey) || '{}'); } catch {}
// Core Set filenames verified against https://www.starrealms.com/card-gallery/data/cards.json
const art = c => `https://www.starrealms.com/card-gallery/images/content/card-gallery/${c.name.toLowerCase().replaceAll(' ', '-')}.webp`;
const el = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; };
const showError = (message, id = 'error') => { $(id).textContent = message; $(id).hidden = !message; };
function lock(value) { busy = value; document.body.classList.toggle('busy', value); document.querySelectorAll('button, select, input').forEach(n => n.disabled = (value && !n.matches('.details, .pile, [data-close], [data-inspect], [data-open], #audio-volume')) || n.dataset.unavailable === 'true'); }
function api(payload) {
  return new Promise((resolve, reject) => {
    const id = ++requestId; waiting.set(id, { resolve, reject }); worker.postMessage({ ...payload, requestId: id });
  });
}
function saveGame(saved) {
  try { if (saved) localStorage.setItem(savedKey, JSON.stringify({ release, saved })); else localStorage.removeItem(savedKey); }
  catch { showError('Browser storage is unavailable. This game works, but cannot resume after leaving this page.'); }
}
const imageObserver = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) { imageObserver.unobserve(entry.target); loadImage(entry.target, entry.target.dataset.art); }
}, { rootMargin: '150px' });
function loadImage(img, path) {
  img.onerror = () => { img.title = 'Card image unavailable. Reconnect to try again.'; };
  img.onload = () => { img.title = ''; };
  img.src = path;
}
function setImage(img, path, immediate = false) {
  img.dataset.art = path;
  if (immediate) loadImage(img, path); else imageObserver.observe(img);
}
async function request(payload, errorTarget = 'error') {
  if (busy) return;
  clearTimeout(timer); showError('', errorTarget); lock(true);
  try {
    const data = await api(payload);
    if (data.models) { models = data.models; renderModels(); }
    if ('saved' in data) saveGame(data.saved);
    if (data.notice) showError(data.notice);
    if ('game' in data) {
      if (data.game) {
        const sameGame = game?.id === data.game.id;
        if (!sameGame || payload.op === 'undo') audio.clear();
        // Reconstructed history is deterministic; only newly applied effects play.
        audio.enqueue(data.game.sounds.slice(sameGame ? game.sounds.length : 0));
      }
      game = data.game;
      if (payload.op === 'undo') dismissedSelection = selectionKey();
      render(); renderStats();
    }
    return data;
  } catch (e) { showError(e.message, errorTarget); $('status').textContent = 'Paused. Your last saved game is preserved. Retry or refresh to resume.'; }
  finally { lock(false); if (game?.status === 'model_thinking' && !$('error').textContent) schedule(); }
}
function schedule() { clearTimeout(timer); timer = setTimeout(() => { if (!busy && game?.status === 'model_thinking') request({ op: 'advance', id: game.id, revision: game.revision }); }, 120); }
function move(id) { if (busy) return; if (openPile === 'hand') $('pile-dialog').close(); request({ op: 'choose', id: game.id, revision: game.revision, action_id: id }, $('decision-dialog').open ? 'decision-error' : 'error'); }
function cancelSelection() {
  if (busy || !game?.can_undo) return;
  const errorTarget = $('scrap-dialog').open ? 'scrap-error' : 'decision-error';
  return request({ op: 'undo', id: game.id, revision: game.revision }, errorTarget);
}
function inspect(card) {
  setImage($('card-large'), art(card), true); $('card-large').alt = card.name; $('card-name').textContent = card.name;
  $('card-description').textContent = `${card.faction.replaceAll('_', ' ')} · ${card.card_type} · Cost ${card.cost}${card.defense ? ` · Defense ${card.defense}` : ''}`;
  $('card-dialog').showModal();
}
function actionName(a) {
  const names = { scrap_card: 'SCRAP', play_card: 'Play', acquire: `Acquire · ${a.amount} trade`, activate_base: 'Use ability', activate_ally: 'Use ally', scrap_for_ability: 'Scrap for ability', attack_base: `Attack base · ${a.amount} combat`, destroy_base: 'DESTROY', free_acquire: 'ACQUIRE FREE', copy_ship: 'COPY', scrap_trade_row: 'SCRAP', discard_card: 'DISCARD' };
  return names[a.kind] || a.label;
}
function renderScrap(actions) {
  const targets = game?.status === 'your_turn' ? actions.filter(a => ['scrap_card', 'scrap_trade_row'].includes(a.kind)) : [];
  const dialog = $('scrap-dialog');
  if (!targets.length || dismissedSelection === selectionKey()) { if (dialog.open) dialog.close(); return; }
  $('scrap-prompt').textContent = game.decision.prompt || 'Choose a card to scrap';
  for (const [zoneName, title] of [['hand', 'Your hand'], ['discard', 'Your discard pile'], ['trade_row', 'Trade row']]) {
    const options = targets.filter(a => a.source_zone === zoneName);
    const section = $('scrap-options-' + zoneName); section.hidden = !options.length;
    section.replaceChildren(el('h3', '', title));
    const list = el('div', 'cards');
    list.append(...options.map(a => cardView(a.kind === 'scrap_trade_row' ? a.target_card_id : a.card_id, [a])));
    section.append(list);
  }
  $('scrap-decline').replaceChildren(...actions.filter(a => a.kind === 'decline').map(a => {
    const button = el('button', 'move', a.label); button.addEventListener('click', () => move(a.id)); return button;
  }));
  if (!dialog.open) {
    if ($('pile-dialog').open) $('pile-dialog').close();
    dialog.showModal();
  }
}
function cardView(id, actions = [], state = '') {
  const c = cards[id]; if (!c) return el('span', 'empty', 'Empty trade slot');
  const node = el('article', `card ${c.card_type !== 'ship' ? 'base' : ''} ${actions.length ? 'actionable' : ''}`);
  node.dataset.faction = c.faction;
  const face = el('button', 'card-face'); face.type = 'button';
  const action = actions.length === 1 ? actions[0] : null;
  face.setAttribute('aria-label', action ? (action.label || `${actionName(action)} ${c.name}`) : `Details for ${c.name}`);
  face.title = face.getAttribute('aria-label');
  face.addEventListener('click', () => action ? move(action.id) : inspect(c));
  const img = el('img'); setImage(img, art(c)); img.alt = c.name; img.loading = 'lazy'; face.append(img); node.append(face);
  const meta = el('div', 'card-meta');
  const heading = el('div', 'card-heading');
  const cost = el('span', 'card-cost', c.cost); cost.setAttribute('aria-label', `Cost ${c.cost}`);
  heading.append(el('span', 'card-name', c.name), cost); meta.append(heading);
  if (state) meta.append(el('span', 'card-state', state));
  const controls = el('div', 'card-actions');
  for (const a of actions) {
    const b = el('button', 'move', actionName(a)); b.type = 'button'; b.title = a.label; b.addEventListener('click', () => move(a.id)); controls.append(b);
  }
  const detail = el('button', 'details', 'Details'); detail.type = 'button'; detail.setAttribute('aria-label', `Details for ${c.name}`); detail.addEventListener('click', () => inspect(c)); controls.append(detail);
  meta.append(controls); node.append(meta); return node;
}
function zone(id, entries) { const node = $(id); node.replaceChildren(...entries); if (!entries.length) node.append(el('span', 'empty', 'No cards')); }
function render() {
  document.body.classList.toggle('active-game', !!game && game.status !== 'complete');
  recordStats();
  imageObserver.disconnect();
  document.body.classList.toggle('playing', !!game && location.hash !== '#stats');
  if (!game) setImage($('welcome-art'), art({ name: 'Scout' }));
  $('opponent-bar').hidden = !game; $('player-bar').hidden = !game;
  $('welcome').hidden = !!game; $('board').hidden = !game;
  if (!game) { $('status').textContent = ''; return; }
  const o = game.observation, d = game.decision, actions = d?.actions || [], main = d?.family === 'main';
  $('opponent-name').textContent = game.model_label;
  $('opponent-last-turn').textContent = opponentTurnSummary(game, cards);
  for (const [id, value] of Object.entries({ authority: o.own_authority, trade: o.trade, combat: o.combat, deck: o.own_deck_count, 'opponent-authority': o.opponent_authority, 'opponent-hand': o.opponent_hand_count, 'opponent-deck': o.opponent_deck_count, 'must-discard': o.pending_discard || 0, 'opponent-must-discard': o.opponent_pending_discard || 0, 'opponent-trade': game.opponent_trade || 0, 'opponent-combat': game.opponent_combat || 0, 'player-hand': o.hand.length })) $(id).textContent = value;
  $('opponent-win').textContent = Number.isFinite(game.opponent_win_probability) ? `${(game.opponent_win_probability * 100).toFixed(1)}%` : '—';
  $('turn').textContent = `TURN ${o.turn}`;
  $('hand-count').textContent = `${o.hand.length} cards`;
  $('supply-count').textContent = `${o.trade_deck_count} trade cards · ${o.explorers_remaining} Explorers`;
  $('discard-count').textContent = o.own_discard.length;
  $('opponent-discard-count').textContent = o.opponent_discard.length;
  const decisionForCard = (id, zoneName) => game.status === 'your_turn' ? actions.filter(a => {
    if (a.kind === 'choose_mode') return zoneName === 'in_play' && a.card_id === id;
    if (['destroy_base', 'free_acquire', 'copy_ship', 'scrap_trade_row'].includes(a.kind)) return a.source_zone === zoneName && a.target_card_id === id;
    return a.kind === 'discard_card' && zoneName === 'hand' && a.card_id === id;
  }) : [];
  const scrapForCard = (id, zoneName) => game.status === 'your_turn' ? actions.filter(a => a.kind === 'scrap_card' && a.card_id === id && a.source_zone === zoneName).slice(0, 1) : [];
  const forCard = (id, kinds, field = 'card_id', zoneName) => main && game.status === 'your_turn' ? actions.filter(a => a[field] === id && kinds.includes(a.kind) && (!zoneName || a.source_zone === zoneName)) : [];
  const handCards = () => o.hand.map(id => cardView(id, [...forCard(id, ['play_card']), ...scrapForCard(id, 'hand'), ...decisionForCard(id, 'hand')]));
  zone('hand', handCards()); zone('hand-inspector', handCards());
  zone('market', [...o.trade_row.map(id => id === null ? el('span', 'empty', 'Empty slot') : cardView(id, [...forCard(id, ['acquire'], 'card_id', 'trade_row'), ...decisionForCard(id, 'trade_row')])), ...(o.explorers_remaining ? [cardView(2, forCard(2, ['acquire'], 'card_id', 'explorer_supply'), `${o.explorers_remaining} available`)] : [])]);
  zone('own-fleet', o.own_in_play.map(i => cardView(i.card, [...forCard(i.card, ['activate_base', 'activate_ally', 'scrap_for_ability']), ...decisionForCard(i.card, 'in_play')], i.copied_from_stealth_needle ? 'Stealth Needle copy' : i.ally_triggered ? 'Ally used' : 'In play')));
  zone('opponent-fleet', o.opponent_in_play.map(i => cardView(i.card, [...forCard(i.card, ['attack_base'], 'target_card_id'), ...decisionForCard(i.card, 'opponent_in_play')], cards[i.card].defense ? `${cards[i.card].card_type} · ${cards[i.card].defense} defense` : 'Ship')));
  zone('discard', o.own_discard.map(id => cardView(id, scrapForCard(id, 'discard'))));
  const canScrapDiscard = actions.some(a => a.kind === 'scrap_card' && a.source_zone === 'discard') && game.status === 'your_turn';
  $('own-discard-pile').classList.toggle('actionable', canScrapDiscard);
  $('own-discard-pile').title = canScrapDiscard ? 'Open discard pile to choose a card to scrap' : 'Inspect your discard pile';
  zone('opponent-discard', o.opponent_discard.map(id => cardView(id)));
  zone('scrap', o.scrap_heap.map(id => cardView(id)));
  $('play-all').hidden = !game.can_play_all;
  $('play-all').title = 'Play the cards currently in your hand, from left to right. Newly drawn cards remain in your hand.';
  const attack = game.status === 'your_turn' && actions.find(a => a.kind === 'attack_player'), end = actions.find(a => a.kind === 'end_turn');
  $('attack').hidden = !attack; $('attack').textContent = attack ? `Attack · ${attack.amount} combat` : 'Attack'; $('attack').onclick = () => attack && move(attack.id);
  $('opponent-attack').dataset.unavailable = String(!attack);
  $('opponent-attack').disabled = !attack;
  $('opponent-attack').title = attack ? `Attack opponent with ${attack.amount} combat` : 'Attack requires combat and no defending outposts; resolve any pending choice first.';
  $('opponent-attack').setAttribute('aria-label', attack ? `Opponent authority ${o.opponent_authority}. Attack with ${attack.amount} combat` : `Opponent authority ${o.opponent_authority}. Attack unavailable`);
  $('opponent-attack').onclick = () => attack && move(attack.id);
  $('end-turn').hidden = false; $('end-turn').dataset.unavailable = String(!end); $('end-turn').disabled = !end; $('end-turn').onclick = () => { if (o.hand.length && !confirm('End your turn and discard the unplayed cards in your hand?')) return; if (end) move(end.id); };
  $('resign').hidden = game.status === 'complete';
  $('resign').onclick = () => { if (busy || game.status === 'complete' || !confirm('Are you sure you want to resign?')) return; request({ op: 'resign', id: game.id, revision: game.revision }); };
  const decisionDialog = $('decision-dialog');
  const selectionDismissed = dismissedSelection === selectionKey();
  for (const id of ['scrap-close', 'scrap-cancel', 'decision-close', 'decision-cancel']) {
    $(id).dataset.unavailable = String(!game.can_undo);
    $(id).disabled = busy || !game.can_undo;
  }
  const needsDecision = !selectionDismissed && game.status === 'your_turn' && d && !main && actions.length && !actions.some(a => ['scrap_card', 'scrap_trade_row'].includes(a.kind));
  $('choices').replaceChildren();
  $('decision-options').replaceChildren();
  if (needsDecision || (selectionDismissed && !main)) for (const a of actions) {
    const button = el('button', 'choice'); const id = a.target_card_id >= 0 ? a.target_card_id : a.card_id;
    if (id >= 0 && a.kind !== 'decline') { const img = el('img'); setImage(img, art(cards[id])); img.alt = ''; button.append(img); }
    button.append(el('span', '', a.label)); button.addEventListener('click', () => move(a.id)); $(selectionDismissed ? 'choices' : 'decision-options').append(button);
  }
  $('choice-note').hidden = !(d && !main); $('choice-note').textContent = 'Resolve this choice to continue. Other actions become available afterward.';
  let status = game.status === 'model_thinking' ? `${game.model_label} is playing…` : 'Your turn · Play cards, use abilities, buy cards, or attack.';
  let title = d ? (main ? 'Your move' : d.prompt) : 'Computer’s turn';
  if (game.status === 'complete') { status = game.result.resigned ? `You resigned. ${game.model_label} wins.` : game.result.truncated ? 'Draw · the game reached its turn or action limit.' : game.result.winner === 0 ? 'Victory! You defeated the champion.' : `${game.model_label} wins. Ready for a rematch?`; title = 'Game complete'; }
  $('status').textContent = status; $('decision-title').textContent = title;
  $('result-banner').hidden = game.status !== 'complete';
  if (game.status === 'complete') {
    const outcome = game.result.truncated ? 'draw' : game.result.winner === 0 ? 'win' : 'loss';
    $('result-banner').dataset.outcome = outcome;
    $('result-title').textContent = {win: 'Victory!', loss: 'Defeat', draw: 'Draw'}[outcome];
    $('result-message').textContent = status;
  }
  if (!needsDecision && decisionDialog.open) decisionDialog.close();
  renderScrap(actions);
  if (needsDecision) {
    $('decision-prompt').textContent = d.prompt || 'Choose an option';
    if (!decisionDialog.open) {
      for (const id of ['pile-dialog', 'card-dialog']) if ($(id).open) $(id).close();
      decisionDialog.showModal();
    }
  }
  if (openPile) renderPile();
  $('log').replaceChildren(...game.action_log.slice().reverse().map(entry => { const li = el('li', entry.player_id ? 'computer' : ''); li.append(el('b', '', `Turn ${entry.turn} · ${entry.player_id ? game.model_label : 'You'}`), document.createTextNode(entry.label)); return li; }));
}
// Shuffle only the public composition, never the game's RNG or hidden zone order.
function shuffled(ids) {
  const result = [...ids];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
function renderPile() {
  const o = game.observation;
  const views = {
    'own-deck': ['Your hand + deck', 'own-deck-cards', [...o.hand, ...(o.own_deck || []), ...(o.own_known_top || [])]],
    'opponent-deck': ['Opponent’s hand + deck', 'opponent-deck-cards', [...(o.opponent_hidden || []), ...(o.opponent_known_hand || []), ...(o.opponent_known_top || [])]],
    hand: ['Your hand', 'hand-inspector', o.hand],
    discard: ['Your discard pile', 'discard', o.own_discard],
    'opponent-discard': ['Opponent’s discard pile', 'opponent-discard', o.opponent_discard],
    scrap: ['Scrap heap', 'scrap', o.scrap_heap],
    log: ['Game log', 'log', null],
  };
  const [title, id, ids] = views[openPile], deck = openPile.endsWith('-deck');
  $('pile-title').textContent = title + (ids ? ` · ${ids.length} cards` : '');
  $('pile-note').textContent = deck ? 'Hand and deck combined, shown in random order. This does not reveal which cards are in the opponent’s hand or the draw order.' : openPile === 'discard' ? 'Cards here may be chosen when a scrap ability allows it.' : '';
  for (const view of Object.values(views)) $(view[1]).hidden = view[1] !== id;
  if (deck) zone(id, shuffled(ids).map(card => cardView(card)));
}
function inspectPile(id) {
  if (!game) return;
  openPile = id; renderPile(); $('pile-dialog').showModal();
}
function renderModels() {
  const selected = $('opponent').value;
  $('opponent').replaceChildren(...models.map(m => { const option = el('option', '', aliases[m.id] || m.name); option.value = m.id; return option; }));
  $('opponent').value = models.some(m => m.id === selected) ? selected : 'level-05';
  $('start').disabled = !models.length;
  $('model-list').replaceChildren(...models.map(m => {
    const form = el('form', 'rename-form'), input = el('input'); input.value = aliases[m.id] || m.name; input.maxLength = 100; input.required = true; input.setAttribute('aria-label', `Name for ${m.name}`);
    const save = el('button', '', 'Save name'); form.append(input, save);
    form.addEventListener('submit', e => { e.preventDefault(); try { aliases[m.id] = input.value.trim() || m.name; localStorage.setItem(aliasesKey, JSON.stringify(aliases)); renderModels(); } catch { showError('Local storage is unavailable.', 'admin-error'); } });
    form.append(el('p', 'muted', m.description)); return form;
  }));
}
const statsPrefix = 'astro-results-v2:' + location.pathname.replace(/index\.html$/, '') + ':';
function recordStats() {
  if (game?.status !== 'complete') return;
  try { recordResult(localStorage, statsPrefix, game); }
  catch { $('stats-error').hidden = false; $('stats-error').textContent = 'Browser storage is unavailable. Results could not be saved.'; }
}
function renderStats() {
  try {
    const stats = readStats(localStorage, statsPrefix);
    const opponents = new Map(Array.from({length: 5}, (_, i) => [`level-${i + 1}`, `Level ${i + 1}`]));
    for (const m of models) opponents.set(m.id, aliases[m.id] || m.name);
    for (const [id, row] of stats) if (!opponents.has(id)) opponents.set(id, row.label);
    let wins = 0, losses = 0, draws = 0;
    $('stats-rows').replaceChildren(...[...opponents].map(([id, name]) => {
      const row = stats.get(id) || {wins: 0, losses: 0, draws: 0};
      wins += row.wins; losses += row.losses; draws += row.draws;
      const tr = el('tr'), heading = el('th', '', name); heading.scope = 'row'; tr.append(heading);
      for (const value of [row.wins, row.losses, row.draws, row.wins + row.losses ? `${Math.round(100 * row.wins / (row.wins + row.losses))}%` : '—']) tr.append(el('td', '', value));
      return tr;
    }));
    $('stats-summary').textContent = `${wins} wins · ${losses} losses · ${draws} draws`;
  } catch { $('stats-error').hidden = false; $('stats-error').textContent = 'Browser storage is unavailable. Stats cannot be loaded.'; }
}
function navigate() {
  const stats = location.hash === '#stats';
  $('game-page').hidden = stats; $('stats-page').hidden = !stats;
  document.body.classList.toggle('viewing-stats', stats);
  document.body.classList.toggle('playing', !!game && !stats);
  if (stats) { renderStats(); $('stats-title').focus(); }
}
// Fixed bars may wrap on narrow screens or with larger browser text sizes.
const barObserver = new ResizeObserver(entries => {
  for (const {target} of entries) document.documentElement.style.setProperty(`--${target.id}-height`, `${target.getBoundingClientRect().height}px`);
});
barObserver.observe($('opponent-bar')); barObserver.observe($('player-bar'));
$('new-game').addEventListener('submit', e => { e.preventDefault(); if (game && game.status !== 'complete' && !confirm('Start a new game and replace this game?')) return; request({ op: 'new', model: $('opponent').value, label: aliases[$('opponent').value] }); });
$('opponent').addEventListener('change', async () => {
  const id = $('opponent').value, m = models.find(m => m.id === id);
  const result = await request({ op: 'load', model: id });
  if (result) { $('status').textContent = `${aliases[id] || m.name} is ready. Start a new game to play this level.`; }
});
$('play-all').addEventListener('click', () => request({ op: 'play_all', id: game.id, revision: game.revision }));
document.querySelectorAll('[data-inspect]').forEach(b => b.addEventListener('click', () => inspectPile(b.dataset.inspect)));
$('pile-dialog').addEventListener('close', () => { if (!$('pile-dialog').open) openPile = null; });
for (const id of ['scrap-dialog', 'decision-dialog']) $(id).addEventListener('cancel', e => { e.preventDefault(); cancelSelection(); });
for (const id of ['scrap-close', 'scrap-cancel', 'decision-close', 'decision-cancel']) $(id).addEventListener('click', cancelSelection);
document.querySelectorAll('[data-open]').forEach(b => b.addEventListener('click', () => $(b.dataset.open).showModal()));
$('models-open').addEventListener('click', () => $('models-dialog').showModal());
document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => $(b.dataset.close).close()));
$('upload-form').addEventListener('submit', async e => {
  e.preventDefault(); const form = new FormData(e.currentTarget), file = form.get('model_file');
  if (file.size > 64 * 1024 * 1024) { showError('Choose a model smaller than 64 MB.', 'admin-error'); return; }
  const result = await request({ op: 'import', name: form.get('name'), bytes: await file.arrayBuffer() }, 'admin-error');
  if (result?.imported) { models.push(result.imported); renderModels(); $('opponent').value = result.imported.id; $('upload-form').reset(); }
});
export async function start(options) {
  window.addEventListener('hashchange', navigate);
  window.addEventListener('storage', renderStats);
  navigate();
  ({ release, base } = options); configureResources(base, release);
  audio = new GameAudio();
  audio.preload();
  for (const event of ['pointerdown', 'keydown', 'click']) document.addEventListener(event, () => audio.unlock(), { capture: true });
  $('audio-volume').addEventListener('input', e => {
    audio.setVolume(Number(e.target.value) / 100);
    e.target.setAttribute('aria-valuetext', e.target.value + '%');
  });
  savedKey = 'astro-game:' + new URL(base).pathname;
  worker = new Worker(resourceURL('assets/worker.js'), { type: 'module' });
  worker.onmessage = ({ data: message }) => {
    if (message.progress) { $('status').textContent = message.progress; return; }
    const pending = waiting.get(message.requestId); if (!pending) return; waiting.delete(message.requestId);
    if (message.error) pending.reject(new Error(message.error)); else pending.resolve(message.data);
  };
  worker.onerror = () => { for (const pending of waiting.values()) pending.reject(new Error('The game worker stopped. Reload to resume.')); waiting.clear(); lock(false); showError('The game worker stopped. Reload to resume.'); };
  let saved = null, changed = false;
  try { const stored = JSON.parse(localStorage.getItem(savedKey) || sessionStorage.getItem(savedKey) || 'null'); if (stored?.release === release) saved = stored.saved; else if (stored) changed = true; } catch {}
  lock(true);
  try {
    const data = await api({ op: 'init', base, release, saved }); cards = data.cards; models = data.models;
    try { models.push(...(await localModels('list')).map(({ bytes, ...m }) => m)); } catch { /* Gameplay does not require IndexedDB. */ }
    game = data.game; renderModels(); render(); renderStats(); saveGame(data.saved);
    if (data.notice) showError(data.notice);
    if (changed) $('status').textContent = '';
  } catch (e) { showError(e.message); $('status').textContent = 'Could not load the game. Reconnect and refresh to retry.'; }
  finally { lock(false); $('start').disabled = !models.length; if (game?.status === 'model_thinking') schedule(); }
}
