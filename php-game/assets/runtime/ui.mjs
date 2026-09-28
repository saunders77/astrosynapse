import { configureResources, resource, resourceURL, sha256 } from './resources.mjs';
import { localModels } from './local-models.mjs';
const $ = id => document.getElementById(id);
let cards = [], models = [], game = null, busy = false, timer = null, worker, release, base, savedKey;
let requestId = 0; const waiting = new Map(), imageMemory = new Map();
const aliasesKey = 'astro-model-names:' + location.pathname.replace(/index\.html$/, '');
let aliases = {}; try { aliases = JSON.parse(localStorage.getItem(aliasesKey) || '{}'); } catch {}
const art = c => `assets/card-art/${c.card_id === 4 ? 'BattlePod' : c.name.replaceAll(' ', '-')}.jpg`;
const el = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; };
const showError = (message, id = 'error') => { $(id).textContent = message; $(id).hidden = !message; };
function lock(value) { busy = value; document.body.classList.toggle('busy', value); document.querySelectorAll('button, select, input').forEach(n => n.disabled = value); }
function api(payload) {
  return new Promise((resolve, reject) => {
    const id = ++requestId; waiting.set(id, { resolve, reject }); worker.postMessage({ ...payload, requestId: id });
  });
}
function saveGame(saved) {
  try { if (saved) sessionStorage.setItem(savedKey, JSON.stringify({ release, saved })); else sessionStorage.removeItem(savedKey); }
  catch { showError('Browser storage is unavailable. This game works, but cannot resume after a refresh.'); }
}
const imageObserver = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) { imageObserver.unobserve(entry.target); loadImage(entry.target, entry.target.dataset.art); }
}, { rootMargin: '150px' });
function loadImage(img, path) {
  if (!imageMemory.has(path)) imageMemory.set(path, resource(path).then(bytes => URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }))).catch(e => { imageMemory.delete(path); throw e; }));
  imageMemory.get(path).then(url => { if (img.dataset.art === path) img.src = url; }).catch(() => { img.title = 'Image unavailable offline. Reconnect to download it.'; });
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
    if ('game' in data) { game = data.game; render(); }
    return data;
  } catch (e) { showError(e.message, errorTarget); $('status').textContent = 'Paused. Your last saved game is preserved. Retry or refresh to resume.'; }
  finally { lock(false); if (game?.status === 'model_thinking' && !$('error').textContent) schedule(); }
}
function schedule() { clearTimeout(timer); timer = setTimeout(() => { if (!busy && game?.status === 'model_thinking') request({ op: 'advance', id: game.id, revision: game.revision }); }, 120); }
function move(id) { request({ op: 'choose', id: game.id, revision: game.revision, action_id: id }); }
function inspect(card) {
  setImage($('card-large'), art(card), true); $('card-large').alt = card.name; $('card-name').textContent = card.name;
  $('card-description').textContent = `${card.faction.replaceAll('_', ' ')} · ${card.card_type} · Cost ${card.cost}${card.defense ? ` · Defense ${card.defense}` : ''}`;
  $('card-dialog').showModal();
}
function cardView(id, actions = [], state = '') {
  const c = cards[id]; if (!c) return el('span', 'empty', 'Empty trade slot');
  const node = el('article', `card ${c.card_type !== 'ship' ? 'base' : ''} ${actions.length ? 'actionable' : ''}`);
  const img = el('img'); setImage(img, art(c)); img.alt = c.name; img.loading = 'lazy'; node.append(img);
  const meta = el('div', 'card-meta'); meta.append(el('span', 'card-name', c.name)); if (state) meta.append(el('span', 'card-state', state));
  const controls = el('div', 'card-actions');
  for (const a of actions) {
    const names = { scrap_card: 'SCRAP', play_card: 'Play', acquire: `Buy · ${a.amount} trade`, activate_base: 'Use ability', activate_ally: 'Use ally', scrap_for_ability: 'Scrap for ability', attack_base: `Attack · ${a.amount} combat`, destroy_base: 'SELECT TARGET', free_acquire: 'SELECT TARGET', copy_ship: 'SELECT TARGET', scrap_trade_row: 'SELECT TARGET', discard_card: 'SELECT TARGET' };
    const b = el('button', 'move', names[a.kind] || a.label); b.type = 'button'; b.title = a.label; b.addEventListener('click', () => move(a.id)); controls.append(b);
  }
  const detail = el('button', 'details', 'Details'); detail.type = 'button'; detail.setAttribute('aria-label', `Details for ${c.name}`); detail.addEventListener('click', () => inspect(c)); controls.append(detail);
  meta.append(controls); node.append(meta); return node;
}
function zone(id, entries) { const node = $(id); node.replaceChildren(...entries); if (!entries.length) node.append(el('span', 'empty', 'No cards')); }
function render() {
  imageObserver.disconnect();
  if (!game) setImage($('welcome-art'), 'assets/card-art/Star-Realms-Back.jpg');
  $('welcome').hidden = !!game; $('board').hidden = !game;
  if (!game) { $('status').textContent = 'Choose your opponent to begin.'; return; }
  const o = game.observation, d = game.decision, actions = d?.actions || [], main = d?.family === 'main';
  $('opponent-name').textContent = game.model_label;
  for (const [id, value] of Object.entries({ authority: o.own_authority, trade: o.trade, combat: o.combat, deck: o.own_deck_count, 'opponent-authority': o.opponent_authority, 'opponent-hand': o.opponent_hand_count, 'opponent-deck': o.opponent_deck_count })) $(id).textContent = value;
  $('turn').textContent = `TURN ${o.turn}`;
  $('hand-count').textContent = `${o.hand.length} cards`;
  $('supply-count').textContent = `${o.trade_deck_count} trade cards · ${o.explorers_remaining} Explorers`;
  $('discard-count').textContent = `(${o.own_discard.length})`;
  $('opponent-discard-count').textContent = `${o.opponent_discard.length} in discard`;
  const decisionForCard = (id, zoneName) => game.status === 'your_turn' ? actions.filter(a => {
    if (a.kind === 'choose_mode') return zoneName === 'in_play' && a.card_id === id;
    if (['destroy_base', 'free_acquire', 'copy_ship', 'scrap_trade_row'].includes(a.kind)) return a.source_zone === zoneName && a.target_card_id === id;
    return a.kind === 'discard_card' && zoneName === 'hand' && a.card_id === id;
  }) : [];
  const scrapForCard = (id, zoneName) => game.status === 'your_turn' ? actions.filter(a => a.kind === 'scrap_card' && a.card_id === id && a.source_zone === zoneName).slice(0, 1) : [];
  const forCard = (id, kinds, field = 'card_id', zoneName) => main ? actions.filter(a => a[field] === id && kinds.includes(a.kind) && (!zoneName || a.source_zone === zoneName)) : [];
  zone('hand', o.hand.map(id => cardView(id, [...forCard(id, ['play_card']), ...scrapForCard(id, 'hand'), ...decisionForCard(id, 'hand')])));
  zone('market', [...o.trade_row.map(id => id === null ? el('span', 'empty', 'Empty slot') : cardView(id, [...forCard(id, ['acquire'], 'card_id', 'trade_row'), ...decisionForCard(id, 'trade_row')], `Cost ${cards[id].cost}`)), ...(o.explorers_remaining ? [cardView(2, forCard(2, ['acquire'], 'card_id', 'explorer_supply'), `${o.explorers_remaining} available`)] : [])]);
  zone('own-fleet', o.own_in_play.map(i => cardView(i.card, [...forCard(i.card, ['activate_base', 'activate_ally', 'scrap_for_ability']), ...decisionForCard(i.card, 'in_play')], i.copied_from_stealth_needle ? 'Stealth Needle copy' : i.ally_triggered ? 'Ally used' : 'In play')));
  zone('opponent-fleet', o.opponent_in_play.map(i => cardView(i.card, [...forCard(i.card, ['attack_base'], 'target_card_id'), ...decisionForCard(i.card, 'opponent_in_play')], cards[i.card].defense ? `${cards[i.card].card_type} · ${cards[i.card].defense} defense` : 'Ship')));
  zone('discard', o.own_discard.map(id => cardView(id, scrapForCard(id, 'discard'))));
  if (actions.some(a => a.kind === 'scrap_card' && a.source_zone === 'discard') && game.status === 'your_turn') $('discard').closest('details').open = true;
  zone('opponent-discard', o.opponent_discard.map(id => cardView(id)));
  zone('scrap', o.scrap_heap.map(id => cardView(id)));
  $('play-all').hidden = !game.can_play_all;
  $('play-all').title = 'Play the cards currently in your hand, from left to right. Newly drawn cards remain in your hand.';
  const attack = actions.find(a => a.kind === 'attack_player'), end = actions.find(a => a.kind === 'end_turn');
  $('attack').hidden = !attack; $('attack').textContent = attack ? `Attack · ${attack.amount} combat` : 'Attack'; $('attack').onclick = () => attack && move(attack.id);
  $('end-turn').hidden = !end; $('end-turn').onclick = () => { if (o.hand.length && !confirm('End your turn and discard the unplayed cards in your hand?')) return; if (end) move(end.id); };
  $('choices').replaceChildren();
  if (d && !main) for (const a of actions) {
    const button = el('button', 'choice'); const id = a.target_card_id >= 0 ? a.target_card_id : a.card_id;
    if (id >= 0 && a.kind !== 'decline') { const img = el('img'); setImage(img, art(cards[id])); img.alt = ''; button.append(img); }
    button.append(el('span', '', a.label)); button.addEventListener('click', () => move(a.id)); $('choices').append(button);
  }
  $('choice-note').hidden = !(d && !main); $('choice-note').textContent = 'Resolve this choice to continue. Other actions become available afterward.';
  let status = game.status === 'model_thinking' ? `${game.model_label} is playing…` : 'Your turn · Play cards, use abilities, buy cards, or attack.';
  let title = d ? (main ? 'Your move' : d.prompt) : 'Computer’s turn';
  if (game.status === 'complete') { status = game.result.truncated ? 'Draw · the game reached its turn or action limit.' : game.result.winner === 0 ? 'Victory! You defeated the champion.' : `${game.model_label} wins. Ready for a rematch?`; title = 'Game complete'; }
  $('status').textContent = status; $('decision-title').textContent = title;
  $('log').replaceChildren(...game.action_log.slice().reverse().map(entry => { const li = el('li', entry.player_id ? 'computer' : ''); li.append(el('b', '', `Turn ${entry.turn} · ${entry.player_id ? game.model_label : 'You'}`), document.createTextNode(entry.label)); return li; }));
}
function renderModels() {
  const selected = $('opponent').value;
  $('opponent').replaceChildren(...models.map(m => { const option = el('option', '', aliases[m.id] || m.name); option.value = m.id; return option; }));
  $('opponent').value = models.some(m => m.id === selected) ? selected : 'level-10';
  $('start').disabled = !models.length;
  $('model-list').replaceChildren(...models.map(m => {
    const form = el('form', 'rename-form'), input = el('input'); input.value = aliases[m.id] || m.name; input.maxLength = 100; input.required = true; input.setAttribute('aria-label', `Name for ${m.name}`);
    const save = el('button', '', 'Save name'); form.append(input, save);
    form.addEventListener('submit', e => { e.preventDefault(); try { aliases[m.id] = input.value.trim() || m.name; localStorage.setItem(aliasesKey, JSON.stringify(aliases)); renderModels(); } catch { showError('Local storage is unavailable.', 'admin-error'); } });
    form.append(el('p', 'muted', m.description)); return form;
  }));
}
$('new-game').addEventListener('submit', e => { e.preventDefault(); if (game && game.status !== 'complete' && !confirm('Start a new game and replace this game?')) return; request({ op: 'new', model: $('opponent').value, label: aliases[$('opponent').value], starts: $('starts').value === 'human' }); });
$('opponent').addEventListener('change', async () => {
  const id = $('opponent').value, m = models.find(m => m.id === id);
  const result = await request({ op: 'load', model: id });
  if (result) { $('status').textContent = `${aliases[id] || m.name} is ready. Start a new game to play this level.`; }
});
$('play-all').addEventListener('click', () => request({ op: 'play_all', id: game.id, revision: game.revision }));
$('models-open').addEventListener('click', () => $('models-dialog').showModal());
document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => $(b.dataset.close).close()));
$('upload-form').addEventListener('submit', async e => {
  e.preventDefault(); const form = new FormData(e.currentTarget), file = form.get('model_file');
  if (file.size > 64 * 1024 * 1024) { showError('Choose a model smaller than 64 MB.', 'admin-error'); return; }
  const result = await request({ op: 'import', name: form.get('name'), bytes: await file.arrayBuffer() }, 'admin-error');
  if (result?.imported) { models.push(result.imported); renderModels(); $('opponent').value = result.imported.id; $('upload-form').reset(); }
});
export async function start(options) {
  ({ release, base } = options); configureResources(base, release); savedKey = 'astro-game:' + new URL(base).pathname;
  worker = new Worker(resourceURL('assets/worker.js'), { type: 'module' });
  worker.onmessage = ({ data: message }) => {
    if (message.progress) { $('status').textContent = message.progress; return; }
    const pending = waiting.get(message.requestId); if (!pending) return; waiting.delete(message.requestId);
    if (message.error) pending.reject(new Error(message.error)); else pending.resolve(message.data);
  };
  worker.onerror = () => { for (const pending of waiting.values()) pending.reject(new Error('The game worker stopped. Reload to resume.')); waiting.clear(); lock(false); showError('The game worker stopped. Reload to resume.'); };
  let saved = null, changed = false;
  try { const stored = JSON.parse(sessionStorage.getItem(savedKey) || 'null'); if (stored?.release === release) saved = stored.saved; else if (stored) changed = true; } catch {}
  lock(true);
  try {
    const data = await api({ op: 'init', base, release, saved }); cards = data.cards; models = data.models;
    try { models.push(...(await localModels('list')).map(({ bytes, ...m }) => m)); } catch { /* Gameplay does not require IndexedDB. */ }
    game = data.game; renderModels(); render(); saveGame(data.saved);
    if (data.notice) showError(data.notice);
    if (changed) $('status').textContent = 'A new release is ready. Start a new game with the updated version.';
  } catch (e) { showError(e.message); $('status').textContent = 'Could not load the game. Reconnect and refresh to retry.'; }
  finally { lock(false); $('start').disabled = !models.length; if (game?.status === 'model_thinking') schedule(); }
  // A visible page checks for a new HTML release on focus and every five minutes.
  let checking = false;
  async function checkRelease() {
    if (checking || document.hidden || !navigator.onLine) return;
    checking = true;
    try {
      const response = await fetch(new URL('index.html', base), { cache: 'no-store' }); if (!response.ok) return;
      const html = new DOMParser().parseFromString(await response.text(), 'text/html').documentElement.outerHTML;
      if (await sha256(new TextEncoder().encode(html)) !== release) $('update-banner').hidden = false;
    } catch { /* Offline play continues. */ } finally { checking = false; }
  }
  $('update-now').addEventListener('click', () => { if (!game || game.status === 'complete' || confirm('Load the new release? This will end the current game.')) location.reload(); });
  window.addEventListener('focus', checkRelease); setInterval(checkRelease, 300000);
  navigator.serviceWorker?.controller?.postMessage({ type: 'release', version: release });
  navigator.serviceWorker?.addEventListener('controllerchange', () => navigator.serviceWorker.controller?.postMessage({ type: 'release', version: release }));
}
