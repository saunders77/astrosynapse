'use strict';
const $ = id => document.getElementById(id);
let cards = [], models = [], game = null, csrf = '', busy = false, timer = null, admin = false;
const art = c => `assets/card-art/${c.card_id === 4 ? 'BattlePod' : c.name.replaceAll(' ', '-')}.jpg`;
const el = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; };
const showError = (message, id = 'error') => { $(id).textContent = message; $(id).hidden = !message; };
function lock(value) { busy = value; document.body.classList.toggle('busy', value); document.querySelectorAll('button, select, input').forEach(n => n.disabled = value); }
async function api(payload) {
  const options = payload ? { method: 'POST', headers: { 'X-CSRF-Token': csrf }, body: payload instanceof FormData ? payload : JSON.stringify(payload) } : {};
  if (payload && !(payload instanceof FormData)) options.headers['Content-Type'] = 'application/json';
  const response = await fetch('api.php', { ...options, credentials: 'same-origin', cache: 'no-store' });
  let data;
  try { data = await response.json(); } catch { throw new Error('The PHP server returned an invalid response. Check that this folder is served by PHP 8.1 or newer.'); }
  if (!response.ok || data.error) throw new Error(data.error || 'The request failed.');
  return data;
}
async function request(payload, errorTarget = 'error') {
  if (busy) return;
  clearTimeout(timer); showError('', errorTarget); lock(true);
  try {
    const data = await api(payload);
    if (data.csrf) csrf = data.csrf;
    if (data.models) { models = data.models; renderModels(); }
    if ('admin' in data) { admin = data.admin; renderAdmin(); }
    if ('game' in data) { game = data.game; render(); }
    return data;
  } catch (e) { showError(e.message, errorTarget); $('status').textContent = 'Request paused. Your last saved game is preserved. Refresh to reconnect.'; }
  finally { lock(false); if (game?.status === 'model_thinking' && !$('error').textContent) schedule(); }
}
function schedule() { clearTimeout(timer); timer = setTimeout(() => { if (!busy && game?.status === 'model_thinking') request({ op: 'advance', id: game.id, revision: game.revision }); }, 120); }
function move(id) { request({ op: 'choose', id: game.id, revision: game.revision, action_id: id }); }
function inspect(card) {
  $('card-large').src = art(card); $('card-large').alt = card.name; $('card-name').textContent = card.name;
  $('card-description').textContent = `${card.faction.replaceAll('_', ' ')} · ${card.card_type} · Cost ${card.cost}${card.defense ? ` · Defense ${card.defense}` : ''}`;
  $('card-dialog').showModal();
}
function cardView(id, actions = [], state = '') {
  const c = cards[id]; if (!c) return el('span', 'empty', 'Empty trade slot');
  const node = el('article', `card ${c.card_type !== 'ship' ? 'base' : ''} ${actions.length ? 'actionable' : ''}`);
  const img = el('img'); img.src = art(c); img.alt = c.name; img.loading = 'lazy'; node.append(img);
  const meta = el('div', 'card-meta'); meta.append(el('span', 'card-name', c.name)); if (state) meta.append(el('span', 'card-state', state));
  const controls = el('div', 'card-actions');
  for (const a of actions) {
    const names = { play_card: 'Play', acquire: `Buy · ${a.amount} trade`, activate_base: 'Use ability', activate_ally: 'Use ally', scrap_for_ability: 'Scrap for ability', attack_base: `Attack · ${a.amount} combat` };
    const b = el('button', 'move', names[a.kind] || a.label); b.type = 'button'; b.title = a.label; b.addEventListener('click', () => move(a.id)); controls.append(b);
  }
  const detail = el('button', 'details', 'Details'); detail.type = 'button'; detail.setAttribute('aria-label', `Details for ${c.name}`); detail.addEventListener('click', () => inspect(c)); controls.append(detail);
  meta.append(controls); node.append(meta); return node;
}
function zone(id, entries) { const node = $(id); node.replaceChildren(...entries); if (!entries.length) node.append(el('span', 'empty', 'No cards')); }
function render() {
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
  const forCard = (id, kinds, field = 'card_id', zoneName) => main ? actions.filter(a => a[field] === id && kinds.includes(a.kind) && (!zoneName || a.source_zone === zoneName)) : [];
  zone('hand', o.hand.map(id => cardView(id, forCard(id, ['play_card']))));
  zone('market', [...o.trade_row.map(id => id === null ? el('span', 'empty', 'Empty slot') : cardView(id, forCard(id, ['acquire'], 'card_id', 'trade_row'), `Cost ${cards[id].cost}`)), ...(o.explorers_remaining ? [cardView(2, forCard(2, ['acquire'], 'card_id', 'explorer_supply'), `${o.explorers_remaining} available`)] : [])]);
  zone('own-fleet', o.own_in_play.map(i => cardView(i.card, forCard(i.card, ['activate_base', 'activate_ally', 'scrap_for_ability']), i.copied_from_stealth_needle ? 'Stealth Needle copy' : i.ally_triggered ? 'Ally used' : 'In play')));
  zone('opponent-fleet', o.opponent_in_play.map(i => cardView(i.card, forCard(i.card, ['attack_base'], 'target_card_id'), cards[i.card].defense ? `${cards[i.card].card_type} · ${cards[i.card].defense} defense` : 'Ship')));
  zone('discard', o.own_discard.map(id => cardView(id)));
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
    if (id >= 0 && a.kind !== 'decline') { const img = el('img'); img.src = art(cards[id]); img.alt = ''; button.append(img); }
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
  $('opponent').replaceChildren(...models.map(m => { const option = el('option', '', m.name); option.value = m.id; return option; }));
  $('opponent').value = models.some(m => m.id === selected) ? selected : models.at(-1)?.id || '';
  $('start').disabled = !models.length;
  $('model-list').replaceChildren(...models.map(m => {
    const form = el('form', 'rename-form'), input = el('input'); input.value = m.name; input.maxLength = 100; input.required = true; input.setAttribute('aria-label', `Name for ${m.name}`);
    const save = el('button', '', 'Save name'); form.append(input, save);
    form.addEventListener('submit', e => { e.preventDefault(); request({ op: 'rename', model: m.id, name: input.value }, 'admin-error'); }); return form;
  }));
}
function renderAdmin() { $('login-form').hidden = admin; $('model-manager').hidden = !admin; }
$('new-game').addEventListener('submit', e => { e.preventDefault(); if (game && game.status !== 'complete' && !confirm('Start a new game and replace this game?')) return; request({ op: 'new', model: $('opponent').value, starts: $('starts').value === 'human' }); });
$('play-all').addEventListener('click', () => request({ op: 'play_all', id: game.id, revision: game.revision }));
$('models-open').addEventListener('click', () => $('models-dialog').showModal());
document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => $(b.dataset.close).close()));
$('login-form').addEventListener('submit', async e => { e.preventDefault(); await request({ op: 'login', password: $('admin-password').value }, 'admin-error'); $('admin-password').value = ''; });
$('logout').addEventListener('click', () => request({ op: 'logout' }, 'admin-error'));
$('upload-form').addEventListener('submit', async e => { e.preventDefault(); const data = new FormData(e.currentTarget); data.set('op', 'upload'); const result = await request(data, 'admin-error'); if (result) $('upload-form').reset(); });
(async () => {
  try { const response = await fetch('assets/cards.json'); if (!response.ok) throw new Error('Card catalog could not be loaded'); cards = await response.json();
    const data = await request(); if (data) $('admin-note').textContent = data.admin_enabled ? 'Add a model or give an opponent a new name. Changes apply to new games.' : 'Model management is disabled until the server owner sets admin_password_hash in config.php. The ten included champions are ready to play.';
  } catch (e) { showError(e.message); }
})();
