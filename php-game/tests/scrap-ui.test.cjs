const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

function board(actions, status = 'your_turn', observation = {}, family = 'scrap') {
  class Element {
    constructor() { this.children = []; this.dataset = {}; this.listeners = {}; this.parent = { open: false }; this.classes = new Set(); this.classList = { toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name) }; this.attributes = {}; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name]; }
    close() { this.open = false; }
    showModal() { this.open = true; }
    closest() { return this.parent; }
  }
  const nodes = new Map();
  const document = {
    body: new Element(),
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
    createElement: () => new Element(),
    createTextNode: text => text,
  };
  const context = vm.createContext({ document, location: { pathname: '/' }, localStorage: { getItem: () => null }, IntersectionObserver: class { observe() {} disconnect() {} } });
  const source = fs.readFileSync(`${__dirname}/../assets/runtime/ui.mjs`, 'utf8').split("// Fixed bars may wrap")[0].replace(/^import .*;$/gm, '');
  vm.runInContext(fs.readFileSync(`${__dirname}/../assets/runtime/turn-summary.mjs`, 'utf8').replace('export function', 'function'), context);
  vm.runInContext(source, context);
  context.fixture = { status, decision: { family, actions }, observation: {
    hand: [0, 0, 1], own_discard: [0, 1], own_in_play: [], opponent_in_play: [],
    trade_row: [], opponent_discard: [0], scrap_heap: [0], explorers_remaining: 0, ...observation,
  }, action_log: [] };
  vm.runInContext(`cards = [0, 1].map(card_id => ({card_id, name: 'Card ' + card_id, card_type: 'ship'})); game = fixture; move = id => { globalThis.chosen = id; }; render();`, context);
  const allButtons = id => document.getElementById(id).children.flatMap(card => card.children[1]?.children.at(-1)?.children || []);
  const buttons = id => allButtons(id).filter(button => button.textContent === 'SCRAP');
  return { buttons, allButtons, context, nodes };
}

test('scrap buttons target the correct zone, including duplicate hand cards', () => {
  const { buttons, context, nodes } = board([
    { id: 3, kind: 'scrap_card', card_id: 0, source_zone: 'hand' },
    { id: 4, kind: 'scrap_card', card_id: 0, source_zone: 'discard' },
  ]);
  assert.equal(buttons('hand').length, 2);
  assert.equal(buttons('discard').length, 1);
  buttons('hand')[1].listeners.click();
  assert.equal(context.chosen, 3);
  buttons('discard')[0].listeners.click();
  assert.equal(context.chosen, 4);
  assert.equal(nodes.get('own-discard-pile').classes.has('actionable'), true);
  assert.equal(buttons('opponent-discard').length, 0);
  assert.equal(buttons('scrap').length, 0);
});

test('hand-only and inactive decisions do not offer discard scraps', () => {
  const actions = [{ id: 3, kind: 'scrap_card', card_id: 0, source_zone: 'hand' }];
  const handOnly = board(actions);
  assert.equal(handOnly.buttons('discard').length, 0);
  assert.equal(handOnly.nodes.get('own-discard-pile').classes.has('actionable'), false);
  assert.equal(board(actions, 'model_thinking').buttons('hand').length, 0);
  assert.equal(board([]).buttons('hand').length, 0);
});


test('mode choices appear together on their source card and dispatch separately', () => {
  const { allButtons, context } = board([
    { id: 10, kind: 'choose_mode', card_id: 0, label: 'Gain 3 trade' },
    { id: 11, kind: 'choose_mode', card_id: 0, label: 'Gain 5 combat' },
  ], 'your_turn', { own_in_play: [{ card: 0 }, { card: 1 }] }, 'ability_mode');
  const choices = allButtons('own-fleet').filter(b => b.textContent.startsWith('Gain'));
  assert.equal(choices.length, 2);
  choices[0].listeners.click(); assert.equal(context.chosen, 10);
  choices[1].listeners.click(); assert.equal(context.chosen, 11);
});

test('target buttons resolve destroy, free acquire, and copy decisions', () => {
  for (const [kind, source_zone, targetZone] of [['destroy_base', 'opponent_in_play', 'opponent-fleet'], ['free_acquire', 'trade_row', 'market'], ['copy_ship', 'in_play', 'own-fleet']]) {
    const { allButtons, context } = board([
      { id: 8, kind, card_id: 0, target_card_id: 1, source_zone },
    ], 'your_turn', { own_in_play: [{ card: 0 }, { card: 1 }], opponent_in_play: [{ card: 1 }], trade_row: [1] });
    for (const zone of ['own-fleet', 'opponent-fleet', 'market']) {
      const targets = allButtons(zone).filter(b => b.textContent === {destroy_base: 'DESTROY', free_acquire: 'ACQUIRE FREE', copy_ship: 'COPY'}[kind]);
      assert.equal(targets.length, zone === targetZone ? 1 : 0);
      if (targets.length) { targets[0].listeners.click(); assert.equal(context.chosen, 8); }
    }
  }
});


test('hand faces play only legal play actions and authority dispatches legal attacks', () => {
  const { context, nodes } = board([
    { id: 12, kind: 'play_card', card_id: 0 },
    { id: 13, kind: 'attack_player', amount: 4 },
  ], 'your_turn', {}, 'main');
  nodes.get('hand').children[0].children[0].listeners.click();
  assert.equal(context.chosen, 12);
  nodes.get('opponent-attack').onclick();
  assert.equal(context.chosen, 13);
  assert.equal(nodes.get('opponent-attack').disabled, false);
  const blocked = board([], 'your_turn', {}, 'main');
  assert.equal(blocked.nodes.get('opponent-attack').disabled, true);
  blocked.nodes.get('opponent-attack').onclick();
  assert.equal(blocked.context.chosen, undefined);
  assert.equal(blocked.nodes.get('hand').children[0].children[0].attributes['aria-label'], 'Details for Card 0');
});

test('pile counts and inspection include known cards exactly once without exposing hidden zones', () => {
  const { context, nodes } = board([], 'your_turn', {
    own_deck_count: 2, own_deck: [1], own_known_top: [0],
    opponent_hand_count: 2, opponent_deck_count: 3,
    opponent_hidden: [0, 1, 1], opponent_known_hand: [0], opponent_known_top: [1],
    pending_discard: 1, opponent_pending_discard: 2,
  });
  assert.equal(nodes.get('deck').textContent, 2);
  assert.equal(nodes.get('opponent-deck').textContent, 5);
  assert.equal(nodes.get('discard-count').textContent, 2);
  assert.equal(nodes.get('opponent-discard-count').textContent, 1);
  assert.equal(nodes.get('opponent-must-discard').textContent, 2);
  assert.equal(nodes.get('must-discard').textContent, 1);
  vm.runInContext("inspectPile('opponent-deck')", context);
  assert.equal(nodes.get('opponent-deck-cards').children.length, 5);
  assert.equal(nodes.get('pile-title').textContent, 'Opponent’s hand + deck · 5 cards');
  const names = nodes.get('opponent-deck-cards').children.map(c => c.children[0].children[0].alt).sort();
  assert.deepEqual(names, ['Card 0', 'Card 0', 'Card 1', 'Card 1', 'Card 1']);
  assert.equal(nodes.get('discard').hidden, true);
  vm.runInContext("inspectPile('own-deck')", context);
  assert.equal(nodes.get('own-deck-cards').children.length, 2);
  assert.equal(nodes.get('pile-title').textContent, 'Your deck · 2 cards');
  assert.deepEqual(nodes.get('own-deck-cards').children.map(c => c.children[0].children[0].alt).sort(), ['Card 0', 'Card 1']);
  vm.runInContext("inspectPile('discard')", context);
  assert.equal(nodes.get('discard').hidden, false);
  assert.equal(nodes.get('own-deck-cards').hidden, true);
  assert.equal(nodes.get('pile-title').textContent, 'Your discard pile · 2 cards');
});


test('single card actions dispatch from the face, including discard and scrap', () => {
  for (const kind of ['discard_card', 'scrap_card']) {
    const {context, nodes} = board([{id: 7, kind, card_id: 0, source_zone: 'hand'}]);
    nodes.get('hand').children[0].children[0].listeners.click();
    assert.equal(context.chosen, 7);
  }
  const {context, nodes} = board([{id: 8, kind: 'acquire', card_id: 1, source_zone: 'trade_row', amount: 2}], 'your_turn', {trade_row: [1]}, 'main');
  nodes.get('market').children[0].children[0].listeners.click();
  assert.equal(context.chosen, 8);
});

test('scrap picker opens automatically, updates across selections, and closes when resolved', () => {
  const {context, nodes} = board([
    {id: 3, kind: 'scrap_card', card_id: 0, source_zone: 'hand'},
    {id: 4, kind: 'scrap_card', card_id: 1, source_zone: 'discard'},
    {id: 5, kind: 'decline', label: 'Done scrapping'},
  ]);
  assert.equal(nodes.get('scrap-dialog').open, true);
  assert.equal(nodes.get('scrap-options-hand').children[1].children.length, 1);
  assert.equal(nodes.get('scrap-options-discard').children[1].children.length, 1);
  nodes.get('scrap-options-discard').children[1].children[0].children[0].listeners.click();
  assert.equal(context.chosen, 4);
  vm.runInContext('game.decision.actions.splice(1, 1); render()', context);
  assert.equal(nodes.get('scrap-dialog').open, true);
  assert.equal(nodes.get('scrap-options-discard').hidden, true);
  nodes.get('scrap-decline').children[0].listeners.click();
  assert.equal(context.chosen, 5);
  vm.runInContext('game.decision.actions = []; render()', context);
  assert.equal(nodes.get('scrap-dialog').open, false);
});

test('multiple card actions keep the face as details', () => {
  const {nodes} = board([
    {id: 1, kind: 'activate_ally', card_id: 0},
    {id: 2, kind: 'scrap_for_ability', card_id: 0},
  ], 'your_turn', {own_in_play: [{card: 0}]}, 'main');
  assert.equal(nodes.get('own-fleet').children[0].children[0].attributes['aria-label'], 'Details for Card 0');
});


test('trade-row scraps use target cards and inactive decisions do not open the picker', () => {
  const actions = [{id: 9, kind: 'scrap_trade_row', target_card_id: 1, source_zone: 'trade_row'}];
  const {context, nodes} = board(actions, 'your_turn', {trade_row: [1]});
  const card = nodes.get('scrap-options-trade_row').children[1].children[0];
  assert.equal(card.children[0].children[0].alt, 'Card 1');
  card.children[0].listeners.click();
  assert.equal(context.chosen, 9);
  assert.equal(board(actions, 'model_thinking').nodes.get('scrap-dialog').open, undefined);
});

test('completed games have a prominent result for wins, losses, and draws', () => {
  const {context, nodes} = board([]);
  for (const [winner, truncated, heading] of [[0, false, 'Victory!'], [1, false, 'Defeat'], [null, true, 'Draw']]) {
    context.result = {winner, truncated};
    vm.runInContext("recordStats = () => {}; game.status = 'complete'; game.result = result; render()", context);
    assert.equal(nodes.get('result-banner').hidden, false);
    assert.equal(nodes.get('result-title').textContent, heading);
  }
  vm.runInContext("game.status = 'your_turn'; render()", context);
  assert.equal(nodes.get('result-banner').hidden, true);
});

test('resign requires confirmation and setup is hidden during play', () => {
  const {context, nodes} = board([], 'your_turn', {}, 'main');
  assert.equal(vm.runInContext("document.body.classes.has('active-game')", context), true);
  vm.runInContext("confirm = message => { globalThis.prompt = message; return false; }; request = payload => { globalThis.sent = payload; };", context);
  nodes.get('resign').onclick();
  assert.equal(context.prompt, 'Are you sure you want to resign?');
  assert.equal(context.sent, undefined);
  vm.runInContext('confirm = () => true;', context);
  nodes.get('resign').onclick();
  assert.equal(context.sent.op, 'resign');
});

test('saved games are written to persistent localStorage', () => {
  const {context} = board([]);
  vm.runInContext("savedKey = 'game'; release = 'build'; localStorage.setItem = (key, value) => { globalThis.persisted = {key, value}; }; saveGame({id: 'resume-me', transcript: ['move']});", context);
  assert.equal(context.persisted.key, 'game');
  assert.deepEqual(JSON.parse(context.persisted.value), {release: 'build', saved: {id: 'resume-me', transcript: ['move']}});
});

test('required decisions open a large picker, dispatch choices, and close after resolution', () => {
  const actions = [
    { id: 10, kind: 'choose_mode', card_id: 0, label: 'Gain trade (3)' },
    { id: 11, kind: 'choose_mode', card_id: 0, label: 'Gain combat (5)' },
  ];
  const { context, nodes } = board(actions, 'your_turn', {}, 'ability_mode');
  assert.equal(nodes.get('decision-dialog').open, true);
  const options = nodes.get('decision-options').children;
  assert.equal(options.length, 2);
  options[1].listeners.click();
  assert.equal(context.chosen, 11);
  vm.runInContext("game.decision = {family: 'main', actions: []}; render();", context);
  assert.equal(nodes.get('decision-dialog').open, false);
  assert.equal(nodes.get('decision-options').children.length, 0);
  assert.ok(!board(actions, 'model_thinking', {}, 'ability_mode').nodes.get('decision-dialog').open);
  assert.ok(!board([{ id: 1, kind: 'scrap_card', card_id: 0, source_zone: 'hand' }]).nodes.get('decision-dialog').open);
});

test('action amounts use a single parenthesized number only when needed', async () => {
  const { Game } = await import('../assets/runtime/engine.mjs');
  assert.equal(Game.label(Game.action('choose_mode', -1, -1, 'gain_trade', '', 3)), 'Choose mode (gain trade) (3)');
  assert.equal(Game.label(Game.action('end_turn')), 'End turn');
});

test('audio enqueues only new effects and clears the queue for a new game', async () => {
  const { context } = board([]);
  context.clearTimeout = () => {};
  vm.runInContext(`
    lock = () => {}; render = () => {}; renderStats = () => {};
    globalThis.heard = []; globalThis.cleared = 0;
    audio = { enqueue: sounds => heard.push(...sounds), clear: () => cleared++ };
    game = {id: 'one', sounds: ['playerturn'], status: 'your_turn'};
    api = async () => ({ game: {id: 'one', sounds: ['playerturn', 'combat', 'trade', 'combat'], status: 'your_turn'} });
  `, context);
  await vm.runInContext('request({})', context);
  assert.deepEqual(Array.from(context.heard), ['combat', 'trade', 'combat']);
  await vm.runInContext('request({})', context);
  assert.equal(context.heard.length, 3, 'Repeated state does not replay sounds');
  vm.runInContext(`api = async () => ({ game: {id: 'two', sounds: ['playerturn'], status: 'your_turn'} });`, context);
  await vm.runInContext('request({})', context);
  assert.equal(context.cleared, 1);
  assert.deepEqual(Array.from(context.heard), ['combat', 'trade', 'combat', 'playerturn']);
});

test('cancel undoes a choice, closes either picker, and leaves restored choices accessible', async () => {
  for (const kind of ['scrap_card', 'choose_mode']) {
    const {context, nodes} = board([{id: 2, kind, card_id: 0, source_zone: 'hand', label: 'Choose card'}]);
    context.clearTimeout = () => {};
    vm.runInContext(`
      game.id = 'test'; game.revision = 2; game.can_undo = true; game.sounds = [];
      lock = value => { busy = value; }; renderStats = () => {};
      audio = {clear() {}, enqueue() {}};
      api = async payload => {
        globalThis.sent = payload;
        return {game: {...game, revision: 3}};
      };
    `, context);
    await vm.runInContext('cancelSelection()', context);
    assert.equal(context.sent.op, 'undo');
    assert.equal(context.sent.revision, 2);
    assert.ok(!nodes.get('scrap-dialog').open);
    assert.ok(!nodes.get('decision-dialog').open);
    assert.equal(nodes.get('choices').children.length, 1);
    nodes.get('choices').children[0].listeners.click();
    assert.equal(context.chosen, 2);
    vm.runInContext('game.revision++; render()', context);
    assert.equal(nodes.get(kind === 'scrap_card' ? 'scrap-dialog' : 'decision-dialog').open, true);
  }
});

test('failed cancel keeps selection open and busy cancel sends no request', async () => {
  const {context, nodes} = board([{id: 2, kind: 'scrap_card', card_id: 0, source_zone: 'hand'}]);
  context.clearTimeout = () => {};
  vm.runInContext(`
    game.can_undo = true; busy = true;
    api = async () => { globalThis.sent = true; throw new Error('Retry cancellation'); };
    lock = value => { busy = value; };
  `, context);
  await vm.runInContext('cancelSelection()', context);
  assert.equal(context.sent, undefined);
  vm.runInContext('busy = false', context);
  await vm.runInContext('cancelSelection()', context);
  assert.equal(nodes.get('scrap-dialog').open, true);
  assert.equal(nodes.get('scrap-error').textContent, 'Retry cancellation');
});

test('home hides the table without discarding the match and supports resuming', () => {
  const { context, nodes } = board([], 'your_turn');
  vm.runInContext("location.hash = '#home'; navigate();", context);
  assert.equal(nodes.get('board').hidden, true);
  assert.equal(nodes.get('opponent-bar').hidden, true);
  assert.equal(nodes.get('player-bar').hidden, true);
  assert.equal(nodes.get('welcome').hidden, false);
  assert.equal(nodes.get('result-banner').hidden, true);
  assert.equal(nodes.get('resume-game').hidden, false);
  assert.equal(vm.runInContext('game === fixture', context), true);
  assert.equal(vm.runInContext("document.body.classes.has('active-game')", context), false);
  vm.runInContext("location.hash = '#game'; navigate();", context);
  assert.equal(nodes.get('board').hidden, false);
  assert.equal(nodes.get('welcome').hidden, true);
  assert.equal(nodes.get('opponent-bar').hidden, false);
  assert.equal(vm.runInContext("document.body.classes.has('active-game')", context), true);
});
