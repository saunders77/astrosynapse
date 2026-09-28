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
      const targets = allButtons(zone).filter(b => b.textContent === 'SELECT TARGET');
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
  assert.equal(nodes.get('opponent-deck').textContent, 3);
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
  assert.equal(nodes.get('own-deck-cards').children.length, 5);
  vm.runInContext("inspectPile('discard')", context);
  assert.equal(nodes.get('discard').hidden, false);
  assert.equal(nodes.get('own-deck-cards').hidden, true);
  assert.equal(nodes.get('pile-title').textContent, 'Your discard pile · 2 cards');
});
