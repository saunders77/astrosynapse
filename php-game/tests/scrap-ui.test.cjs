const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

function board(actions, status = 'your_turn', observation = {}, family = 'scrap') {
  class Element {
    constructor() { this.children = []; this.dataset = {}; this.listeners = {}; this.parent = { open: false }; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute() {}
    closest() { return this.parent; }
  }
  const nodes = new Map();
  const document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
    createElement: () => new Element(),
    createTextNode: text => text,
  };
  const context = vm.createContext({ document, location: { pathname: '/' }, localStorage: { getItem: () => null }, IntersectionObserver: class { observe() {} disconnect() {} } });
  const source = fs.readFileSync(`${__dirname}/../assets/runtime/ui.mjs`, 'utf8').split("$('new-game').addEventListener")[0].replace(/^import .*;$/gm, '');
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
  assert.equal(nodes.get('discard').parent.open, true);
  assert.equal(buttons('opponent-discard').length, 0);
  assert.equal(buttons('scrap').length, 0);
});

test('hand-only and inactive decisions do not offer discard scraps', () => {
  const actions = [{ id: 3, kind: 'scrap_card', card_id: 0, source_zone: 'hand' }];
  const handOnly = board(actions);
  assert.equal(handOnly.buttons('discard').length, 0);
  assert.equal(handOnly.nodes.get('discard').parent.open, false);
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
