const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const test = require('node:test');

function board(actions, status = 'your_turn') {
  class Element {
    constructor() { this.children = []; this.listeners = {}; this.parent = { open: false }; }
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
  const context = vm.createContext({ document });
  const source = fs.readFileSync(`${__dirname}/../assets/game.js`, 'utf8').split("$('new-game').addEventListener")[0];
  vm.runInContext(source, context);
  context.fixture = { status, decision: { family: 'scrap', actions }, observation: {
    hand: [0, 0, 1], own_discard: [0, 1], own_in_play: [], opponent_in_play: [],
    trade_row: [], opponent_discard: [0], scrap_heap: [0], explorers_remaining: 0,
  }, action_log: [] };
  vm.runInContext(`cards = [0, 1].map(card_id => ({card_id, name: 'Card ' + card_id, card_type: 'ship'})); game = fixture; move = id => { globalThis.chosen = id; }; render();`, context);
  const buttons = id => document.getElementById(id).children.flatMap(card => card.children[1]?.children[1]?.children || []).filter(button => button.textContent === 'SCRAP');
  return { buttons, context, nodes };
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
