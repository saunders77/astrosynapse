import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';
const source = await readFile(new URL('../app/card-actions.ts', import.meta.url), 'utf8');
const context = vm.createContext({ exports: {} });
vm.runInContext(ts.transpile(source, { module: ts.ModuleKind.CommonJS }), context);
const { matchesCardAction, cardActionLabel } = context.exports;

test('manual and ally abilities belong to the source in play', () => {
  for (const kind of ['activate_base', 'activate_ally', 'choose_mode']) {
    const action = { kind, card_id: 12, label: 'Gain 5 combat' };
    assert.equal(matchesCardAction(action, 12, 'in_play'), true);
    assert.equal(matchesCardAction(action, 12, 'hand'), false);
    assert.equal(matchesCardAction(action, 12, 'opponent_in_play'), false);
    assert.equal(matchesCardAction(action, 13, 'in_play'), false);
  }
  assert.equal(cardActionLabel({ kind: 'choose_mode', label: 'Gain 5 combat' }), 'Gain 5 combat');
  assert.equal(cardActionLabel({ kind: 'choose_mode', label: 'Gain 3 trade' }), 'Gain 3 trade');
});

test('target actions bind the destination, never the source or wrong zone', () => {
  for (const [kind, zone] of [['destroy_base', 'opponent_in_play'], ['free_acquire', 'trade_row'], ['copy_ship', 'in_play'], ['scrap_trade_row', 'trade_row']]) {
    const action = { kind, card_id: 12, target_card_id: 13, source_zone: zone };
    assert.equal(matchesCardAction(action, 13, zone), true);
    assert.equal(matchesCardAction(action, 12, zone), false);
    assert.equal(matchesCardAction(action, 13, 'discard'), false);
    assert.equal(cardActionLabel(action), 'SELECT TARGET');
  }
  assert.equal(matchesCardAction({ kind: 'decline', card_id: 12 }, 12, 'in_play'), false);
});
