import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { Game, InPlay } from '../assets/runtime/engine.mjs';
import { Encoder } from '../assets/runtime/encoder.mjs';
Game.catalog = JSON.parse(fs.readFileSync(new URL('../assets/cards.json', import.meta.url)));
for (const name of ['Explorer', 'Ram']) for (const copyFirst of [false, true]) for (const scrapCopy of [false, true]) {
  test(`${name}: copy first=${copyFirst}, scrap copy=${scrapCopy}`, () => {
    const g = new Game(17), p = g.players[0], id = Game.catalog.find(c => c.name === name).card_id;
    g.explorers_remaining = 9;
    const original = new InPlay(100, id), copy = new InPlay(101, 23); copy.card = id;
    p.in_play = copyFirst ? [copy, original] : [original, copy];
    const actions = g.mainActions(p).filter(a => a.kind === 'scrap_for_ability');
    assert.equal(actions.length, 2);
    for (const version of [1, 2, 3]) {
      const enc = new Encoder(version), o = g.observation(0);
      assert.notDeepEqual(enc.action(actions[0], o), enc.action(actions[1], o));
    }
    const a = actions.find(a => (a.target_card_id === 23) === scrapCopy), supply = g.explorers_remaining;
    g.apply(p, a);
    assert.deepEqual(p.in_play, scrapCopy ? [original] : [copy]);
    assert.equal(g.explorers_remaining, supply + +(name === 'Explorer' && !scrapCopy));
    if (scrapCopy) { assert.ok(g.scrap_heap.includes(23)); assert.match(Game.label(a), /scraps Stealth Needle/); }
    const remaining = g.mainActions(p).filter(a => a.kind === 'scrap_for_ability');
    assert.equal(remaining.length, 1);
    g.apply(p, remaining[0]); assert.equal(p.in_play.length, 0);
  });
}
