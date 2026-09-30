import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Game } from '../assets/runtime/engine.mjs';
import { Session } from '../assets/runtime/session.mjs';
Game.catalog = JSON.parse(fs.readFileSync(new URL('../assets/cards.json', import.meta.url)));
const actor = { winProbability: () => 0.5, choose: () => 0 };
const run = Game.prototype.run;
try {
  for (const name of ['Trade Bot', 'Battle Pod', 'Patrol Mech', 'Brain World']) {
    const card = Game.catalog.find(c => c.name === name).card_id;
    Game.prototype.run = function () {
      this.players[0].hand = [card, 0, 1];
      return run.call(this);
    };
    const session = Session.start({ id: 'test', name: 'Test' }, true, 17);
    let state = Session.advance(session, actor);
    assert.equal(state.can_undo, false);
    assert.throws(() => Session.advance(structuredClone(session), actor, 'undo'));
    let previous;
    for (let step = 0; step < 3 && state.decision.family === 'main'; step++) {
      previous = state;
      const action = state.decision.actions.find(a => a.card_id === card && ['play_card', 'activate_base'].includes(a.kind));
      assert.ok(action, name);
      state = Session.advance(session, actor, 'choose', action.id);
    }
    assert.notEqual(state.decision.family, 'main', name);
    assert.equal(state.can_undo, true);
    const revision = state.revision;
    const restored = Session.advance(session, actor, 'undo');
    assert.deepEqual(restored.observation, previous.observation, name);
    assert.deepEqual(restored.decision, previous.decision, name);
    assert.deepEqual(restored.action_log, previous.action_log, name);
    assert.deepEqual(restored.sounds, previous.sounds, name);
    assert.equal(restored.revision, revision + 1);
    assert.deepEqual(Session.advance(structuredClone(session), actor), restored);
  }
} finally { Game.prototype.run = run; }
console.log('Undo restores scrap, trade-row scrap, mode, and base activation choices.');
