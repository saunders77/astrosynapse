import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { Game, InPlay } from '../assets/runtime/engine.mjs';

Game.catalog = JSON.parse(readFileSync(new URL('../assets/cards.json', import.meta.url)));
const card = name => Game.catalog.find(c => c.name === name).card_id;
for (const lateBase of [false, true]) {
  for (const ship of ['Embassy Yacht', 'Stealth Needle']) {
    test(`${ship} checks bases only on play (late base: ${lateBase})`, () => {
      const game = new Game(1);
      const player = game.players[0];
      const base = card('Battle Station');
      player.in_play = [new InPlay(100, base)];
      if (!lateBase) player.in_play.push(new InPlay(101, card('Space Station')));
      if (ship === 'Stealth Needle') {
        const yacht = new InPlay(102, card('Embassy Yacht'));
        yacht.activated = true;
        player.in_play.push(yacht);
      }
      player.deck = [card('Scout'), card('Explorer'), card('Scout')];
      player.hand = [card(ship)];
      game.play(player, 0);
      assert.equal(player.in_play.at(-1).activated, true);
      if (lateBase) {
        assert.deepEqual(player.hand, []);
        player.hand.push(card('Space Station'));
        game.play(player, 0);
      }
      const expected = lateBase ? [] : [card('Scout'), card('Explorer')];
      assert.deepEqual(player.hand, expected);
      game.allies(player);
      assert.deepEqual(player.hand, expected);
    });
  }
}
