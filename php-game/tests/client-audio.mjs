import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Game, InPlay } from '../assets/runtime/engine.mjs';
import { Session } from '../assets/runtime/session.mjs';
import { GameAudio, SOUNDS } from '../assets/runtime/audio.mjs';
Game.catalog = JSON.parse(fs.readFileSync(new URL('../assets/cards.json', import.meta.url)));
const card = name => Game.catalog.find(c => c.name === name).card_id;
for (const pid of [0, 1]) {
  const g = new Game(17), p = g.players[pid];
  p.hand = [card('Cutter')]; p.in_play = [];
  g.play(p, 0);
  assert.deepEqual(g.sounds, ['trade', 'authority']);
  g.sounds = []; p.hand = [card('Ram'), card('Ram')]; p.in_play = [];
  g.play(p, 0); g.play(p, 0);
  assert.deepEqual(g.sounds, ['combat', 'combat', 'combat', 'combat']);
  g.sounds = []; p.deck = []; p.discard = [0];
  g.draw(p, 1); g.draw(p, 1);
  assert.deepEqual(g.sounds, ['shuffle']);
  g.sounds = []; g.scrap(2); g.effect(p, 'gain_combat', 2);
  assert.deepEqual(g.sounds, ['scrap', 'combat']);
  g.sounds = []; g.apply(p, Game.action('attack_player'));
  assert.deepEqual(g.sounds, ['attack']);
  g.sounds = []; const base = new InPlay(99, card('Battle Station'));
  g.players[1-pid].in_play = [base]; p.combat = 10;
  g.apply(p, Game.action('attack_base', -1, base.card, '', '', 0, 0, [99]));
  assert.deepEqual(g.sounds, ['attack']);
  g.sounds = []; p.hand = []; p.deck = []; p.discard = [0]; p.in_play = [];
  g.cleanup(p); assert.deepEqual(g.sounds, ['shuffle']);
  g.sounds = []; const clone = g.clone(); clone.effect(clone.players[pid], 'gain_trade', 3);
  assert.deepEqual(g.sounds, [], 'Planning clones cannot emit live sounds');
}
const actor = { winProbability: () => 0.5, choose: () => 0 };
const session = Session.start({id:'test', name:'Test'}, true, 1234);
const opening = Session.advance(session, actor);
assert.deepEqual(opening.sounds, ['playerturn']);
const played = Session.advance(session, actor, 'play_all');
assert.equal(played.sounds.length - opening.sounds.length, 3, 'Each starter card emits its own sound');
assert.deepEqual(Session.advance(session, actor).sounds, played.sounds, 'Replay preserves event offsets');

let now = 0, timerId = 0;
const timers = new Map(), starts = [], loads = [], decodes = [];
class Context {
  state = 'suspended'; destination = {};
  get currentTime() { return now / 1000; }
  addEventListener() {}
  createGain() { return { connect() {}, gain: { setValueAtTime: value => { this.volume = value; } } }; }
  async resume() { this.state = 'running'; }
  async decodeAudioData(bytes) { decodes.push(bytes); return { duration: 2 }; }
  createBufferSource() { return { connect() {}, disconnect() {}, stop() {}, start() { starts.push(now); } }; }
}
const audio = new GameAudio({ Context, load: async path => { loads.push(path); return new ArrayBuffer(1); },
  later: (fn, delay) => { timers.set(++timerId, {fn, at:now+delay}); return timerId; }, cancel: id => timers.delete(id) });
function tick(ms) { now += ms; for (const [id,timer] of [...timers]) if (timer.at <= now) { timers.delete(id); timer.fn(); } }
await audio.preload(); await audio.preload();
assert.equal(loads.length, SOUNDS.length); assert.equal(decodes.length, SOUNDS.length);
audio.enqueue(['combat', 'trade', 'combat']); assert.deepEqual(starts, []);
audio.unlock(); await Promise.resolve(); assert.deepEqual(starts, [0]);
tick(249); assert.deepEqual(starts, [0]); tick(1); assert.deepEqual(starts, [0,250]);
tick(250); assert.deepEqual(starts, [0,250,500], 'Long buffers overlap with independent starts');
tick(300); audio.enqueue(['scrap']); assert.deepEqual(starts, [0,250,500,800], 'Idle queue starts immediately');
tick(100); audio.enqueue(['authority']); tick(149); assert.equal(starts.length,4); tick(1); assert.equal(starts.at(-1),1050);
audio.setVolume(0.3); assert.equal(audio.context.volume,0.3);
audio.enqueue(['attack']); audio.clear(); tick(300); assert.equal(starts.length,5);
const failed = new GameAudio({ Context, load: async () => { throw new Error('offline'); } });
await failed.preload(); failed.enqueue(['trade']); failed.unlock(); await Promise.resolve();
assert.equal(failed.queue.length,0,'Unavailable audio does not block the game');
// Browser timers require the Window receiver, unlike Node's native timers.
const nativeSetTimeout = globalThis.setTimeout, nativeClearTimeout = globalThis.clearTimeout;
try {
  globalThis.setTimeout = function(fn, delay) {
    assert.equal(this, globalThis, 'setTimeout must receive the browser global');
    timers.set(++timerId, {fn, at:now+delay}); return timerId;
  };
  globalThis.clearTimeout = function(id) {
    assert.equal(this, globalThis, 'clearTimeout must receive the browser global');
    timers.delete(id);
  };
  const browserAudio = new GameAudio({ Context, load: async () => new ArrayBuffer(1) });
  await browserAudio.preload();
  await browserAudio.context.resume();
  const before = starts.length;
  browserAudio.enqueue(['combat', 'trade', 'attack']);
  assert.equal(starts.length, before + 1);
  tick(250);
  assert.equal(starts.length, before + 2);
  browserAudio.clear();
  tick(250);
  assert.equal(starts.length, before + 2, 'Clearing cancels the pending browser timer');
  assert.equal(timers.size, 0);
} finally {
  globalThis.setTimeout = nativeSetTimeout;
  globalThis.clearTimeout = nativeClearTimeout;
}
console.log('PASS audio effects for both players, replay, batching, shuffle, preload, volume, 250 ms overlapping queue and browser timer receivers');
