import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import { Game } from '../assets/runtime/engine.mjs';
import { Encoder } from '../assets/runtime/encoder.mjs';
import { Actor } from '../assets/runtime/actor.mjs';
import { importModel } from '../assets/runtime/import-model.mjs';
Game.catalog=JSON.parse(fs.readFileSync(new URL('../assets/cards.json',import.meta.url)));
const setup=()=> { const game=new Game(42,0),p=game.players[1]; p.hand=[0,0,0,0,0]; p.discard=Array(8).fill(1); p.in_play=[]; p.known_top=[]; return {game,p}; };
for(let remaining=1;remaining<=5;remaining++) test(`cleanup remembers ${remaining} cards through shuffle, clone, and removal`,()=> {
  const {game,p}=setup(); p.deck=Array.from({length:remaining},(_,i)=>3+i); const expected=[...p.deck];
  game.cleanup(p);
  assert.deepEqual(game.observation(0).opponent_inferred_hand,expected);
  assert.deepEqual(game.observation(0).opponent_known_hand,[]);
  const clone=game.clone(); clone.discardHand(clone.players[1],clone.players[1].hand.indexOf(3));
  assert.deepEqual(clone.observation(0).opponent_inferred_hand,expected.slice(1));
  assert.deepEqual(game.observation(0).opponent_inferred_hand,expected);
  game.discardHand(p,p.hand.indexOf(3));
  assert.deepEqual(game.observation(0).opponent_inferred_hand,expected.slice(1));
});
test('known tops do not double count; later top draws of the same type are separate copies',()=> {
  const {game,p}=setup(); p.deck=[3,4]; p.known_top=[4]; game.cleanup(p);
  assert.deepEqual(game.observation(0).opponent_known_hand,[4]);
  assert.deepEqual(game.observation(0).opponent_inferred_hand,[3,4]);
  game.place(p,3,true); game.draw(p,1);
  assert.deepEqual(game.observation(0).opponent_inferred_hand,[3,3,4]);
  game.discardHand(p,p.hand.indexOf(3));
  assert.deepEqual(game.observation(0).opponent_inferred_hand,[3,4]);
  p.deck=[];
  assert.deepEqual(game.observation(0).opponent_inferred_hand,[...p.hand].sort((a,b)=>a-b));
});
test('six unknown cards do not expose the five drawn',()=> {
  const {game,p}=setup(); p.deck=[3,4,5,6,7,8]; game.cleanup(p);
  assert.deepEqual(game.observation(0).opponent_inferred_hand,[]);
});
test('arch3 extends the old prefix with unordered inferred counts',()=> {
  const {game}=setup(),obs={...game.observation(0),opponent_inferred_hand:[3,3,4]};
  const old=new Encoder(2).state(obs),encoder=new Encoder(3),state=encoder.state(obs);
  assert.equal(state.length,1341); assert.deepEqual(state.slice(0,1292),old);
  assert.equal(state[1295],2); assert.equal(state[1296],1);
  assert.deepEqual(encoder.state({...obs,trade_deck:[...obs.trade_deck].reverse(),opponent_inferred_hand:[4,3,3]}),state);
  const {opponent_inferred_hand,...legacy}=obs;
  assert.deepEqual(encoder.state(legacy).slice(1292),encoder.state({...legacy,opponent_inferred_hand:legacy.opponent_known_hand}).slice(1292));
});
test('arch3 NPZ and packaged model import preserve weights and validate dimensions',async()=> {
  const registry=JSON.parse(fs.readFileSync(new URL('../models/registry.json',import.meta.url))),entry=registry.at(-1);
  assert.equal(entry.name,'Level 5 (1283 ELO)');
  const buffer=b=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);
  const packaged=await importModel(buffer(fs.readFileSync(new URL('../models/'+entry.file,import.meta.url))));
  const native=await importModel(buffer(fs.readFileSync(new URL('../../'+entry.source,import.meta.url))));
  const actor=new Actor(packaged),npz=new Actor(native);
  assert.equal(actor.spec.encoder_version,3); assert.equal(actor.spec.state_size,1341);
  assert.deepEqual(actor.spec,npz.spec);
  for(const name of Object.keys(actor.tensors)) assert.deepEqual(actor.tensors[name],npz.tensors[name]);
  const corrupted=new TextEncoder().encode(new TextDecoder().decode(new Uint8Array(packaged,12,new DataView(packaged).getUint32(8,true))).replace('"state_size":1341','"state_size":1292'));
  const bad=packaged.slice(0); new Uint8Array(bad,12,corrupted.length).set(corrupted);
  assert.throws(()=>new Actor(bad),/Unsupported model architecture/);
});
