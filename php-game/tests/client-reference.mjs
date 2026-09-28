import assert from 'node:assert/strict';
import fs from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { Game } from '../assets/runtime/engine.mjs';
import { Encoder } from '../assets/runtime/encoder.mjs';
import { Actor } from '../assets/runtime/actor.mjs';
import { Lethal } from '../assets/runtime/lethal.mjs';
Game.catalog=JSON.parse(fs.readFileSync(new URL('../assets/cards.json',import.meta.url)));
const fixture=JSON.parse(fs.readFileSync(process.argv[2]||'/tmp/astro-js-reference.json'));
let total=0,lethal=0;
for(const f of fixture.games) {
  let cursor=0;
  const g=new Game(f.seed,f.starts,(g,p,d)=>f.decisions[cursor].selected);
  g.decision_hook=(g,p,d,a)=> {
    const expected=f.decisions[cursor],context=`Seed ${f.seed}, decision ${cursor}`;
    assert.equal(p,expected.player,context); assert.equal(d.family,expected.family,context);
    assert.deepEqual(d.observation,expected.observation,context+' observation');
    assert.deepEqual(d.actions.map(({opaque,...a})=>a),expected.actions,context+' actions');
    assert.equal(Game.key(a),Game.key(d.actions[expected.selected]),context+' chosen');
    if(expected.lethal) { const plan=Lethal.plan(g,d).map(([family,key])=>[family,Object.values(JSON.parse(key))]); assert.deepEqual(plan,expected.lethal,context+' lethal'); lethal++; }
    cursor++;
  };
  g.run(); assert.equal(cursor,f.decisions.length); assert.deepEqual(g.result,f.result); total+=cursor;
}
console.log(`PASS ${fixture.games.length} games, ${total} decisions, ${lethal} lethal plans match Python`);
const raw=gunzipSync(fs.readFileSync(new URL('../models/level-10.astro.gz',import.meta.url)));
const actor=new Actor(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)),enc=new Encoder(2); let maxError=0;
for(const f of fixture.neural) {
  const state=enc.state(f.observation),actions=f.actions.map(a=>enc.action(a,f.observation));
  state.forEach((v,j)=>assert.ok(Math.abs(v-f.state[j])<1e-6,`State ${j}`));
  actions.forEach((a,i)=>a.forEach((v,j)=>assert.ok(Math.abs(v-f.encoded_actions[i][j])<1e-6,`${f.family} action ${i}/${j}`)));
  const scores=actor.scores(state,actions,Encoder.FAMILIES[f.family]);
  scores.forEach((v,i)=> { const err=Math.abs(v-f.scores[i]); maxError=Math.max(maxError,err); assert.ok(err<0.002,`${f.family}: error ${err}`); });
  assert.equal(scores.indexOf(Math.max(...scores)),f.scores.indexOf(Math.max(...f.scores)),f.family+' chosen action');
}
console.log(`PASS all ${fixture.neural.length} neural families; maximum logit error ${maxError}`);

for (const level of fixture.levels || []) {
  const raw=gunzipSync(fs.readFileSync(new URL(`../models/${level.id}.astro.gz`,import.meta.url)));
  const actor=new Actor(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
  for(const check of level.checks) {
    const decision=fixture.neural.find(d=>d.family===check.family),e=actor.encoder;
    const state=e.state(decision.observation),actions=decision.actions.map(a=>e.action(a,decision.observation));
    state.forEach((v,j)=>assert.ok(Math.abs(v-check.state[j])<1e-6));
    actions.forEach((a,i)=>a.forEach((v,j)=>assert.ok(Math.abs(v-check.encoded_actions[i][j])<1e-6)));
    const scores=actor.scores(state,actions,Encoder.FAMILIES[check.family]);
    if(check.win_probability!==undefined) assert.ok(Math.abs(actor.winProbability(decision)-check.win_probability)<0.0001, `${level.id} ${check.family} win estimate matches Python`);
    scores.forEach((v,i)=>assert.ok(Math.abs(v-check.scores[i])<0.002,`${level.id} ${check.family}`));
    assert.equal(scores.indexOf(Math.max(...scores)),check.scores.indexOf(Math.max(...check.scores)),`${level.id} ${check.family} choice`);
  }
}
if(fixture.levels) console.log(`PASS all ${fixture.levels.length} levels: encodings, scores, win estimates (when included), and chosen actions match NumPy across eight families`);
