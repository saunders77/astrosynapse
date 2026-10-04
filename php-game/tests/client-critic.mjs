import assert from 'node:assert/strict';
import fs from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Critic } from '../assets/runtime/critic.mjs';
import { Actor } from '../assets/runtime/actor.mjs';
import { Encoder } from '../assets/runtime/encoder.mjs';
import { Game } from '../assets/runtime/engine.mjs';
import { Session } from '../assets/runtime/session.mjs';

const root=new URL('../../',import.meta.url), arena=new URL('../',import.meta.url);
const registry=JSON.parse(fs.readFileSync(new URL('models/registry.json',arena)));
const level=registry.find(m=>m.id==='level-05');
const load=file=>{const b=gunzipSync(fs.readFileSync(new URL('models/'+file,arena)));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const critic=new Critic(load(level.critic.file)), actor=new Actor(load(level.file));
Game.catalog=JSON.parse(fs.readFileSync(new URL('assets/cards.json',arena)));
assert.deepEqual(registry.filter(m=>m.critic).map(m=>m.id),['level-05']);
// Compare real game states across all families against the original NumPy model.
const inputs=[], actual=[];
let session=Session.start(level,false,31415), view=Session.advance(session,actor);
for(let step=0;step<24;step++) {
  for(const [family,index] of Object.entries(Encoder.FAMILIES)) {
    const input=[...critic.encoder.state(view.observation),...Array(8).fill(0)];
    input[input.length-8+index]=1;
    inputs.push(input);actual.push(critic.winProbability({family,observation:view.observation}));
  }
  view=Session.advance(session,actor,view.status==='model_thinking'?'advance':'choose',0);
}
const expected=JSON.parse(execFileSync(fileURLToPath(new URL('astrosynapse2/.venv/bin/python',root)),['-c',`
import json, sys, numpy as np
sys.path.insert(0, 'astrosynapse2/backend')
from astro2.critic import IndependentCritic
model, _, _ = IndependentCritic.load(sys.argv[1])
print(json.dumps(model.predict(np.asarray(json.load(sys.stdin),dtype=np.float32)).tolist()))
`,fileURLToPath(new URL(level.critic.source,root))],{cwd:fileURLToPath(root),input:JSON.stringify(inputs),maxBuffer:4e6}));
const error=Math.max(...actual.map((p,i)=>Math.abs(p-expected[i])));
assert.ok(error<1e-5,`NumPy parity: ${error}`);
assert.throws(()=>new Critic(new ArrayBuffer(12)),/format/);
assert.throws(()=>new Critic(load(level.critic.file).slice(0,-4)),/tensor/);

// Exercise actual worker loading, hash rejection/retry, both perspectives,
// deterministic replay, and the unchanged level-4 fallback.
globalThis.self=globalThis;
const downloads=[];
let corrupt=true;
globalThis.fetch=async url=>{
  const path=new URL(url).pathname.slice(1);downloads.push(path);
  let bytes=fs.readFileSync(new URL(path,arena));
  if(path.endsWith(level.critic.file)&&corrupt) { bytes=Buffer.from(bytes);bytes[20]^=1; }
  return new Response(bytes);
};
const pending=new Map();let requestId=0;
globalThis.postMessage=message=>{if(message.requestId)pending.get(message.requestId)(message);};
await import('../assets/runtime/worker.mjs');
const request=data=>new Promise(resolve=>{const id=++requestId;pending.set(id,resolve);self.onmessage({data:{...data,requestId:id}});});
assert.match((await request({op:'init',base:'https://arena.test/',release:'critic-test'})).error,/Critic integrity/);
corrupt=false;
assert.equal((await request({op:'init',base:'https://arena.test/',release:'critic-test'})).error,undefined);
assert.equal(downloads.filter(p=>p.endsWith(level.critic.file)).length,2);
for(const humanStarts of [true,false]) {
  const saved=Session.start(level,humanStarts,1234);
  const plain=Session.advance(structuredClone(saved),actor);
  const chosen=actor.choose.bind(actor);
  const withCritic={choose:chosen,winProbability:d=>critic.winProbability(d)};
  const expected=Session.advance(structuredClone(saved),withCritic);
  const response=await request({op:'init',base:'https://arena.test/',release:'critic-test',saved});
  assert.deepEqual(response.data.game,expected);
  assert.notEqual(expected.opponent_win_probability,plain.opponent_win_probability);
  const next=await request({op:humanStarts?'choose':'advance',id:saved.id,revision:saved.revision,action_id:0});
  const old=structuredClone(saved);
  Session.advance(old,actor,humanStarts?'choose':'advance',0);
  assert.deepEqual(next.data.saved.transcript,old.transcript,'Critic must not change moves');
}
const level4=registry.find(m=>m.id==='level-04'), saved=Session.start(level4,true,4321);
const response=await request({op:'init',base:'https://arena.test/',release:'critic-test',saved});
assert.deepEqual(response.data.game,Session.advance(structuredClone(saved),new Actor(load(level4.file))));
console.log(`Critic: ${inputs.length} NumPy comparisons, max error ${error}; worker integrity, retry, perspectives, replay, policy and level-4 checks passed.`);
