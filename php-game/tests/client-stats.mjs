import assert from 'node:assert/strict';
import {recordResult, readStats} from '../assets/runtime/stats.mjs';
import {Session} from '../assets/runtime/session.mjs';
const entries = new Map();
const storage = {get length(){return entries.size;}, key:i=>[...entries.keys()][i], getItem:k=>entries.get(k)??null, setItem:(k,v)=>entries.set(k,v)};
const game = {id:'one', model_id:'level-3', model_label:'Level 3', status:'complete', result:{winner:0}};
recordResult(storage,'test:',game); recordResult(storage,'test:',game);
recordResult(storage,'test:',{...game,id:'two',result:{winner:1}});
recordResult(storage,'test:',{...game,id:'three',result:{winner:null,truncated:true}});
recordResult(storage,'test:',{...game,id:'four',status:'your_turn'});
recordResult(storage,'test:',{...game,id:'five',model_id:'level-8'});
entries.set('test:bad','invalid json');
assert.deepEqual(readStats(storage,'test:').get('level-3'),{label:'Level 3',wins:1,losses:1,draws:1});
assert.equal(readStats(storage,'test:').get('level-8').wins,1);
assert.equal(readStats(storage,'other:').size,0);
const original = crypto.getRandomValues;
try {
  for (const bit of [0,1]) {
    crypto.getRandomValues = a => {a[0]=bit;return a;};
    const session = Session.start({id:'level-1',name:'Level 1'});
    assert.equal(session.starts,bit);
    assert.equal(JSON.parse(JSON.stringify(session)).starts,bit);
  }
} finally {crypto.getRandomValues=original;}
console.log('Stats persistence, deduplication, level attribution, draws, and random starters passed.');
