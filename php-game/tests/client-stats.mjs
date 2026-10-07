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

const {readLevelStats, winInterval} = await import('../assets/runtime/stats.mjs');
entries.set('test:padded', JSON.stringify({model:'level-03', outcome:'wins', completedAt:20}));
entries.set('test:earlier', JSON.stringify({model:'level-3', outcome:'losses', completedAt:10}));
for (const key of ['test:one', 'test:two', 'test:three']) { const record = JSON.parse(entries.get(key)); delete record.completedAt; entries.set(key, JSON.stringify(record)); }
const rows = readLevelStats(storage, 'test:');
assert.deepEqual(rows.map(r => r.level), [5,4,3,2,1]);
assert.equal(rows.length, 5);
assert.deepEqual([rows[2].wins, rows[2].losses, rows[2].draws], [2,2,1]);
assert.equal(rows[2].points.length, 4);
assert.equal(rows[2].points.at(-1).rate, .5);
assert.deepEqual(rows[2].points.slice(-2).map(p => p.completedAt), [10,20]);
assert.equal(rows[0].points.length, 0);
assert.equal(winInterval(0,0), null);
assert.ok(Math.abs(winInterval(1,1)[0] - .206549314) < 1e-8);
assert.ok(Math.abs(winInterval(0,1)[1] - .793450686) < 1e-8);
assert.ok(winInterval(50,100)[1] - winInterval(50,100)[0] < winInterval(5,10)[1] - winInterval(5,10)[0]);
console.log('Five descending levels, legacy ID merging, chronological cumulative rates, and Wilson intervals passed.');
