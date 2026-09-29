import assert from 'node:assert/strict';
import fs from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {Game,InPlay,Pause} from '../assets/runtime/engine.mjs';
import {Session} from '../assets/runtime/session.mjs';
import {Actor} from '../assets/runtime/actor.mjs';
import {Lethal} from '../assets/runtime/lethal.mjs';
Game.catalog=JSON.parse(fs.readFileSync(new URL('../assets/cards.json',import.meta.url)));
let count=0;
const ok=(v,message)=> {assert.ok(v,message);count++;};
function setup(hand) { const g=new Game(17),p=g.players[0]; Object.assign(p,{hand,discard:[],deck:[],known_top:[],in_play:[]}); return g; }
const plan=g=>g.playAllPlan({family:'main'});
let g=setup([0,0]),before=JSON.stringify([g.players,g.trade_row]); ok(plan(g).length===2,'Duplicate Scouts'); ok(JSON.stringify([g.players,g.trade_row])===before,'No mutation on probe');
for(const [hand,n] of [[[0],0],[[22,0],0],[[25,0],0],[[0,25],2],[[21,0],2],[[9,16],2],[[0,23],2],[[0,1,23],0],[[23,0],2]]) ok(plan(setup(hand)).length===n,'Play all '+hand);
g=setup([21,0]);g.players[1].in_play=[new InPlay(90,15)];ok(!plan(g).length,'Optional destroy blocks batch');
g=setup([27,0]);g.players[0].deck=[25];ok(plan(g).length===2,'New draws outside batch');
g=setup([0,0]);g.active_player=1;ok(!plan(g).length,'Computer cannot batch');
g=setup([0,0]);ok(!g.mainActions(g.players[0]).some(a=>a.kind==='play_all'),'No model batch action');
g=setup([2]);g.play(g.players[0],0);g.apply(g.players[0],g.mainActions(g.players[0]).find(a=>a.kind==='scrap_for_ability'));ok(g.explorers_remaining===11&&g.players[0].combat===2,'Explorer recycles');
g=setup([1,29,1]);for(let j=0;j<3;j++)g.play(g.players[0],0);ok(g.players[0].combat===3,'Fleet HQ timing');
g=setup([39]);g.players[0].deck=[0,1];g.play(g.players[0],0);ok(g.players[0].hand.length===2,'Command ship draws');
g=setup([9]);g.play(g.players[0],0);g.players[0].blob_cards_played=3;g.players[0].deck=[0,0,1,1];g.chooser=()=>1;g.effect(g.players[0],'blob_world',0,g.players[0].in_play[0]);ok(g.players[0].hand.length===3,'Blob draw count');
g=setup([1]);g.players[0].combat=9;g.players[1].authority=5;g.players[1].in_play=[new InPlay(50,48),new InPlay(51,8)];ok(!g.mainActions(g.players[0]).some(a=>a.kind==='attack_player'||a.target_card_id===8),'Outposts protect');ok(Lethal.plan(g,{family:'main',observation:g.observation(0)}).length>0,'Lethal outpost plan');
const raw=gunzipSync(fs.readFileSync(new URL('../models/level-10.astro.gz',import.meta.url))),actor=new Actor(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)),m={id:'level-10',name:'Level 10'};
let s=Session.start(m,true,1234),a=Session.advance(s,actor),b=Session.advance(s,actor);assert.deepEqual(a,b);count++;
const original=structuredClone(s);a=Session.advance(s,actor,'play_all');ok(a.observation.hand.length===0&&s.transcript.length===3,'Batch records each move');ok(a.revision===1,'One batch revision');
s=structuredClone(original);assert.throws(()=>Session.advance(s,actor,'choose',999));assert.deepEqual(s,original);count++;
s=Session.start(m,false,1234);a=Session.advance(s,actor);let moves=0;
const start=performance.now();while(a.status==='model_thinking'&&moves++<30)a=Session.advance(s,actor,'advance');ok(a.status==='your_turn','Full champion turn');
const replay=Session.advance(structuredClone(s),actor);assert.deepEqual(replay,a);count++;
const championTurnMs=performance.now()-start;
// Human-only pauses must survive deduplication, batching, and replay.
g=setup([0,0,0]); g.manual_player=0; g.players[0].must_discard=2;
let discardChoices=0;
g.chooser=(game,pid,d)=> {
  if(d.family==='discard') { discardChoices++; return 0; }
  throw new Pause(d);
};
assert.throws(()=>g.takeTurn(g.players[0]),Pause);
ok(discardChoices===2&&g.players[0].hand.length===1,'Identical discards each require a human choice');
g=setup([0]); g.manual_player=0; g.players[0].must_discard=1;
g.chooser=()=> { throw new Pause({family:'discard'}); };
assert.throws(()=>g.takeTurn(g.players[0]),Pause);
ok(g.players[0].hand.length===1&&g.players[0].must_discard===1,'The last card is not automatically discarded');
g=setup([]); g.manual_player=0; g.chooser=(game,pid,d)=> { throw new Pause(d); };
assert.throws(()=>g.takeTurn(g.players[0]),e=>e instanceof Pause&&e.decision.actions.length===1&&e.decision.actions[0].kind==='end_turn');
ok(g.active_player===0,'End Turn waits even when it is the only action');
let manualEnd=false;
for(let seed=1;seed<=100&&!manualEnd;seed++) {
  const session=Session.start(m,true,seed);
  let state=Session.advance(session,actor);
  state=Session.advance(session,actor,'play_all');
  const attack=state.decision?.actions.find(a=>a.kind==='attack_player');
  if(attack) state=Session.advance(session,actor,'choose',attack.id);
  if(state.decision?.actions.length!==1||state.decision.actions[0].kind!=='end_turn') continue;
  ok(state.status==='your_turn','Play all and attack never auto-end a spent turn');
  assert.deepEqual(Session.advance(structuredClone(session),actor),state);count++;
  const next=Session.advance(session,actor,'choose',state.decision.actions[0].id);
  ok(next.status==='model_thinking','Only explicit End Turn starts the opponent');
  manualEnd=true;
}
ok(manualEnd,'Exercised a one-action end-turn session');
// Check value-head semantics independently of action policy logits.
const valueActor=Object.create(Actor.prototype);
valueActor.spec={objective_version:2,bootstrap_heads:2};
valueActor.encoder={state:()=>[]}; valueActor.stateFeatures=()=>[];
valueActor.linear=()=>Float64Array.from([0,Math.log(3)]);
assert.ok(Math.abs(valueActor.winProbability({observation:{},family:'main'})-0.625)<1e-12);count++;
const view=Session.advance(Session.start(m,true,1234),actor);
ok(view.opponent_win_probability>=0&&view.opponent_win_probability<=1,'Opponent win estimate is a probability');
// Session advances into turn 1 before evaluation; compare to its actual observation.
const evaluated=actor.winProbability({family:'main',observation:{...view.observation,action_number:view.observation.action_number-1},actions:view.decision.actions});
ok(Math.abs(view.opponent_win_probability-(1-evaluated))<1e-12,'Human-turn estimate is complemented to opponent perspective');
// An asymmetric estimate makes a reversed perspective visible on either turn.
const fixedValueActor={winProbability:()=>0.8};
for(const humanStarts of [true,false]) {
  const state=Session.advance(Session.start(m,humanStarts,1234),fixedValueActor);
  ok(Math.abs(state.opponent_win_probability-(humanStarts?0.2:0.8))<1e-12,
    `AI win probability stays in AI perspective when ${humanStarts?'human':'AI'} starts`);
}
console.log(`PASS ${count} regressions; champion opening turn ${championTurnMs.toFixed(1)} ms (${moves} decisions)`);

// Resignation is terminal, replayable, and does not advance either player's turn.
for (const humanStarts of [true, false]) {
  const session = Session.start(m, humanStarts, 1234);
  const before = Session.advance(session, actor);
  const transcript = [...session.transcript];
  const resigned = Session.advance(session, actor, 'resign');
  assert.equal(resigned.status, 'complete');
  assert.deepEqual(resigned.result, {winner: 1, truncated: false, resigned: true});
  assert.equal(resigned.revision, before.revision + 1);
  assert.equal(resigned.decision, null);
  assert.equal(resigned.can_play_all, false);
  assert.deepEqual(session.transcript, transcript);
  assert.deepEqual(Session.advance(JSON.parse(JSON.stringify(session)), actor), resigned);
  assert.throws(() => Session.advance(session, actor, 'advance'), /complete/);
}
console.log('Resignation and restore regressions passed');
