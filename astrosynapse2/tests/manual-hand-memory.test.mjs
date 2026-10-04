import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const source=await readFile(new URL('../app/manual-hard-ai-match.tsx',import.meta.url),'utf8');
const parsed=ts.createSourceFile('manual.tsx',source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
const names=new Set(['uid','card','undefinedCards','starterPool','createMatch','originalCard','effectiveCardId','effectiveDefinition','definitionList','remainingTradeDeck','buildObservation','drawHardCards','inferredHardHand','hiddenHardHandCard','discardRemainingHardHand','loadSavedMatch','hardLegalActionKinds','hardAttackTargets','affordableHardAcquisitions']);
const selected=parsed.statements.filter(s=>ts.isFunctionDeclaration(s)&&names.has(s.name?.text)).map(s=>s.getText(parsed)).join('\n');
const context=vm.createContext({});
vm.runInContext(ts.transpile('let uidSequence=0; const STORAGE_KEY="test";\n'+selected,{target:ts.ScriptTarget.ES2022}),context);
const catalog=JSON.parse(await readFile(new URL('../../php-game/assets/cards.json',import.meta.url),'utf8'));
const definitions=new Map(catalog.map(c=>[c.card_id,c]));
const ids=cards=>Array.from(cards,c=>c.cardId).sort((a,b)=>a-b);
const make=()=>context.createMatch('arch3-champion','astro5');
const cards=ids=>ids.map(id=>context.card(id));
const observation=m=>context.buildObservation(m,catalog,definitions);
function checkPool(m) {
  const o=observation(m);
  assert.equal(o.opponent_hidden.length+o.opponent_known_hand.length+o.opponent_known_top.length,o.opponent_hand_count+o.opponent_deck_count);
  assert.ok(o.opponent_inferred_hand.length<=o.opponent_hand_count);
  return o;
}
for(let count=1;count<=5;count++) test(`remembers ${count} exhausted-deck cards through shuffle and saved match`,()=> {
  let m=make();m.hard={...m.hard,handCount:0,deckCount:count,hidden:cards(Array.from({length:count},(_,i)=>i+3)),discard:cards(Array(8).fill(0))};
  m=context.drawHardCards(m,5);checkPool(m);
  assert.deepEqual(ids(context.inferredHardHand(m.hard)),Array.from({length:count},(_,i)=>i+3));
  context.window={localStorage:{getItem:()=>JSON.stringify(m)}};
  const restored=context.loadSavedMatch(make());
  assert.deepEqual(ids(context.inferredHardHand(restored.hard)),ids(context.inferredHardHand(m.hard)));
  assert.deepEqual(Array.from(observation(restored).opponent_inferred_hand,c=>c.card_id).sort((a,b)=>a-b),ids(context.inferredHardHand(m.hard)));
});
test('six-card deck reveals no hand membership',()=> {
  let m=make();m.hard={...m.hard,handCount:0,deckCount:6,hidden:cards([3,4,5,6,7,8]),discard:[]};
  m=context.drawHardCards(m,5);checkPool(m);
  assert.deepEqual(ids(context.inferredHardHand(m.hard)),[]);
});
test('mid-turn exhaustion remembers old hand before mixing in discard',()=> {
  let m=make();m.hard={...m.hard,handCount:2,deckCount:1,hidden:cards([3,4,5]),discard:cards([0,0,0,0])};
  const old=m;
  m=context.drawHardCards(m,2);checkPool(m);
  assert.deepEqual(ids(context.inferredHardHand(m.hard)),[3,4,5]);
  assert.ok(old.hard.hidden.every(c=>!c.inferredInHand),'draw does not mutate previous state');
});
test('known top draws are distinct copies and deck count includes them once',()=> {
  let m=make();m.hard={...m.hard,handCount:0,deckCount:2,hidden:cards([3]),knownTop:cards([3]),discard:cards([0,0,0,0,0])};
  checkPool(m);m=context.drawHardCards(m,5);checkPool(m);
  assert.deepEqual(ids(m.hard.knownHand),[3]);
  assert.deepEqual(ids(context.inferredHardHand(m.hard)),[3,3]);
  // A later top acquisition does not invalidate the existing inferred copy.
  m.hard.knownTop.unshift(context.card(3));m.hard.deckCount++;
  checkPool(m);m=context.drawHardCards(m,1);checkPool(m);
  assert.deepEqual(ids(context.inferredHardHand(m.hard)),[3,3,3]);
});
test('hand removal selects a remembered copy before a same-type unknown copy',()=> {
  const m=make();m.hard.hidden=cards([3,3]);m.hard.hidden[1].inferredInHand=true;
  const found=context.hiddenHardHandCard(m.hard,3);
  assert.equal(found.uid,m.hard.hidden[1].uid);
  assert.equal(context.originalCard(found).inferredInHand,undefined);
});
test('cleanup discards known leftover hand and forgets its old membership',()=> {
  let m=make();m.hard={...m.hard,handCount:2,deckCount:1,hidden:cards([3,4]),knownHand:cards([5]),discard:cards([0,0,0,0,0])};
  m.hard.hidden[0].inferredInHand=true;
  m=context.discardRemainingHardHand(m);
  assert.deepEqual(ids(m.hard.hidden),[4]);assert.equal(m.hard.handCount,0);
  assert.deepEqual(ids(m.hard.discard),[0,0,0,0,0,3,5]);
  m=context.drawHardCards(m,5);checkPool(m);
  assert.deepEqual(ids(context.inferredHardHand(m.hard)),[4]);
});
test('unknown leftover hand must be recorded, never guessed from hidden deck order',()=> {
  const m=make();assert.throws(()=>context.discardRemainingHardHand(m),/remaining hand cards as discards/);
});
test('old saved matches without flags retain only deductions justified now',()=> {
  const m=make();context.window={localStorage:{getItem:()=>JSON.stringify(m)}};
  const restored=context.loadSavedMatch(make());
  assert.deepEqual(ids(context.inferredHardHand(restored.hard)),[]);
});

test('cleanup discards are recordable when there is no pending discard effect',()=> {
  const m=make();
  const kinds=Array.from(context.hardLegalActionKinds(m,definitions,''));
  assert.ok(kinds.includes('discard')); assert.ok(kinds.includes('end_turn'));
  m.hard.handCount=0;
  assert.ok(!Array.from(context.hardLegalActionKinds(m,definitions,'')).includes('discard'));
});
