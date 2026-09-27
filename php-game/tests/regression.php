<?php
declare(strict_types=1);
require __DIR__.'/../src/bootstrap.php';
use Astro\Game;
use Astro\Pause;
use Astro\InPlay;
use Astro\Session;
use Astro\Lethal;
function ok(bool $condition,string $message): void { if(!$condition) throw new RuntimeException($message); echo "PASS: $message\n"; }
function setup(array $hand): Game { $g=new Game(17); $p=$g->players[0]; $p->hand=$hand; $p->discard=$p->deck=$p->known_top=[]; $p->in_play=[]; $g->active_player=0; return $g; }
function plan(Game $g): array { return $g->playAllPlan(['family'=>'main']); }
$g=setup([0,0]); $before=serialize([$g->players,$g->trade_row]); ok(count(plan($g))===2,'Play all handles duplicate Scouts'); ok($before===serialize([$g->players,$g->trade_row]),'Play all preflight does not mutate the game');
ok(!plan(setup([0])),'Play all requires at least two cards');
ok(!plan(setup([22,0])),'Patrol Mech mode blocks Play all');
ok(!plan(setup([25,0])),'Scrap choice blocks Play all');
ok(count(plan(setup([0,25])))===2,'No remaining scrap target means no extra choice');
ok(count(plan(setup([21,0])))===2,'Missile Mech with no enemy bases does not need a choice');
$g=setup([21,0]); $g->players[1]->in_play=[new InPlay(90,15)]; ok(!plan($g),'Optional destroy target blocks Play all');
ok(count(plan(setup([9,16])))===2,'Bases with later manual abilities can be played together');
ok(count(plan(setup([0,23])))===2,'Single forced copy target allows Play all');
ok(!plan(setup([0,1,23])),'Copy targets are checked at the time of play');
ok(count(plan(setup([23,0])))===2,'Stealth Needle without a target resolves without choice');
$g=setup([27,0]); $g->players[0]->deck=[25]; ok(count(plan($g))===2,'Newly drawn cards stay outside the current Play all batch');
$g=setup([0,0]); $g->active_player=1; ok(!plan($g),'Computer cannot use Play all');
$g=setup([0,0]); ok(!array_filter($g->mainActions($g->players[0]),fn($a)=>$a['kind']==='play_all'),'Play all is absent from model action space');
$g=setup([2]); $g->play($g->players[0],0); $a=array_values(array_filter($g->mainActions($g->players[0]),fn($a)=>$a['kind']==='scrap_for_ability'))[0]; $g->apply($g->players[0],$a); ok($g->explorers_remaining===11 && $g->players[0]->combat===2,'Explorer scrap recycles into supply');
$g=setup([1,29,1]); foreach([0,0,0] as $j) $g->play($g->players[0],$j); ok($g->players[0]->combat===3,'Fleet HQ boosts ships played after it enters play');
$g=setup([39]); $g->players[0]->deck=[0,1]; $g->play($g->players[0],0); ok(count($g->players[0]->hand)===2,'Command Ship draws two');
$g=setup([9]); $g->play($g->players[0],0); $p=$g->players[0]; $p->blob_cards_played=3; $p->deck=[0,0,1,1]; $g->chooser=fn($g,$pid,$d)=>1; $g->effect($p,'blob_world',0,$p->in_play[0]); ok(count($p->hand)===3,'Blob World draws its full Blob count');
$g=setup([1]); $p=$g->players[0]; $op=$g->players[1]; $p->combat=9; $op->authority=5; $op->in_play=[new InPlay(50,48),new InPlay(51,8)];
$a=$g->mainActions($p); ok(!array_filter($a,fn($a)=>$a['kind']==='attack_player' || $a['target_card_id']===8),'Outposts protect authority and non-outpost bases');
$plan=Lethal::plan($g,['family'=>'main','observation'=>$g->observation(0),'actions'=>$a]); ok(count($plan)>0,'Lethal finisher finds an outpost-breaking win');
$s=Session::start('champion-10',true); $s['seed']=1234; $a=Session::advance($s); $b=Session::advance($s); ok($a===$b,'Refreshing deterministically reconstructs the saved game');
$original=$s; $a=Session::advance($s,'play_all'); ok(count($a['observation']['hand'])===0 && count($s['transcript'])===3,'Human Play all records individual engine moves');
ok($a['revision']===1,'A batch advances the revision once');
$copy=$original; try { Session::advance($copy,'choose',999); throw new RuntimeException('Illegal move was accepted'); } catch(InvalidArgumentException $e) { ok($copy===$original,'Illegal action cannot mutate the game'); }
// A full real champion turn, one inference decision per request.
$s=Session::start('champion-10',false); $s['seed']=1234; $a=Session::advance($s); $count=0;
while($a['status']==='model_thinking' && $count++<30) $a=Session::advance($s,'advance');
ok($a['status']==='your_turn','Champion takes a complete server-side turn');
ok(!isset($a['observation']['opponent_hand']) && !isset($a['observation']['opponent_deck']),'Client does not receive hidden opponent assignments');
echo 'Peak memory: '.round(memory_get_peak_usage(true)/1048576)." MB\n";
