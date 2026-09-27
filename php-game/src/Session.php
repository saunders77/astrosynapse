<?php
declare(strict_types=1);
namespace Astro;
final class Session {
    public static function start(string $model,bool $humanStarts): array {
        $m=Models::get($model);
        return ['id'=>bin2hex(random_bytes(12)),'seed'=>random_int(1,2000000000),'starts'=>$humanStarts?0:1,'model'=>$m['id'],'label'=>$m['name'],'transcript'=>[],'revision'=>0,'lethal'=>[]];
    }
    public static function advance(array &$session,string $operation='state',?int $actionId=null): array {
        $history=$session['transcript']; $cursor=0; $live=false; $pending=null; $pendingPlayer=null; $budget=$operation==='advance'?1:0;
        $humanSubmitted=false; $batch=[]; $batchStarted=false; $actor=null;
        $game=new Game($session['seed'],$session['starts']);
        $game->decision_hook=function($g,$pid,$d,$a) use(&$session,&$live): void {
            if(!$live || !$session['lethal']) return;
            if($session['lethal'][0]===[$d['family'],Game::key($a)]) array_shift($session['lethal']); else $session['lethal']=[];
        };
        $game->chooser=function(Game $g,int $pid,array $d) use(&$session,$history,&$cursor,&$live,&$budget,&$humanSubmitted,&$batch,&$batchStarted,&$actor,$operation,$actionId): int {
            if($cursor<count($history)) { $key=$history[$cursor++]; foreach($d['actions'] as $j=>$a) if(Game::key($a)===$key) return $j; throw new \RuntimeException('Game replay mismatch'); }
            $live=true;
            if($pid===0) {
                if($operation==='choose' && !$humanSubmitted) {
                    if($actionId===null || !isset($d['actions'][$actionId])) throw new \InvalidArgumentException('This move is no longer available');
                    $selected=$actionId; $humanSubmitted=true;
                } elseif($operation==='play_all') {
                    if(!$batchStarted) { $batch=$g->playAllPlan($d); $batchStarted=true; if(!$batch) throw new \InvalidArgumentException('Play all is not available for this hand'); }
                    if(!$batch || $d['family']!=='main') throw new Pause($d);
                    $key=array_shift($batch); $selected=null; foreach($d['actions'] as $j=>$a) if(Game::key($a)===$key) { $selected=$j; break; }
                    if($selected===null) throw new \RuntimeException('Play all plan no longer legal');
                } else throw new Pause($d);
            } else {
                if($budget<=0) throw new Pause($d); $budget--;
                if(!$session['lethal'] && $d['family']==='main') $session['lethal']=Lethal::plan($g,$d);
                $selected=null;
                if($session['lethal'] && $session['lethal'][0][0]===$d['family']) foreach($d['actions'] as $j=>$a) if(Game::key($a)===$session['lethal'][0][1]) { $selected=$j; break; }
                if($selected===null) {
                    $session['lethal']=[]; $indices=array_keys($d['actions']);
                    if($d['family']==='main' && array_filter($d['actions'],fn($a)=>in_array($a['kind'],['play_card','activate_base','activate_ally','attack_base','attack_player'],true))) $indices=array_values(array_filter($indices,fn($j)=>$d['actions'][$j]['kind']!=='end_turn'));
                    if(count($indices)===1) $selected=$indices[0]; else { $actor??=new Actor(Models::path(Models::get($session['model']))); $selected=$actor->choose($d,$indices); }
                }
            }
            $session['transcript'][]=Game::key($d['actions'][$selected]);
            return $selected;
        };
        try { $game->run(); } catch(Pause $e) { $pending=$e->decision; $pendingPlayer=$game->active_player; }
        if($operation==='choose' && !$humanSubmitted) throw new \InvalidArgumentException('The game is not waiting for your move');
        if($operation==='play_all' && !$batchStarted) throw new \InvalidArgumentException('Play all is not available now');
        if(count($session['transcript'])!==count($history)) $session['revision']++;
        $o=$game->observation(0);
        // Never send exact opponent hand/deck assignments or any shuffled order.
        $actions=[]; if($pendingPlayer===0) foreach($pending['actions'] as $j=>$a) { unset($a['opaque']); $actions[]=array_merge($a,['id'=>$j,'label'=>Game::label($a)]); }
        return ['id'=>$session['id'],'revision'=>$session['revision'],'model_label'=>$session['label'],
            'status'=>$game->result!==null?'complete':($pendingPlayer===0?'your_turn':'model_thinking'),
            'observation'=>$o,'decision'=>$pendingPlayer===0?['family'=>$pending['family'],'prompt'=>$pending['prompt'],'actions'=>$actions]:null,
            'can_play_all'=>$pendingPlayer===0 && (bool)$game->playAllPlan($pending),
            'action_log'=>array_slice($game->log,-180),'result'=>$game->result];
    }
}
