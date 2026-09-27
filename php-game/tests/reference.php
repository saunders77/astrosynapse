<?php
declare(strict_types=1);
require __DIR__.'/../src/bootstrap.php';
use Astro\Game;
use Astro\Encoder;
use Astro\Actor;
use Astro\Lethal;
function check(bool $ok,string $message): void { if(!$ok) throw new RuntimeException($message); }
function difference($a,$b,string $path=''): ?string {
    if(is_array($a) && is_array($b)) { if(count($a)!==count($b)) return $path.' count '.count($a).' != '.count($b); foreach($a as $k=>$v) { if(!array_key_exists($k,$b)) return $path.' missing '.$k; $d=difference($v,$b[$k],$path.'/'.$k); if($d!==null) return $d; } return null; }
    return $a==$b?null:$path.' '.json_encode($a).' != '.json_encode($b);
}
$fixture=json_decode(file_get_contents($argv[1]??'/tmp/astro-reference.json'),true,512,JSON_THROW_ON_ERROR);
if(($argv[2]??'')!=='neural') {
    $total=0;
    foreach($fixture['games'] as $f) {
        $cursor=0;
        $g=new Game($f['seed'],$f['starts'],function($g,$p,$d) use(&$cursor,$f) { return $f['decisions'][$cursor]['selected']; });
        $g->decision_hook=function($g,$p,$d,$a) use(&$cursor,$f) {
            $expected=$f['decisions'][$cursor];
            check($p===$expected['player'],'Player mismatch'); check($d['family']===$expected['family'],'Family mismatch');
            $diff=difference($d['observation'],$expected['observation']); check($diff===null,'Seed '.$f['seed'].' decision '.$cursor.' observation '.$diff);
            $actions=array_map(function($a) { unset($a['opaque']); return $a; },$d['actions']);
            $diff=difference($actions,$expected['actions']); check($diff===null,'Seed '.$f['seed'].' decision '.$cursor.' actions '.$diff);
            check(Game::key($a)===Game::key($d['actions'][$expected['selected']]),'Selected action mismatch');
            if(isset($expected['lethal'])) {
                $plan=array_map(fn($v)=>[$v[0],array_values(json_decode($v[1],true))],Lethal::plan($g,$d));
                $diff=difference($plan,$expected['lethal']); check($diff===null,'Lethal plan mismatch seed '.$f['seed'].' decision '.$cursor.' '.$diff);
            }
            $cursor++;
        };
        $g->run(); check($cursor===count($f['decisions']),'Decision count mismatch'); check($g->result===$f['result'],'Result mismatch'); $total+=$cursor;
    }
    echo 'PASS: '.count($fixture['games'])." complete games / $total decisions match Python rules and observations.\n";
}
if(($argv[2]??'')!=='rules') {
    $actor=new Actor(__DIR__.'/../models/champion-10.model.php'); $enc=new Encoder(2); $maxError=0;
    foreach($fixture['neural'] as $f) {
        $state=$enc->state($f['observation']); $actions=array_map(fn($a)=>$enc->action($a,$f['observation']),$f['actions']);
        foreach($state as $j=>$v) check(abs($v-$f['state'][$j])<1e-6,'State encoding mismatch '.$j);
        foreach($actions as $i=>$a) foreach($a as $j=>$v) check(abs($v-$f['encoded_actions'][$i][$j])<1e-6,'Action encoding mismatch '.$f['family'].' '.$i.'/'.$j);
        $scores=$actor->scores($state,$actions,Encoder::FAMILIES[$f['family']]);
        foreach($scores as $i=>$v) { $err=abs($v-$f['scores'][$i]); $maxError=max($maxError,$err); check($err<0.002,'Model score mismatch '.$f['family'].' '.$err); }
        check(array_search(max($scores),$scores)===array_search(max($f['scores']),$f['scores']),'Champion selected a different action');
        echo 'PASS: '.$f['family'].' encoding, inference and selected action ('.count($actions)." options).\n";
    }
    echo "Maximum logit error: $maxError; peak PHP memory: ".round(memory_get_peak_usage(true)/1048576)." MB\n";
}
