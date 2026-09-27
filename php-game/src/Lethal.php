<?php
declare(strict_types=1);
namespace Astro;
final class SearchBranch extends \RuntimeException { public function __construct(public int $choices) { parent::__construct(); } }
final class SearchDeadEnd extends \RuntimeException {}

// Same bounded, public-information finisher used by the deployed Python actor.
final class Lethal {
    public static function combat(int $id): int { $c=Game::card($id); return $c['combat']+($c['ally']==='gain_combat'?$c['ally_amount']:0)+($c['scrap']==='gain_combat'?$c['scrap_amount']:0)+(['blob_world'=>5,'patrol_mech'=>5,'defense_center'=>2][$c['primary']]??0); }
    private static function possible(array $o): bool {
        $cards=array_merge($o['hand'],array_column($o['own_in_play'],'card'));
        $blobs=$o['blob_cards_played']+count(array_filter($o['hand'],fn($id)=>Game::card($id)['faction']==='blob'));
        $draws=fn($s)=>['draw'=>1,'draw_two'=>2,'draw_destroy'=>1,'scrap_two_draw'=>2,'draw_then_scrap'=>1,'recycle'=>2,'embassy_yacht'=>2,'blob_world'=>$blobs][$s]??0;
        $n=0; foreach($o['hand'] as $id) { $c=Game::card($id); $n+=$draws($c['primary'])+$draws($c['ally'])+$draws($c['scrap']); }
        foreach($o['own_in_play'] as $i) { $c=Game::card($i['card']); $n+=($i['activated']?0:$draws($c['primary']))+($i['ally_triggered']?0:$draws($c['ally']))+$draws($c['scrap']); }
        if(in_array(23,$o['hand'],true)) { $best=0; foreach($cards as $id) if(Game::ship($id)) { $c=Game::card($id); $best=max($best,$draws($c['primary'])+$draws($c['ally'])+$draws($c['scrap'])); } $n+=$best; }
        if($o['own_deck_count'] && $n>=$o['own_deck_count']) $cards=array_merge($cards,$o['own_deck'],$o['own_known_top']);
        $total=$o['combat']+array_sum(array_map([self::class,'combat'],$cards));
        foreach($o['own_in_play'] as $i) { $c=Game::card($i['card']); $total-=$c['combat']; if($i['ally_triggered'] && $c['ally']==='gain_combat') $total-=$c['ally_amount']; if($i['activated']) $total-=['blob_world'=>5,'patrol_mech'=>5,'defense_center'=>2][$c['primary']]??0; }
        if(in_array(29,$cards,true)) $total+=count(array_filter($cards,[Game::class,'ship']));
        $copies=count(array_filter($cards,fn($id)=>Game::card($id)['primary']==='copy_ship'));
        $best=0; foreach($cards as $id) if(Game::ship($id)) $best=max($best,self::combat($id)+1); $total+=$copies*$best;
        $destroy=$copies*2; foreach($cards as $id) { $c=Game::card($id); foreach(['primary','ally','scrap'] as $s) $destroy+=(int)in_array($c[$s],['destroy_base','destroy_and_scrap','draw_destroy'],true); }
        $def=[]; foreach($o['opponent_in_play'] as $i) if(Game::card($i['card'])['card_type']==='outpost') $def[]=Game::card($i['card'])['defense']; rsort($def,SORT_NUMERIC);
        return $total >= $o['opponent_authority']+array_sum(array_slice($def,$destroy));
    }
    public static function plan(Game $game,array $decision,int $budget=256): array {
        if(!self::possible($decision['observation'])) return [];
        $pid=$decision['observation']['player_id']; $prefixes=[[]]; $visited=[];
        for($step=0;$step<$budget && $prefixes;$step++) {
            $prefix=array_pop($prefixes); $branch=clone $game; $own=$branch->players[$pid]; sort($own->deck,SORT_NUMERIC); $own->known_top=$own->revealed_hand=[]; $branch->searchDraw=true;
            $original=array_count_values($own->hand); $plan=[]; $used=0;
            $branch->choose_override=function(Game $g,Player $p,string $family,array $options,string $prompt) use(&$plan,&$used,&$original,&$visited,$prefix,$game,$own,$pid): array {
                if(count($plan)>=220-$game->turn_actions) throw new SearchDeadEnd();
                if($own->deck) $options=array_values(array_filter($options,fn($a)=>!($a['kind']==='play_card' || (in_array($a['kind'],['scrap_card','discard_card'],true) && $a['source_zone']==='hand')) || ($original[$a['card_id']]??0)>0));
                if($family==='main') {
                    $attacks=array_values(array_filter($options,fn($a)=>$a['kind']==='attack_player' && $own->combat >= $g->players[1-$pid]->authority));
                    if($attacks) $options=$attacks;
                    else {
                        $options=array_values(array_filter($options,function($a) {
                            if(!in_array($a['kind'],['play_card','activate_base','activate_ally','scrap_for_ability','attack_base'],true)) return false;
                            if($a['kind']==='attack_base' && Game::card($a['target_card_id'])['card_type']!=='outpost') return false;
                            if($a['kind']==='scrap_for_ability' && !in_array($a['ability'],['gain_combat','draw','draw_destroy'],true)) return false;
                            return true;
                        }));
                        usort($options,function($a,$b) {
                            $key=function($v) { $k=$v['kind']; $rank=['play_card'=>0,'activate_ally'=>1,'activate_base'=>2,'scrap_for_ability'=>3,'attack_base'=>4][$k];
                                if($k!=='play_card') return [$rank,-($k==='attack_base'?$v['amount']:0),0,0,0,0,0];
                                $c=Game::card($v['card_id']); return [$rank,$c['primary']==='copy_ship',Game::ship($v['card_id']),!in_array($c['primary'],['draw','draw_two'],true),$c['primary']==='embassy_yacht',-self::combat($v['card_id']),$v['card_id']]; };
                            return $key($a)<=>$key($b);
                        });
                    }
                    $hand=$own->hand; sort($hand); $in=array_map(fn($i)=>[$i->card,$i->original,$i->activated,$i->ally_triggered],$own->in_play); sort($in); $op=array_map(fn($i)=>$i->card,$g->players[1-$pid]->in_play); sort($op); $discard=$own->discard; sort($discard); $orig=$original; ksort($orig);
                    $key=json_encode([$hand,$in,$own->combat,$own->blob_cards_played,count($own->deck),$orig,$op,$discard]);
                    if($used===count($prefix)) { if(isset($visited[$key])) throw new SearchDeadEnd(); $visited[$key]=true; }
                } elseif(in_array($family,['scrap_trade_row','free_acquire'],true)) $options=array_values(array_filter($options,fn($a)=>$a['kind']==='decline'));
                elseif($family==='destroy_base') { $targets=array_values(array_filter($options,fn($a)=>$a['kind']==='destroy_base')); usort($targets,fn($a,$b)=>Game::card($b['target_card_id'])['defense']<=>Game::card($a['target_card_id'])['defense']); if($targets) $options=[$targets[0]]; }
                elseif($family==='ability_mode') usort($options,fn($a,$b)=>[$a['ability']!=='gain_combat',!in_array($a['ability'],['draw','cycle'],true)]<=>[$b['ability']!=='gain_combat',!in_array($b['ability'],['draw','cycle'],true)]);
                elseif($family==='copy_ship') usort($options,fn($a,$b)=>self::combat($b['target_card_id'])<=>self::combat($a['target_card_id']));
                elseif(in_array($family,['scrap','discard'],true)) usort($options,fn($a,$b)=>[$a['kind']!=='decline',$a['source_zone']!=='discard',$a['card_id']>=0?self::combat($a['card_id']):0]<=>[$b['kind']!=='decline',$b['source_zone']!=='discard',$b['card_id']>=0?self::combat($b['card_id']):0]);
                if(!$options) throw new SearchDeadEnd();
                if(count($options)>1) { if($used===count($prefix)) throw new SearchBranch(count($options)); $selected=$options[$prefix[$used++]]; } else $selected=$options[0];
                $plan[]=[$family,Game::key($selected)];
                if($own->deck && ($selected['kind']==='play_card' || (in_array($selected['kind'],['scrap_card','discard_card'],true) && $selected['source_zone']==='hand'))) $original[$selected['card_id']]=($original[$selected['card_id']]??0)-1;
                return $selected;
            };
            try {
                while($branch->winner===null) { $a=$branch->choose($own,'main',$branch->mainActions($own),'Main phase'); $branch->apply($own,$a); }
                return $plan;
            } catch(SearchBranch $e) { for($i=$e->choices-1;$i>=0;$i--) $prefixes[]=array_merge($prefix,[$i]); }
            catch(SearchDeadEnd $e) {}
        }
        return [];
    }
}
