<?php
declare(strict_types=1);
namespace Astro;

// Checkpoint-stable Astro2 encoder v1/v2. Missing-card sentinels and effect
// ordering intentionally preserve the training encoder's historical contract.
final class Encoder {
    const FAMILIES=['main'=>0,'discard'=>1,'scrap'=>2,'destroy_base'=>3,'scrap_trade_row'=>4,'copy_ship'=>5,'free_acquire'=>6,'ability_mode'=>7];
    const KINDS=['play_card'=>1,'activate_base'=>2,'activate_ally'=>2,'scrap_for_ability'=>3,'attack_base'=>4,'attack_player'=>5,'acquire'=>6,'end_turn'=>7,'discard_card'=>8,'scrap_card'=>10,'destroy_base'=>12,'scrap_trade_row'=>14,'copy_ship'=>16,'free_acquire'=>18,'decline'=>24];
    const MODES=['gain_combat'=>19,'gain_attack'=>19,'gain_trade'=>20,'gain_authority'=>21,'draw'=>22,'cycle'=>23,'recycle'=>23];
    const ZONES=['hand'=>0,'own_hand'=>0,'discard'=>3,'own_discard'=>3,'in_play'=>4,'own_in_play'=>4,'opponent_in_play'=>9,'trade_row'=>10,'explorer_supply'=>13];
    public int $version;
    public function __construct(int $version=2) { $this->version=$version; }
    public static function zones(array $o): array { return [$o['hand'],$o['own_deck'],$o['own_known_top'],$o['own_discard'],array_column($o['own_in_play'],'card'),$o['opponent_hidden'],$o['opponent_known_hand'],$o['opponent_known_top'],$o['opponent_discard'],array_column($o['opponent_in_play'],'card'),$o['trade_row'],$o['trade_deck'],$o['scrap_heap'],$o['explorer_supply']]; }
    public function state(array $o): array {
        $r=array_fill(0,1292,0.0);
        $s=[(float)$o['is_starting_player'],$o['own_authority']/50,$o['opponent_authority']/50,$o['combat']/20,$o['trade']/15,$o['pending_discard']/5,$o['opponent_pending_discard']/5,(float)$o['next_ship_to_top'],$o['blob_cards_played']/5,(float)$o['all_allied'],(float)$o['fleet_active'],$o['turn']/50,$o['action_number']/50,$o['own_deck_count']/20,$o['opponent_hand_count']/10,$o['opponent_deck_count']/20,$o['trade_deck_count']/80,$o['explorers_remaining']/10];
        foreach($s as $j=>$v) $r[$j]=$v;
        $cursor=18; foreach(self::zones($o) as $zone) { foreach($zone as $id) if($id!==null) $r[$cursor+$id]++; $cursor+=49; }
        foreach(['own_known_top','opponent_known_top'] as $zone) { foreach(array_slice($o[$zone],0,3) as $j=>$id) $r[$cursor+$j*49+$id]=1.0; $cursor+=147; }
        foreach(['own_in_play','opponent_in_play'] as $zone) foreach(['ready','ally_triggered','copied_from_stealth_needle'] as $status) { foreach($o[$zone] as $i) if($status==='ready'?!$i['activated']:$i[$status]) $r[$cursor+$i['card']]++; $cursor+=49; }
        return $r;
    }
    private static function attributes(int $id,bool $zero=false): array {
        $r=array_fill(0,13,0.0); if($zero) return $r;
        if($id<0) { $r[9]=$r[10]=1.0; return $r; }
        $c=Game::card($id); foreach(['cost','combat','authority','trade','defense'] as $j=>$k) $r[$j]=(float)$c[$k];
        $r[5+['machine_cult'=>0,'blob'=>1,'trade_federation'=>2,'star_empire'=>3,'unaligned'=>4][$c['faction']]]=1.0;
        $r[10+['ship'=>0,'base'=>1,'outpost'=>2][$c['card_type']]]=1.0; return $r;
    }
    private static function effect(string $s): int {
        if($s==='' || $s==='none' || $s==='used') return 0;
        if((strpos($s,'attack')!==false)||(strpos($s,'combat')!==false)) return 1;
        if((strpos($s,'trade')!==false)) return 2;
        if((strpos($s,'authority')!==false)) return 3;
        if((strpos($s,'draw')!==false) || $s==='recycle') return 4;
        if((strpos($s,'discard')!==false)) return 5;
        if((strpos($s,'scrap')!==false) && !(strpos($s,'row')!==false) && !(strpos($s,'destroy')!==false)) return 6;
        if((strpos($s,'kill')!==false) || (strpos($s,'destroy')!==false)) return 7;
        if((strpos($s,'row')!==false)) return 8;
        if((strpos($s,'free')!==false)) return 9;
        if((strpos($s,'copy')!==false)) return 10;
        if((strpos($s,'top')!==false)) return 11;
        if((strpos($s,'ally')!==false)) return 12;
        return 13;
    }
    public function action(array $a,array $o): array {
        $k=$a['kind']==='choose_mode'?(self::MODES[$a['ability']]??0):(self::KINDS[$a['kind']]??0);
        $s=$a['card_id']; $t=$a['target_card_id']; $sz=self::ZONES[$a['source_zone']]??-1; $tz=-1;
        if($k===6) { $t=$s; $s=-1; }
        if(in_array($k,[4,6,12,14,16,18],true)) { $tz=$sz; $sz=-1; }
        if($sz<0) { if(in_array($k,[1,8],true)) $sz=0; elseif($k===10) $sz=$a['source_zone']==='discard'?3:0; elseif($k===3) $sz=4; }
        if($tz<0) { if(in_array($k,[6,18,14],true)) $tz=10; elseif(in_array($k,[4,12],true)) $tz=9; elseif($k===16) $tz=4; }
        $r=array_fill(0,$this->version>=2?219:203,0.0); $r[$k]=1.0;
        if($s>=0) $r[25+$s]=1.0;
        foreach(self::attributes($s,$k===6) as $j=>$v) $r[74+$j]=$v;
        if($t>=0) $r[87+$t]=1.0;
        foreach(self::attributes($t) as $j=>$v) $r[136+$j]=$v;
        if($sz>=0) $r[149+$sz]=1.0; if($tz>=0) $r[163+$tz]=1.0;
        $r[177+self::effect($a['ability'])]=1.0;
        $n=[(float)$a['amount'],in_array($k,[6,18],true)?(float)$a['amount']:0.0,$k===4?(float)$a['amount2']:($k===5?(float)$a['amount']:0.0),$k===4?(float)$a['amount']:0.0,0.0,0.0,0.0,0.0,(float)($k===24),0.0,(float)($k===18),0.0];
        foreach($n as $j=>$v) $r[191+$j]=$v;
        if($this->version>=2) foreach($this->relations($o,$k,$s,$t,$sz) as $j=>$v) $r[203+$j]=$v;
        return $r;
    }
    private function relations(array $o,int $k,int $s,int $t,int $sz): array {
        $id=$t>=0?$t:$s; $c=$id>=0?Game::card($id):null; $f=$c['faction']??'unaligned'; $aligned=$f!=='unaligned';
        $zones=self::zones($o); $total=0; $owned=$draw=$play=$allied=0;
        foreach([0,1,2,3,4] as $z) { $total+=count($zones[$z]); foreach($zones[$z] as $cid) { $card=Game::card($cid); if($aligned && $card['faction']===$f) { $owned++; if($card['ally']!=='') $allied++; if(in_array($z,[1,2,3],true)) $draw++; if($z===4) $play++; } } }
        $ally=(bool)($c['ally']??'');
        if($aligned && $s>=0) { if(in_array($sz,[0,1,2,3,4],true)) { $owned=max(0,$owned-1); if($ally) $allied=max(0,$allied-1); } if(in_array($sz,[1,2,3],true)) $draw=max(0,$draw-1); if($sz===4) $play=max(0,$play-1); }
        $knownCombat=$knownTrade=$knownDraw=0; $knownFactions=[];
        foreach($o['opponent_known_top'] as $cid) { $v=Game::card($cid); $knownCombat+=$v['combat']; $knownTrade+=$v['trade']; $knownDraw+=(int)(in_array($v['primary'],['draw','draw_two'],true)||in_array($v['ally'],['draw','draw_two'],true)); $knownFactions[]=$v['faction']; }
        $target=$t>=0?Game::card($t):null; $def=in_array($k,[4,12],true)?($target['defense']??0):0;
        return [$owned/10,$draw/10,$play/5,(float)$ally,(float)($owned>0 && ($ally || $allied>0)),$owned/max(1,$total),min(1,count($o['opponent_known_top'])/3),$knownCombat/15,$knownTrade/10,$knownDraw/3,$def/10,$def?min(2,$o['combat']/max(1,$def)):0.0,(float)($def>0 && $target['faction']!=='unaligned' && in_array($target['faction'],$knownFactions,true)),max(-2.5,min(2.5,($o['combat']-$o['opponent_authority'])/20)),in_array($k,[6,18],true)?max(-1,min(1.5,($o['trade']-($c['cost']??0))/10)):0.0,$k===3 && $o['turn']<=18?($c['cost']??0)/10:0.0];
    }
}
