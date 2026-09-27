<?php
declare(strict_types=1);
namespace Astro;

// Port of astro2/engine.py, rules version 2. Decisions remain ordinary single
// actions; Play all is deliberately implemented by the human session adapter.
final class Rng {
    public function __construct(public int $state) { $this->state = ($state % 2147483646) + 1; }
    public function next(): int { return $this->state = (int)(($this->state * 48271) % 2147483647); }
    public function shuffle(array &$a): void { for ($i=count($a)-1;$i>0;$i--) { $j=$this->next()%($i+1); [$a[$i],$a[$j]]=[$a[$j],$a[$i]]; } }
}
final class Player {
    public array $deck=[], $hand=[], $discard=[], $in_play=[], $known_top=[], $revealed_hand=[];
    public int $authority=50, $combat=0, $trade=0, $must_discard=0, $blob_cards_played=0;
    public bool $next_ship_top=false;
    public function __construct(public int $id, public Rng $rng) {}
    public function __clone() { $this->rng=clone $this->rng; $this->in_play=array_map(fn($i)=>clone $i,$this->in_play); }
}
final class InPlay {
    public bool $activated=false, $ally_triggered=false;
    public int $original;
    public function __construct(public int $uid, public int $card) { $this->original=$card; }
}
class Pause extends \RuntimeException {
    public function __construct(public array $decision) { parent::__construct('Waiting for a choice'); }
}
class Limit extends \RuntimeException {}
final class Game {
    public array $players=[], $trade_deck=[], $trade_row=[], $scrap_heap=[], $log=[];
    public int $explorers_remaining=10, $turns=0, $decisions=0, $turn_actions=0, $uid=0, $active_player, $starting_player;
    public ?int $winner=null;
    public ?array $result=null;
    public $chooser;
    public $decision_hook=null;
    public $choose_override=null;
    public bool $searchDraw=false;
    private static ?array $catalog=null;
    const RESOURCE=['gain_combat','gain_trade','gain_authority'];
    const AUTO=['gain_combat','gain_trade','gain_authority','draw','draw_two','ship_top'];
    public static function cards(): array { return self::$catalog ??= json_decode(file_get_contents(__DIR__.'/../assets/cards.json'),true,512,JSON_THROW_ON_ERROR); }
    public static function card(int $id): array { return self::cards()[$id]; }
    public static function ship(int $id): bool { return self::card($id)['card_type']==='ship'; }
    public static function manual(int $id): bool { $c=self::card($id); return !self::ship($id) && $c['primary']!=='' && !in_array($c['primary'],['all_ally','fleet_hq','ship_top'],true); }
    public static function action(string $kind,int $card_id=-1,int $target_card_id=-1,string $ability='',string $source_zone='',int $amount=0,int $amount2=0,array $opaque=[]): array {
        return compact('kind','card_id','target_card_id','ability','source_zone','amount','amount2','opaque');
    }
    public static function key(array $a): string { unset($a['opaque'],$a['label'],$a['id']); return json_encode($a,JSON_THROW_ON_ERROR); }
    public static function dedup(array $actions): array { $seen=[]; $out=[]; foreach($actions as $a) { $k=self::key($a); if(!isset($seen[$k])) { $seen[$k]=true; $out[]=$a; } } return $out; }
    public static function label(array $a): string {
        $label=str_replace('_',' ',$a['kind']);
        if($a['card_id']>=0) $label.=' '.self::card($a['card_id'])['name'];
        if($a['kind']==='scrap_card') $label.=' from '.str_replace('_',' ',$a['source_zone']);
        if($a['target_card_id']>=0) $label.=' → '.self::card($a['target_card_id'])['name'];
        if($a['ability']) $label.=' ('.str_replace('_',' ',$a['ability']).')';
        if($a['amount'] || $a['amount2']) $label.=' ['.$a['amount'].','.$a['amount2'].']';
        return ucfirst($label);
    }
    public function __construct(public int $seed,int $starts=0,?callable $chooser=null) {
        $this->starting_player=$this->active_player=$starts;
        $this->chooser=$chooser ?? fn($g,$p,$d)=>0;
        $rng=new Rng($seed+101);
        foreach(self::cards() as $c) for($i=0;$i<$c['copies'];$i++) $this->trade_deck[]=$c['card_id'];
        $rng->shuffle($this->trade_deck);
        for($i=0;$i<5;$i++) $this->trade_row[]=array_pop($this->trade_deck);
        for($i=0;$i<2;$i++) { $p=new Player($i,new Rng($seed+202+$i*101)); $p->deck=[0,0,0,0,0,0,0,0,1,1]; $p->rng->shuffle($p->deck); $this->players[]=$p; }
        $this->draw($this->players[$starts],3); $this->draw($this->players[1-$starts],5);
    }
    public function __clone() { $this->players=array_map(fn($p)=>clone $p,$this->players); $this->log=[]; $this->decision_hook=null; }
    public function run(): void {
        try {
            while($this->winner===null && $this->turns<240) { $this->takeTurn($this->players[$this->active_player]); if($this->winner===null) $this->active_player=1-$this->active_player; }
            $this->result=['winner'=>$this->winner,'turns'=>$this->turns,'truncated'=>$this->winner===null];
        } catch(Limit $e) { $this->result=['winner'=>null,'turns'=>$this->turns,'truncated'=>true]; }
    }
    public function choose(Player $p,string $family,array $actions,string $prompt): array {
        if($this->choose_override) return ($this->choose_override)($this,$p,$family,self::dedup($actions),$prompt);
        if($this->turn_actions>=220) throw new Limit('Action limit');
        $actions=self::dedup($actions); if(!$actions) throw new \LogicException('Empty decision');
        // Observation is captured before the decision counter advances, matching Python.
        $d=['family'=>$family,'observation'=>$this->observation($p->id),'actions'=>$actions,'prompt'=>$prompt];
        $this->decisions++; $this->turn_actions++;
        $index=count($actions)===1 ? 0 : ($this->chooser)($this,$p->id,$d);
        if(!is_int($index)||!isset($actions[$index])) throw new \InvalidArgumentException('Illegal action');
        $a=$actions[$index];
        $this->log[]=['player_id'=>$p->id,'turn'=>$this->turns,'family'=>$family,'label'=>self::label($a)];
        if($this->decision_hook) ($this->decision_hook)($this,$p->id,$d,$a);
        return $a;
    }
    public function takeTurn(Player $p): void {
        $this->turns++; $this->turn_actions=0; $p->combat=$p->trade=$p->blob_cards_played=0; $p->next_ship_top=false;
        foreach($p->in_play as $i) { $i->ally_triggered=false; $i->activated=!self::manual($i->card); $this->resources($p,$i->card); if(self::card($i->card)['primary']==='ship_top') $p->next_ship_top=true; }
        while($p->must_discard>0 && $p->hand) { $a=$this->choose($p,'discard',$this->handActions($p,'discard_card'),'Choose a card to discard'); $this->discardHand($p,$a['opaque'][0]); $p->must_discard--; }
        if(!$p->hand) $p->must_discard=0;
        $this->allies($p);
        while($this->winner===null) { $a=$this->choose($p,'main',$this->mainActions($p),'Main phase'); if($this->apply($p,$a)) break; }
        if($this->winner===null) $this->cleanup($p);
    }
    public function handActions(Player $p,string $kind): array { $a=[]; foreach($p->hand as $j=>$c) $a[]=self::action($kind,card_id:$c,source_zone:'hand',opaque:[$j]); return $a; }
    public function targets(Player $p): array { $all=$this->players[1-$p->id]->in_play; $out=array_values(array_filter($all,fn($i)=>self::card($i->card)['card_type']==='outpost')); return $out ?: array_values(array_filter($all,fn($i)=>!self::ship($i->card))); }
    public function mainActions(Player $p): array {
        $a=$this->handActions($p,'play_card');
        foreach($p->in_play as $i) {
            $c=self::card($i->card);
            if(!in_array($c['ally'],self::AUTO,true) && $this->allyAvailable($p,$i)) $a[]=self::action('activate_ally',$i->card,ability:$c['ally'],source_zone:'in_play',amount:$c['ally_amount'],opaque:[$i->uid]);
            if(!$i->activated && self::manual($i->card)) $a[]=self::action('activate_base',$i->card,ability:$c['primary'],source_zone:'in_play',opaque:[$i->uid]);
            if($c['scrap']) $a[]=self::action('scrap_for_ability',$i->card,ability:$c['scrap'],source_zone:'in_play',amount:$c['scrap_amount'],opaque:[$i->uid]);
        }
        if($p->combat>0) {
            foreach($this->targets($p) as $i) { $c=self::card($i->card); if($p->combat >= $c['defense']) $a[]=self::action('attack_base',target_card_id:$i->card,source_zone:'opponent_in_play',amount:$c['defense'],amount2:$p->combat,opaque:[$i->uid]); }
            if(!array_filter($this->players[1-$p->id]->in_play,fn($i)=>self::card($i->card)['card_type']==='outpost')) $a[]=self::action('attack_player',amount:$p->combat);
        }
        foreach($this->trade_row as $j=>$id) if($id!==null && self::card($id)['cost']<=$p->trade) $a[]=self::action('acquire',$id,source_zone:'trade_row',amount:self::card($id)['cost'],opaque:[$j]);
        if($this->explorers_remaining>0 && $p->trade>=2) $a[]=self::action('acquire',2,source_zone:'explorer_supply',amount:2);
        $a[]=self::action('end_turn'); return self::dedup($a);
    }
    public function apply(Player $p,array $a): bool {
        $op=$this->players[1-$p->id];
        switch($a['kind']) {
            case 'play_card': $this->play($p,$a['opaque'][0]); break;
            case 'activate_base': $i=$this->find($p,$a['opaque'][0]); $i->activated=true; $this->effect($p,self::card($i->card)['primary'],0,$i); $this->allies($p); break;
            case 'activate_ally': $i=$this->find($p,$a['opaque'][0]); $i->ally_triggered=true; $c=self::card($i->card); $this->effect($p,$c['ally'],$c['ally_amount'],$i); break;
            case 'scrap_for_ability': $i=$this->find($p,$a['opaque'][0]); $this->remove($p,$i); $this->scrap($i->original); $c=self::card($i->card); $this->effect($p,$c['scrap'],$c['scrap_amount'],$i); break;
            case 'attack_base': $i=$this->find($op,$a['opaque'][0]); $p->combat-=self::card($i->card)['defense']; $this->remove($op,$i); $op->discard[]=$i->original; break;
            case 'attack_player': $op->authority-=$p->combat; $p->combat=0; if($op->authority<=0) $this->winner=$p->id; break;
            case 'acquire': if($a['source_zone']==='explorer_supply') { $p->trade-=2; $this->explorers_remaining--; $this->place($p,2); } else $this->acquire($p,$a['opaque'][0],$a['amount']); break;
            case 'end_turn': return true;
            default: throw new \LogicException('Unknown main action');
        } return false;
    }
    public function play(Player $p,int $j): void {
        $id=array_splice($p->hand,$j,1)[0]; $this->forget($p,$id); $i=new InPlay(++$this->uid,$id); $p->in_play[]=$i;
        if(self::card($id)['faction']==='blob' && $id!==23) $p->blob_cards_played++;
        $this->allies($p,true); $this->resources($p,$id);
        if(self::ship($id) && $this->fleet($p)) $p->combat++;
        $this->primary($p,$i); $this->allies($p);
    }
    public function primary(Player $p,InPlay $i): void {
        $c=self::card($i->card);
        if($c['primary']==='copy_ship') {
            $a=[]; foreach($p->in_play as $t) if($t->uid!==$i->uid && self::ship($t->card)) $a[]=self::action('copy_ship',$i->original,$t->card,'copy_ship','in_play',opaque:[$t->uid]);
            if($a) { $s=$this->choose($p,'copy_ship',$a,'Stealth Needle: copy a ship'); $i->card=$this->find($p,$s['opaque'][0])->card; }
            if($i->card!==$c['card_id']) { $this->resources($p,$i->card); $this->primary($p,$i); } else $i->activated=true;
            return;
        }
        if($c['primary']==='embassy_yacht') { $i->activated=false; return; }
        $i->activated=!self::manual($i->card);
        if($c['primary'] && (self::ship($i->card) || $c['primary']==='ship_top')) $this->effect($p,$c['primary'],0,$i);
    }
    public function resources(Player $p,int $id): void { $c=self::card($id); $p->combat+=$c['combat']; $p->trade+=$c['trade']; $p->authority+=$c['authority']; }
    public function allyAvailable(Player $p,InPlay $i): bool {
        $c=self::card($i->card); if(!$c['ally'] || $i->ally_triggered) return false;
        foreach($p->in_play as $o) if($o->uid!==$i->uid && ($o->card===19 || self::card($o->card)['faction']===$c['faction'] || ($o->original===23 && $c['faction']==='machine_cult'))) return true;
        return false;
    }
    public function allies(Player $p,bool $resourcesOnly=false): void {
        foreach($p->in_play as $i) {
            $c=self::card($i->card);
            if(!$resourcesOnly && $c['primary']==='embassy_yacht' && !$i->activated && count(array_filter($p->in_play,fn($v)=>!self::ship($v->card)))>=2) { $i->activated=true; $this->draw($p,2); }
            if(in_array($c['ally'],$resourcesOnly?self::RESOURCE:self::AUTO,true) && $this->allyAvailable($p,$i)) { $i->ally_triggered=true; $this->effect($p,$c['ally'],$c['ally_amount'],$i); }
        }
    }
    public function effect(Player $p,string $effect,int $amount,InPlay $s): void {
        switch($effect) {
            case 'gain_combat': $p->combat+=$amount; break;
            case 'gain_trade': $p->trade+=$amount; break;
            case 'gain_authority': $p->authority+=$amount; break;
            case 'draw': $this->draw($p,1); break;
            case 'draw_two': $this->draw($p,2); break;
            case 'all_ally': case 'fleet_hq': case 'copy_ship': break;
            case 'opponent_discard': $this->players[1-$p->id]->must_discard++; break;
            case 'ship_top': $p->next_ship_top=true; break;
            case 'embassy_yacht': if(count(array_filter($p->in_play,fn($i)=>!self::ship($i->card)))>=2) $this->draw($p,2); break;
            case 'scrap_trade_row': $this->rowChoice($p,$s,false); break;
            case 'free_ship': $this->rowChoice($p,$s,true); break;
            case 'blob_world': $this->mode($p,$s,['gain_combat'=>5,'draw'=>$p->blob_cards_played]); break;
            case 'patrol_mech': $this->mode($p,$s,['gain_combat'=>5,'gain_trade'=>3]); break;
            case 'barter_world': $this->mode($p,$s,['gain_authority'=>2,'gain_trade'=>2]); break;
            case 'defense_center': $this->mode($p,$s,['gain_combat'=>2,'gain_authority'=>3]); break;
            case 'trading_post': $this->mode($p,$s,['gain_authority'=>1,'gain_trade'=>1]); break;
            case 'scrap_any': $this->scrapAny($p,$s); break;
            case 'scrap_two_draw': $n=$this->scrapAny($p,$s); $n+=$this->scrapAny($p,$s); $this->draw($p,$n); break;
            case 'draw_then_scrap': $this->draw($p,1); if($p->hand) { $a=$this->choose($p,'scrap',$this->handActions($p,'scrap_card'),'Scrap a card from hand'); $id=array_splice($p->hand,$a['opaque'][0],1)[0]; $this->forget($p,$id); $this->scrap($id); } break;
            case 'destroy_base': $this->destroy($p,$s); break;
            case 'destroy_and_scrap': $this->destroy($p,$s); $this->rowChoice($p,$s,false); break;
            case 'draw_destroy': $this->draw($p,1); $this->destroy($p,$s); break;
            case 'recycle':
                $a=$this->choose($p,'ability_mode',[self::action('choose_mode',$s->card,ability:'gain_trade',amount:1),self::action('choose_mode',$s->card,ability:'cycle',amount:2)],'Recycling Station: gain trade or cycle up to two cards');
                if($a['ability']==='gain_trade') { $p->trade++; break; }
                $n=0; for($j=0;$j<2 && $p->hand;$j++) { $a=$this->handActions($p,'discard_card'); $a[]=self::action('decline',$s->card,ability:'cycle'); $v=$this->choose($p,'discard',$a,'Discard a card to replace'); if($v['kind']==='decline') break; $this->discardHand($p,$v['opaque'][0]); $n++; } $this->draw($p,$n); break;
            default: throw new \LogicException('Unknown effect '.$effect);
        }
    }
    public function mode(Player $p,InPlay $s,array $modes): void { $a=[]; foreach($modes as $effect=>$amount) $a[]=self::action('choose_mode',$s->card,ability:$effect,amount:$amount); $v=$this->choose($p,'ability_mode',$a,self::card($s->card)['name'].': choose mode'); if($v['ability']==='draw') $this->draw($p,$v['amount']); else $this->effect($p,$v['ability'],$v['amount'],$s); }
    public function scrapAny(Player $p,InPlay $s): int {
        $a=[]; foreach($p->discard as $j=>$id) $a[]=self::action('scrap_card',$id,source_zone:'discard',opaque:[$j]);
        $a=array_merge($a,$this->handActions($p,'scrap_card')); $a[]=self::action('decline',$s->card,ability:'scrap_any');
        $v=$this->choose($p,'scrap',$a,'Scrap a card from hand or discard'); if($v['kind']==='decline') return 0;
        $z=$v['source_zone']; $id=array_splice($p->$z,$v['opaque'][0],1)[0]; if($z==='hand') $this->forget($p,$id); $this->scrap($id); return 1;
    }
    public function destroy(Player $p,InPlay $s): void {
        $a=[]; foreach($this->targets($p) as $t) $a[]=self::action('destroy_base',$s->card,$t->card,'destroy_base','opponent_in_play',opaque:[$t->uid]);
        if(!$a) return; $a[]=self::action('decline',$s->card,ability:'destroy_base'); $v=$this->choose($p,'destroy_base',$a,'Optionally destroy a base');
        if($v['kind']!=='decline') { $op=$this->players[1-$p->id]; $i=$this->find($op,$v['opaque'][0]); $this->remove($op,$i); $op->discard[]=$i->original; }
    }
    public function rowChoice(Player $p,InPlay $s,bool $free): void {
        $kind=$free?'free_acquire':'scrap_trade_row'; $effect=$free?'free_ship_to_top':'scrap_trade_row'; $a=[];
        foreach($this->trade_row as $j=>$id) if($id!==null && (!$free || self::ship($id))) $a[]=self::action($kind,$s->card,$id,$effect,'trade_row',opaque:[$j]);
        if(!$a) return; $a[]=self::action('decline',$s->card,ability:$effect); $v=$this->choose($p,$kind,$a,$free?'Optionally acquire a ship free onto your deck':'Optionally scrap a trade-row card');
        if($v['kind']==='decline') return; $j=$v['opaque'][0];
        if($free) $this->acquire($p,$j,0,true); else { $this->scrap_heap[]=$this->trade_row[$j]; $this->trade_row[$j]=array_pop($this->trade_deck); }
    }
    public function scrap(int $id): void { if($id===2) $this->explorers_remaining++; else $this->scrap_heap[]=$id; }
    public function acquire(Player $p,int $slot,int $cost,bool $top=false): void { $p->trade-=$cost; $this->place($p,$this->trade_row[$slot],$top); $this->trade_row[$slot]=array_pop($this->trade_deck); }
    public function place(Player $p,int $id,bool $top=false): void { if(self::ship($id) && ($top || $p->next_ship_top)) { $p->deck[]=$id; $p->known_top[]=$id; $p->next_ship_top=false; } else $p->discard[]=$id; }
    public function draw(Player $p,int $n): void {
        for($j=0;$j<$n;$j++) {
            if(!$p->deck) { if($this->searchDraw || !$p->discard) break; $p->deck=$p->discard; $p->discard=[]; $p->rng->shuffle($p->deck); $p->known_top=[]; }
            $id=array_pop($p->deck); $p->hand[]=$id;
            if($p->known_top && end($p->known_top)===$id) { array_pop($p->known_top); $p->revealed_hand[]=$id; }
        }
    }
    public function cleanup(Player $p): void {
        $p->discard=array_merge($p->discard,$p->hand); $p->hand=$p->revealed_hand=[];
        foreach($p->in_play as $i) if(self::ship($i->card)) { $this->remove($p,$i); $p->discard[]=$i->original; }
        $p->combat=$p->trade=$p->blob_cards_played=0; $p->next_ship_top=false; $this->draw($p,5);
    }
    public function find(Player $p,int $uid): InPlay { foreach($p->in_play as $i) if($i->uid===$uid) return $i; throw new \LogicException('Missing card'); }
    public function remove(Player $p,InPlay $i): void { $p->in_play=array_values(array_filter($p->in_play,fn($v)=>$v->uid!==$i->uid)); }
    public function forget(Player $p,int $id): void { $j=array_search($id,$p->revealed_hand,true); if($j!==false) array_splice($p->revealed_hand,$j,1); }
    public function discardHand(Player $p,int $j): void { $id=array_splice($p->hand,$j,1)[0]; $this->forget($p,$id); $p->discard[]=$id; }
    public function fleet(Player $p): bool { foreach($p->in_play as $i) if($i->card===29) return true; return false; }
    public function observation(int $id): array {
        $p=$this->players[$id]; $op=$this->players[1-$id];
        $unknown=fn($v)=>count($v->known_top)?array_slice($v->deck,0,-count($v->known_top)):$v->deck;
        $hidden=$op->hand; foreach($op->revealed_hand as $c) { $j=array_search($c,$hidden,true); if($j!==false) array_splice($hidden,$j,1); }
        $sorted=function($ids) { sort($ids,SORT_NUMERIC); return $ids; };
        $items=fn($v)=>array_map(fn($i)=>['card'=>$i->card,'activated'=>$i->activated,'ally_triggered'=>$i->ally_triggered,'copied_from_stealth_needle'=>$i->original===23 && $i->card!==23],$v->in_play);
        return ['version'=>2,'player_id'=>$id,'active_player'=>$this->active_player,'starting_player'=>$this->starting_player,'is_starting_player'=>$id===$this->starting_player,'turn'=>$this->turns,'action_number'=>$this->turn_actions,
            'own_authority'=>$p->authority,'opponent_authority'=>$op->authority,'combat'=>$this->active_player===$id?$p->combat:0,'trade'=>$this->active_player===$id?$p->trade:0,'pending_discard'=>$p->must_discard,'opponent_pending_discard'=>$op->must_discard,
            'hand'=>$p->hand,'own_deck_count'=>count($p->deck),'own_deck'=>$sorted($unknown($p)),'own_known_top'=>array_reverse($p->known_top),'own_discard'=>$sorted($p->discard),'own_in_play'=>$items($p),
            'opponent_hand_count'=>count($op->hand),'opponent_known_hand'=>$op->revealed_hand,'opponent_hidden'=>$sorted(array_merge($unknown($op),$hidden)),'opponent_deck_count'=>count($op->deck),'opponent_known_top'=>array_reverse($op->known_top),'opponent_discard'=>$sorted($op->discard),'opponent_in_play'=>$items($op),
            'trade_row'=>$this->trade_row,'trade_deck_count'=>count($this->trade_deck),'trade_deck'=>$sorted($this->trade_deck),'explorers_remaining'=>$this->explorers_remaining,'explorer_supply'=>array_fill(0,$this->explorers_remaining,2),'scrap_heap'=>$sorted($this->scrap_heap),
            'next_ship_to_top'=>$p->next_ship_top,'blob_cards_played'=>$p->blob_cards_played,'all_allied'=>in_array(19,array_map(fn($i)=>$i->card,$p->in_play),true),'fleet_active'=>$this->fleet($p)];
    }
    // Probe the actual card resolution, including copy targets and primary
    // effects. Forced single options do not require additional user choices.
    public function playAllPlan(array $decision): array {
        if($this->active_player!==0 || $decision['family']!=='main' || count($this->players[0]->hand)<2) return [];
        $probe=clone $this; $probe->chooser=fn($g,$p,$d)=>throw new Pause($d); $plan=[];
        try {
            // Only cards in the hand at click time; newly drawn cards stay for
            // the next human decision. Order is the visible left-to-right order.
            foreach($this->players[0]->hand as $id) {
                $j=array_search($id,$probe->players[0]->hand,true);
                $a=self::action('play_card',$id,source_zone:'hand',opaque:[$j]);
                $plan[]=self::key($a); $probe->apply($probe->players[0],$a);
            }
        } catch(Pause|Limit $e) { return []; }
        return $plan;
    }
}
