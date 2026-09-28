import { Game } from './engine.mjs';
class SearchBranch extends Error { constructor(choices) { super(); this.choices=choices; } }
class SearchDeadEnd extends Error {}
const sum=a=>a.reduce((x,y)=>x+y,0);
const cmp=(a,b)=> { for(let i=0;i<a.length;i++) { const d=Array.isArray(a[i])?cmp(a[i],b[i]):Number(a[i])-Number(b[i]); if(d) return d; } return 0; };
const num=(a,b)=>a-b;
const bonus={blob_world:5,patrol_mech:5,defense_center:2};
export class Lethal {
  static combat(id) { const c=Game.card(id); return c.combat+(c.ally==='gain_combat'?c.ally_amount:0)+(c.scrap==='gain_combat'?c.scrap_amount:0)+(bonus[c.primary]??0); }
  static possible(o) {
    let cards=[...o.hand,...o.own_in_play.map(i=>i.card)];
    const blobs=o.blob_cards_played+o.hand.filter(id=>Game.card(id).faction==='blob').length;
    const draws=s=>({draw:1,draw_two:2,draw_destroy:1,scrap_two_draw:2,draw_then_scrap:1,recycle:2,embassy_yacht:2,blob_world:blobs}[s]??0);
    let n=0;
    for(const id of o.hand) { const c=Game.card(id); n+=draws(c.primary)+draws(c.ally)+draws(c.scrap); }
    for(const i of o.own_in_play) { const c=Game.card(i.card); n+=(i.activated?0:draws(c.primary))+(i.ally_triggered?0:draws(c.ally))+draws(c.scrap); }
    if(o.hand.includes(23)) { let best=0; for(const id of cards) if(Game.ship(id)) { const c=Game.card(id); best=Math.max(best,draws(c.primary)+draws(c.ally)+draws(c.scrap)); } n+=best; }
    if(o.own_deck_count&&n>=o.own_deck_count) cards.push(...o.own_deck,...o.own_known_top);
    let total=o.combat+sum(cards.map(id=>this.combat(id)));
    for(const i of o.own_in_play) { const c=Game.card(i.card); total-=c.combat; if(i.ally_triggered&&c.ally==='gain_combat') total-=c.ally_amount; if(i.activated) total-=bonus[c.primary]??0; }
    if(cards.includes(29)) total+=cards.filter(id=>Game.ship(id)).length;
    const copies=cards.filter(id=>Game.card(id).primary==='copy_ship').length;
    let best=0; for(const id of cards) if(Game.ship(id)) best=Math.max(best,this.combat(id)+1); total+=copies*best;
    let destroy=copies*2;
    for(const id of cards) { const c=Game.card(id); for(const s of ['primary','ally','scrap']) destroy+=+['destroy_base','destroy_and_scrap','draw_destroy'].includes(c[s]); }
    const def=o.opponent_in_play.filter(i=>Game.card(i.card).card_type==='outpost').map(i=>Game.card(i.card).defense).sort((a,b)=>b-a);
    return total>=o.opponent_authority+sum(def.slice(destroy));
  }
  static plan(game,decision,budget=256) {
    if(!this.possible(decision.observation)) return [];
    const pid=decision.observation.player_id,prefixes=[[]],visited=new Set();
    for(let step=0;step<budget&&prefixes.length;step++) {
      const prefix=prefixes.pop(),branch=game.clone(),own=branch.players[pid]; own.deck.sort(num); own.known_top=[]; own.revealed_hand=[]; branch.searchDraw=true;
      const original={}; for(const id of own.hand) original[id]=(original[id]??0)+1;
      const plan=[]; let used=0;
      branch.choose_override=(g,p,family,options)=> {
        if(plan.length>=220-game.turn_actions) throw new SearchDeadEnd();
        if(own.deck.length) options=options.filter(a=>!(a.kind==='play_card'||(['scrap_card','discard_card'].includes(a.kind)&&a.source_zone==='hand'))||(original[a.card_id]??0)>0);
        if(family==='main') {
          const attacks=options.filter(a=>a.kind==='attack_player'&&own.combat>=g.players[1-pid].authority);
          if(attacks.length) options=attacks;
          else {
            options=options.filter(a=>['play_card','activate_base','activate_ally','scrap_for_ability','attack_base'].includes(a.kind)&&!(a.kind==='attack_base'&&Game.card(a.target_card_id).card_type!=='outpost')&&!(a.kind==='scrap_for_ability'&&!['gain_combat','draw','draw_destroy'].includes(a.ability)));
            const key=v=> { const k=v.kind,rank={play_card:0,activate_ally:1,activate_base:2,scrap_for_ability:3,attack_base:4}[k];
              if(k!=='play_card') return [rank,-(k==='attack_base'?v.amount:0),0,0,0,0,0];
              const c=Game.card(v.card_id); return [rank,c.primary==='copy_ship',Game.ship(v.card_id),!['draw','draw_two'].includes(c.primary),c.primary==='embassy_yacht',-this.combat(v.card_id),v.card_id]; };
            options.sort((a,b)=>cmp(key(a),key(b)));
          }
          const key=JSON.stringify([[...own.hand].sort(num),own.in_play.map(i=>[i.card,i.original,i.activated,i.ally_triggered]).sort(cmp),own.combat,own.blob_cards_played,own.deck.length,Object.entries(original).sort((a,b)=>Number(a[0])-Number(b[0])),g.players[1-pid].in_play.map(i=>i.card).sort(num),[...own.discard].sort(num)]);
          if(used===prefix.length) { if(visited.has(key)) throw new SearchDeadEnd(); visited.add(key); }
        } else if(['scrap_trade_row','free_acquire'].includes(family)) options=options.filter(a=>a.kind==='decline');
        else if(family==='destroy_base') { const targets=options.filter(a=>a.kind==='destroy_base').sort((a,b)=>Game.card(b.target_card_id).defense-Game.card(a.target_card_id).defense); if(targets.length) options=[targets[0]]; }
        else if(family==='ability_mode') options.sort((a,b)=>cmp([a.ability!=='gain_combat',!['draw','cycle'].includes(a.ability)],[b.ability!=='gain_combat',!['draw','cycle'].includes(b.ability)]));
        else if(family==='copy_ship') options.sort((a,b)=>this.combat(b.target_card_id)-this.combat(a.target_card_id));
        else if(['scrap','discard'].includes(family)) { const key=a=>[a.kind!=='decline',a.source_zone!=='discard',a.card_id>=0?this.combat(a.card_id):0]; options.sort((a,b)=>cmp(key(a),key(b))); }
        if(!options.length) throw new SearchDeadEnd();
        let selected;
        if(options.length>1) { if(used===prefix.length) throw new SearchBranch(options.length); selected=options[prefix[used++]]; } else selected=options[0];
        plan.push([family,Game.key(selected)]);
        if(own.deck.length&&(selected.kind==='play_card'||(['scrap_card','discard_card'].includes(selected.kind)&&selected.source_zone==='hand'))) original[selected.card_id]=(original[selected.card_id]??0)-1;
        return selected;
      };
      try { while(branch.winner===null) { const a=branch.choose(own,'main',branch.mainActions(own),'Main phase'); branch.apply(own,a); } return plan; }
      catch(e) { if(e instanceof SearchBranch) { for(let i=e.choices-1;i>=0;i--) prefixes.push([...prefix,i]); } else if(!(e instanceof SearchDeadEnd)) throw e; }
    } return [];
  }
}
