import { Game } from './engine.mjs';
// Checkpoint-stable v1/v2 layout, including historical missing-card sentinels.
export class Encoder {
  static FAMILIES={main:0,discard:1,scrap:2,destroy_base:3,scrap_trade_row:4,copy_ship:5,free_acquire:6,ability_mode:7};
  static KINDS={play_card:1,activate_base:2,activate_ally:2,scrap_for_ability:3,attack_base:4,attack_player:5,acquire:6,end_turn:7,discard_card:8,scrap_card:10,destroy_base:12,scrap_trade_row:14,copy_ship:16,free_acquire:18,decline:24};
  static MODES={gain_combat:19,gain_attack:19,gain_trade:20,gain_authority:21,draw:22,cycle:23,recycle:23};
  static ZONES={hand:0,own_hand:0,discard:3,own_discard:3,in_play:4,own_in_play:4,opponent_in_play:9,trade_row:10,explorer_supply:13};
  constructor(version=2) { this.version=version; }
  static zones(o) { return [o.hand,o.own_deck,o.own_known_top,o.own_discard,o.own_in_play.map(i=>i.card),o.opponent_hidden,o.opponent_known_hand,o.opponent_known_top,o.opponent_discard,o.opponent_in_play.map(i=>i.card),o.trade_row,o.trade_deck,o.scrap_heap,o.explorer_supply]; }
  state(o) {
    const r=Array(1292).fill(0),s=[+o.is_starting_player,o.own_authority/50,o.opponent_authority/50,o.combat/20,o.trade/15,o.pending_discard/5,o.opponent_pending_discard/5,+o.next_ship_to_top,o.blob_cards_played/5,+o.all_allied,+o.fleet_active,o.turn/50,o.action_number/50,o.own_deck_count/20,o.opponent_hand_count/10,o.opponent_deck_count/20,o.trade_deck_count/80,o.explorers_remaining/10];
    s.forEach((v,j)=>r[j]=v); let cursor=18;
    for(const zone of Encoder.zones(o)) { for(const id of zone) if(id!==null) r[cursor+id]++; cursor+=49; }
    for(const zone of ['own_known_top','opponent_known_top']) { o[zone].slice(0,3).forEach((id,j)=>r[cursor+j*49+id]=1); cursor+=147; }
    for(const zone of ['own_in_play','opponent_in_play']) for(const status of ['ready','ally_triggered','copied_from_stealth_needle']) { for(const i of o[zone]) if(status==='ready'?!i.activated:i[status]) r[cursor+i.card]++; cursor+=49; }
    return r;
  }
  static attributes(id,zero=false) {
    const r=Array(13).fill(0); if(zero) return r; if(id<0) { r[9]=r[10]=1; return r; }
    const c=Game.card(id); ['cost','combat','authority','trade','defense'].forEach((k,j)=>r[j]=c[k]);
    r[5+{machine_cult:0,blob:1,trade_federation:2,star_empire:3,unaligned:4}[c.faction]]=1;
    r[10+{ship:0,base:1,outpost:2}[c.card_type]]=1; return r;
  }
  static effect(s) {
    if(['','none','used'].includes(s)) return 0;
    if(s.includes('attack')||s.includes('combat')) return 1;
    if(s.includes('trade')) return 2;
    if(s.includes('authority')) return 3;
    if(s.includes('draw')||s==='recycle') return 4;
    if(s.includes('discard')) return 5;
    if(s.includes('scrap')&&!s.includes('row')&&!s.includes('destroy')) return 6;
    if(s.includes('kill')||s.includes('destroy')) return 7;
    if(s.includes('row')) return 8;
    if(s.includes('free')) return 9;
    if(s.includes('copy')) return 10;
    if(s.includes('top')) return 11;
    if(s.includes('ally')) return 12; return 13;
  }
  action(a,o) {
    const k=a.kind==='choose_mode'?(Encoder.MODES[a.ability]??0):(Encoder.KINDS[a.kind]??0);
    let s=a.card_id,t=a.target_card_id,sz=Encoder.ZONES[a.source_zone]??-1,tz=-1;
    if(k===6) { t=s; s=-1; }
    if([4,6,12,14,16,18].includes(k)) { tz=sz; sz=-1; }
    if(sz<0) { if([1,8].includes(k)) sz=0; else if(k===10) sz=a.source_zone==='discard'?3:0; else if(k===3) sz=4; }
    if(tz<0) { if([6,18,14].includes(k)) tz=10; else if([4,12].includes(k)) tz=9; else if(k===16) tz=4; }
    const r=Array(this.version>=2?219:203).fill(0); r[k]=1; if(s>=0) r[25+s]=1;
    Encoder.attributes(s,k===6).forEach((v,j)=>r[74+j]=v); if(t>=0) r[87+t]=1;
    Encoder.attributes(t).forEach((v,j)=>r[136+j]=v); if(sz>=0) r[149+sz]=1; if(tz>=0) r[163+tz]=1;
    r[177+Encoder.effect(a.ability)]=1;
    const n=[a.amount,[6,18].includes(k)?a.amount:0,k===4?a.amount2:(k===5?a.amount:0),k===4?a.amount:0,0,0,0,0,+(k===24),0,+(k===18),0];
    n.forEach((v,j)=>r[191+j]=v); if(this.version>=2) this.relations(o,k,s,t,sz).forEach((v,j)=>r[203+j]=v); return r;
  }
  relations(o,k,s,t,sz) {
    const id=t>=0?t:s,c=id>=0?Game.card(id):null,f=c?.faction??'unaligned',aligned=f!=='unaligned',zones=Encoder.zones(o);
    let total=0,owned=0,draw=0,play=0,allied=0;
    for(const z of [0,1,2,3,4]) { total+=zones[z].length; for(const cid of zones[z]) { const card=Game.card(cid); if(aligned&&card.faction===f) { owned++; if(card.ally!=='') allied++; if([1,2,3].includes(z)) draw++; if(z===4) play++; } } }
    const ally=!!c?.ally;
    if(aligned&&s>=0) { if([0,1,2,3,4].includes(sz)) { owned=Math.max(0,owned-1); if(ally) allied=Math.max(0,allied-1); } if([1,2,3].includes(sz)) draw=Math.max(0,draw-1); if(sz===4) play=Math.max(0,play-1); }
    let knownCombat=0,knownTrade=0,knownDraw=0; const knownFactions=[];
    for(const cid of o.opponent_known_top) { const v=Game.card(cid); knownCombat+=v.combat; knownTrade+=v.trade; knownDraw+=+(['draw','draw_two'].includes(v.primary)||['draw','draw_two'].includes(v.ally)); knownFactions.push(v.faction); }
    const target=t>=0?Game.card(t):null,def=[4,12].includes(k)?(target?.defense??0):0;
    return [owned/10,draw/10,play/5,+ally,+(owned>0&&(ally||allied>0)),owned/Math.max(1,total),Math.min(1,o.opponent_known_top.length/3),knownCombat/15,knownTrade/10,knownDraw/3,def/10,def?Math.min(2,o.combat/Math.max(1,def)):0,+(def>0&&target.faction!=='unaligned'&&knownFactions.includes(target.faction)),Math.max(-2.5,Math.min(2.5,(o.combat-o.opponent_authority)/20)),[6,18].includes(k)?Math.max(-1,Math.min(1.5,(o.trade-(c?.cost??0))/10)):0,k===3&&o.turn<=18?(c?.cost??0)/10:0];
  }
}
