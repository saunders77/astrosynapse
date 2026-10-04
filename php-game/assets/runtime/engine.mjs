// Rules version 2. Portable deterministic RNG matches the former PHP engine.
export class Rng {
  constructor(state) { this.state = state % 2147483646 + 1; }
  next() { return this.state = this.state * 48271 % 2147483647; }
  shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = this.next() % (i + 1); [a[i], a[j]] = [a[j], a[i]]; } }
}
export class Player {
  constructor(id, rng) { Object.assign(this, { id, rng, deck: [], hand: [], discard: [], in_play: [], known_top: [], revealed_hand: [], inferred_hand: [], authority: 50, combat: 0, trade: 0, must_discard: 0, blob_cards_played: 0, next_ship_top: false }); }
}
export class InPlay {
  constructor(uid, card) { Object.assign(this, { uid, card, original: card, activated: false, ally_triggered: false }); }
}
export class Pause extends Error { constructor(decision) { super('Waiting for a choice'); this.decision = decision; } }
export class Limit extends Error {}
const sorted = a => [...a].sort((a,b) => a-b);
const pop = a => a.length ? a.pop() : null;
export class Game {
  static catalog = [];
  static RESOURCE = ['gain_combat','gain_trade','gain_authority'];
  static AUTO = [...Game.RESOURCE,'draw','draw_two','ship_top','opponent_discard'];
  static cards() { return this.catalog; }
  static card(id) { return this.catalog[id]; }
  static ship(id) { return this.card(id).card_type === 'ship'; }
  static manual(id) { const c=this.card(id); return !this.ship(id) && c.primary !== '' && !['all_ally','fleet_hq','ship_top'].includes(c.primary); }
  static action(kind,card_id=-1,target_card_id=-1,ability='',source_zone='',amount=0,amount2=0,opaque=[]) { return {kind,card_id,target_card_id,ability,source_zone,amount,amount2,opaque}; }
  static key(a) { return JSON.stringify({kind:a.kind,card_id:a.card_id,target_card_id:a.target_card_id,ability:a.ability,source_zone:a.source_zone,amount:a.amount,amount2:a.amount2}); }
  static dedup(actions) { const seen=new Set(); return actions.filter(a=> { const k=this.key(a); if(seen.has(k)) return false; seen.add(k); return true; }); }
  static label(a) {
    let s=a.kind.replaceAll('_',' ');
    if(a.card_id>=0) s+=' '+this.card(a.card_id).name;
    if(a.kind==='scrap_card') s+=' from '+a.source_zone.replaceAll('_',' ');
    if(a.target_card_id>=0) s+=' → '+this.card(a.target_card_id).name;
    if(a.ability) s+=' ('+a.ability.replaceAll('_',' ')+')';
    if(a.amount) s+=` (${a.amount})`;
    return s[0].toUpperCase()+s.slice(1);
  }
  constructor(seed,starts=0,chooser=()=>0) {
    Object.assign(this,{seed,starting_player:starts,active_player:starts,chooser,players:[],trade_deck:[],trade_row:[],scrap_heap:[],sounds:[],log:[],explorers_remaining:10,turns:0,decisions:0,turn_actions:0,uid:0,winner:null,result:null,decision_hook:null,choose_override:null,manual_player:null,searchDraw:false});
    const rng=new Rng(seed+101);
    for(const c of Game.cards()) for(let i=0;i<c.copies;i++) this.trade_deck.push(c.card_id);
    rng.shuffle(this.trade_deck);
    for(let i=0;i<5;i++) this.trade_row.push(pop(this.trade_deck));
    for(let i=0;i<2;i++) { const p=new Player(i,new Rng(seed+202+i*101)); p.deck=[0,0,0,0,0,0,0,0,1,1]; p.rng.shuffle(p.deck); this.players.push(p); }
    this.draw(this.players[starts],3); this.draw(this.players[1-starts],5);
  }
  clone() {
    const g=Object.assign(Object.create(Game.prototype),this);
    for(const k of ['trade_deck','trade_row','scrap_heap']) g[k]=[...this[k]];
    g.players=this.players.map(p=> { const q=Object.assign(Object.create(Player.prototype),p); q.rng=Object.assign(Object.create(Rng.prototype),p.rng); for(const k of ['deck','hand','discard','known_top','revealed_hand','inferred_hand']) q[k]=[...p[k]]; q.in_play=p.in_play.map(i=>Object.assign(Object.create(InPlay.prototype),i)); return q; });
    g.log=[]; g.sounds=[]; g.decision_hook=null; return g;
  }
  run() {
    try { while(this.winner===null && this.turns<240) { this.takeTurn(this.players[this.active_player]); if(this.winner===null) this.active_player=1-this.active_player; } this.result={winner:this.winner,turns:this.turns,truncated:this.winner===null}; }
    catch(e) { if(!(e instanceof Limit)) throw e; this.result={winner:null,turns:this.turns,truncated:true}; }
  }
  choose(p,family,actions,prompt) {
    if(this.choose_override) return this.choose_override(this,p,family,Game.dedup(actions),prompt);
    if(this.turn_actions>=220) throw new Limit('Action limit');
    actions=Game.dedup(actions); if(!actions.length) throw new Error('Empty decision');
    const d={family,observation:this.observation(p.id),actions,prompt}; this.decisions++; this.turn_actions++;
    const manual = p.id === this.manual_player && (family === 'discard' || actions[0].kind === 'end_turn');
    const index=actions.length===1&&!manual?0:this.chooser(this,p.id,d);
    if(!Number.isInteger(index)||!actions[index]) throw new Error('Illegal action');
    const a=actions[index]; this.log.push({player_id:p.id,turn:this.turns,family,kind:a.kind,card_id:a.card_id,target_card_id:a.target_card_id,label:Game.label(a)});
    if(this.decision_hook) this.decision_hook(this,p.id,d,a); return a;
  }
  takeTurn(p) {
    if(p.id===0) this.sounds.push('playerturn');
    this.turns++; this.turn_actions=0; p.combat=p.trade=p.blob_cards_played=0; p.next_ship_top=false;
    for(const i of p.in_play) { i.ally_triggered=false; i.activated=!Game.manual(i.card); this.resources(p,i.card); if(Game.card(i.card).primary==='ship_top') p.next_ship_top=true; }
    while(p.must_discard>0 && p.hand.length) { const a=this.choose(p,'discard',this.handActions(p,'discard_card'),'Choose a card to discard'); this.discardHand(p,a.opaque[0]); p.must_discard--; }
    if(!p.hand.length) p.must_discard=0; this.allies(p);
    while(this.winner===null) { const a=this.choose(p,'main',this.mainActions(p),'Main phase'); if(this.apply(p,a)) break; }
    if(this.winner===null) this.cleanup(p);
  }
  handActions(p,kind) { return p.hand.map((c,j)=>Game.action(kind,c,-1,'','hand',0,0,[j])); }
  targets(p) { const all=this.players[1-p.id].in_play, out=all.filter(i=>Game.card(i.card).card_type==='outpost'); return out.length?out:all.filter(i=>!Game.ship(i.card)); }
  mainActions(p) {
    const a=this.handActions(p,'play_card');
    for(const i of p.in_play) { const c=Game.card(i.card);
      if(!Game.AUTO.includes(c.ally) && this.allyAvailable(p,i)) a.push(Game.action('activate_ally',i.card,-1,c.ally,'in_play',c.ally_amount,0,[i.uid]));
      if(!i.activated && Game.manual(i.card)) a.push(Game.action('activate_base',i.card,-1,c.primary,'in_play',0,0,[i.uid]));
      if(c.scrap) a.push(Game.action('scrap_for_ability',i.card,-1,c.scrap,'in_play',c.scrap_amount,0,[i.uid]));
    }
    if(p.combat>0) { for(const i of this.targets(p)) { const c=Game.card(i.card); if(p.combat>=c.defense) a.push(Game.action('attack_base',-1,i.card,'','opponent_in_play',c.defense,p.combat,[i.uid])); }
      if(!this.players[1-p.id].in_play.some(i=>Game.card(i.card).card_type==='outpost')) a.push(Game.action('attack_player',-1,-1,'','',p.combat)); }
    this.trade_row.forEach((id,j)=> { if(id!==null && Game.card(id).cost<=p.trade) a.push(Game.action('acquire',id,-1,'','trade_row',Game.card(id).cost,0,[j])); });
    if(this.explorers_remaining>0 && p.trade>=2) a.push(Game.action('acquire',2,-1,'','explorer_supply',2));
    a.push(Game.action('end_turn')); return Game.dedup(a);
  }
  apply(p,a) {
    const op=this.players[1-p.id]; let i,c;
    switch(a.kind) {
      case 'play_card': this.play(p,a.opaque[0]); break;
      case 'activate_base': i=this.find(p,a.opaque[0]); i.activated=true; this.effect(p,Game.card(i.card).primary,0,i); this.allies(p); break;
      case 'activate_ally': i=this.find(p,a.opaque[0]); i.ally_triggered=true; c=Game.card(i.card); this.effect(p,c.ally,c.ally_amount,i); break;
      case 'scrap_for_ability': i=this.find(p,a.opaque[0]); this.remove(p,i); this.scrap(i.original); c=Game.card(i.card); this.effect(p,c.scrap,c.scrap_amount,i); break;
      case 'attack_base': this.sounds.push('attack'); i=this.find(op,a.opaque[0]); p.combat-=Game.card(i.card).defense; this.remove(op,i); op.discard.push(i.original); break;
      case 'attack_player': this.sounds.push('attack'); op.authority-=p.combat; p.combat=0; if(op.authority<=0) this.winner=p.id; break;
      case 'acquire': if(a.source_zone==='explorer_supply') { p.trade-=2; this.explorers_remaining--; this.place(p,2); } else this.acquire(p,a.opaque[0],a.amount); break;
      case 'end_turn': return true;
      default: throw new Error('Unknown main action');
    } return false;
  }
  play(p,j) {
    const id=p.hand.splice(j,1)[0]; this.forget(p,id); const i=new InPlay(++this.uid,id); p.in_play.push(i);
    if(Game.card(id).faction==='blob' && id!==23) p.blob_cards_played++;
    this.allies(p,true); this.resources(p,id); if(Game.ship(id)&&this.fleet(p)) this.gain(p,'combat',1);
    this.primary(p,i); this.allies(p);
  }
  primary(p,i) {
    const c=Game.card(i.card);
    if(c.primary==='copy_ship') { const a=p.in_play.filter(t=>t.uid!==i.uid&&Game.ship(t.card)).map(t=>Game.action('copy_ship',i.original,t.card,'copy_ship','in_play',0,0,[t.uid]));
      if(a.length) { const s=this.choose(p,'copy_ship',a,'Stealth Needle: copy a ship'); i.card=this.find(p,s.opaque[0]).card; }
      if(i.card!==c.card_id) { this.resources(p,i.card); this.primary(p,i); } else i.activated=true; return;
    }
    i.activated=!Game.manual(i.card); if(c.primary&&(Game.ship(i.card)||c.primary==='ship_top')) this.effect(p,c.primary,0,i);
  }
  gain(p,resource,amount) { p[resource]+=amount; if(amount>0) this.sounds.push(resource); }
  resources(p,id) { const c=Game.card(id); for(const resource of ['combat','trade','authority']) this.gain(p,resource,c[resource]); }
  allyAvailable(p,i) { const c=Game.card(i.card); return !!c.ally && !i.ally_triggered && p.in_play.some(o=>o.uid!==i.uid&&(o.card===19||Game.card(o.card).faction===c.faction||(o.original===23&&c.faction==='machine_cult'))); }
  allies(p,resourcesOnly=false) {
    for(const i of p.in_play) { const c=Game.card(i.card);
      if((resourcesOnly?Game.RESOURCE:Game.AUTO).includes(c.ally)&&this.allyAvailable(p,i)) { i.ally_triggered=true; this.effect(p,c.ally,c.ally_amount,i); }
    }
  }
  effect(p,effect,amount,s) {
    let a,n,v,id;
    switch(effect) {
      case 'gain_combat': this.gain(p,'combat',amount); break;
      case 'gain_trade': this.gain(p,'trade',amount); break;
      case 'gain_authority': this.gain(p,'authority',amount); break;
      case 'draw': this.draw(p,1); break;
      case 'draw_two': this.draw(p,2); break;
      case 'all_ally': case 'fleet_hq': case 'copy_ship': break;
      case 'opponent_discard': this.players[1-p.id].must_discard++; break;
      case 'ship_top': p.next_ship_top=true; break;
      case 'embassy_yacht': if(p.in_play.filter(i=>!Game.ship(i.card)).length>=2) this.draw(p,2); break;
      case 'scrap_trade_row': this.rowChoice(p,s,false); break;
      case 'free_ship': this.rowChoice(p,s,true); break;
      case 'blob_world': this.mode(p,s,{gain_combat:5,draw:p.blob_cards_played}); break;
      case 'patrol_mech': this.mode(p,s,{gain_combat:5,gain_trade:3}); break;
      case 'barter_world': this.mode(p,s,{gain_authority:2,gain_trade:2}); break;
      case 'defense_center': this.mode(p,s,{gain_combat:2,gain_authority:3}); break;
      case 'trading_post': this.mode(p,s,{gain_authority:1,gain_trade:1}); break;
      case 'scrap_any': this.scrapAny(p,s); break;
      case 'scrap_two_draw': n=this.scrapAny(p,s); n+=this.scrapAny(p,s); this.draw(p,n); break;
      case 'draw_then_scrap': this.draw(p,1); if(p.hand.length) { a=this.choose(p,'scrap',this.handActions(p,'scrap_card'),'Scrap a card from hand'); id=p.hand.splice(a.opaque[0],1)[0]; this.forget(p,id); this.scrap(id); } break;
      case 'destroy_base': this.destroy(p,s); break;
      case 'destroy_and_scrap': this.destroy(p,s); this.rowChoice(p,s,false); break;
      case 'draw_destroy': this.draw(p,1); this.destroy(p,s); break;
      case 'recycle':
        a=this.choose(p,'ability_mode',[Game.action('choose_mode',s.card,-1,'gain_trade','',1),Game.action('choose_mode',s.card,-1,'cycle','',2)],'Recycling Station: gain trade or cycle up to two cards');
        if(a.ability==='gain_trade') { this.gain(p,'trade',1); break; }
        n=0; for(let j=0;j<2&&p.hand.length;j++) { a=this.handActions(p,'discard_card'); a.push(Game.action('decline',s.card,-1,'cycle')); v=this.choose(p,'discard',a,'Discard a card to replace'); if(v.kind==='decline') break; this.discardHand(p,v.opaque[0]); n++; } this.draw(p,n); break;
      default: throw new Error('Unknown effect '+effect);
    }
  }
  mode(p,s,modes) { const a=Object.entries(modes).map(([effect,amount])=>Game.action('choose_mode',s.card,-1,effect,'',amount)); const v=this.choose(p,'ability_mode',a,Game.card(s.card).name+': choose mode'); if(v.ability==='draw') this.draw(p,v.amount); else this.effect(p,v.ability,v.amount,s); }
  scrapAny(p,s) {
    const a=p.discard.map((id,j)=>Game.action('scrap_card',id,-1,'','discard',0,0,[j])); a.push(...this.handActions(p,'scrap_card'),Game.action('decline',s.card,-1,'scrap_any'));
    const v=this.choose(p,'scrap',a,'Scrap a card from hand or discard'); if(v.kind==='decline') return 0;
    const z=v.source_zone,id=p[z].splice(v.opaque[0],1)[0]; if(z==='hand') this.forget(p,id); this.scrap(id); return 1;
  }
  destroy(p,s) {
    const a=this.targets(p).map(t=>Game.action('destroy_base',s.card,t.card,'destroy_base','opponent_in_play',0,0,[t.uid])); if(!a.length) return;
    a.push(Game.action('decline',s.card,-1,'destroy_base')); const v=this.choose(p,'destroy_base',a,'Optionally destroy a base');
    if(v.kind!=='decline') { this.sounds.push('attack'); const op=this.players[1-p.id],i=this.find(op,v.opaque[0]); this.remove(op,i); op.discard.push(i.original); }
  }
  rowChoice(p,s,free) {
    const kind=free?'free_acquire':'scrap_trade_row',effect=free?'free_ship_to_top':'scrap_trade_row',a=[];
    this.trade_row.forEach((id,j)=> { if(id!==null&&(!free||Game.ship(id))) a.push(Game.action(kind,s.card,id,effect,'trade_row',0,0,[j])); });
    if(!a.length) return; a.push(Game.action('decline',s.card,-1,effect)); const v=this.choose(p,kind,a,free?'Optionally acquire a ship free onto your deck':'Optionally scrap a trade-row card');
    if(v.kind==='decline') return; const j=v.opaque[0]; if(free) this.acquire(p,j,0,true); else { this.sounds.push('scrap'); this.scrap_heap.push(this.trade_row[j]); this.trade_row[j]=pop(this.trade_deck); }
  }
  scrap(id) { this.sounds.push('scrap'); if(id===2) this.explorers_remaining++; else this.scrap_heap.push(id); }
  acquire(p,slot,cost,top=false) { p.trade-=cost; this.place(p,this.trade_row[slot],top); this.trade_row[slot]=pop(this.trade_deck); }
  place(p,id,top=false) { if(Game.ship(id)&&(top||p.next_ship_top)) { p.deck.push(id); p.known_top.push(id); p.next_ship_top=false; } else p.discard.push(id); }
  draw(p,n) {
    for(let j=0;j<n;j++) { if(!p.deck.length) { if(this.searchDraw||!p.discard.length) break; p.deck=p.discard; p.discard=[]; p.rng.shuffle(p.deck); this.sounds.push('shuffle'); p.known_top=[]; }
      const id=pop(p.deck); p.hand.push(id); if(p.known_top.length&&p.known_top.at(-1)===id) { p.known_top.pop(); p.revealed_hand.push(id); }
    }
  }
  cleanup(p) {
    p.discard.push(...p.hand); p.hand=[]; p.revealed_hand=[]; p.inferred_hand=[];
    for(const i of p.in_play) if(Game.ship(i.card)) { this.remove(p,i); p.discard.push(i.original); }
    p.combat=p.trade=p.blob_cards_played=0; p.next_ship_top=false;
    // After cleanup the remaining deck multiset is public. Remember it when
    // all of it will enter the new hand, even across a discard shuffle.
    if(p.deck.length<=5) p.inferred_hand.push(...(p.known_top.length?p.deck.slice(0,-p.known_top.length):p.deck));
    this.draw(p,5);
  }
  find(p,uid) { const i=p.in_play.find(i=>i.uid===uid); if(!i) throw new Error('Missing card'); return i; }
  remove(p,i) { p.in_play=p.in_play.filter(v=>v.uid!==i.uid); }
  forget(p,id) { for(const cards of [p.revealed_hand,p.inferred_hand]) { const j=cards.indexOf(id); if(j>=0) { cards.splice(j,1); return; } } }
  knownHand(p) { return sorted(p.deck.length===p.known_top.length?p.hand:[...p.inferred_hand,...p.revealed_hand]); }
  discardHand(p,j) { const id=p.hand.splice(j,1)[0]; this.forget(p,id); p.discard.push(id); }
  fleet(p) { return p.in_play.some(i=>i.card===29); }
  observation(id) {
    const p=this.players[id],op=this.players[1-id],unknown=v=>v.known_top.length?v.deck.slice(0,-v.known_top.length):v.deck;
    const hidden=[...op.hand]; for(const c of op.revealed_hand) { const j=hidden.indexOf(c); if(j>=0) hidden.splice(j,1); }
    const items=v=>v.in_play.map(i=>({card:i.card,activated:i.activated,ally_triggered:i.ally_triggered,copied_from_stealth_needle:i.original===23&&i.card!==23}));
    return {version:2,player_id:id,active_player:this.active_player,starting_player:this.starting_player,is_starting_player:id===this.starting_player,turn:this.turns,action_number:this.turn_actions,
      own_authority:p.authority,opponent_authority:op.authority,combat:this.active_player===id?p.combat:0,trade:this.active_player===id?p.trade:0,pending_discard:p.must_discard,opponent_pending_discard:op.must_discard,
      hand:[...p.hand],own_deck_count:p.deck.length,own_deck:sorted(unknown(p)),own_known_top:[...p.known_top].reverse(),own_discard:sorted(p.discard),own_in_play:items(p),
      opponent_hand_count:op.hand.length,opponent_known_hand:[...op.revealed_hand],opponent_inferred_hand:this.knownHand(op),opponent_hidden:sorted([...unknown(op),...hidden]),opponent_deck_count:op.deck.length,opponent_known_top:[...op.known_top].reverse(),opponent_discard:sorted(op.discard),opponent_in_play:items(op),
      trade_row:[...this.trade_row],trade_deck_count:this.trade_deck.length,trade_deck:sorted(this.trade_deck),explorers_remaining:this.explorers_remaining,explorer_supply:Array(this.explorers_remaining).fill(2),scrap_heap:sorted(this.scrap_heap),
      next_ship_to_top:p.next_ship_top,blob_cards_played:p.blob_cards_played,all_allied:p.in_play.some(i=>i.card===19),fleet_active:this.fleet(p)};
  }
  playAllPlan(decision) {
    if(this.active_player!==0||decision.family!=='main'||this.players[0].hand.length<2) return [];
    const probe=this.clone(),plan=[]; probe.chooser=(g,p,d)=> { throw new Pause(d); };
    try { for(const id of this.players[0].hand) { const j=probe.players[0].hand.indexOf(id),a=Game.action('play_card',id,-1,'','hand',0,0,[j]); plan.push(Game.key(a)); probe.apply(probe.players[0],a); } }
    catch(e) { if(e instanceof Pause||e instanceof Limit) return []; throw e; } return plan;
  }
}
