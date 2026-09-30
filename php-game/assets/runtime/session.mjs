import { Game, Pause } from './engine.mjs';
import { Lethal } from './lethal.mjs';
export class Session {
  static start(model,humanStarts=crypto.getRandomValues(new Uint32Array(1))[0]%2===0,seed=crypto.getRandomValues(new Uint32Array(1))[0]%2000000000+1) {
    return {id:crypto.randomUUID(),seed,starts:humanStarts?0:1,model:model.id,label:model.name,transcript:[],revision:0,lethal:[]};
  }
  // Replay only applies recorded choices; it never repeats past model inference.
  static advance(session,actor,operation='state',actionId=null) {
    if(session.resigned && operation !== 'state') throw new Error('This game is complete');
    const undoing = operation === 'undo';
    if(undoing) operation = 'state';
    let lastHuman = -1;
    const resigning = operation === 'resign';
    if(resigning) operation = 'state';
    const history=[...session.transcript]; let cursor=0,live=false,pending=null,pendingPlayer=null,budget=operation==='advance'?1:0,humanSubmitted=false,batch=[],batchStarted=false;
    const game=new Game(session.seed,session.starts);
    game.manual_player=0;
    game.decision_hook=(g,pid,d,a)=> { if(!live||!session.lethal.length) return; if(session.lethal[0][0]===d.family&&session.lethal[0][1]===Game.key(a)) session.lethal.shift(); else session.lethal=[]; };
    game.chooser=(g,pid,d)=> {
      if(cursor<history.length) { if(pid===0) lastHuman=cursor; const key=history[cursor++],j=d.actions.findIndex(a=>Game.key(a)===key); if(j<0) throw new Error('Game replay mismatch'); return j; }
      live=true; let selected;
      if(pid===0) {
        if(operation==='choose'&&!humanSubmitted) { if(!Number.isInteger(actionId)||!d.actions[actionId]) throw new Error('This move is no longer available'); selected=actionId; humanSubmitted=true; }
        else if(operation==='play_all') { if(!batchStarted) { batch=g.playAllPlan(d); batchStarted=true; if(!batch.length) throw new Error('Play all is not available for this hand'); }
          if(!batch.length||d.family!=='main') throw new Pause(d); const key=batch.shift(); selected=d.actions.findIndex(a=>Game.key(a)===key); if(selected<0) throw new Error('Play all plan no longer legal'); }
        else throw new Pause(d);
      } else {
        if(budget<=0) throw new Pause(d); budget--;
        if(!session.lethal.length&&d.family==='main') session.lethal=Lethal.plan(g,d);
        selected=-1;
        if(session.lethal.length&&session.lethal[0][0]===d.family) selected=d.actions.findIndex(a=>Game.key(a)===session.lethal[0][1]);
        if(selected<0) {
          session.lethal=[]; let indices=d.actions.map((_,j)=>j);
          if(d.family==='main'&&d.actions.some(a=>['play_card','activate_base','activate_ally','attack_base','attack_player'].includes(a.kind))) indices=indices.filter(j=>d.actions[j].kind!=='end_turn');
          selected=indices.length===1?indices[0]:actor.choose(d,indices);
        }
      }
      if(pid===0) lastHuman=session.transcript.length;
      session.transcript.push(Game.key(d.actions[selected])); return selected;
    };
    try { game.run(); } catch(e) { if(!(e instanceof Pause)) throw e; pending=e.decision; pendingPlayer=game.active_player; }
    if(cursor!==history.length) throw new Error('Saved game contains extra decisions');
    if(undoing) {
      if(game.result || pendingPlayer!==0 || pending.family==='main' || lastHuman<0) throw new Error('There is no selection to cancel');
      session.transcript.length=lastHuman; session.lethal=[]; session.revision++;
      return Session.advance(session,actor);
    }
    if(operation==='choose'&&!humanSubmitted) throw new Error('The game is not waiting for your move');
    if(operation==='play_all'&&!batchStarted) throw new Error('Play all is not available now');
    if(session.transcript.length!==history.length) session.revision++;
    if(resigning && !game.result) { session.resigned = true; session.revision++; session.lethal = []; }
    if(session.resigned) { game.result = {winner:1,truncated:false,resigned:true}; pending = null; pendingPlayer = null; }
    const actions=pendingPlayer===0?pending.actions.map((a,j)=> { const {opaque,...publicAction}=a; return {...publicAction,id:j,label:Game.label(a)}; }):[];
    // Evaluate the side to move with the opponent's model, then express the
    // estimate from the opponent's perspective. Never treat policy scores as odds.
    const value=pending?actor.winProbability(pending):null;
    const opponentWin=game.result ? (game.result.winner===null?0.5:game.result.winner===1?1:0) : pending.observation.player_id===1?value:1-value;
    return {sounds:game.sounds,opponent_win_probability:opponentWin,opponent_trade:game.active_player===1?game.players[1].trade:0,opponent_combat:game.active_player===1?game.players[1].combat:0,id:session.id,revision:session.revision,model_id:session.model,model_label:session.label,status:game.result!==null?'complete':pendingPlayer===0?'your_turn':'model_thinking',observation:game.observation(0),decision:pendingPlayer===0?{family:pending.family,prompt:pending.prompt,actions}:null,can_undo:!game.result&&pendingPlayer===0&&pending.family!=='main'&&lastHuman>=0,can_play_all:pendingPlayer===0&&game.playAllPlan(pending).length>0,action_log:game.log,result:game.result};
  }
}
