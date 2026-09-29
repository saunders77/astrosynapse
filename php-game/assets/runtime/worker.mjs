import { Game } from './engine.mjs';
import { importModel } from './import-model.mjs';
import { Actor } from './actor.mjs';
import { Session } from './session.mjs';
import { configureResources, resource, forgetResource, sha256, decompress } from './resources.mjs';
import { localModels } from './local-models.mjs';
let models=[],session=null;
const actors=new Map();
async function actorFor(id) {
  if(actors.has(id)) return actors.get(id);
  const m=models.find(m=>m.id===id); let bytes;
  if(m) {
    postMessage({progress:`Loading ${m.name}… ${(m.bytes/1000000).toFixed(1)} MB on first use`});
    bytes=await resource('models/'+m.file);
    if(await sha256(bytes)!==m.sha256) { await forgetResource('models/'+m.file); throw new Error('Model integrity check failed. Reload and retry.'); }
    bytes=await decompress(bytes);
  } else { const custom=await localModels('get',id); if(!custom) throw new Error('This local model is no longer available. Choose another level.'); bytes=custom.bytes; }
  const actor=new Actor(bytes); actors.set(id,actor); return actor;
}
async function handle(p) {
  if(p.op==='init') {
    configureResources(p.base,p.release);
    [Game.catalog,models]=await Promise.all(['assets/cards.json','models/registry.json'].map(async path=>JSON.parse(new TextDecoder().decode(await resource(path)))));
    await actorFor('level-10');
    if(p.saved) { session=p.saved; try { const actor=await actorFor(session.model); return {models,cards:Game.catalog,game:Session.advance(session,actor),saved:session}; } catch(e) { session=null; return {models,cards:Game.catalog,game:null,saved:null,notice:'The saved game could not be restored: '+e.message}; } }
    return {models,cards:Game.catalog,game:null,saved:null};
  }
  if(p.op==='load') { await actorFor(p.model); return {}; }
  if(p.op==='import') {
    const bytes=await importModel(p.bytes);
    const actor=new Actor(bytes),id='local-'+crypto.randomUUID(),name=p.name.trim().slice(0,100);
    if(!name) throw new Error('Enter a model name');
    await localModels('put',{id,name,bytes,description:'Imported on this browser'}); actors.set(id,actor); return {imported:{id,name,description:'Imported on this browser'}};
  }
  if(p.op==='new') {
    const actor=await actorFor(p.model),m=models.find(m=>m.id===p.model)||await localModels('get',p.model);
    const next=Session.start({...m,name:p.label||m.name}),game=Session.advance(next,actor); session=next; return {game,saved:session};
  }
  if(!session) throw new Error('Start a game first');
  if(p.id!==session.id||p.revision!==session.revision) throw new Error('This move is out of date. Reload the saved game.');
  if(!['choose','play_all','advance','resign'].includes(p.op)) throw new Error('Unknown operation');
  const next=structuredClone(session),actor=await actorFor(next.model),game=Session.advance(next,actor,p.op,p.action_id);
  session=next; return {game,saved:session};
}
// Serialize messages so model downloads and new-game requests cannot race.
let queue=Promise.resolve();
self.onmessage=({data})=> { queue=queue.then(async()=> { try { postMessage({requestId:data.requestId,data:await handle(data)}); } catch(e) { postMessage({requestId:data.requestId,error:e.message}); } }); };
