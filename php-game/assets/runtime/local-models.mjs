// All model management is local to this browser; no upload endpoint exists.
let dbPromise;
function database() {
  return dbPromise??=new Promise((resolve,reject)=> {
    const r=indexedDB.open('astro-local-models:'+new URL('../',import.meta.url).pathname,1);
    r.onupgradeneeded=()=>r.result.createObjectStore('models',{keyPath:'id'});
    r.onsuccess=()=>resolve(r.result); r.onerror=()=>reject(r.error);
  });
}
export async function localModels(operation,value) {
  const db=await database(); return new Promise((resolve,reject)=> {
    const tx=db.transaction('models',operation==='list'||operation==='get'?'readonly':'readwrite'),s=tx.objectStore('models');
    const request=operation==='list'?s.getAll():operation==='get'?s.get(value):operation==='put'?s.put(value):s.delete(value);
    tx.oncomplete=()=>resolve(request.result); tx.onerror=()=>reject(tx.error); tx.onabort=()=>reject(tx.error||new Error('Local model storage failed'));
  });
}
