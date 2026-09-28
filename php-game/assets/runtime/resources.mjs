// Shared by the page and worker. Cache Storage is best-effort; RAM reuse works
// even when persistent storage is unavailable or the browser evicts old files.
let base,release;
const pending=new Map();
export function configureResources(root,version) { base=new URL(root); release=version; }
export function resourceURL(path) { const u=new URL(path,base); if(u.origin!==base.origin||!u.pathname.startsWith(base.pathname)) throw new Error('Invalid resource path'); u.searchParams.set('v',release); return u.href; }
export async function resource(path) {
  const url=resourceURL(path);
  if(!pending.has(url)) pending.set(url,(async()=> {
    let cache;
    try { cache=await caches.open(`astro-assets:${base.pathname}:${release}`); const hit=await cache.match(url); if(hit) return await hit.arrayBuffer(); } catch { /* Private browsing / storage quota. */ }
    const response=await fetch(url,{cache:'force-cache'});
    if(!response.ok) throw new Error(`Could not download ${path} (${response.status}). Reconnect and retry.`);
    const bytes=await response.arrayBuffer();
    try { await cache?.put(url,new Response(bytes,{headers:response.headers})); } catch { /* RAM still holds this resource. */ }
    return bytes;
  })().catch(e=> { pending.delete(url); throw e; }));
  return pending.get(url);
}
export async function forgetResource(path) { const url=resourceURL(path); pending.delete(url); try { const cache=await caches.open(`astro-assets:${base.pathname}:${release}`); await cache.delete(url); } catch {} }
export async function sha256(bytes) { return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join(''); }
export async function decompress(bytes) {
  // Browsers decompress HTTP Content-Encoding themselves. Static .gz files are
  // normally delivered as application/gzip, but tolerate hosts that decode them.
  const view=new Uint8Array(bytes);
  if(view[0]!==31||view[1]!==139) return bytes;
  return readLimited(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')));
}

export async function readLimited(stream, limit=128*1024*1024) {
  const reader=stream.getReader(),chunks=[];let total=0;
  try { while(true) { const {done,value}=await reader.read();if(done)break;total+=value.byteLength;if(total>limit)throw new Error('Expanded model is too large');chunks.push(value); } }
  catch(e) { await reader.cancel();throw e; }
  const out=new Uint8Array(total);let offset=0;for(const chunk of chunks){out.set(chunk,offset);offset+=chunk.byteLength;}return out.buffer;
}
