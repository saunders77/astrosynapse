/* Static-only service worker. HTML is always network-first. Every asset URL
   includes the SHA-256 of the HTML; a changed document gets a fresh cache. */
const scope=new URL(self.registration.scope),shell='astro-shell:'+scope.pathname;
const assetPrefix='astro-assets:'+scope.pathname+':';
self.addEventListener('install',event=>event.waitUntil((async()=> {
  try { const r=await fetch(new URL('index.html',scope),{cache:'no-store'}); if(r.ok) { const c=await caches.open(shell); await c.put(new URL('index.html',scope),r); } } catch {}
  await self.skipWaiting();
})()));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
self.addEventListener('fetch',event=> {
  const r=event.request,u=new URL(r.url);
  if(r.method!=='GET'||u.origin!==scope.origin||!u.pathname.startsWith(scope.pathname)) return;
  if(r.mode==='navigate'&&(u.pathname===scope.pathname||u.pathname===scope.pathname+'index.html')) {
    event.respondWith((async()=> {
      try { const response=await fetch(r,{cache:'no-store',signal:AbortSignal.timeout(5000)}); if(response.ok) { try { const c=await caches.open(shell); await c.put(new URL('index.html',scope),response.clone()); } catch {} return response; } throw new Error('HTML unavailable'); }
      catch { const hit=await caches.match(new URL('index.html',scope),{cacheName:shell}); return hit||new Response('Reconnect to load Astrosynapse.',{status:503,headers:{'Content-Type':'text/plain'}}); }
    })()); return;
  }
  const version=u.searchParams.get('v');
  if(!version||!/^[a-f0-9]{64}$/.test(version)) return;
  event.respondWith((async()=> {
    let cache; try { cache=await caches.open(assetPrefix+version); const hit=await cache.match(r); if(hit) return hit; } catch {}
    const response=await fetch(r); if(response.ok) { try { await cache?.put(r,response.clone()); } catch {} } return response;
  })());
});
// Delete obsolete releases only when no open page still uses them.
const versions=new Map();
self.addEventListener('message',event=> {
  if(event.data?.type!=='release'||!/^[a-f0-9]{64}$/.test(event.data.version)) return;
  if(event.source?.id) versions.set(event.source.id,event.data.version);
  event.waitUntil((async()=> {
    const clients=await self.clients.matchAll({type:'window'}),live=new Set(clients.map(c=>c.id));
    for(const id of versions.keys()) if(!live.has(id)) versions.delete(id);
    // Unknown tabs might still use an earlier cache; retain it until they report.
    if(clients.some(c=>!versions.has(c.id))) return;
    const active=new Set(versions.values());
    for(const key of await caches.keys()) if(key.startsWith(assetPrefix)&&!active.has(key.slice(assetPrefix.length))) await caches.delete(key);
  })());
});
