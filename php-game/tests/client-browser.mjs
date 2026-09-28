// npm install playwright; npx playwright install chromium
// PLAYWRIGHT_MODULE may point to an isolated installation instead.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=fileURLToPath(new URL('../',import.meta.url)),requests=[];
let newRelease=false;
const server=http.createServer(async(req,res)=> {
  const url=new URL(req.url,'http://localhost'),name=url.pathname==='/'?'index.html':decodeURIComponent(url.pathname.slice(1));
  requests.push(name);
  try {
    if(name.includes('..')) throw new Error('bad path');
    let data=await fs.readFile(path.join(root,name));
    if(name==='index.html'&&newRelease) data=Buffer.from(data.toString().replace('Your next opponent is ready.','Your next opponent is ready for this release.'));
    const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.css':'text/css','.jpg':'image/jpeg','.gz':'application/gzip'}[path.extname(name)]||'application/octet-stream';
    res.writeHead(200,{'Content-Type':mime,'Cache-Control':name==='index.html'||name==='sw.js'?'no-cache':'public, max-age=31536000, immutable'});res.end(data);
  } catch {res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const url=`http://127.0.0.1:${server.address().port}/`;
const browser=await chromium.launch({headless:true});
try {
  const context=await browser.newContext({viewport:{width:1440,height:1100}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=> {if(m.type()==='error'&&!m.text().includes('404')&&!m.text().includes('ERR_INTERNET_DISCONNECTED')&&!m.text().includes('Failed to load resource: net::ERR_FAILED')) errors.push(m.text());});
  page.on('dialog',d=>d.accept());
  await page.goto(url); await page.locator('#start:not([disabled])').waitFor({timeout:30000});
  assert.equal(await page.locator('#opponent').inputValue(),'level-10');
  assert.deepEqual(requests.filter(x=>x.endsWith('.astro.gz')),['models/level-10.astro.gz']);
  await page.waitForFunction(()=>document.getElementById('welcome-art').naturalWidth>0);
  assert.equal(requests.filter(x=>x.endsWith('.jpg')).length,1,'Only welcome image loads initially');
  console.log('PASS default level 10 only, lazy welcome artwork');
  await page.locator('#start').click(); await page.locator('#play-all:not([hidden])').waitFor(); await page.locator('#play-all').click();
  await page.waitForFunction(()=>document.getElementById('hand-count').textContent==='0 cards');
  const before=await page.locator('#log').innerText();
  await page.reload();await page.locator('#start:not([disabled])').waitFor();assert.equal(await page.locator('#log').innerText(),before);
  assert.equal(requests.filter(x=>x.endsWith('.astro.gz')).length,1,'Model reused across refresh');
  await page.locator('#end-turn').click();
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Your turn')&&document.getElementById('turn').textContent!=='TURN 1');
  await page.screenshot({path:'/tmp/astro-client-desktop.png',fullPage:true});
  console.log('PASS Play all, refresh resume, full worker-driven AI turn');
  await page.locator('#opponent').selectOption('level-01');
  await page.waitForFunction(()=>document.getElementById('status').textContent==='Level 1 is ready. Start a new game to play this level.');
  await page.locator('#start').click();await page.waitForFunction(()=>document.getElementById('opponent-name').textContent==='Level 1');await page.locator('#play-all:not([hidden])').waitFor();
  assert.equal(await page.locator('#opponent-name').innerText(),'Level 1');
  await page.locator('#opponent').selectOption('level-08');
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Level 8 is ready.'));
  assert.deepEqual(requests.filter(x=>x.endsWith('.astro.gz')),['models/level-10.astro.gz','models/level-01.astro.gz','models/level-08.astro.gz']);
  console.log('PASS levels 1 and 8 download only on selection');
  await context.setOffline(true);await page.reload();await page.locator('#start:not([disabled])').waitFor();
  assert.equal(await page.locator('#opponent-name').innerText(),'Level 1');
  await page.locator('#play-all').click();await page.waitForFunction(()=>document.getElementById('hand-count').textContent==='0 cards');
  await page.locator('#end-turn').click();await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Your turn')&&document.getElementById('turn').textContent!=='TURN 1');
  console.log('PASS offline refresh and gameplay with downloaded models');
  await context.setOffline(false);
  // Trigger the same focus check with unchanged HTML: must not offer an update.
  await page.evaluate(()=>window.dispatchEvent(new Event('focus')));
  await page.waitForTimeout(150);assert.equal(await page.locator('#update-banner').isVisible(),false);
  newRelease=true;await page.evaluate(()=>window.dispatchEvent(new Event('focus')));await page.locator('#update-banner:not([hidden])').waitFor();
  const downloaded=requests.filter(x=>x.endsWith('.astro.gz')).length;
  await page.locator('#update-now').click();await page.locator('#start:not([disabled])').waitFor();
  assert.equal(requests.filter(x=>x.endsWith('.astro.gz')).length,downloaded+1,'HTML release fetches default model anew');
  assert.equal(await page.locator('#board').isVisible(),false);
  assert.match(await page.locator('#status').innerText(),/new release/);
  console.log('PASS unchanged HTML keeps cache; changed HTML resets assets and saved game');
  await page.setViewportSize({width:390,height:844});await page.locator('#start').click();await page.locator('#play-all:not([hidden])').waitFor();
  for(const img of await page.locator('#market img, #hand img').all()) { await img.scrollIntoViewIfNeeded(); await img.evaluate(img=>new Promise((resolve,reject)=> { if(img.naturalWidth) return resolve(); img.addEventListener('load',resolve,{once:true}); setTimeout(()=>img.naturalWidth?resolve():reject(new Error('Image did not load')),5000); })); }
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:'/tmp/astro-client-mobile.png',fullPage:true});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'No mobile horizontal overflow');
  assert.ok(!requests.some(x=>x.endsWith('.php')),'No PHP requests');
  assert.deepEqual(errors,[]);
  console.log('PASS mobile layout, no PHP calls or browser exceptions');
  await page.locator('#models-open').click();
  await page.locator('#upload-form input[name=name]').fill('My local champion');
  const registry=JSON.parse(await fs.readFile(path.join(root,'models/registry.json')));
  const source=path.resolve(root,'..',registry[0].source);
  await page.locator('#upload-form input[type=file]').setInputFiles(source);
  await page.locator('#upload-form button').click();
  await page.waitForFunction(()=>document.getElementById('opponent').selectedOptions[0].textContent==='My local champion');
  await page.locator('[data-close=models-dialog]').click();
  await page.locator('#start').click();await page.waitForFunction(()=>document.getElementById('opponent-name').textContent==='My local champion');
  await page.reload();await page.locator('#start:not([disabled])').waitFor();
  assert.equal(await page.locator('#opponent-name').innerText(),'My local champion');
  console.log('PASS native NPZ import, local persistence, and imported-model gameplay');
  await context.close();
} finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
