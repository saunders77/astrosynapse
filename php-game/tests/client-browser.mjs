// npm install playwright; npx playwright install chromium
// PLAYWRIGHT_MODULE may point to an isolated installation instead.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Game, InPlay} from '../assets/runtime/engine.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
async function assertTableFits(page) {
  const layout = await page.evaluate(() => {
    const ids = ['opponent-fleet', 'market', 'hand', 'own-fleet'];
    return {
      scrollHeight: document.documentElement.scrollHeight, height: innerHeight,
      bars: ['opponent-bar','player-bar'].map(id=>{const n=document.getElementById(id),r=n.getBoundingClientRect();return {id,position:getComputedStyle(n).position,top:r.top,bottom:r.bottom,left:r.left,right:r.right};}),
      scrollWidth: document.documentElement.scrollWidth, width: innerWidth,
      boxes: ids.flatMap(id => [document.getElementById(id), ...document.querySelectorAll(`#${id} .card`)]).map(n => {
        const r = n.getBoundingClientRect(); return { name: n.id || n.className, top:r.top, bottom:r.bottom, left:r.left, right:r.right };
      }),
    };
  });
  assert.ok(layout.scrollHeight <= layout.height + 1, 'No vertical page scrolling');
  assert.ok(layout.scrollWidth <= layout.width, 'No horizontal page scrolling');
  const rows = ['opponent-fleet', 'market', 'own-fleet', 'hand'].map(id => layout.boxes.find(b => b.name === id));
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i].top >= rows[i - 1].bottom, 'Fleets, trade row, and hand stay in vertical order');
  assert.equal(layout.bars[0].position, 'fixed'); assert.equal(layout.bars[1].position, 'fixed');
  assert.equal(layout.bars[0].top, 0); assert.equal(layout.bars[1].bottom, layout.height);
  for (const r of layout.boxes) assert.ok(r.top >= layout.bars[0].bottom && r.left >= 0 && r.bottom <= layout.bars[1].top + 1 && r.right <= layout.width + 1, `${r.name} fits in viewport`);
}
const root=fileURLToPath(new URL('../',import.meta.url)),requests=[];
let newRelease=false;
const server=http.createServer(async(req,res)=> {
  const url=new URL(req.url,'http://localhost'),name=url.pathname==='/'?'index.html':decodeURIComponent(url.pathname.slice(1));
  requests.push(name);
  try {
    if(name.includes('..')) throw new Error('bad path');
    let data=await fs.readFile(path.join(root,name));
    if(name==='index.html'&&newRelease) data=Buffer.from(data.toString().replace('Your opponent is ready.','Your opponent is ready for this release.'));
    const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.css':'text/css','.jpg':'image/jpeg','.gz':'application/gzip'}[path.extname(name)]||'application/octet-stream';
    res.writeHead(200,{'Content-Type':mime,'Cache-Control':name==='index.html'?'no-cache':'public, max-age=31536000, immutable'});res.end(data);
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
  assert.equal(await page.locator('#opponent').inputValue(),'level-05');
  assert.deepEqual(requests.filter(x=>x.endsWith('.astro.gz')),['models/level-05.astro.gz']);
  await page.waitForFunction(()=>document.getElementById('welcome-art').naturalWidth>0);
  assert.equal(requests.filter(x=>/\.(jpg|webp)$/.test(x)).length,0,'No artwork loads from Astrosynapse');
  assert.equal(await page.locator('#welcome-art').getAttribute('src'),'https://www.starrealms.com/card-gallery/images/content/card-gallery/scout.webp');
  console.log('PASS default level 10 only, lazy welcome artwork');
  await page.locator('#start').click(); await page.locator('#play-all:not([hidden])').waitFor();
  const openingHand = Number(await page.locator('#player-hand').textContent());
  const opponentCards = Number(await page.locator('#opponent-hand').textContent()) + Number(await page.locator('#opponent-deck').textContent());
  const opponentDiscard = Number(await page.locator('#opponent-discard-count').textContent());
  assert.equal(await page.locator('#starts').count(), 0);
  await page.locator('.stats-link').click();
  await page.locator('#stats-title').waitFor();
  const configuredModels = JSON.parse(await fs.readFile(path.join(root, 'models/registry.json')));
  const statsOpponents = new Set([...Array.from({length:10}, (_, i) => `level-${i + 1}`), ...configuredModels.map(m => m.id)]);
  assert.equal(await page.locator('#stats-rows tr').count(), statsOpponents.size);
  await page.locator('#stats-page a').click();
  for (const [pile, title, count] of [['own-deck', 'Your hand + deck', 10], ['opponent-deck', 'Opponent’s hand + deck', opponentCards], ['discard', 'Your discard pile', 0], ['opponent-discard', 'Opponent’s discard pile', opponentDiscard]]) {
    await page.locator(`[data-inspect="${pile}"]`).first().click();
    assert.equal(await page.locator('#pile-title').textContent(), `${title} · ${count} cards`);
    assert.equal(await page.locator('#pile-dialog .cards:not([hidden]) .card').count(), count);
    await page.locator('[data-close="pile-dialog"]').click();
  }
  assert.match(await page.locator('#opponent-win').textContent(), /^\d+(\.\d+)?%$/);
  await page.locator('[data-inspect=hand]').click();
  assert.equal(await page.locator('#hand-inspector .card').count(), openingHand);
  await page.locator('[data-close=pile-dialog]').click();
  await page.locator('#hand .card-face').first().click();
  await page.waitForFunction(count=>document.getElementById('hand-count').textContent===`${count - 1} cards`, openingHand);
  await page.locator('#play-all').click();
  await page.waitForFunction(()=>document.getElementById('hand-count').textContent==='0 cards');
  if (await page.locator('#opponent-attack').isEnabled()) {
    const authority = Number(await page.locator('#opponent-authority').textContent());
    const combat = Number(await page.locator('#combat').textContent());
    await page.locator('#opponent-attack').click();
    await page.waitForFunction(expected=>Number(document.getElementById('opponent-authority').textContent)===expected, authority-combat);
    assert.equal(await page.locator('#opponent-attack').isEnabled(), false);
  }
  console.log('PASS deck/discard inspectors, hand-face play, and authority attack availability');
  const before=await page.locator('#log').textContent();
  await page.reload();await page.locator('#start:not([disabled])').waitFor();assert.equal(await page.locator('#log').textContent(),before);
  assert.equal(requests.filter(x=>x.endsWith('.astro.gz')).length,1,'Model reused across refresh');
  await page.locator('#end-turn').click();
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Your turn')&&document.getElementById('turn').textContent!=='TURN 1');
  await page.setViewportSize({width:1440,height:900});
  await assertTableFits(page);
  await page.screenshot({path:'/tmp/astro-client-desktop.png',fullPage:true});
  console.log('PASS Play all, refresh resume, full worker-driven AI turn');
  await page.locator('#opponent').selectOption('level-01');
  await page.waitForFunction(()=>document.getElementById('status').textContent==='Level 1 is ready. Start a new game to play this level.');
  await page.locator('#start').click();await page.waitForFunction(()=>document.getElementById('opponent-name').textContent==='Level 1');await page.locator('#play-all:not([hidden])').waitFor();
  assert.equal(await page.locator('#opponent-name').innerText(),'Level 1');
  await page.locator('#opponent').selectOption('level-04');
  await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Level 8 is ready.'));
  assert.deepEqual(requests.filter(x=>x.endsWith('.astro.gz')),['models/level-05.astro.gz','models/level-01.astro.gz','models/level-04.astro.gz']);
  console.log('PASS levels 1 and 8 download only on selection');
  await context.setOffline(true);
  assert.equal(await page.locator('#opponent-name').innerText(),'Level 1');
  await page.locator('#play-all').click();await page.waitForFunction(()=>document.getElementById('hand-count').textContent==='0 cards');
  await page.locator('#end-turn').click();await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Your turn')&&document.getElementById('turn').textContent!=='TURN 1');
  console.log('PASS dropped-connection gameplay with downloaded models');
  await context.setOffline(false);
  newRelease=true;
  const downloaded=requests.filter(x=>x.endsWith('.astro.gz')).length;
  await page.reload();await page.locator('#start:not([disabled])').waitFor();
  assert.equal(requests.filter(x=>x.endsWith('.astro.gz')).length,downloaded+1,'Changed HTML fetches the default model anew');
  assert.equal(await page.locator('#board').isVisible(),false);
  assert.match(await page.locator('#status').innerText(),/previous saved game is incompatible/);
  console.log('PASS changed HTML resets resource URLs and the saved game');
  await page.setViewportSize({width:390,height:844});await page.locator('#start').click();await page.locator('#play-all:not([hidden])').waitFor();
  for(const img of await page.locator('#market img, #hand img').all()) { await img.scrollIntoViewIfNeeded(); await img.evaluate(img=>new Promise((resolve,reject)=> { if(img.naturalWidth) return resolve(); img.addEventListener('load',resolve,{once:true}); setTimeout(()=>img.naturalWidth?resolve():reject(new Error('Image did not load')),5000); })); }
  await page.evaluate(()=>window.scrollTo(0,0));
  await assertTableFits(page);
  await page.screenshot({path:'/tmp/astro-client-mobile.png',fullPage:true});
  await page.setViewportSize({width:1280,height:720}); await assertTableFits(page);
  await page.setViewportSize({width:390,height:844});
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
  // Render a dense, public game observation through a mock worker to exercise
  // mixed ship/base/outpost rows and every on-card control at small sizes.
  Game.catalog = JSON.parse(await fs.readFile(path.join(root, 'assets/cards.json')));
  const dense = new Game(17);
  Object.assign(dense.players[0], {hand:[0,1,2,8,15,16,17,18,19,29], trade:20, combat:12, must_discard:0,
    in_play:[2,8,15,16,17,18,19,29].map((id,j)=>new InPlay(j+1,id))});
  Object.assign(dense.players[1], {must_discard:3, in_play:[8,15,16,17,18,19,29,32].map((id,j)=>new InPlay(j+30,id))});
  const fixture = {id:'dense', revision:0, model_label:'Crowded fleet', status:'your_turn',
    opponent_win_probability:0.375,opponent_trade:0,opponent_combat:0,observation:dense.observation(0), action_log:[], result:null, can_play_all:false,
    decision:{family:'main', actions:dense.mainActions(dense.players[0]).map((a,id)=>({...a,id,label:Game.label(a)}))}};
  const denseContext = await browser.newContext({viewport:{width:1280,height:720}});
  await denseContext.addInitScript(({fixture,cards})=>{
    window.Worker = class {
      postMessage(message) {
        if (message.op === 'init') setTimeout(()=>this.onmessage({data:{requestId:message.requestId,data:{cards,models:[{id:'level-10',name:'Level 10'}],game:fixture,saved:null}}}),0);
      }
    };
  }, {fixture,cards:Game.catalog});
  const densePage = await denseContext.newPage();
  await densePage.goto(url); await densePage.locator('#hand .card').first().waitFor();
  for (const viewport of [{width:1280,height:720},{width:1440,height:900},{width:390,height:844}]) {
    await densePage.setViewportSize(viewport); await assertTableFits(densePage);
    const controls = await densePage.evaluate(()=>['hand','own-fleet','opponent-fleet','market'].flatMap(id=>{
      const cards=[...document.querySelectorAll(`#${id} .card`)];
      return cards.map(card=>{
        const r=card.getBoundingClientRect();
        return {zone:id,height:r.height,bottom:r.bottom,buttons:[...card.querySelectorAll('.card-actions button')].map(b=>{const q=b.getBoundingClientRect();return {top:q.top,bottom:q.bottom,left:q.left,right:q.right};}),top:r.top,left:r.left,right:r.right};
      });
    }));
    for (const zone of ['hand','own-fleet','opponent-fleet','market']) {
      const row=controls.filter(c=>c.zone===zone);
      assert.ok(Math.max(...row.map(c=>c.height))-Math.min(...row.map(c=>c.height))<1, `${zone}: ships, bases, and outposts have equal height`);
    }
    for (const c of controls) for (const b of c.buttons) assert.ok(b.top>=c.top && b.bottom<=c.bottom+1 && b.left>=c.left && b.right<=c.right+1, 'Card controls fit inside their enclosing box');
    assert.equal(await densePage.locator('#opponent-attack').isEnabled(), false, 'Outposts block authority attack');
    assert.equal(await densePage.locator('#opponent-must-discard').textContent(), '3');
    assert.equal(await densePage.locator('#opponent-win').textContent(), '37.5%');
    assert.equal(await densePage.locator('#player-bar #end-turn').isVisible(), true);
    if(viewport.width===1280) {
      await densePage.locator('#own-fleet img').first().evaluate(img=>new Promise(resolve=>{if(img.naturalWidth)resolve();else img.addEventListener('load',resolve,{once:true});}));
      await densePage.screenshot({path:'/tmp/astro-client-dense.png',fullPage:true});
    }
  }
  await denseContext.close();
  const scrapFixture = structuredClone(fixture);
  scrapFixture.decision = {family:'scrap', prompt:'Scrap up to 2 cards', actions:[
    {id:0, kind:'scrap_card', card_id:0, source_zone:'hand', label:'Scrap Scout from hand'},
    {id:1, kind:'scrap_card', card_id:1, source_zone:'discard', label:'Scrap Viper from discard'},
    {id:2, kind:'decline', label:'Done scrapping'},
  ]};
  scrapFixture.observation.own_discard = [1];
  const scrapContext = await browser.newContext({viewport:{width:1280,height:720}});
  await scrapContext.addInitScript(({fixture,cards})=>{
    window.Worker = class {
      postMessage(message) {
        if (message.op === 'choose') {
          fixture.decision.actions = fixture.decision.actions.filter(a=>a.id!==message.action_id);
          if (!fixture.decision.actions.some(a=>a.kind==='scrap_card')) {
            fixture.status='complete'; fixture.result={winner:0,truncated:false}; fixture.decision=null;
          }
          fixture.revision++;
        }
        setTimeout(()=>this.onmessage({data:{requestId:message.requestId,data:{cards,models:[{id:'level-10',name:'Level 10'}],game:fixture,saved:null}}}),0);
      }
    };
  }, {fixture:scrapFixture,cards:Game.catalog});
  const scrapPage = await scrapContext.newPage();
  await scrapPage.goto(url);
  await scrapPage.locator('#scrap-dialog[open]').waitFor();
  assert.equal(await scrapPage.locator('#scrap-dialog .card').count(), 2);
  assert.equal(await scrapPage.locator('#scrap-decline button').textContent(), 'Done scrapping');
  for (const viewport of [{width:1280,height:720},{width:390,height:844}]) {
    await scrapPage.setViewportSize(viewport);
    const bounds = await scrapPage.evaluate(()=>{
      const dialog=document.getElementById('scrap-dialog'), r=dialog.getBoundingClientRect();
      const button=dialog.querySelector('.move'), card=button.closest('.card'), info=card.querySelector('.details');
      return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:innerWidth,height:innerHeight,
        white:getComputedStyle(button).backgroundColor,cardLeft:card.getBoundingClientRect().left,infoLeft:info.getBoundingClientRect().left};
    });
    assert.ok(bounds.left>=0 && bounds.right<=bounds.width && bounds.top>=0 && bounds.bottom<=bounds.height);
    assert.equal(bounds.white, 'rgb(255, 255, 255)');
    assert.ok(bounds.infoLeft-bounds.cardLeft<5, 'Info icon is at the left edge');
  }
  await scrapPage.locator('#scrap-options-hand .card-face').click();
  await scrapPage.locator('#scrap-options-hand').waitFor({state:'hidden'});
  assert.equal(await scrapPage.locator('#scrap-dialog').isVisible(), true);
  await scrapPage.locator('#scrap-options-discard .card-face').click();
  await scrapPage.locator('#scrap-dialog').waitFor({state:'hidden'});
  await scrapPage.locator('#result-banner').waitFor();
  assert.equal(await scrapPage.locator('#result-title').textContent(), 'Victory!');
  await scrapPage.screenshot({path:'/tmp/astro-client-result.png',fullPage:true});
  await scrapContext.close();
  console.log('PASS automatic scrap picker, grouped targets, card-face selection, white actions, left info icons, and victory banner');
  console.log('PASS crowded fleets, equal base/outpost height, on-card control bounds, discard counter, and outpost protection');
  await context.close();
} finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
