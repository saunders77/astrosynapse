// Run with PLAYWRIGHT_MODULE pointing to a local Playwright installation.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Game} from '../assets/runtime/engine.mjs';
import {Session} from '../assets/runtime/session.mjs';
import {opponentTurnSummary} from '../assets/runtime/turn-summary.mjs';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('../', import.meta.url));
const cards = JSON.parse(await fs.readFile(path.join(root, 'assets/cards.json')));
Game.catalog = cards;
const session = Session.start({id:'level-05',name:'Test opponent'}, true, 42);
const game = Session.advance(session, {winProbability:()=>.5});
const model = {id:'level-05', name:'Test opponent', level:5};
let instrumentUI = true;
const server = http.createServer(async (req, res) => {
  try {
    const name = new URL(req.url, 'http://localhost').pathname.slice(1) || 'index.html';
    if (name.includes('..')) throw Error('Invalid path');
    let data;
    if (name === 'assets/game.js' && instrumentUI) {
      data = (await fs.readFile(path.join(root, 'assets/runtime/ui.mjs'), 'utf8')).replaceAll("from './", "from './runtime/");
      data += '\nwindow.fixture = {set(next) {game=next; render(); lock(false);}, get() {return game;}, acquire: animateAcquisitions};';
    } else if (name === 'assets/worker.js') {
      data = `onmessage = ({data}) => postMessage({requestId:data.requestId, data:${JSON.stringify({cards,models:[model],game})}});`;
    } else data = await fs.readFile(path.join(root, name));
    const type = {'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.json':'application/json'}[path.extname(name)] || 'application/octet-stream';
    res.writeHead(200, {'Content-Type':type}); res.end(data);
  } catch {res.writeHead(404); res.end();}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({headless:true});
try {
  const page = await browser.newPage({viewport:{width:1440,height:900}});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('https://www.starrealms.com/**', route => route.fulfill({contentType:'image/svg+xml', body:'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="700"><rect width="500" height="700" fill="#314253"/></svg>'}));
  await page.goto(`http://127.0.0.1:${server.address().port}/#game`);
  await page.waitForFunction(() => window.fixture?.get());
  await page.waitForTimeout(300);
  assert.equal(await page.locator('#status').textContent(), '');
  assert.equal(await page.locator('#player-bar .decision .inspectors button').count(), 3);
  await page.locator('#hand .details').first().click();
  await page.mouse.click(2,2);
  assert.equal(await page.locator('#card-dialog').evaluate(n => n.open), false);
  for (const kind of ['scrap_card','discard_card','choose_mode']) {
    const fixture = structuredClone(game);
    fixture.revision++;
    fixture.decision = {family:kind, prompt:'Choose a card', actions:[{id:0,kind,card_id:0,source_zone:'hand',label:'Choose Scout'}]};
    await page.evaluate(g => window.fixture.set(g), fixture);
    const dialog = kind === 'scrap_card' ? '#scrap-dialog' : '#decision-dialog';
    await page.locator(dialog).waitFor({state:'visible'});
    if (kind === 'discard_card') await page.locator('#decision-cancel').click();
    else await page.mouse.click(2,2);
    await page.locator(dialog).waitFor({state:'hidden'});
    assert.equal(await page.evaluate(() => window.fixture.get().revision), fixture.revision);
    await page.locator('[data-inspect="own-deck"]').click();
    await page.locator('#pile-dialog').waitFor({state:'visible'});
    await page.locator('[data-close="pile-dialog"]').click();
    await page.locator('#choose-bar').click();
    await page.locator(dialog).waitFor({state:'visible'});
    assert.equal(await page.locator('#choose-bar').isVisible(), false);
    await page.evaluate(g => window.fixture.set(g), game);
  }
  for (const [player_id, kind] of [[0,'acquire'],[1,'acquire'],[0,'free_acquire']]) {
    await page.evaluate(g => window.fixture.set(g), game);
    const next = structuredClone(game);
    next.action_log.push({kind,player_id,card_id:game.observation.trade_row[0],target_card_id:game.observation.trade_row[0]});
    const duration = await page.evaluate(async next => {
      const promise = window.fixture.acquire(next);
      const ghost = document.querySelector('.acquiring-card');
      const animation = ghost.getAnimations()[0];
      const duration = animation.effect.getTiming().duration;
      if (animation.effect.getTiming().easing !== 'cubic-bezier(0.42, 0, 1, 1)') throw Error('Acquisition should accelerate from rest');
      const entry = next.action_log.at(-1);
      const target = document.querySelector(`[data-inspect="${entry.player_id ? 'opponent-deck' : 'own-deck'}"]`).getBoundingClientRect();
      const from = ghost.getBoundingClientRect();
      const transform = new DOMMatrix(animation.effect.getKeyframes().at(-1).transform);
      if (Math.abs(from.x + from.width / 2 + transform.m41 - target.x - target.width / 2) > 1 ||
          Math.abs(from.y + from.height / 2 + transform.m42 - target.y - target.height / 2) > 1) throw Error('Incorrect acquisition destination');
      await promise;
      if (document.querySelector('.acquiring-card')) throw Error('Animation was not cleaned up');
      const source = document.querySelector(`#market .card[data-card-id="${entry.card_id}"]`);
      if (getComputedStyle(source).visibility !== 'hidden') throw Error('Acquired card reappeared before the board update');
      next.observation.trade_row[0] = 0;
      window.fixture.set(next);
      const replacement = document.querySelector('#market .card');
      if (replacement.dataset.cardId !== '0' || getComputedStyle(replacement).visibility !== 'visible') throw Error('Replacement card is not visible');
      return duration;
    }, next);
    assert.equal(duration,350);
  }
  for (const viewport of [{width:1440,height:900},{width:390,height:844},{width:320,height:568},{width:667,height:375}]) {
    await page.setViewportSize(viewport);
    const heights = [];
    for (const contents of [[],[cards.find(c=>c.card_type !== 'ship').card_id],[0],Array(48).fill(0)]) {
      const fixture = structuredClone(game);
      fixture.observation.own_in_play = contents.map(card=>({card}));
      await page.evaluate(g=>window.fixture.set(g),fixture);
      await page.waitForTimeout(500);
      const layout = await page.evaluate(() => {
        const container = document.querySelector('#own-fleet');
        const rect = container.getBoundingClientRect();
        return {height:rect.height, rows:[...container.querySelectorAll('.card')].map(n=>{const r=n.getBoundingClientRect();return {top:r.top,bottom:r.bottom,right:r.right};}),bottom:rect.bottom,right:rect.right,scroll:document.documentElement.scrollHeight,viewport:innerHeight};
      });
      heights.push(layout.height);
      assert.ok(layout.scroll <= layout.viewport + 1);
      for (const card of layout.rows) {assert.ok(card.bottom<=layout.bottom+1, JSON.stringify({viewport,layout}));}
      if (contents.length === 48) {
        assert.equal(new Set(layout.rows.map(r=>r.top)).size, 1, 'Crowded rows stay on one line');
        assert.ok(layout.rows[0].bottom - layout.rows[0].top >= layout.height - 1, `Cards fill the row height: ${JSON.stringify({viewport,layout})}`);
        await page.getByRole('button', {name:'Scroll right: In play', exact:true}).click();
        await page.waitForTimeout(400);
        assert.ok(await page.locator('#own-fleet').evaluate(n=>n.scrollLeft > 0), 'Carousel moves right');
      }
      const market = await page.locator('#market').evaluate(n=>({width:n.clientWidth,scroll:n.scrollWidth,cards:[...n.children].map(c=>({w:c.clientWidth,h:c.clientHeight}))}));
      assert.equal(market.cards.length,6);
      assert.ok(market.scroll<=market.width+1, 'Six market cards fit without overflow');
      assert.ok(market.cards.every(c=>c.h>c.w), 'All market cards use portrait slots');
      assert.equal(await page.locator('#must-discard').isVisible(),false);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth), 'No page overflow');
    }
    assert.ok(heights[0] < heights[1] && heights[1] < heights[2], 'Empty and base-only rows reserve less space');
    const discards = structuredClone(game);
    discards.model_label = 'Level 5';
    discards.observation.pending_discard = 1;
    discards.observation.opponent_pending_discard = 3;
    await page.evaluate(g=>window.fixture.set(g), discards);
    assert.equal(await page.locator('#must-discard').isVisible(), true);
    assert.equal(await page.locator('#opponent-must-discard').isVisible(), true);
    assert.equal(await page.locator('#opponent-name').textContent(), 'Lv. 5');
    assert.equal(await page.locator('#authority').evaluate(n=>getComputedStyle(n).color), 'rgb(0, 0, 0)');
    assert.ok(await page.locator('#win-estimate').evaluate(n=>{const r=n.getBoundingClientRect();return r.left>=0 && r.right<=innerWidth;}), 'Win probability stays inside the viewport with discard counters');
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:`/tmp/php-game-ui-${viewport.width}.png`});
  }
  assert.equal(opponentTurnSummary({status:'your_turn',observation:{turn:3},action_log:[
    {player_id:1,turn:2,kind:'play_card',card_id:0}, {player_id:1,turn:2,kind:'play_card',card_id:0},
    {player_id:1,turn:2,kind:'play_card',card_id:1}, {player_id:1,turn:2,kind:'acquire',card_id:2},
  ]},cards),'Last turn: Played Scout x2, Viper. Acquired Explorer');
  instrumentUI = false;
  await page.reload();
  await page.locator('#hand .card').first().waitFor();
  assert.equal(await page.evaluate(() => typeof window.fixture), 'undefined', 'Smoke test uses the production bundle');
  await page.locator('#hand .details').first().click();
  await page.mouse.click(2,2);
  assert.equal(await page.locator('#card-dialog').evaluate(n=>n.open), false);
  assert.deepEqual(errors, []);
  console.log('PASS minimize/restore, pile inspection, detail backdrop, acquisition timing for both players, summaries, desktop/mobile sizing and carousels');
} finally {await browser.close(); await new Promise(resolve=>server.close(resolve));}
