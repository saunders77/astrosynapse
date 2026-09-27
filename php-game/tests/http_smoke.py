#!/usr/bin/env python3
"""Protocol checks against a running PHP server; no browser dependencies."""
import http.cookiejar,json,sys,time,urllib.request,urllib.error
base=(sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8092').rstrip('/')
cookies=http.cookiejar.CookieJar(); client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
def request(payload=None, token=None, status=200):
    data=json.dumps(payload).encode() if payload is not None else None
    headers={'Content-Type':'application/json'} if data else {}
    if token: headers['X-CSRF-Token']=token
    r=urllib.request.Request(base+'/api.php',data=data,headers=headers)
    try: result=client.open(r)
    except urllib.error.HTTPError as e: result=e
    body=result.read()
    assert result.status==status,(result.status,body)
    return json.loads(body)
def move(op,g,**kw): return {'op':op,'id':g['id'],'revision':g['revision'],**kw}
assert client.open(base+'/').status==200
for path in ['assets/game.js','assets/style.css','assets/cards.json','assets/card-art/BattlePod.jpg']:
    assert client.open(base+'/'+path).status==200
r=request(); token=r['csrf']; assert len(r['models'])==10
assert any(c.name=='astrosynapse_game' and c.has_nonstandard_attr('HttpOnly') for c in cookies)
request({'op':'new','model':'champion-10','starts':True},status=403)
g=request({'op':'new','model':'champion-10','starts':True},token)['game']; assert g['status']=='your_turn' and g['can_play_all']
assert request()['game']==g
request(move('choose',g,action_id=9999),token,status=400)
assert request()['game']==g
old=g; g=request(move('play_all',g),token)['game']; assert not g['observation']['hand']
request(move('play_all',old),token,status=409)
assert request()['game']==g
end=next(a for a in g['decision']['actions'] if a['kind']=='end_turn')
g=request(move('choose',g,action_id=end['id']),token)['game']; assert g['status']=='model_thinking'
request(move('play_all',g),token,status=400)
request(move('choose',g,action_id=0),token,status=400)
assert request()['game']==g
n=0; t=time.monotonic()
while g['status']=='model_thinking' and n<40:
    g=request(move('advance',g),token)['game']; n+=1
assert g['status']=='your_turn'
request({'op':'rename','model':'champion-10','name':'Not authorized'},token,status=403)
assert request()['models'][-1]['name']=='Level 10 · Astro6 champion 10'
assert all(a['kind']!='play_all' for a in g['decision']['actions'])
for path in ['models/champion-10.model.php','models/registry.php']:
    try: client.open(base+'/'+path); raise AssertionError('Model exposed')
    except urllib.error.HTTPError as e: assert e.code==404
print(f'PASS: assets, 10 models, sessions, CSRF, illegal/stale moves, batch play, hidden model files, admin protection, and {n} server-side computer decisions ({time.monotonic()-t:.2f}s).')
