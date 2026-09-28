#!/usr/bin/env python3
"""Export float32 Astro2 checkpoints for the static JS arena (NumPy, build time only).
Single model: export_client_models.py checkpoint.actor.npz opponent.astro.gz
Bundled levels: export_client_models.py --levels
"""
import argparse, gzip, hashlib, json, sqlite3, struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent

def export(source,target):
    with np.load(source,allow_pickle=False) as z:
        spec=json.loads(z['__spec_json__'].tobytes())
        spec.setdefault('encoder_version',1); spec.setdefault('objective_version',1)
        tensors={k:np.asarray(z[k],dtype='<f4') for k in z.files if k!='__spec_json__'}
    header=json.dumps({'format':1,'spec':spec,'tensors':{k:{'shape':list(v.shape),'bytes':v.nbytes} for k,v in tensors.items()}},separators=(',',':')).encode()
    header+=b' '*(-len(header)%4)
    raw=b'ASTROJS1'+struct.pack('<I',len(header))+header+b''.join(v.tobytes() for v in tensors.values())
    compressed=gzip.compress(raw,compresslevel=9,mtime=0)
    Path(target).write_bytes(compressed)
    return {'sha256':hashlib.sha256(compressed).hexdigest(),'bytes':len(compressed),'uncompressed_bytes':len(raw),'parameters':sum(v.size for v in tensors.values()),'spec':spec}

def levels():
    db=sqlite3.connect(f'file:{REPO}/astrosynapse2/data/astrosynapse2.sqlite3?immutable=1',uri=True)
    # Earliest checkpoint, then selected promoted champions spanning Astro2–Astro6.
    first=db.execute('select id from checkpoints order by created_at limit 1').fetchone()[0]
    selections=[(first,'First Astrosynapse2 checkpoint · untrained'),('1b822120f1634e46','Astro2 · 3,303,168 games'),('246e56c917644759','Astro3 · 2,602,496 games'),('05ef55aaf4c548c5','Astro4 · 98,176 games'),('0ecf69b96351463d','Astro4 · 502,656 games'),('08aa018c672847d9','Astro5 · 5,954,048 games')]
    sources=[]
    for cid,label in selections:
        row=db.execute('select actor_path from checkpoints where id=?',(cid,)).fetchone()
        sources.append((Path(row[0]),label,{'checkpoint_id':cid}))
    state=json.loads((REPO/'astrosynapse2/data/progressive/evolution-20260923/state.json').read_text())
    for generation in [1,2,6,10]:
        p=state['promotions'][generation-1]
        assert p.get('passed') is True
        sources.append((Path(p['actor']),f'Astro6 generation {generation}',{'generation':generation}))
    entries=[]
    for level,(source,label,provenance) in enumerate(sources,1):
        filename=f'level-{level:02d}.astro.gz'
        meta=export(source,ROOT/'models'/filename)
        entries.append({'id':f'level-{level:02d}','name':f'Level {level}','level':level,'file':filename,'description':label,'source':str(source.relative_to(REPO)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),**provenance,**meta})
        print(f'Level {level}: {label}; {meta["bytes"]:,} bytes')
    (ROOT/'models/registry.json').write_text(json.dumps(entries,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source',nargs='?'); p.add_argument('target',nargs='?'); p.add_argument('--levels',action='store_true'); a=p.parse_args()
    if a.levels: levels()
    elif a.source and a.target: print(json.dumps(export(a.source,a.target)))
    else: p.error('provide SOURCE TARGET or --levels')
