#!/usr/bin/env python3
"""Export float32 Astro2 checkpoints for the static JS arena (NumPy, build time only).
Single model: export_client_models.py checkpoint.actor.npz opponent.astro.gz
Bundled levels: export_client_models.py --levels
"""
import argparse, gzip, hashlib, json, sqlite3, struct
from pathlib import Path
import numpy as np
from export_client_critic import export_level5_critic
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

def checkpoint_training_games(db, checkpoint_id):
    """Count cumulative games once per ancestral run, including cross-run parents."""
    runs, visited = {}, set()
    while checkpoint_id:
        if checkpoint_id in visited:
            raise ValueError(f'Cyclic checkpoint ancestry: {checkpoint_id}')
        visited.add(checkpoint_id)
        row = db.execute('select run_id, games, parent_id from checkpoints where id=?',
                         (checkpoint_id,)).fetchone()
        if row is None:
            raise ValueError(f'Missing checkpoint ancestor: {checkpoint_id}')
        run_id, games, checkpoint_id = row
        runs[run_id] = max(runs.get(run_id, 0), games)
    return sum(runs.values())


def model_training_games(db, model):
    """Resolve database checkpoints and file-based training experiment sources."""
    row = db.execute('select id from checkpoints where id=? or path=? or actor_path=?',
                     (model, model, model)).fetchone()
    if row:
        return checkpoint_training_games(db, row[0])
    path = Path(model)
    if not path.is_absolute():
        path = REPO / 'astrosynapse2' / path
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    # Experiment filenames contain that experiment's cumulative game count.
    games = int(path.name.split('.')[0].removeprefix('g'))
    return games + model_training_games(db, manifest['model'])


def levels():
    db=sqlite3.connect(f'file:{REPO}/astrosynapse2/data/astrosynapse2.sqlite3?immutable=1',uri=True)
    # Five measured opponents spanning the September 30 Elo field.
    first=db.execute('select id from checkpoints order by created_at limit 1').fetchone()[0]
    selections=[(first,113),('a4a66dd88a0a41e7',743),('1b822120f1634e46',914)]
    sources=[]
    for cid,elo in selections:
        row=db.execute('select actor_path from checkpoints where id=?',(cid,)).fetchone()
        sources.append((Path(row[0]),elo,{'checkpoint_id':cid}))
    sources.append((REPO/'astrosynapse2/data/analysis/astro5-champion-8844ddc7295a4f60.actor.npz',1093,{'checkpoint_id':'8844ddc7295a4f60'}))
    sources.append((REPO/'astrosynapse2/data/autopilot/828813dc45d24cf086e019c6537d1b02/tasks/policy-00004/g00020000.actor.npz',1283,{'checkpoint_id':'auto-828813dc45d24cf086e019c6537d1b02-pd9dd811c376c'}))
    entries=[]
    for level,(source,elo,provenance) in enumerate(sources,1):
        filename=f'level-{level:02d}.astro.gz'
        label=f'{elo} ELO'
        meta=export(source,ROOT/'models'/filename)
        entries.append({'id':f'level-{level:02d}','name':f'Level {level} ({elo} ELO)','level':level,'file':filename,'description':label,'elo':elo,'source':str(source.relative_to(REPO)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),**provenance,**meta})
        print(f'Level {level}: {label}; {meta["bytes"]:,} bytes')
    entries[-1]['critic']=export_level5_critic()
    (ROOT/'models/registry.json').write_text(json.dumps(entries,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source',nargs='?'); p.add_argument('target',nargs='?'); p.add_argument('--levels',action='store_true'); a=p.parse_args()
    if a.levels: levels()
    elif a.source and a.target: print(json.dumps(export(a.source,a.target)))
    else: p.error('provide SOURCE TARGET or --levels')
