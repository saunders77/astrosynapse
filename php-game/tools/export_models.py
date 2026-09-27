#!/usr/bin/env python3
"""Local export helper only. The uploaded game never invokes Python.

Single model: python export_models.py input.actor.npz output.model.php
Bundled lineage: python export_models.py --champions path/to/state.json
Requires NumPy on the exporting computer.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

PREFIX = b'<?php http_response_code(404); exit; __halt_compiler();\n'
ROOT = Path(__file__).resolve().parents[1]

def export(source, target):
    with np.load(source, allow_pickle=False) as z:
        spec = json.loads(bytes(z['__spec_json__'].tolist()))
        spec.setdefault('encoder_version', 1)
        spec.setdefault('objective_version', 1)
        tensors = {k: np.asarray(z[k], dtype='<f4') for k in z.files if k != '__spec_json__'}
    header = json.dumps({'format': 1, 'spec': spec, 'tensors': {
        k: {'shape': list(v.shape), 'bytes': v.nbytes} for k, v in tensors.items()
    }}, separators=(',', ':')).encode()
    with open(target, 'wb') as f:
        f.write(PREFIX + struct.pack('<I', len(header)) + header)
        for value in tensors.values():
            f.write(value.tobytes())
    return hashlib.sha256(Path(target).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', nargs='?')
    p.add_argument('target', nargs='?')
    p.add_argument('--champions')
    args=p.parse_args()
    if args.champions:
        state=json.loads(Path(args.champions).read_text())
        promotions=state['promotions'][-10:]
        entries=[]
        for level, item in enumerate(promotions, 1):
            filename=f'champion-{level:02d}.model.php'
            sha=export(item['actor'], ROOT/'models'/filename)
            entries.append({'id': f'champion-{level:02d}', 'name': f'Level {level} · Astro6 champion {level}',
                            'file': filename, 'sha256': sha, 'level': level,
                            'description': 'Current champion' if item['actor']==state['champion'] else 'Earlier promoted champion'})
        (ROOT/'models'/'registry.php').write_bytes(PREFIX + json.dumps(entries,indent=2).encode())
        print(f'Exported {len(entries)} actual champions.')
    elif args.source and args.target:
        print(export(args.source,args.target))
    else:
        p.error('provide SOURCE TARGET or --champions STATE_JSON')
if __name__=='__main__': main()
