#!/usr/bin/env python3
"""Package the incumbent Arch3 critic's inference weights for level 5."""
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'astrosynapse2/data/autopilot/828813dc45d24cf086e019c6537d1b02/inputs/critic.npz'


def export_level5_critic():
    with np.load(SOURCE, allow_pickle=False) as z:
        metadata = json.loads(z['metadata'].tobytes())
        tensors = {k: np.asarray(z[k], dtype='<f4') for k in
                   ('mean', 'scale', 'w0', 'b0', 'w1', 'b1', 'w2', 'b2')}
    header = json.dumps({'format': 1, 'encoder_version': metadata['encoder_version'],
                         'state_size': metadata['state_size'], 'families': metadata['families'],
                         'tensors': {k: {'shape': list(v.shape), 'bytes': v.nbytes}
                                     for k, v in tensors.items()}}, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 4)
    raw = b'ASTROCR1' + struct.pack('<I', len(header)) + header + b''.join(v.tobytes() for v in tensors.values())
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    filename = 'level-05-critic.astro.gz'
    (ROOT / 'models' / filename).write_bytes(compressed)
    return {'file': filename, 'source': str(SOURCE.relative_to(ROOT.parent)),
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'sha256': hashlib.sha256(compressed).hexdigest(), 'bytes': len(compressed),
            'encoder_version': metadata['encoder_version'], 'best_epoch': metadata['best_epoch'],
            'training_games': metadata['config']['games']}


if __name__ == '__main__':
    path = ROOT / 'models/registry.json'
    registry = json.loads(path.read_text())
    next(m for m in registry if m['id'] == 'level-05')['critic'] = export_level5_critic()
    path.write_text(json.dumps(registry, indent=2) + '\n')
