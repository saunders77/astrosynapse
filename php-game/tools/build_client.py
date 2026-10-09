#!/usr/bin/env python3
"""Bundle the browser and worker, then stamp HTML with a content-derived cache key.
Uses this repository's esbuild only at build time. Hosting is entirely static.
"""
import hashlib,json,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ESBUILD=ROOT.parent/'astrosynapse2/node_modules/.bin/esbuild'
for source,target in [('ui','game'),('worker','worker')]:
    subprocess.run([str(ESBUILD),str(ROOT/f'assets/runtime/{source}.mjs'),'--bundle','--format=esm','--target=es2022','--minify',f'--outfile={ROOT}/assets/{target}.js'],check=True)
subprocess.run([str(ESBUILD),str(ROOT/'assets/runtime/home-charts.tsx'),'--bundle','--format=esm','--target=es2022','--minify','--define:process.env.NODE_ENV="production"',f'--outfile={ROOT}/assets/home-charts.js'],check=True,env={**os.environ,'NODE_PATH':str(ROOT.parent/'astrosynapse2/node_modules')})
paths=[ROOT/'assets/home-charts.js',ROOT/'assets/game.js',ROOT/'assets/worker.js',ROOT/'assets/style.css',ROOT/'assets/cards.json',ROOT/'models/registry.json']
paths += sorted((ROOT/'assets/icons').glob('*'))
paths += sorted((ROOT/'assets/audio').glob('*.mp3'))
build=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
template=(ROOT/'tools/index.template.html').read_text()
script=(ROOT/'tools/boot.js').read_text()
import base64
csp_hash=base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
html=template.replace('{{BUILD}}',build).replace('{{BOOT_HASH}}',csp_hash).replace('{{BOOT}}',script)
(ROOT/'index.html').write_text(html)
print('Build:',build)
