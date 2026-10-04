#!/usr/bin/env python3
"""Create a static hosting archive without development or obsolete PHP files."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
target=ROOT.parent/'php-game-upload.zip'
files=[ROOT/'index.html',ROOT/'.htaccess',ROOT/'README.md',ROOT/'assets/game.js',ROOT/'assets/home-charts.js',ROOT/'assets/home-charts.js.LICENSE.txt',ROOT/'assets/worker.js',ROOT/'assets/style.css',ROOT/'assets/cards.json',*sorted((ROOT/'assets/audio').glob('*.mp3')),*sorted((ROOT/'models').glob('*')),ROOT/'models/.htaccess']
with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=9) as archive:
    for p in dict.fromkeys(files):
        if p.is_file() and p.suffix!='.php' and p.name!='.DS_Store': archive.write(p,p.relative_to(ROOT))
print(f'{target}: {target.stat().st_size:,} bytes')
