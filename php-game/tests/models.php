<?php
declare(strict_types=1);
require __DIR__.'/../src/bootstrap.php';
use Astro\Actor;
use Astro\Models;
$rows=Models::read(); if(count($rows)!==10) throw new RuntimeException('Expected 10 champions');
foreach($rows as $m) {
    $path=Models::path($m); if(hash_file('sha256',$path)!==$m['sha256']) throw new RuntimeException('Checkpoint hash mismatch');
    $raw=file_get_contents($path); $p=strlen(Actor::PREFIX); $n=unpack('V',substr($raw,$p,4))[1]; $h=json_decode(substr($raw,$p+4,$n),true,512,JSON_THROW_ON_ERROR); $payload=substr($raw,$p+4+$n);
    Models::validate($h,$payload);
    echo 'PASS: '.$m['name']." checkpoint structure and SHA-256.\n";
}
$registry=__DIR__.'/../models/registry.php'; $original=file_get_contents($registry); $created=[];
try {
    $entry=Models::import(__DIR__.'/../models/champion-10.model.php','Import check'); $created[]=Models::path($entry);
    if(Models::get($entry['id'])['name']!=='Import check') throw new RuntimeException('Import failed');
    Models::update(function($rows) use($entry) { foreach($rows as &$r) if($r['id']===$entry['id']) $r['name']='Renamed champion'; return $rows; });
    if(Models::get($entry['id'])['name']!=='Renamed champion') throw new RuntimeException('Rename failed');
    echo "PASS: preconverted model import and rename.\n";
    if(isset($argv[1])) { $entry=Models::import($argv[1],'NPZ check'); $created[]=Models::path($entry); $actor=new Actor(Models::path($entry)); if($actor->spec!==$h['spec']) throw new RuntimeException('NPZ spec mismatch'); echo "PASS: native PHP NPZ import.\n"; }
    $h['spec']['state_size']=1;
    try { Models::validate($h,$payload); throw new RuntimeException('Invalid architecture accepted'); } catch(InvalidArgumentException $e) { echo "PASS: invalid architecture rejected.\n"; }
} finally { file_put_contents($registry,$original); foreach($created as $p) unlink($p); }
