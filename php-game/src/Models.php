<?php
declare(strict_types=1);
namespace Astro;
final class Models {
    public static function read(): array {
        // Writers replace the complete registry atomically; readers see one
        // coherent version and do not need a lock or write permissions.
        $raw=file_get_contents(__DIR__.'/../models/registry.php');
        if($raw===false || strncmp($raw,Actor::PREFIX,strlen(Actor::PREFIX))!==0) throw new \RuntimeException('Cannot read model registry');
        return json_decode(substr($raw,strlen(Actor::PREFIX)),true,512,JSON_THROW_ON_ERROR);
    }
    public static function publicList(): array { return array_map(fn($m)=>array_intersect_key($m,array_flip(['id','name','level','description'])),self::read()); }
    public static function get(string $id): array { foreach(self::read() as $m) if($m['id']===$id) return $m; throw new \InvalidArgumentException('Unknown opponent'); }
    public static function path(array $m): string { if(basename($m['file'])!==$m['file'] || !preg_match('/^[a-zA-Z0-9._-]+\.php$/D',$m['file'])) throw new \RuntimeException('Invalid model path'); return __DIR__.'/../models/'.$m['file']; }
    public static function update(callable $change): void {
        $dir=__DIR__.'/../models/'; $path=$dir.'registry.php';
        $f=fopen($dir.'registry.lock.php','c+');
        if(!$f || !flock($f,LOCK_EX)) throw new \RuntimeException('Models directory must be writable');
        $temporary=$dir.'registry-'.bin2hex(random_bytes(8)).'.php';
        try {
            if(fstat($f)['size']===0) fwrite($f,Actor::PREFIX);
            $data=Actor::PREFIX.json_encode($change(self::read()),JSON_PRETTY_PRINT|JSON_THROW_ON_ERROR);
            if(file_put_contents($temporary,$data)!==strlen($data) || !rename($temporary,$path)) throw new \RuntimeException('Unable to save model names');
        } finally {
            if(is_file($temporary)) unlink($temporary);
            flock($f,LOCK_UN); fclose($f);
        }
    }
    public static function name(string $name): string { $name=trim($name); if($name==='' || strlen($name)>100 || preg_match('/[\x00-\x1F]/',$name)) throw new \InvalidArgumentException('Use a model name of 1–100 characters'); return $name; }
    // Validate before writing executable-extension files. Never copy the PHP
    // prefix from an upload; it is replaced with our fixed non-executing guard.
    public static function import(string $source,string $name): array {
        $name=self::name($name); $raw=file_get_contents($source); if(strlen($raw)>40*1024*1024) throw new \InvalidArgumentException('Model exceeds 40 MB');
        if((strncmp($raw,Actor::PREFIX,strlen(Actor::PREFIX))===0)) {
            $offset=strlen(Actor::PREFIX); $size=unpack('V',substr($raw,$offset,4))[1]; if($size>1000000) throw new \InvalidArgumentException('Invalid header');
            $header=json_decode(substr($raw,$offset+4,$size),true,512,JSON_THROW_ON_ERROR); $payload=substr($raw,$offset+4+$size);
        } elseif(substr($raw,0,2)==='PK') {
            if(!class_exists('ZipArchive')) throw new \InvalidArgumentException('NPZ import needs PHP ZipArchive. Export a .model.php file on your computer instead.');
            $zip=new \ZipArchive(); if($zip->open($source)!==true) throw new \InvalidArgumentException('Invalid NPZ archive');
            $tensors=[]; $payload=''; $spec=null; $total=0;
            try {
                if($zip->numFiles>300) throw new \InvalidArgumentException('Too many tensors');
                for($i=0;$i<$zip->numFiles;$i++) { $stat=$zip->statIndex($i); $total+=$stat['size']; if($total>40*1024*1024) throw new \InvalidArgumentException('Expanded model exceeds 40 MB');
                    $key=$stat['name']; if(!preg_match('/^([a-zA-Z0-9_.]+)\.npy$/D',$key,$m)) throw new \InvalidArgumentException('Unexpected archive entry');
                    [$shape,$type,$bytes]=self::npy($zip->getFromIndex($i)); $key=$m[1];
                    if($key==='__spec_json__') { if($type!=='|u1') throw new \InvalidArgumentException('Invalid model spec'); $spec=json_decode($bytes,true,32,JSON_THROW_ON_ERROR); }
                    else { if($type!=='<f4' && $type!=='=f4') throw new \InvalidArgumentException('Only float32 actor exports are supported'); $tensors[$key]=['shape'=>$shape,'bytes'=>strlen($bytes)]; $payload.=$bytes; }
                }
            } finally { $zip->close(); }
            $header=['format'=>1,'spec'=>$spec,'tensors'=>$tensors];
        } else throw new \InvalidArgumentException('Choose an Astro2 .actor.npz or exported .model.php file');
        $header['spec']+=['encoder_version'=>1,'objective_version'=>1];
        self::validate($header,$payload);
        $id='custom-'.bin2hex(random_bytes(8)); $file=$id.'.model.php'; $encoded=json_encode($header,JSON_THROW_ON_ERROR);
        $data=Actor::PREFIX.pack('V',strlen($encoded)).$encoded.$payload;
        $path=__DIR__.'/../models/'.$file;
        if(file_put_contents($path,$data,LOCK_EX)!==strlen($data)) throw new \RuntimeException('Models directory must be writable');
        $entry=['id'=>$id,'name'=>$name,'file'=>$file,'sha256'=>hash('sha256',$data),'description'=>'Custom model'];
        try { self::update(function($rows) use($entry) { $rows[]=$entry; return $rows; }); } catch(\Throwable $e) { unlink($path); throw $e; }
        return $entry;
    }
    private static function npy(string $data): array {
        if(substr($data,0,6)!=="\x93NUMPY") throw new \InvalidArgumentException('Invalid NPY tensor');
        $version=ord($data[6]); $start=$version===1?10:12; $length=unpack($version===1?'v':'V',substr($data,8,$version===1?2:4))[1]; $h=substr($data,$start,$length);
        if(!preg_match("/'descr':\\s*'([^']+)'/",$h,$d) || !preg_match("/'shape':\\s*\\(([^)]*)\\)/",$h,$s) || !preg_match("/'fortran_order':\\s*False/",$h)) throw new \InvalidArgumentException('Unsupported NPY layout');
        $shape=array_map('intval',array_filter(array_map('trim',explode(',',$s[1])),fn($v)=>$v!==''));
        return [$shape,$d[1],substr($data,$start+$length)];
    }
    public static function validate(array $h,string $payload): void {
        $s=$h['spec']??[];
        $s+=['encoder_version'=>1,'objective_version'=>1];
        foreach(['state_size','action_size','families','hidden_size','action_hidden_size','residual_blocks','bootstrap_heads'] as $k) if(!isset($s[$k]) || !is_int($s[$k])) throw new \InvalidArgumentException('Invalid architecture');
        if(($h['format']??0)!==1 || !in_array($s['encoder_version'],[1,2],true) || !in_array($s['objective_version'],[1,2],true) || $s['state_size']!==1292 || $s['action_size']!==($s['encoder_version']===2?219:203) || $s['families']!==8 || $s['hidden_size']<1 || $s['hidden_size']>384 || $s['action_hidden_size']<1 || $s['action_hidden_size']>192 || $s['residual_blocks']<0 || $s['residual_blocks']>6 || $s['bootstrap_heads']<1 || $s['bootstrap_heads']>8 || ($s['layer_norm_eps']??0)<=0) throw new \InvalidArgumentException('Unsupported Astro2 architecture');
        $expected=[];
        $linear=function($p,$out,$in) use(&$expected) { $expected[$p.'.weight']=[$out,$in]; $expected[$p.'.bias']=[$out]; };
        $norm=function($p,$n) use(&$expected) { $expected[$p.'.weight']=[$n]; $expected[$p.'.bias']=[$n]; };
        $block=function($p,$n) use($linear,$norm) { $norm($p.'.norm',$n); $linear($p.'.fc1',$n*2,$n); $linear($p.'.fc2',$n,$n*2); };
        $w=$s['hidden_size']; $aw=$s['action_hidden_size'];
        $linear('state_in',$w,1292); $norm('state_norm',$w); $linear('action_in',$aw,$s['action_size']); $norm('action_norm',$aw); $block('action_blocks.0',$aw); $linear('fusion_in',$w,$w+$aw); $norm('fusion_norm',$w);
        for($j=0;$j<$s['residual_blocks'];$j++) { $block('state_blocks.'.$j,$w); $block('fusion_blocks.'.$j,$w); }
        if($s['objective_version']>=2) { for($j=0;$j<$s['bootstrap_heads'];$j++) { $block('head_blocks.'.$j,$w); $linear('head_outputs.'.$j,8,$w); } $linear('value_output',8*$s['bootstrap_heads'],$w); } else $linear('output',8*$s['bootstrap_heads'],$w);
        if(count($h['tensors']??[])!==count($expected)) throw new \InvalidArgumentException('Unexpected tensor count');
        if(strlen($payload)>12*1024*1024) throw new \InvalidArgumentException('Model weights exceed the 12 MB shared-host limit');
        $offset=0; foreach($h['tensors'] as $name=>$meta) {
            if(!isset($expected[$name]) || $meta['shape']!==$expected[$name] || $meta['bytes']!==array_product($expected[$name])*4) throw new \InvalidArgumentException('Invalid tensor '.$name);
            if($offset+$meta['bytes']>strlen($payload)) throw new \InvalidArgumentException('Truncated tensor');
            // Inspect one tensor at a time to keep memory bounded during upload.
            foreach(unpack('g*',substr($payload,$offset,$meta['bytes'])) as $v) if(!is_finite($v)) throw new \InvalidArgumentException('Non-finite model weights');
            $offset+=$meta['bytes'];
        }
        if($offset!==strlen($payload)) throw new \InvalidArgumentException('Unexpected model data');
    }
}
