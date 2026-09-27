<?php
declare(strict_types=1);
namespace Astro;

// Native PHP inference, without Python, external services, FFI or extensions.
// Tensor storage is little-endian float32; arithmetic uses PHP doubles.
final class Actor {
    public array $spec;
    private array $tensors=[];
    const PREFIX="<?php http_response_code(404); exit; __halt_compiler();\n";
    public function __construct(string $path) {
        $handle=fopen($path,'rb'); if(!$handle) throw new \RuntimeException('Model not found');
        if(fread($handle,strlen(self::PREFIX))!==self::PREFIX) throw new \RuntimeException('Invalid model format');
        $length=unpack('V',fread($handle,4))[1]; if($length>1000000) throw new \RuntimeException('Invalid model header');
        $header=json_decode(fread($handle,$length),true,512,JSON_THROW_ON_ERROR); $this->spec=$header['spec'];
        foreach($header['tensors'] as $name=>$meta) {
            $bytes=$meta['bytes']; $data=''; while(strlen($data)<$bytes && !feof($handle)) $data.=fread($handle,$bytes-strlen($data));
            if(strlen($data)!==$bytes) throw new \RuntimeException('Truncated model');
            $this->tensors[$name]=['shape'=>$meta['shape'],'data'=>unpack('g*',$data)];
        }
        fclose($handle);
    }
    private function linear(array $x,string $prefix,?int $row=null): array {
        $w=$this->tensors[$prefix.'.weight']; $b=$this->tensors[$prefix.'.bias']['data']; $cols=$w['shape'][1]; $weights=$w['data']; $out=[];
        $start=$row??0; $end=$row===null?$w['shape'][0]:$row+1;
        // The encoded state and action vectors are sparse; avoiding zero
        // multiplies is especially useful on ordinary shared PHP hosting.
        $sparse=count($x)>300 || $prefix==='action_in';
        if($sparse) $x=array_filter($x,fn($v)=>$v!=0.0);
        for($i=$start;$i<$end;$i++) { $sum=$b[$i+1]; $offset=$i*$cols+1; foreach($x as $j=>$v) $sum+=$v*$weights[$offset+$j]; $out[]=$sum; }
        return $out;
    }
    private function norm(array $x,string $prefix): array {
        $n=count($x); $mean=array_sum($x)/$n; $variance=0.0; foreach($x as $v) $variance+=($v-$mean)**2;
        $scale=1/sqrt($variance/$n+$this->spec['layer_norm_eps']); $w=$this->tensors[$prefix.'.weight']['data']; $b=$this->tensors[$prefix.'.bias']['data'];
        foreach($x as $j=>$v) $x[$j]=($v-$mean)*$scale*$w[$j+1]+$b[$j+1]; return $x;
    }
    private static function silu(array $x): array { foreach($x as $j=>$v) $x[$j]=$v/(1+exp(-max(-40,min(40,$v)))); return $x; }
    private function residual(array $x,string $prefix): array {
        $y=$this->linear(self::silu($this->linear($this->norm($x,$prefix.'.norm'),$prefix.'.fc1')),$prefix.'.fc2');
        foreach($x as $j=>$v) $x[$j]=($v+$y[$j])*0.7071067811865475; return $x;
    }
    public function scores(array $state,array $actions,int $family): array {
        $s=self::silu($this->norm($this->linear($state,'state_in'),'state_norm'));
        for($i=0;$i<$this->spec['residual_blocks'];$i++) $s=$this->residual($s,'state_blocks.'.$i);
        $scores=[];
        foreach($actions as $a) {
            $v=$this->residual(self::silu($this->norm($this->linear($a,'action_in'),'action_norm')),'action_blocks.0');
            $v=self::silu($this->norm($this->linear(array_merge($s,$v),'fusion_in'),'fusion_norm'));
            for($i=0;$i<$this->spec['residual_blocks'];$i++) $v=$this->residual($v,'fusion_blocks.'.$i);
            $heads=[];
            for($i=0;$i<$this->spec['bootstrap_heads'];$i++) {
                $heads[]=$this->spec['objective_version']>=2 ? $this->linear($this->residual($v,'head_blocks.'.$i),'head_outputs.'.$i,$family)[0] : $this->linear($v,'output',$family*$this->spec['bootstrap_heads']+$i)[0];
            }
            $scores[]=array_sum($heads)/count($heads);
        }
        return $scores;
    }
    public function choose(array $d,?array $eligible=null): int {
        $indices=$eligible??array_keys($d['actions']);
        if(count($indices)===1) return $indices[0];
        $enc=new Encoder($this->spec['encoder_version']);
        $values=$this->scores($enc->state($d['observation']),array_map(fn($i)=>$enc->action($d['actions'][$i],$d['observation']),$indices),Encoder::FAMILIES[$d['family']]);
        return $indices[array_search(max($values),$values,true)];
    }
}
