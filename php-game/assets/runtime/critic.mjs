import { Encoder } from './encoder.mjs';

export class Critic {
  constructor(buffer) {
    const bytes=new Uint8Array(buffer), decoder=new TextDecoder();
    if(bytes.length<12||decoder.decode(bytes.subarray(0,8))!=='ASTROCR1') throw new Error('Invalid critic format');
    const length=new DataView(buffer).getUint32(8,true);
    if(length>1000000||length%4||12+length>bytes.length) throw new Error('Invalid critic header');
    const header=JSON.parse(decoder.decode(bytes.subarray(12,12+length)));
    if(header.format!==1||![1,2,3].includes(header.encoder_version)||header.state_size!==(header.encoder_version===3?1341:1292)||header.families!==8) throw new Error('Unsupported critic architecture');
    this.encoder=new Encoder(header.encoder_version);
    this.size=header.state_size+header.families;
    this.tensors=Object.create(null);
    let offset=12+length;
    for(const [name,meta] of Object.entries(header.tensors)) {
      const size=meta.shape.reduce((a,b)=>a*b,1);
      if(!Number.isSafeInteger(size)||size<=0||meta.bytes!==size*4||offset+meta.bytes>bytes.length) throw new Error('Invalid critic tensor');
      const data=new Float32Array(buffer,offset,size);
      if(data.some(v=>!Number.isFinite(v))) throw new Error('Non-finite critic weights');
      this.tensors[name]={shape:meta.shape,data}; offset+=meta.bytes;
    }
    if(offset!==bytes.length) throw new Error('Unexpected critic data');
    const width=this.tensors.b0?.shape[0];
    if(![32,64,128,256].includes(width)) throw new Error('Invalid critic width');
    const shapes={mean:[this.size],scale:[this.size],w0:[this.size,width],b0:[width],w1:[width,width/2],b1:[width/2],w2:[width/2,1],b2:[1]};
    for(const [name,shape] of Object.entries(shapes)) if(JSON.stringify(this.tensors[name]?.shape)!==JSON.stringify(shape)) throw new Error('Invalid critic tensor: '+name);
    if(this.tensors.scale.data.some(v=>v<=0)) throw new Error('Invalid critic normalization');
  }
  predict(input) {
    if(input.length!==this.size) throw new Error('Invalid critic input size');
    const t=this.tensors;
    let x=Float32Array.from(input,(v,i)=>Math.max(-10,Math.min(10,Math.fround(Math.fround(Math.fround(v)-t.mean.data[i])/t.scale.data[i]))));
    for(let layer=0;layer<3;layer++) {
      const w=t['w'+layer], b=t['b'+layer].data, out=new Float32Array(b.length);
      for(let j=0;j<out.length;j++) {
        let sum=0;
        for(let i=0;i<x.length;i++) sum+=x[i]*w.data[i*out.length+j];
        sum=Math.fround(Math.fround(sum)+b[j]);
        out[j]=layer<2?Math.tanh(sum):sum;
      }
      x=out;
    }
    const probability=1/(1+Math.exp(-Math.max(-40,Math.min(40,x[0]))));
    if(!Number.isFinite(probability)) throw new Error('Invalid critic win estimate');
    return probability;
  }
  winProbability(decision) {
    const family=Encoder.FAMILIES[decision.family];
    if(family===undefined) throw new Error('Invalid critic decision family');
    const input=[...this.encoder.state(decision.observation),...Array(8).fill(0)];
    input[input.length-8+family]=1;
    return this.predict(input);
  }
}
