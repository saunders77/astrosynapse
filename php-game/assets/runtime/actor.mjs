import { Encoder } from './encoder.mjs';
// ASTROJS1 + uint32 header length + padded JSON header + little-endian float32.
// Weights stay float32; accumulations use JS doubles, as the former PHP actor did.
export class Actor {
  constructor(buffer) {
    const bytes=new Uint8Array(buffer), view=new DataView(buffer), decoder=new TextDecoder();
    if(buffer.byteLength<12||decoder.decode(bytes.subarray(0,8))!=='ASTROJS1') throw new Error('Invalid model format');
    const length=view.getUint32(8,true);
    if(length>1000000||length%4||12+length>buffer.byteLength) throw new Error('Invalid model header');
    const header=JSON.parse(decoder.decode(bytes.subarray(12,12+length)));
    this.spec={encoder_version:1,objective_version:1,layer_norm_eps:1e-5,...header.spec}; this.tensors=Object.create(null);
    let offset=12+length;
    for(const [name,meta] of Object.entries(header.tensors)) {
      const size=meta.shape.reduce((a,b)=>a*b,1);
      if(!Number.isSafeInteger(size)||size<=0||meta.bytes!==size*4||offset+meta.bytes>buffer.byteLength) throw new Error('Invalid tensor '+name);
      const data=new Float32Array(buffer,offset,size);
      if(data.some(v=>!Number.isFinite(v))) throw new Error('Non-finite model weights');
      this.tensors[name]={shape:meta.shape,data}; offset+=meta.bytes;
    }
    if(offset!==buffer.byteLength) throw new Error('Unexpected model data');
    if(![1,2].includes(this.spec.encoder_version)||![1,2].includes(this.spec.objective_version)||this.spec.state_size!==1292||this.spec.action_size!==(this.spec.encoder_version===2?219:203)) throw new Error('Unsupported model architecture');
    const s=this.spec;
    for(const key of ['hidden_size','action_hidden_size','residual_blocks','bootstrap_heads','families']) if(!Number.isInteger(s[key])||s[key]<1||s[key]>2048) throw new Error('Invalid model specification');
    if(s.families!==8||s.residual_blocks>16||s.bootstrap_heads>32||!Number.isFinite(s.layer_norm_eps)||s.layer_norm_eps<=0) throw new Error('Unsupported model specification');
    const check=(name,shape)=> { const t=this.tensors[name];if(!t||JSON.stringify(t.shape)!==JSON.stringify(shape))throw new Error('Missing or invalid tensor: '+name); };
    const linear=(name,input,output)=> {check(name+'.weight',[output,input]);check(name+'.bias',[output]);};
    const norm=(name,width)=> {check(name+'.weight',[width]);check(name+'.bias',[width]);};
    const residual=(name,width)=> {norm(name+'.norm',width);linear(name+'.fc1',width,width*2);linear(name+'.fc2',width*2,width);};
    const h=s.hidden_size,ah=s.action_hidden_size;
    linear('state_in',s.state_size,h);norm('state_norm',h);linear('action_in',s.action_size,ah);norm('action_norm',ah);residual('action_blocks.0',ah);
    linear('fusion_in',h+ah,h);norm('fusion_norm',h);
    for(let i=0;i<s.residual_blocks;i++){residual('state_blocks.'+i,h);residual('fusion_blocks.'+i,h);}
    if(s.objective_version>=2) {for(let i=0;i<s.bootstrap_heads;i++){residual('head_blocks.'+i,h);linear('head_outputs.'+i,h,s.families);}linear('value_output',h,s.families*s.bootstrap_heads);}
    else linear('output',h,s.families*s.bootstrap_heads);
    this.encoder=new Encoder(this.spec.encoder_version);
  }
  linear(x,prefix,row=null) {
    const w=this.tensors[prefix+'.weight'],b=this.tensors[prefix+'.bias'].data,cols=w.shape[1],weights=w.data;
    const start=row??0,end=row===null?w.shape[0]:row+1,out=new Float64Array(end-start);
    if(x.length!==cols) throw new Error('Model input size mismatch: '+prefix);
    if(x.length>300||prefix==='action_in') {
      const indices=[]; for(let j=0;j<x.length;j++) if(x[j]!==0) indices.push(j);
      for(let i=start;i<end;i++) { let sum=b[i],offset=i*cols; for(const j of indices) sum+=x[j]*weights[offset+j]; out[i-start]=sum; }
    } else for(let i=start;i<end;i++) { let sum=b[i],offset=i*cols; for(let j=0;j<cols;j++) sum+=x[j]*weights[offset+j]; out[i-start]=sum; }
    return out;
  }
  norm(x,prefix) {
    const n=x.length,mean=x.reduce((a,b)=>a+b,0)/n; let variance=0;
    for(const v of x) variance+=(v-mean)**2;
    const scale=1/Math.sqrt(variance/n+this.spec.layer_norm_eps),w=this.tensors[prefix+'.weight'].data,b=this.tensors[prefix+'.bias'].data;
    return x.map((v,j)=>(v-mean)*scale*w[j]+b[j]);
  }
  static silu(x) { return x.map(v=>v/(1+Math.exp(-Math.max(-40,Math.min(40,v))))); }
  residual(x,prefix) { const y=this.linear(Actor.silu(this.linear(this.norm(x,prefix+'.norm'),prefix+'.fc1')),prefix+'.fc2'); return x.map((v,j)=>(v+y[j])*0.7071067811865475); }
  stateFeatures(state) {
    let s=Actor.silu(this.norm(this.linear(state,'state_in'),'state_norm'));
    for(let i=0;i<this.spec.residual_blocks;i++) s=this.residual(s,'state_blocks.'+i);
    return s;
  }
  scores(state,actions,family,probabilities=false) {
    const s=this.stateFeatures(state);
    return actions.map(a=> {
      let v=this.residual(Actor.silu(this.norm(this.linear(a,'action_in'),'action_norm')),'action_blocks.0');
      v=Actor.silu(this.norm(this.linear([...s,...v],'fusion_in'),'fusion_norm'));
      for(let i=0;i<this.spec.residual_blocks;i++) v=this.residual(v,'fusion_blocks.'+i);
      let sum=0; for(let i=0;i<this.spec.bootstrap_heads;i++) {
        const logit=this.spec.objective_version>=2?this.linear(this.residual(v,'head_blocks.'+i),'head_outputs.'+i,family)[0]:this.linear(v,'output',family*this.spec.bootstrap_heads+i)[0];
        sum+=probabilities?Actor.sigmoid(logit):logit;
      }
      return sum/this.spec.bootstrap_heads;
    });
  }
  static sigmoid(logit) { return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, logit)))); }
  winProbability(d) {
    const e=this.encoder, state=e.state(d.observation), family=Encoder.FAMILIES[d.family];
    let probability;
    if(this.spec.objective_version>=2) {
      // Policy logits are not win probabilities. Use the trained state-value heads.
      const values=this.linear(this.stateFeatures(state),'value_output');
      const start=family*this.spec.bootstrap_heads;
      probability=values.slice(start,start+this.spec.bootstrap_heads).reduce((sum,v)=>sum+Actor.sigmoid(v),0)/this.spec.bootstrap_heads;
    } else {
      // Earlier models predict outcomes for actions; evaluate their preferred move.
      const actions=d.actions.map(a=>e.action(a,d.observation)), scores=this.scores(state,actions,family);
      const best=scores.indexOf(Math.max(...scores));
      probability=this.scores(state,[actions[best]],family,true)[0];
    }
    if(!Number.isFinite(probability)) throw new Error('Invalid model win estimate');
    return probability;
  }
  choose(d,eligible=d.actions.map((_,i)=>i)) {
    if(eligible.length===1) return eligible[0];
    const e=this.encoder,scores=this.scores(e.state(d.observation),eligible.map(i=>e.action(d.actions[i],d.observation)),Encoder.FAMILIES[d.family]);
    if(scores.some(v=>!Number.isFinite(v))) throw new Error('Invalid model output');
    return eligible[scores.indexOf(Math.max(...scores))];
  }
}
