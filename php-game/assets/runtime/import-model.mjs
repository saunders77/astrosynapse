// Data-only readers for the previous PHP export and NumPy's ZIP/NPY format.
// No Python, PHP, eval, filesystem extraction, or third-party runtime is used.
import { decompress, readLimited } from './resources.mjs';
const decoder=new TextDecoder(),encoder=new TextEncoder();
function pack(spec,tensors) {
  const meta={}; for(const [name,t] of Object.entries(tensors)) meta[name]={shape:t.shape,bytes:t.data.length};
  let header=encoder.encode(JSON.stringify({format:1,spec,tensors:meta}));
  const size=header.length+(4-header.length%4)%4,total=12+size+Object.values(tensors).reduce((s,t)=>s+t.data.length,0),out=new Uint8Array(total);
  out.set(encoder.encode('ASTROJS1'));new DataView(out.buffer).setUint32(8,size,true);out.fill(32,12,12+size);out.set(header,12);
  let offset=12+size;for(const t of Object.values(tensors)){out.set(t.data,offset);offset+=t.data.length;}return out.buffer;
}
function crc32(bytes) { let crc=-1;for(const b of bytes){crc^=b;for(let i=0;i<8;i++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}return (crc^-1)>>>0; }
async function npz(buffer) {
  const v=new DataView(buffer),bytes=new Uint8Array(buffer),tensors={};let end=-1,spec,total=0;
  for(let i=bytes.length-22;i>=Math.max(0,bytes.length-65557);i--)if(v.getUint32(i,true)===0x06054b50){end=i;break;}
  if(end<0||v.getUint16(end+4,true)||v.getUint16(end+6,true))throw new Error('Invalid NPZ directory');
  const count=v.getUint16(end+10,true);let offset=v.getUint32(end+16,true);
  if(count>2048)throw new Error('Too many NPZ tensors');
  for(let i=0;i<count;i++) {
    if(offset+46>bytes.length||v.getUint32(offset,true)!==0x02014b50)throw new Error('Invalid NPZ entry');
    const flags=v.getUint16(offset+8,true),method=v.getUint16(offset+10,true),crc=v.getUint32(offset+16,true),compressed=v.getUint32(offset+20,true),expanded=v.getUint32(offset+24,true),nameLength=v.getUint16(offset+28,true),extraLength=v.getUint16(offset+30,true),commentLength=v.getUint16(offset+32,true),local=v.getUint32(offset+42,true);
    const name=decoder.decode(bytes.subarray(offset+46,offset+46+nameLength));offset+=46+nameLength+extraLength+commentLength;
    total+=expanded;if(flags&1||total>128*1024*1024||![0,8].includes(method)||!name.endsWith('.npy'))throw new Error('Unsupported NPZ entry');
    if(local+30>bytes.length||v.getUint32(local,true)!==0x04034b50)throw new Error('Invalid NPZ data');
    const start=local+30+v.getUint16(local+26,true)+v.getUint16(local+28,true);
    if(start+compressed>bytes.length)throw new Error('Truncated NPZ data');
    let data=bytes.slice(start,start+compressed);
    if(method===8)data=new Uint8Array(await readLimited(new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw')),expanded));
    if(data.length!==expanded||crc32(data)!==crc)throw new Error('NPZ checksum mismatch');
    if(decoder.decode(data.subarray(0,6))!=='\x93NUMPY'&&!(data[0]===147&&decoder.decode(data.subarray(1,6))==='NUMPY'))throw new Error('Invalid NPY tensor');
    const dv=new DataView(data.buffer,data.byteOffset,data.byteLength),version=data[6],hstart=version===1?10:12;
    if(![1,2,3].includes(version))throw new Error('Unsupported NPY version');
    const hsize=version===1?dv.getUint16(8,true):dv.getUint32(8,true),header=decoder.decode(data.subarray(hstart,hstart+hsize));
    const dtype=header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/)?.[1],shapeText=header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/)?.[1];
    if(!/['"]fortran_order['"]\s*:\s*False/.test(header)||shapeText===undefined)throw new Error('Unsupported NPY layout');
    const shape=shapeText.split(',').map(x=>x.trim()).filter(Boolean).map(Number),body=data.slice(hstart+hsize),key=name.slice(0,-4);
    if(shape.some(x=>!Number.isSafeInteger(x)||x<=0)||Object.hasOwn(tensors,key))throw new Error('Invalid NPY shape/name');
    const size=shape.reduce((a,b)=>a*b,1);
    if(key==='__spec_json__') {if(dtype!=='|u1'||body.length!==size)throw new Error('Invalid model spec');spec=JSON.parse(decoder.decode(body));}
    else {if(dtype!=='<f4'||body.length!==size*4)throw new Error('Models must contain little-endian float32 weights');tensors[key]={shape,data:body};}
  }
  if(!spec)throw new Error('NPZ model specification is missing');return pack(spec,tensors);
}
export async function importModel(buffer) {
  const bytes=await decompress(buffer),b=new Uint8Array(bytes);
  if(b[0]===80&&b[1]===75)return npz(bytes);
  const prefix='<?php http_response_code(404); exit; __halt_compiler();\n';
  if(decoder.decode(b.subarray(0,5))==='<?php') {
    if(decoder.decode(b.subarray(0,prefix.length))!==prefix||b.length<prefix.length+4)throw new Error('Unsupported PHP model header');
    const size=new DataView(bytes).getUint32(prefix.length,true);
    if(size>1000000||prefix.length+4+size>b.length)throw new Error('Invalid PHP model header');
    const header=b.slice(prefix.length+4,prefix.length+4+size),pad=(4-size%4)%4,out=new Uint8Array(12+size+pad+b.length-prefix.length-4-size);
    out.set(encoder.encode('ASTROJS1'));new DataView(out.buffer).setUint32(8,size+pad,true);out.set(header,12);out.fill(32,12+size,12+size+pad);out.set(b.subarray(prefix.length+4+size),12+size+pad);return out.buffer;
  }
  return bytes;
}
