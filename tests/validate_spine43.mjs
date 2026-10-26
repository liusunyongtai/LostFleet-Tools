/** Optional validation against the official runtime, not the desktop Spine editor. */
import {readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const spine=await import(process.env.SPINE_CORE_MODULE || '@esotericsoftware/spine-core');
const {SkeletonJson,AtlasAttachmentLoader,TextureAtlasRegion,Skeleton,AnimationState,AnimationStateData,Physics}=spine;
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../examples/xqdz-spine');
const json=JSON.parse(readFileSync(path.join(root,'xqdz-4.3.json'),'utf8'));
const regions=new Map();
const loader=new AtlasAttachmentLoader({findRegion(name){
  if(!regions.has(name)){
    const png=readFileSync(path.join(root,'images',name+'.png'));
    const width=png.readUInt32BE(16),height=png.readUInt32BE(20);
    const r=new TextureAtlasRegion({regions:[]});
    Object.assign(r,{name,width,height,originalWidth:width,originalHeight:height,offsetX:0,offsetY:0,u:0,v:0,u2:1,v2:1,degrees:0});regions.set(name,r);
  }
  return regions.get(name);
}});
const data=new SkeletonJson(loader).readSkeletonData(json);
const results={runtime:'@esotericsoftware/spine-core 4.3.13',targetEditor:'4.3.23',desktopEditorTested:false,
  notes:'Runtime parse, timeline application, finite world transforms and attachment references. No desktop/editor import claim.',bones:data.bones.length,slots:data.slots.length,animations:{},samples:[]};
results.skeletonSHA256=createHash('sha256').update(readFileSync(path.join(root,'xqdz-4.3.json'))).digest('hex');
assert.equal(data.bones.length,13);assert.equal(data.slots.length,10);assert.equal(data.animations.length,5);
for(const animation of data.animations){
  const count=Math.round(animation.duration*30);
  const skeleton=new Skeleton(data);
  const state=new AnimationState(new AnimationStateData(data));
  const entry=state.setAnimation(0,animation.name,false);
  let shown=0;
  for(let frame=0;frame<=count;frame++){
    const time=frame===count?animation.duration:frame/30;
    skeleton.setupPose();entry.trackTime=time;state.apply(skeleton);skeleton.updateWorldTransform(Physics.none);
    const bones={};
    for(const bone of skeleton.bones){
      const p=bone.appliedPose;
      const values=[p.a,p.b,p.c,p.d,p.worldX,p.worldY];assert.ok(values.every(Number.isFinite));
      bones[bone.data.name]=values;
    }
    const flash=skeleton.findSlot('muzzle_flash').appliedPose.attachment;
    if(flash)shown++;
    if(animation.name!=='attack')assert.equal(flash,null);
    results.samples.push({animation:animation.name,time,bones,flash:flash?.name||null});
  }
  if(animation.name==='attack')assert.equal(shown,3);
  results.animations[animation.name]={duration:animation.duration,frames:count+1,flashFrames:shown};
  console.log(`PASS runtime: ${animation.name}, ${count+1} poses, ${shown} flash frames`);
}
const output=process.env.SPINE_REPORT || path.join(root,'runtime-validation.json');
writeFileSync(output,JSON.stringify(results,null,2)+'\n');
console.log('PASS official Spine 4.3 runtime validation; desktop 4.3.23 import remains untested');
