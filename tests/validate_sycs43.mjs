/** Load both editable skeletons with official Spine 4.3, sample at 60Hz. */
import {readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const {SkeletonJson,AtlasAttachmentLoader,TextureAtlasRegion,Skeleton,AnimationState,AnimationStateData,Physics}=await import(process.env.SPINE_CORE_MODULE||'@esotericsoftware/spine-core');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../examples/sycs-spine');
const regions=new Map(),loader=new AtlasAttachmentLoader({findRegion(name){
 if(!regions.has(name)){
  const png=readFileSync(path.join(root,'images',name+'.png')),w=png.readUInt32BE(16),h=png.readUInt32BE(20),r=new TextureAtlasRegion({regions:[]});
  Object.assign(r,{name,width:w,height:h,originalWidth:w,originalHeight:h,offsetX:0,offsetY:0,u:0,v:0,u2:1,v2:1,degrees:0});regions.set(name,r);
 }
 return regions.get(name);
}});
function validate(filename,reportFile,names,bonesCount,slotsCount){
 const bytes=readFileSync(path.join(root,filename)),data=new SkeletonJson(loader).readSkeletonData(JSON.parse(bytes));
 assert.deepEqual(data.animations.map(a=>a.name).sort(),names.sort());assert.equal(data.bones.length,bonesCount);assert.equal(data.slots.length,slotsCount);
 const report={runtime:'@esotericsoftware/spine-core 4.3.13',targetEditor:'4.3.23',desktopEditorTested:false,skeletonSHA256:createHash('sha256').update(bytes).digest('hex'),sampleHz:60,
 meshSampling:'Every vertex checked finite; every eighth vertex and last stored for independent comparison.',animations:{},samples:[]};
 for(const animation of data.animations){
  const skeleton=new Skeleton(data),state=new AnimationState(new AnimationStateData(data)),entry=state.setAnimation(0,animation.name,false),frames=Math.round(animation.duration*60);
  for(let i=0;i<=frames;i++){
   const time=i===frames?animation.duration:Math.fround(i/60);skeleton.setupPose();entry.trackTime=time;state.apply(skeleton);skeleton.updateWorldTransform(Physics.none);
   const bones={},meshes={},attachments={},alpha={};
   for(const b of skeleton.bones){const p=b.appliedPose,v=[p.a,p.b,p.c,p.d,p.worldX,p.worldY];assert.ok(v.every(Number.isFinite));bones[b.data.name]=v;}
   for(const s of skeleton.slots){
    alpha[s.data.name]=s.appliedPose.color.a;const a=s.appliedPose.attachment;attachments[s.data.name]=a?.name??null;if(!a)continue;
    const points=[];if(a.worldVerticesLength!==undefined)a.computeWorldVertices(skeleton,s,0,a.worldVerticesLength,points,0,2);else a.computeWorldVertices(s,a.getOffsets(s.appliedPose),points,0,2);assert.ok(points.every(Number.isFinite));
    const count=points.length/2,indices=[];for(let n=0;n<count;n+=8)indices.push(n);if(indices.at(-1)!==count-1)indices.push(count-1);
    meshes[s.data.name]={count,indices,vertices:indices.flatMap(n=>[points[n*2],points[n*2+1]])};
   }
   report.samples.push({animation:animation.name,time,bones,meshes,attachments,alpha,drawOrder:skeleton.drawOrder.appliedPose.map(s=>s.data.name)});
  }
  report.animations[animation.name]={duration:animation.duration,poses:frames+1};console.log(`PASS ${filename}: ${animation.name}, ${frames+1} poses`);
 }
 writeFileSync(path.join(root,reportFile),JSON.stringify(report,null,2)+'\n');return report.samples.length;
}
const actor=validate('sycs-4.3.json','runtime-validation.json',['stand','dead','hit','move','attack'],42,28);
console.log(`PASS: ${actor} SYCS poses at 60Hz; desktop editor not tested.`);
