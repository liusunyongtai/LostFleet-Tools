/** Official Spine 4.3 load/60Hz pose and weighted-mesh validation. */
import {readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const {SkeletonJson,AtlasAttachmentLoader,TextureAtlasRegion,Skeleton,AnimationState,AnimationStateData,Physics}=await import(process.env.SPINE_CORE_MODULE||'@esotericsoftware/spine-core');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../examples/jbjs-spine');
const filename=path.join(root,'jbjs-4.3.json'),bytes=readFileSync(filename),json=JSON.parse(bytes);
const regions=new Map();const loader=new AtlasAttachmentLoader({findRegion(name){
  if(!regions.has(name)){const png=readFileSync(path.join(root,'images',name+'.png'));const w=png.readUInt32BE(16),h=png.readUInt32BE(20);
    const r=new TextureAtlasRegion({regions:[]});Object.assign(r,{name,width:w,height:h,originalWidth:w,originalHeight:h,offsetX:0,offsetY:0,u:0,v:0,u2:1,v2:1,degrees:0});regions.set(name,r);}
  return regions.get(name);
}});
const data=new SkeletonJson(loader).readSkeletonData(json);
assert.deepEqual(data.animations.map(a=>a.name).sort(),['stand','dead','hit','attack','move'].sort());
assert.equal(data.bones.length,15);assert.equal(data.slots.length,8);
const report={runtime:'@esotericsoftware/spine-core 4.3.13',targetEditor:'4.3.23',desktopEditorTested:false,
  skeletonSHA256:createHash('sha256').update(bytes).digest('hex'),sampleHz:60,
  meshSampling:'All vertices checked finite; every eighth vertex and last vertex stored for independent comparison.',animations:{},samples:[]};
for(const animation of data.animations){
  const skeleton=new Skeleton(data),state=new AnimationState(new AnimationStateData(data)),entry=state.setAnimation(0,animation.name,false);
  const frames=Math.round(animation.duration*60);
  for(let i=0;i<=frames;i++){
    const time=i===frames?animation.duration:Math.fround(i/60);skeleton.setupPose();entry.trackTime=time;state.apply(skeleton);skeleton.updateWorldTransform(Physics.none);
    const bones={};for(const bone of skeleton.bones){const p=bone.appliedPose;const values=[p.a,p.b,p.c,p.d,p.worldX,p.worldY];assert.ok(values.every(Number.isFinite));bones[bone.data.name]=values;}
    const meshes={},attachments={};
    for(const slot of skeleton.slots){const attachment=slot.appliedPose.attachment;attachments[slot.data.name]=attachment?.name??null;if(!attachment)continue;const points=[];
      attachment.computeWorldVertices(skeleton,slot,0,attachment.worldVerticesLength,points,0,2);assert.ok(points.every(Number.isFinite));
      const count=points.length/2,indices=[];for(let n=0;n<count;n+=8)indices.push(n);if(indices.at(-1)!==count-1)indices.push(count-1);
      meshes[slot.data.name]={count,indices,vertices:indices.flatMap(index=>[points[index*2],points[index*2+1]])};
    }
    report.samples.push({animation:animation.name,time,bones,meshes,attachments,drawOrder:skeleton.drawOrder.appliedPose.map(slot=>slot.data.name)});
  }
  report.animations[animation.name]={duration:animation.duration,poses:frames+1};
  console.log(`PASS official runtime: ${animation.name}, ${frames+1} poses including between-key samples`);
}
writeFileSync(path.join(root,'runtime-validation.json'),JSON.stringify(report,null,2)+'\n');
console.log(`PASS: ${report.samples.length} 60Hz poses, all mesh vertices finite; desktop editor not tested`);
