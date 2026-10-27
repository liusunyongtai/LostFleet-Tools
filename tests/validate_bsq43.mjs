/** Official Spine runtime loading and pose sampling; no desktop-editor claim. */
import {readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const {SkeletonJson,AtlasAttachmentLoader,TextureAtlasRegion,Skeleton,AnimationState,AnimationStateData,Physics}=await import(process.env.SPINE_CORE_MODULE||'@esotericsoftware/spine-core');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../examples/bsq-spine');
const report={runtime:'@esotericsoftware/spine-core 4.3.13',targetEditor:'4.3.23',desktopEditorTested:false,files:{},samples:[]};
for(const [kind,file,expected] of [['role','bsq-4.3.json',['stand','dead','hit','attack_1','attack_2','move']],['regions','bsq-regions-4.3.json',['stand','dead','hit','attack_1','attack_2','move']],['bullet','bullet-4.3.json',['bullet']]]){
  const bytes=readFileSync(path.join(root,file));const json=JSON.parse(bytes);
  const regions=new Map();
  const loader=new AtlasAttachmentLoader({findRegion(name){
    if(!regions.has(name)){const png=readFileSync(path.join(root,'images',name+'.png'));const width=png.readUInt32BE(16),height=png.readUInt32BE(20);
      const region=new TextureAtlasRegion({regions:[]});Object.assign(region,{name,width,height,originalWidth:width,originalHeight:height,offsetX:0,offsetY:0,u:0,v:0,u2:1,v2:1,degrees:0});regions.set(name,region);}
    return regions.get(name);
  }});
  const data=new SkeletonJson(loader).readSkeletonData(json);
  assert.deepEqual(data.animations.map(a=>a.name).sort(),[...expected].sort());
  assert.equal(data.bones.length,kind!=='bullet'?16:5);assert.equal(data.slots.length,kind!=='bullet'?7:4);
  const details={sha256:createHash('sha256').update(bytes).digest('hex'),bones:data.bones.length,slots:data.slots.length,animations:{}};
  for(const animation of data.animations){
    const skeleton=new Skeleton(data),state=new AnimationState(new AnimationStateData(data));
    const entry=state.setAnimation(0,animation.name,false);const count=Math.round(animation.duration*30);const spit=[];
    for(let frame=0;frame<=count;frame++){
      const time=frame===count?animation.duration:Math.fround(frame/30);skeleton.setupPose();entry.trackTime=time;state.apply(skeleton);skeleton.updateWorldTransform(Physics.none);
      const bones={};for(const bone of skeleton.bones){const p=bone.appliedPose;const v=[p.a,p.b,p.c,p.d,p.worldX,p.worldY];assert.ok(v.every(Number.isFinite));bones[bone.data.name]=v;}
      const visible={};for(const slot of skeleton.slots){visible[slot.data.name]=slot.appliedPose.attachment?.name||null;}
      if(kind!=='bullet'){
        if(visible.spit_fx)spit.push(frame);
        if(animation.name!=='attack_2')assert.equal(visible.spit_fx,null);
      }else{assert.ok(Math.abs(bones.root[4])<1e-6&&Math.abs(bones.root[5])<1e-6);}
      const drawOrder=skeleton.drawOrder.appliedPose.map(slot=>slot.data.name);
      let mesh=null;
      if(kind==='role'){
        const slot=skeleton.findSlot('shenti'),attachment=slot.appliedPose.attachment;
        const vertices=[];attachment.computeWorldVertices(skeleton,slot,0,attachment.worldVerticesLength,vertices,0,2);
        assert.ok(vertices.every(Number.isFinite));
        mesh=vertices;
      }
      report.samples.push({kind,animation:animation.name,time,bones,visible,drawOrder,mesh});
    }
    if(animation.name==='attack_2')assert.deepEqual(spit,[12,13,14,15]);
    details.animations[animation.name]={duration:animation.duration,frames:count+1,spitFrames:spit};
    console.log(`PASS official runtime: ${kind}/${animation.name}, ${count+1} poses`);
  }
  report.files[file]=details;
}
writeFileSync(path.join(root,'runtime-validation.json'),JSON.stringify(report,null,2)+'\n');
console.log(`PASS: ${report.samples.length} official runtime pose samples; desktop Spine 4.3.23 not tested`);
