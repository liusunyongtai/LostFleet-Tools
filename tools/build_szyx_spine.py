"""Reproducible six-action SZYX PRO rig. Source PNGs remain unmodified."""
from __future__ import annotations
import argparse,json,math,sys,hashlib
from copy import deepcopy
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageSequence
from tools.build_xqdz_spine import rotation,rounded,interpolate_controls,pose,load_images,save_gif
from tools.build_btmg_spine import grid,nearest_arc,solve_paws
from tools.build_bsq_spine import render,mesh_world,smoothstep,write_json
from tools.spine_mesh_topology import perimeter_edges,editor_mesh_check
BASE=Path(__file__).resolve().parents[1]/'examples/szyx-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.6,'move':1.2,'attack':1.2,'attack_2':1.2}
curve=interpolate_controls

def weights(n,point,c):
 if n=='shenti':
  w=smoothstep(600,860,point[0]);return {'hips':w,'chest':1-w}
 if n in c['chains']:
  d=c['chains'][n];pts=d['points'];arc=nearest_arc(point,pts);lengths=[math.dist(a,b) for a,b in zip(pts,pts[1:])];out={};offset=0
  # Blend at joints, while preserving the terminal curvature and attached base.
  for i,l in enumerate(lengths):
   left=1 if i==0 else smoothstep(offset-24,offset+24,arc)
   right=1 if i==len(lengths)-1 else 1-smoothstep(offset+l-24,offset+l+24,arc)
   out[d['bones'][i]]=max(0,min(left,right));offset+=l
  return out
 limb=next(((k,d) for k,d in c['limbs'].items() if d['image']==n),None)
 if limb:
  k,d=limb;pts=[d[x] for x in ('hip','knee','ankle','toe')];arc=nearest_arc(point,pts);a=math.dist(pts[0],pts[1]);b=math.dist(pts[1],pts[2]);u=1-smoothstep(a-30,a+30,arc);p=smoothstep(a+b-22,a+b+22,arc)
  return {k+'_upper':u,k+'_lower':max(0,1-u-p),k+'_paw':p}
 return {c['bindings'][n]:1}

def build_rig(source,c):
 defs={b['name']:b for b in c['bones']};indices={n:i for i,n in enumerate(defs)};origin=np.array(c['origin'])
 a={'skeleton':{'spine':'4.3.23','images':'./images/','fps':30,'x':-650,'y':0,'width':1300,'height':990},'bones':[],'slots':[],'skins':[],'animations':{},'events':{'bite_hit':{},'claw_hit':{},'impact':{},'footstep':{},'death_end':{}}}
 for b in c['bones']:
  p=defs.get(b.get('parent'));pos=np.array(b['position']);local=rotation(-p['rotation'])@(pos-p['position']) if p else pos-origin
  e={'name':b['name'],'x':rounded(local[0]),'y':rounded(local[1]),'rotation':rounded(b['rotation']-(p['rotation'] if p else 0)),'length':rounded(b['length'])}
  if p:e['parent']=p['name']
  a['bones'].append(e)
 skin={}
 for slot in source['slots']:
  n=slot['name'];reg=source['skins']['default'][n][n];a['slots'].append({'name':n,'bone':c['bindings'][n],'attachment':n})
  cols,rows=(18,16) if n=='shenti' else ((12,18) if n in c['chains'] or n in [d['image'] for d in c['limbs'].values()] else (1,1))
  pts,tris,hull=grid(cols,rows);uv=[];verts=[]
  for col,row in pts:
   u,v=col/cols,row/rows;uv.extend([u,v]);p=np.array([reg['x']+(u-.5)*reg['width'],reg['y']+(.5-v)*reg['height']]);ww={b:w for b,w in weights(n,p,c).items() if w>1e-8};total=sum(ww.values());verts.append(len(ww))
   for b,w in ww.items():
    q=rotation(-defs[b]['rotation'])@(p-defs[b]['position']);verts.extend([indices[b],rounded(q[0]),rounded(q[1]),w/total])
  mesh={'type':'mesh','path':n,'width':reg['width'],'height':reg['height'],'uvs':uv,'triangles':tris,'vertices':verts,'hull':hull,'edges':perimeter_edges(hull)};assert not editor_mesh_check(mesh),(n,editor_mesh_check(mesh));skin[n]={n:mesh}
 a['skins']=[{'name':'default','attachments':skin}];return a

def controls(name,t,c):
 d=DURATIONS[name];ph=t/d*math.tau;names=[b['name'] for b in c['bones']];r={n:0. for n in names};p={n:[0.,0.] for n in names};s={n:[1.,1.] for n in names};targets={k:list(v['ankle']) for k,v in c['limbs'].items()};pitches={k:0 for k in targets}
 wind=reach=fall=0
 if name in ('stand','move'):
  breath=(1-math.cos(ph))/2;p['hips'][1]=-6*breath;p['chest'][1]=9*breath;r['chest']=1.5*math.sin(ph);r['head']=-3.5*math.sin(ph);r['jaw']=3*breath
  if name=='move':
   p['hips'][1]=-48-35*math.sin(ph)**2-20*(math.sqrt(math.sin(ph)**2+.01**2)-.01);p['chest'][1]=8*math.sin(ph*2);r['chest']=2*math.sin(ph);r['head']=-4*math.sin(ph*2)
   for k,l in c['limbs'].items():
    q=ph+l['phase'];step=math.sin(q);lift=max(0,step)**2;targets[k]=[l['ankle'][0]+52*math.cos(q),l['ankle'][1]+65*lift];pitches[k]=-10*lift
 elif name in ('attack','attack_2'):
  wind=curve(t,[(0,0),(.3,1),(.4,.8),(.5,0),(d,0)]);reach=curve(t,[(0,0),(.3,0),(.5,1),(.6,.9),(.9,-.09),(d,0)]);lag=curve(t,[(0,0),(.45,0),(.65,1),(.9,-.25),(d,0)])
  p['hips']=[35*wind-90*reach,-32*wind-10*reach];p['chest']=[15*wind-32*reach,-5*wind+8*reach];r['chest']=-3*wind+4*reach;r['head']=-9*wind+8*reach-5*lag
  if name=='attack':
   gape=curve(t,[(0,0),(.25,1),(.43,1),(.53,-.35),(.65,.15),(.9,0),(d,0)]);r['jaw']=25*gape;p['head']=[12*wind-40*reach,5*reach]
   for k,l in c['limbs'].items():targets[k]=[l['ankle'][0]-(105 if 'front' in k else 65)*reach,l['ankle'][1]+18*max(0,reach)];pitches[k]=-5*max(0,reach)
  else:
   r['jaw']=6*reach;r['head']=-15*reach-6*wind;p['head']=[70*reach,50*reach];r['chest']+=3*wind-5*reach;p['hips'][1]-=65*max(0,reach)+30*abs(reach)
   # The other three paws support the body's weight; the near front claw rakes left.
   lift=curve(t,[(0,0),(.3,1),(.42,1),(.56,.7),(.8,.15),(1.,0),(d,0)])
   targets['near_front']=[c['limbs']['near_front']['ankle'][0]+115*wind-350*reach,c['limbs']['near_front']['ankle'][1]+130*lift];pitches['near_front']=22*wind-42*reach
   for k,l in c['limbs'].items():
    if k!='near_front':targets[k]=[l['ankle'][0]-35*reach,l['ankle'][1]]
 elif name=='hit':
  shock=curve(t,[(0,0),(.1,1),(.18,.85),(.33,-.18),(d,0)]);lag=curve(t,[(0,0),(.16,1),(.36,-.2),(d,0)]);p['hips']=[35*shock,-22*shock-55*abs(shock)];p['chest']=[15*shock,-8*shock];r['chest']=-6*shock;r['head']=-12*lag;r['jaw']=14*lag
 elif name=='dead':
  wind=curve(t,[(0,0),(.2,1),(.5,0),(d,0)]);fall=curve(t,[(0,0),(.25,0),(.9,1),(1.08,.96),(1.4,1),(d,1)]);bounce=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.12),(1.4,0),(d,0)])
  p['hips']=[35*fall,-240*fall-30*wind];p['chest']=[-15*fall,-20*fall];r['hips']=-5*fall;r['chest']=9*fall;r['head']=-12*wind+4*bounce;p['head'][1]=35*fall;r['jaw']=8*fall+12*wind
  for k in c['limbs']:
   targets[k]=[c['limbs'][k]['ankle'][0]+{'far_front':-95,'near_front':-150,'far_hind':110,'near_hind':130}[k]*fall,c['limbs'][k]['ankle'][1]+100*fall];pitches[k]=(-8 if 'front' in k else 8)*fall
 for j,(img,chain) in enumerate(c['chains'].items()):
  for i,b in enumerate(chain['bones']):
   amp=(4 if img=='weiba' else 8)+i*(1 if img=='weiba' else 2)
   if name in ('stand','move'):
    rate=1 if name=='stand' else 2;phase=ph*rate-j*.65-i*.8;r[b]=amp*(math.sin(phase)-math.sin(-j*.65-i*.8))
   elif name=='dead':r[b]=((-5 if img=='weiba' else 7)+i*2)*fall
   else:
    lag=curve(t,[(0,0),(.2,0),(.42,1),(.7,-.6),(d,0)]) if name!='hit' else curve(t,[(0,0),(.18,1),(.38,-.4),(d,0)])
    r[b]=(4+i*2)*lag*(1 if j%2 else -1)
 return r,p,s,targets,pitches

def make_animations(a,c):
 result={};defs={b['name']:b for b in c['bones']}
 for name,d in DURATIONS.items():
  tracks={n:{'rotate':[],'translate':[],'scale':[]} for n in defs}
  for frame in range(round(d*30)+1):
   t=frame/30;r,p,s,targets,pitches=controls(name,t,c)
   solve_paws(a,c,r,p,s,targets,pitches)
   for n in defs:
    par=defs.get(defs[n].get('parent'));offset=rotation(-par['rotation'])@p[n] if par else p[n];prev=tracks[n]['rotate']
    value=r[n]
    if prev:value=prev[-1]['value']+(value-prev[-1]['value']+180)%360-180
    prev.append({'time':t,'value':rounded(value)});tracks[n]['translate'].append({'time':t,'x':rounded(offset[0]),'y':rounded(offset[1])});tracks[n]['scale'].append({'time':t,'x':s[n][0],'y':s[n][1]})
  for n in list(tracks):
   for kind in list(tracks[n]):
    keys=['value'] if kind=='rotate' else ['x','y'];neutral=1 if kind=='scale' else 0
    if n!='root' and all(all(abs(k[x]-neutral)<1e-7 for x in keys) for k in tracks[n][kind]):del tracks[n][kind]
   if not tracks[n]:del tracks[n]
  result[name]={'bones':tracks}
  events={'attack':[(.53,'bite_hit')],'attack_2':[(.5,'claw_hit')],'hit':[(.1,'impact')],'dead':[(1.4,'death_end')],'move':[(.6,'footstep'),(1.2,'footstep')]}
  if name in events:result[name]['events']=[{'time':t,'name':e} for t,e in events[name]]
 return result

def scaled_preview(a,f):
 b=deepcopy(a);b['bones'][0]['scaleX']=f;b['bones'][0]['scaleY']=f
 for an in b['animations'].values():
  for k in an['bones']['root']['translate']:k['x']*=f;k['y']*=f
 return b

def correct_ground(a,images):
 tr=a['animations']['dead']['bones']['root']['translate'];cam=(1500,2200);size=(3000,2600)
 im=render(a,images,'dead',1.6,size=size,camera=cam);correction=im.getbbox()[3]-cam[1]
 for k in tr:k['y']=rounded(correction*curve(k['time'],[(0,0),(.25,0),(.9,1),(1.6,1)]))
 for k in tr:
  im=render(a,images,'dead',k['time'],size=size,camera=cam);k['y']=rounded(k['y']+max(0,im.getbbox()[3]-cam[1]))

def make_previews(a,images,only=None):
 out=BASE/'previews';out.mkdir(exist_ok=True);size=(860,640);camera=(475,565);factor=.49;rig=scaled_preview(a,factor)
 bg=Image.new('RGBA',size,'#233039');dr=ImageDraw.Draw(bg)
 for y in range(0,size[1],20):
  for x in range(0,size[0],20):
   if (x//20+y//20)%2:dr.rectangle((x,y,x+19,y+19),fill='#293842')
 dr.line((0,camera[1],860,camera[1]),fill='#759b8a');sheet=Image.new('RGB',(1720,len(DURATIONS)*340),'#151e26');labels=ImageDraw.Draw(sheet)
 for row,(n,d) in enumerate(DURATIONS.items()):
  if only and n not in only:
   with Image.open(out/(n+'.gif')) as cached:frames=[im.convert('RGB').copy() for im in ImageSequence.Iterator(cached)]
  else:
   frames=[];mobile=[]
   for f in range(round(d*30)+1):
    image=render(rig,images,n,f/30,size=size,camera=camera,background=bg).convert('RGB');ImageDraw.Draw(image).text((15,15),f'{n}   {f/30:.2f}s',fill='#a9e3d5');frames.append(image)
    phone=Image.new('RGB',(420,250),'#19252e');phone.paste(image.resize((236,176),Image.Resampling.LANCZOS),(92,42));ImageDraw.Draw(phone).text((10,12),f'{n} / character ~128px',fill='#a9e3d5');mobile.append(phone)
   save_gif(frames,out/(n+'.gif'),n in ('stand','move','attack','attack_2'));save_gif(mobile,out/('mobile-'+n+'.gif'),True)
  times=[0,.3,.5,.73] if n.startswith('attack') else [0,d*.25,d*.65,d]
  for col,t in enumerate(times):sheet.paste(frames[min(round(t*30),len(frames)-1)].resize((430,320)),(col*430,row*340+20))
  labels.text((12,row*340+4),n,fill='#a9e3d5')
 sheet.save(out/'animation-contact.png');render(rig,images,size=size,camera=camera,background=bg,rig=True).convert('RGB').save(out/'rig-setup.png')
 cards=''.join(f'<section><h2>{n} · {d}s</h2><img src="previews/{n}.gif"><img class="phone" src="previews/mobile-{n}.gif"></section>' for n,d in DURATIONS.items())
 (BASE/'preview.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>SZYX 动画预览</title><style>body{background:#101820;color:#d6e6e8;font:16px system-ui;margin:24px}main{max-width:1300px;margin:auto}section{background:#1a2832;padding:18px;margin:20px 0;border:1px solid #38505a;border-radius:12px}img{max-width:70%;height:auto}.phone{max-width:29%}a{color:#8fe5c0}.wide{max-width:100%}</style><main><h1>SZYX · 四足异形</h1><p>加权四足、触手延迟、蓄力与回弹。移动原地循环；攻击在游戏中单次播放，GIF 重复便于查看。</p><p>独立软件渲染预览；不是 Spine 桌面截图。导入说明见 README.md。</p>'+cards+'<section><h2>关键帧与骨架</h2><img class="wide" src="previews/animation-contact.png"><img src="previews/rig-setup.png"></section></main>',encoding='utf8')

def build(previews=True):
 c=json.loads((BASE/'rig-config.json').read_text());source=json.loads((BASE/'source/photoshop.json').read_text());images=load_images(BASE/'images');a=build_rig(source,c);a['animations']=make_animations(a,c);correct_ground(a,images);write_json(BASE/'szyx-4.3.json',a)
 report={'skeletonSHA256':hashlib.sha256((BASE/'szyx-4.3.json').read_bytes()).hexdigest(),'desktopEditorTested':False,'meshes':[{'name':n,'vertices':len(m['uvs'])//2,'errors':editor_mesh_check(m)} for n,entries in a['skins'][0]['attachments'].items() for m in entries.values()]};write_json(BASE/'editor-topology-validation.json',report)
 if previews:make_previews(a,images)
 print('Built SZYX:',len(a['bones']),'bones,',len(a['slots']),'meshes;',list(a['animations']))
 return a
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--no-previews',action='store_true');build(not p.parse_args().no_previews)
