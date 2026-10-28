"""Fungal infected: expressive claw swipe, red slash, spitting and independent poison."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from tools.build_xqdz_spine import rotation,rounded,interpolate_controls,pose,load_images,save_gif
from tools.build_bsq_spine import mesh_world,render as shared_render,smoothstep,write_json

BASE=Path(__file__).resolve().parents[1]/'examples'/'zhshz-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'attack_1':1.2,'attack_2':1.4,'move':1.}
SIZE=(2000,1500)
CAMERA=(1150,1330)


def grid(columns,rows):
    boundary=([(x,0) for x in range(columns+1)]+[(columns,y) for y in range(1,rows+1)]
              +[(x,rows) for x in range(columns-1,-1,-1)]+[(0,y) for y in range(rows-1,0,-1)])
    points=boundary+[(x,y) for y in range(1,rows) for x in range(1,columns)]
    lookup={point:i for i,point in enumerate(points)};triangles=[]
    for y in range(rows):
        for x in range(columns):
            a,b,c,d=[lookup[p] for p in [(x,y),(x+1,y),(x+1,y+1),(x,y+1)]]
            triangles.extend([a,b,c,a,c,d])
    return points,triangles,len(boundary)


def nearest_arc(point,chain):
    best=(float('inf'),0);offset=0
    for a,b in zip(chain,chain[1:]):
        a,b=np.array(a,float),np.array(b,float);vector=b-a;length=np.linalg.norm(vector)
        u=float(np.clip(np.dot(point-a,vector)/(length*length),0,1))
        distance=np.linalg.norm(point-(a+u*vector))
        if distance<best[0]:best=(distance,offset+u*length)
        offset+=length
    return best[1]


def vertex_weights(name,point,config):
    if name=='qugan':
        pelvis=smoothstep(835,935,-point[1]);return {'body':1-pelvis,'pelvis':pelvis}
    if name=='tou':
        crown=.7*(1-smoothstep(200,470,-point[1]));return {'head':1-crown,'crown':crown}
    binding={'claw_fx':'slash_anchor','spit_fx':'mouth'}.get(name,config['bindings'].get(name))
    return {binding:1.}


def weighted_attachment(name,region,config):
    columns,rows={'qugan':(10,14),'tou':(12,14)}.get(name,(1,1))
    points,triangles,hull=grid(columns,rows);definitions={b['name']:b for b in config['bones']};indices={b['name']:i for i,b in enumerate(config['bones'])}
    uvs=[];vertices=[]
    for col,row in points:
        u,v=col/columns,row/rows;uvs.extend([u,v])
        point=np.array([region['x']+(u-.5)*region['width'],region['y']+(.5-v)*region['height']])
        weights={bone:weight for bone,weight in vertex_weights(name,point,config).items() if weight>1e-8};total=sum(weights.values())
        vertices.append(len(weights))
        for bone,weight in weights.items():
            data=definitions[bone];local=rotation(-data['rotation'])@(point-data['position'])
            vertices.extend([indices[bone],rounded(local[0]),rounded(local[1]),weight/total])
    return {'type':'mesh','path':name,'width':region['width'],'height':region['height'],
            'uvs':uvs,'triangles':triangles,'vertices':vertices,'hull':hull}


def build_rig(source,config):
    definitions={b['name']:b for b in config['bones']};origin=np.array(config['origin'],float)
    result={'skeleton':{'spine':config['target_version'],'images':'./images/','fps':FPS,
                        'x':-1000,'y':0,'width':1500,'height':1400},'bones':[],'slots':[],'skins':[],
            'events':{'claw_hit':{'string':'front_arm_paw'},'spawn_bullet':{'string':'projectile_spawn'},'impact':{},'footstep':{'string':'back_leg'},'death_end':{}},'animations':{}}
    for b in config['bones']:
        parent=definitions.get(b.get('parent'));position=np.array(b['position'],float)
        local=rotation(-parent['rotation'])@(position-parent['position']) if parent else position-origin
        entry={'name':b['name'],'x':rounded(local[0]),'y':rounded(local[1]),
               'rotation':rounded(b['rotation']-(parent['rotation'] if parent else 0)),'length':rounded(b['length'])}
        if parent:entry['parent']=parent['name']
        result['bones'].append(entry)
    skin={}
    for slot in source['slots']:
        name=slot['name'];result['slots'].append({'name':name,'bone':config['bindings'][name],'attachment':name})
        skin[name]={name:weighted_attachment(name,source['skins']['default'][name][name],config)}
    for name,region,bone in [('claw_fx',{'x':440,'y':-780,'width':440,'height':350},'slash_anchor'),('spit_fx',{'x':425,'y':-565,'width':300,'height':180},'mouth')]:
        attachment=weighted_attachment(name,region,config);attachment['path']='fx/'+name
        skin[name]={name:attachment};result['slots'].append({'name':name,'bone':bone,'blend':'additive' if name=='claw_fx' else 'normal'})
    result['skins']=[{'name':'default','attachments':skin}]
    return result


def controls_world(rig,config,angles,positions,scales):
    definitions={b['name']:b for b in config['bones']};world={}
    for bone in rig['bones']:
        name=bone['name'];parent=bone.get('parent');parent_rotation=definitions[parent]['rotation'] if parent else 0
        offset=rotation(-parent_rotation)@positions[name]
        local=np.eye(3);local[:2,:2]=rotation(bone.get('rotation',0)+angles[name])@np.diag(scales[name])
        local[:2,2]=[bone.get('x',0)+offset[0],bone.get('y',0)+offset[1]]
        world[name]=world[parent]@local if parent else local
    return world


def solve_paws(rig,config,angles,positions,scales,targets,pitches):
    world=controls_world(rig,config,angles,positions,scales);definitions={b['name']:b for b in config['bones']}
    for name,data in config['limbs'].items():
        upper=definitions[name+'_upper'];lower=definitions[name+'_lower'];paw=definitions[name+'_paw']
        parent=world[data['parent']];parent_angle=math.degrees(math.atan2(parent[1,0],parent[0,0]))
        hip=world[name+'_upper'][:2,2];target=np.array(targets[name])-config['origin'];vector=target-hip
        length=np.linalg.norm(vector);a,b=upper['length'],lower['length']
        u=np.array(data['knee'])-data['hip'];v=np.array(data['ankle'])-data['knee'];sign=1 if np.linalg.det(np.array([u,v]))>=0 else -1
        distance=np.clip(length,abs(a-b)+1e-5,a+b-1e-5)
        bend=sign*math.acos(np.clip((distance**2-a*a-b*b)/(2*a*b),-1,1))
        theta=math.atan2(vector[1],vector[0])-math.atan2(b*math.sin(bend),a+b*math.cos(bend))
        first=math.degrees(theta);second=first+math.degrees(bend)
        # atan2 wraps at ±180; normalize deltas so interpolation cannot spin a leg 360°.
        delta=lambda value:(value+180)%360-180
        angles[name+'_upper']=delta(first-parent_angle-(upper['rotation']-definitions[data['parent']]['rotation']))
        angles[name+'_lower']=delta((second-first)-(lower['rotation']-upper['rotation']))
        angles[name+'_paw']=delta(paw['rotation']+pitches[name]-second-(paw['rotation']-lower['rotation']))


def curve(t,points):return interpolate_controls(t,points)


def make_animations(rig,config):
    definitions={b['name']:b for b in config['bones']};animations={}
    for name,duration in DURATIONS.items():
        tracks={n:{'rotate':[],'translate':[],'scale':[]} for n in definitions}
        for f in range(round(duration*FPS)+1):
            t=f/FPS;phase=t/duration*math.tau;r={n:0. for n in definitions};p={n:[0.,0.] for n in definitions};s={n:[1.,1.] for n in definitions}
            targets={n:list(d['ankle']) for n,d in config['limbs'].items()};pitches={n:0 for n in targets}
            arm_target=None;arm_pitch=0
            if name=='stand':
                breath=(1-math.cos(phase))/2;p['pelvis'][1]=-6*breath;p['body'][1]=16*breath
                r['body']=3*math.sin(phase);r['neck']=-2*math.sin(phase);r['head']=6*math.sin(phase)
                r['crown']=-5*(math.sin(phase-.4)-math.sin(-.4));s['body']=[1+.018*breath,1+.008*breath]
                for arm,sign in [('front_arm',1),('back_arm',-1)]:r[arm+'_upper']=sign*7*math.sin(phase);r[arm+'_lower']=-sign*10*math.sin(phase);r[arm+'_paw']=5*breath
            elif name=='move':
                stride=math.sin(phase);bob=(math.sqrt(stride*stride+.04**2)-.04)/(math.sqrt(1+.04**2)-.04)
                p['pelvis'][1]=-30*bob;r['body']=6*(1-math.cos(phase))/2+3*stride;r['neck']=-2*stride;r['head']=-5*stride
                r['crown']=-7*(math.sin(phase-.35)-math.sin(-.35))
                for arm,sign in [('front_arm',1),('back_arm',-1)]:r[arm+'_upper']=sign*23*stride;r[arm+'_lower']=-sign*14*stride;r[arm+'_paw']=sign*9*stride
                for n,d in config['limbs'].items():
                    step=math.sin(phase+d['phase']);lift=max(step,0)**2;targets[n]=[d['ankle'][0]-50*step,d['ankle'][1]+35*lift];pitches[n]=-10*lift
            elif name=='attack_1':
                wind=curve(t,[(0,0),(.2,1),(.23,1),(.5,0),(duration,0)])
                reach=curve(t,[(0,0),(.23,0),(.5,1),(16/FPS,.97),(.95,-.08),(duration,0)])
                rake=curve(t,[(0,0),(.5,0),(.7,1),(.95,-.15),(duration,0)])
                p['pelvis']=[20*wind-75*reach,-30*wind-35*reach];p['body']=[30*wind-65*reach,-15*wind-20*reach]
                r['body']=-9*wind+12*reach;r['neck']=4*wind-4*reach;r['head']=-7*wind-8*reach+5*rake;r['crown']=9*wind-12*rake
                arm_target=[968+32*wind-523*max(reach,0),-920+160*wind+130*max(reach,0)-40*rake];arm_pitch=-38*wind-77*reach-12*rake
                if t<=.5:arm_target[1]-=90*math.sin(math.pi*max(0,min(1,reach)))
                r['back_arm_upper']=25*wind+20*reach;r['back_arm_lower']=-20*wind-25*reach;r['back_arm_paw']=15*rake
                for n,d in config['limbs'].items():targets[n]=[d['ankle'][0]-55*reach,d['ankle'][1]+5*max(reach,0)]
            elif name=='attack_2':
                wind=curve(t,[(0,0),(.3,1),(.37,1),(.5,0),(duration,0)])
                spit=curve(t,[(0,0),(.37,0),(.5,1),(.6,.85),(.95,-.12),(duration,0)])
                lag=curve(t,[(0,0),(.5,0),(.63,1),(.95,-.2),(duration,0)])
                p['pelvis']=[25*wind-65*spit,-18*wind-25*spit];p['body']=[25*wind-35*spit,-15*wind-15*spit]
                r['body']=-9*wind+10*spit;r['neck']=-5*wind+4*spit;r['head']=-9*wind+6*spit-5*lag;r['crown']=8*wind-13*lag
                p['head']=[-40*spit,0];s['body']=[1+.065*wind-.025*spit,1-.025*wind+.015*spit]
                r['front_arm_upper']=20*wind-20*spit;r['front_arm_lower']=-22*wind+10*spit;r['front_arm_paw']=10*lag
                r['back_arm_upper']=-10*wind+15*spit;r['back_arm_lower']=15*wind-12*spit
                for n,d in config['limbs'].items():targets[n]=[d['ankle'][0]-25*spit,d['ankle'][1]]
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(.17,.8),(.3,-.18),(duration,0)]);lag=curve(t,[(0,0),(.14,1),(.28,-.2),(duration,0)])
                p['pelvis']=[25*shock,-15*shock];p['body']=[15*shock,-10*shock];r['body']=-10*shock;r['neck']=-5*lag;r['head']=-13*lag;r['crown']=12*lag
                for arm in ('front_arm','back_arm'):r[arm+'_upper']=20*shock;r[arm+'_lower']=-25*lag;r[arm+'_paw']=15*lag
            elif name=='dead':
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)]);fall=curve(t,[(0,0),(.25,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)]);impact=curve(t,[(0,0),(.85,0),(1,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['pelvis']=[-120*fall,-160*fall-35*buckle];r['pelvis']=82*fall;r['body']=3*fall+5*buckle;r['neck']=-5*fall;r['head']=-8*fall-6*impact;r['crown']=12*impact
                for arm in ('front_arm','back_arm'):r[arm+'_upper']=-10*fall;r[arm+'_lower']=-15*fall;r[arm+'_paw']=15*fall
                r['back_arm_upper']=20*fall;r['back_arm_lower']=30*fall;r['back_arm_paw']=0
                for n in config['limbs']:r[n+'_upper']=35*fall-25*buckle;r[n+'_lower']=50*buckle;r[n+'_paw']=-25*buckle
            if name!='dead':solve_paws(rig,config,r,p,s,targets,pitches)
            if arm_target is not None:solve_paws(rig,{**config,'limbs':{'front_arm':config['arms']['front_arm']}},r,p,s,{'front_arm':arm_target},{'front_arm':arm_pitch})
            for n in definitions:
                parent=definitions.get(definitions[n].get('parent'));local=rotation(-parent['rotation'])@p[n] if parent else p[n]
                angle=r[n]
                if tracks[n]['rotate']:
                    prev=tracks[n]['rotate'][-1]['value'];angle=prev+(angle-prev+180)%360-180
                tracks[n]['rotate'].append({'time':t,'value':rounded(angle)});tracks[n]['translate'].append({'time':t,'x':rounded(local[0]),'y':rounded(local[1])});tracks[n]['scale'].append({'time':t,'x':rounded(s[n][0]),'y':rounded(s[n][1])})
        for n in list(tracks):
            for kind in list(tracks[n]):
                fields=('value',) if kind=='rotate' else ('x','y');neutral=1 if kind=='scale' else 0
                if n!='root' and all(all(k[field]==neutral for field in fields) for k in tracks[n][kind]):del tracks[n][kind]
            if not tracks[n]:del tracks[n]
        slots={n:{'attachment':[{'time':0,'name':None}],'alpha':[{'time':0,'value':0}]} for n in ('claw_fx','spit_fx')};a={'bones':tracks,'slots':slots}
        if name=='attack_1':
            a['events']=[{'time':.5,'name':'claw_hit'}];slots['claw_fx']={'attachment':[{'time':0,'name':None},{'time':.5,'name':'claw_fx'},{'time':.8,'name':None}],'alpha':[{'time':0,'value':0},{'time':14/FPS,'value':0},{'time':.5,'value':1},{'time':.6,'value':.85},{'time':.8,'value':0}]}
            names=[x['name'] for x in rig['slots']];moving=['zuogebo','zuoshoubi','zuoshou'];order=[n for n in names if n not in moving];index=order.index('claw_fx');order[index:index]=moving
            offsets=[{'slot':n,'offset':order.index(n)-i} for i,n in enumerate(names) if order.index(n)!=i]
            a['drawOrder']=[{'time':0},{'time':.3,'offsets':offsets},{'time':1.1}]
        if name=='attack_2':
            a['events']=[{'time':.5,'name':'spawn_bullet'}];slots['spit_fx']={'attachment':[{'time':0,'name':None},{'time':.5,'name':'spit_fx'},{'time':.7,'name':None}],'alpha':[{'time':0,'value':0},{'time':14/FPS,'value':0},{'time':.5,'value':1},{'time':.6,'value':.8},{'time':.7,'value':0}]}
        if name=='hit':a['events']=[{'time':.1,'name':'impact'}]
        if name=='dead':a['events']=[{'time':1.4,'name':'death_end'}]
        if name=='move':a['events']=[{'time':.5,'name':'footstep','string':'back_leg'},{'time':1.,'name':'footstep','string':'front_leg'}]
        animations[name]=a
    return animations


def make_effects():
    """Explicit procedural VFX; never change the fourteen supplied body textures."""
    from PIL import ImageFilter
    directory=BASE/'images/fx';scale=3;size=(440,350);im=Image.new('RGBA',(size[0]*scale,size[1]*scale));d=ImageDraw.Draw(im)
    # Three parallel tapering hooked scars, extending down and forward (left).
    for i in range(3):
        x=65+i*80;y=35+i*25
        points=[(x+190,y),(x+102,y+73),(x+53,y+160),(x+20,y+245),(x+52,y+180),(x+97,y+114),(x+155,y+64)]
        d.polygon([(int(a*scale),int(b*scale)) for a,b in points],fill=(245,24,47,240))
        d.line([(int((x+120)*scale),int((y+44)*scale)),(int((x+65)*scale),int((y+110)*scale)),(int((x+38)*scale),int((y+170)*scale))],fill=(255,125,119,255),width=5*scale)
    glow=im.filter(ImageFilter.GaussianBlur(8*scale));glow.alpha_composite(im);glow.resize(size,Image.Resampling.LANCZOS).save(directory/'claw_fx.png')
    # A short leftward jet; its right edge is anchored at the mouth.
    im=Image.new('RGBA',(900,540));d=ImageDraw.Draw(im);d.polygon([(900,250),(620,185),(340,175),(95,255),(340,315),(600,310),(900,290)],fill=(105,132,30,225));d.polygon([(900,258),(610,225),(360,240),(155,270),(550,275),(900,280)],fill=(194,214,67,245))
    for x,y,r in [(120,230,32),(260,350,25),(420,155,17),(70,315,17),(530,350,11)]:d.ellipse((x-r,y-r,x+r,y+r),fill=(149,175,43,230))
    im.resize((300,180),Image.Resampling.LANCZOS).save(directory/'spit_fx.png')


def render(skeleton,images,animation=None,t=0,size=SIZE,camera=CAMERA,background=None,rig=False):
    return shared_render(skeleton,images,animation,t,size,camera,background,rig)


def correct_ground(skeleton,images):
    track=skeleton['animations']['dead']['bones']['root']['translate'];size=(3000,2400);camera=(1700,2100)
    correction=render(skeleton,images,'dead',1.6,size=size,camera=camera).getbbox()[3]-camera[1]
    for k in track:k['y']=rounded(correction*curve(k['time'],[(0,0),(.25,0),(.95,1),(1.6,1)]))
    for k in track:
        bottom=render(skeleton,images,'dead',k['time'],size=size,camera=camera).getbbox()[3]
        k['y']=rounded(k['y']+max(0,bottom-camera[1]))


def scaled_preview(skeleton,factor=.4):
    result=deepcopy(skeleton);result['bones'][0]['scaleX']=factor;result['bones'][0]['scaleY']=factor
    for a in result['animations'].values():
        for k in a.get('bones',{}).get('root',{}).get('translate',[]):k['x']*=factor;k['y']*=factor
    return result


# The independent poison rig is shared; its image provenance is recorded in source/.
from tools.build_zsx_spine import build_bullet


def make_previews(skeleton,images,directory,bullet):
    factor=.4;stage=tuple(round(n*factor) for n in SIZE);camera=tuple(round(n*factor) for n in CAMERA);viewrig=scaled_preview(skeleton,factor)
    bg=Image.new('RGBA',stage,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,stage[0],16):
        for y in range(0,stage[1],16):
            if (x//16+y//16)%2:d.rectangle((x,y,x+15,y+15),fill='#2d3d47')
    d.line((0,camera[1],stage[0],camera[1]),fill='#6e9f90')
    sheet=Image.new('RGB',(1440,len(DURATIONS)*300),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for f in range(round(duration*FPS)+1):
            t=f/FPS;image=render(viewrig,images,name,t,size=stage,camera=camera,background=bg).convert('RGB');view=image.resize((840,630),Image.Resampling.LANCZOS);ImageDraw.Draw(view).text((12,12),f'{name} / {t:.2f}s',fill='#9cdedc');frames.append(view)
            phone=Image.new('RGB',(420,264),'#172027');phone.paste(image.resize((224,168),Image.Resampling.LANCZOS),(98,48));ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack_1','attack_2'));save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=([0,.23,.5,.7] if name=='attack_1' else [0,.3,.5,.65] if name=='attack_2' else [0,duration*.3,duration*.65,duration])
        for col,t in enumerate(times):sheet.paste(frames[min(round(t*FPS),len(frames)-1)].resize((360,270),Image.Resampling.LANCZOS),(col*360,row*300+25))
        labels.text((12,row*300+6),name,fill='#9cdedc')
    sheet.save(directory/'animation-contact.png');render(viewrig,images,size=stage,camera=camera,background=bg,rig=True).convert('RGB').save(directory/'rig-setup.png')
    frames=[]
    for f in range(19):
        im=render(bullet,images,'bullet',f/FPS,size=(640,360),camera=(320,180),background=Image.new('RGBA',(640,360),'#26343e')).convert('RGB');ImageDraw.Draw(im).text((12,12),'bullet / stationary root / LEFT-facing poison',fill='#9cdedc');frames.append(im)
    save_gif(frames,directory/'bullet.gif',True);frames[4].save(directory/'bullet-frame.png')


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text());config=json.loads((BASE/'rig-config.json').read_text());make_effects();images=load_images(BASE/'images')
    for n,entries in source['skins']['default'].items():
        a=entries[n]
        if images[n].size!=(a['width'],a['height']):raise ValueError('Image size mismatch '+n)
    actor=build_rig(source,config);actor['animations']=make_animations(actor,config);correct_ground(actor,images);write_json(BASE/'zhshz-4.3.json',actor)
    bullet=build_bullet(config);write_json(BASE/'poison-bullet-4.3.json',bullet)
    if previews:make_previews(actor,images,BASE/'previews',bullet)
    print('Built ZHSHZ:',len(actor['bones']),'bones;',len(actor['slots']),'slots;',list(actor['animations']));return actor


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true');args=parser.parse_args();build(not args.no_previews)

if __name__=='__main__':main()
