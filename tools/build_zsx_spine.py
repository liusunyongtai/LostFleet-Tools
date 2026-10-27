"""Venom creature: anticipatory melee, belly-powered spit, stationary slime projectile."""
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

BASE=Path(__file__).resolve().parents[1]/'examples'/'zsx-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'attack_1':1.2,'attack_2':1.4,'move':1.}
SIZE=(2400,1800)
CAMERA=(1400,1500)


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
    x,y=point;depth=-y
    if name=='spit_fx':return {'spit_fx':1.}
    if name=='shenti':
        legs=smoothstep(1030,1080,depth);front=smoothstep(700,770,x)
        sac=.9*smoothstep(650,750,depth)*(1-smoothstep(1020,1080,depth))*(1-smoothstep(860,940,x))
        hips=smoothstep(950,1040,depth);remain=(1-legs)*(1-sac)
        weights={'sac':(1-legs)*sac,'body':remain*(1-hips),'pelvis':remain*hips}
        for key,share in [('front_leg',front),('back_leg',1-front)]:
            d=config['limbs'][key];lower=smoothstep(-d['knee'][1]-35,-d['knee'][1]+35,depth)
            paw=smoothstep(-d['ankle'][1]-14,-d['ankle'][1]+14,depth)
            weights.update({key+'_upper':legs*share*(1-lower),key+'_lower':legs*share*max(0,lower-paw),key+'_paw':legs*share*paw})
        return weights
    if name=='tou':
        points=[[514,-551],[437,-611],[328,-705]];arc=nearest_arc(point,points);distance=1e9
        for a,b in zip(points,points[1:]):
            a,b=np.array(a),np.array(b);v=b-a;u=np.clip(np.dot(point-a,v)/np.dot(v,v),0,1)
            distance=min(distance,np.linalg.norm(point-(a+u*v)))
        tongue=(1-smoothstep(35,70,distance))*(1-smoothstep(530,570,x))*smoothstep(520,550,depth)
        tip=smoothstep(85,115,arc);jaw=smoothstep(565,620,depth)*smoothstep(485,555,x)*(1-tongue)
        return {'head':max(0,1-tongue-jaw),'jaw':jaw,'tongue_base':tongue*(1-tip),'tongue_tip':tongue*tip}
    key,d=next((k,v) for k,v in config['arms'].items() if v['image']==name)
    points=[d[k] for k in ('hip','knee','ankle','toe')];arc=nearest_arc(point,points)
    a=math.dist(*points[:2]);b=math.dist(*points[1:3]);upper=1-smoothstep(a-35,a+35,arc);paw=smoothstep(a+b-24,a+b+24,arc)
    return {key+'_upper':upper,key+'_lower':max(0,1-upper-paw),key+'_paw':paw}


def weighted_attachment(name,region,config):
    columns,rows={'shenti':(20,28),'tou':(24,26),'youbi':(10,16),'zuobi':(10,16)}.get(name,(1,1))
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
                        'x':-1100,'y':0,'width':1800,'height':1500},'bones':[],'slots':[],'skins':[],
            'events':{'melee_hit':{'string':'melee_tip'},'spawn_bullet':{'string':'projectile_spawn'},'impact':{},'footstep':{'string':'back_leg'},'death_end':{}},'animations':{}}
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
    region={'x':300,'y':-550,'width':300,'height':150}
    fx=weighted_attachment('spit_fx',region,config);fx['path']='fx/poison-flight'
    skin['spit_fx']={'spit_fx':fx};result['slots'].append({'name':'spit_fx','bone':'spit_fx'})
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
        timelines={bone:{'rotate':[],'translate':[],'scale':[]} for bone in definitions}
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;phase=t/duration*math.tau
            r={bone:0. for bone in definitions};p={bone:[0.,0.] for bone in definitions};s={bone:[1.,1.] for bone in definitions}
            targets={key:list(value['ankle']) for key,value in config['limbs'].items()};pitches={key:0 for key in targets}
            if name=='stand':
                breath=(1-math.cos(phase))/2;p['pelvis'][1]=-4*breath;p['body'][1]=10*breath
                r['body']=2*math.sin(phase);r['neck']=-2*math.sin(phase);r['head']=7*math.sin(phase)
                s['sac']=[1+.04*breath,1+.06*breath];r['jaw']=2*breath
                r['tongue_base']=3*math.sin(phase);r['tongue_tip']=4*(math.sin(phase-.25)-math.sin(-.25))
                r['back_arm_upper']=5*math.sin(phase);r['front_arm_upper']=-6*math.sin(phase)
            elif name=='move':
                stride=math.sin(phase);bob=(math.sqrt(stride*stride+.04**2)-.04)/(math.sqrt(1+.04**2)-.04)
                p['pelvis'][1]=-16*bob;r['body']=4*stride;r['neck']=-2*stride;r['head']=-7*stride
                r['back_arm_upper']=13*stride;r['front_arm_upper']=-13*stride
                r['back_arm_lower']=-7*stride;r['front_arm_lower']=7*stride
                r['tongue_base']=5*(math.sin(phase-.2)-math.sin(-.2));r['tongue_tip']=6*(math.sin(phase-.45)-math.sin(-.45))
                for key,d in config['limbs'].items():
                    step=math.sin(phase+d['phase']);lift=max(step,0)**2
                    targets[key]=[d['ankle'][0]-32*step,d['ankle'][1]+18*lift];pitches[key]=-6*lift
            elif name=='attack_1':
                wind=curve(t,[(0,0),(.2,1),(7/FPS,.95),(.5,0),(duration,0)])
                reach=curve(t,[(0,0),(7/FPS,0),(.5,1),(16/FPS,.96),(.95,-.06),(duration,0)])
                recoil=curve(t,[(0,0),(.5,0),(.7,1),(1.,-.15),(duration,0)])
                p['pelvis']=[20*wind-45*reach,-10*wind-6*reach];p['body']=[8*wind-15*reach,-5*wind]
                r['body']=-7*wind+10*reach;r['neck']=4*wind-6*reach;r['head']=-8*wind-4*reach-4*recoil
                p['head']=[20*reach,6*recoil];s['sac']=[1-.035*wind+.025*reach,1+.045*wind-.02*reach]
                r['front_arm_upper']=35*wind-130*reach+20*recoil
                r['front_arm_lower']=-20*wind-10*reach+20*recoil;r['front_arm_paw']=30*reach-15*recoil
                s['front_arm_upper']=[1+.08*reach,1.];s['front_arm_lower']=[1+.06*reach,1.]
                r['back_arm_upper']=-15*wind-55*reach;r['back_arm_lower']=12*wind+20*reach
                r['jaw']=8*wind-4*reach;r['tongue_base']=-4*wind+5*recoil;r['tongue_tip']=6*recoil
                for key,d in config['limbs'].items():targets[key]=[d['ankle'][0]-42*reach,d['ankle'][1]+5*max(reach,0)]
            elif name=='attack_2':
                inhale=curve(t,[(0,0),(.3,1),(.4,1),(.5,0),(duration,0)])
                spit=curve(t,[(0,0),(.4,0),(.5,1),(.6,.95),(.95,-.08),(duration,0)])
                recoil=curve(t,[(0,0),(.5,0),(.72,1),(1.1,-.15),(duration,0)])
                p['pelvis']=[18*inhale-32*spit,-8*inhale-6*spit];p['body']=[8*inhale-12*spit,7*inhale-5*spit]
                r['body']=-7*inhale+12*spit;r['neck']=-5*inhale+5*spit;r['head']=-12*inhale+6*spit-7*recoil
                p['head']=[-45*spit+20*recoil,8*inhale]
                s['sac']=[1+.10*inhale-.06*spit+.025*recoil,1+.08*inhale+.03*spit]
                r['jaw']=9*inhale-3*spit+3*recoil;r['tongue_base']=-4*inhale+5*spit;r['tongue_tip']=-5*inhale+7*spit
                r['front_arm_upper']=22*inhale+30*spit;r['front_arm_lower']=-15*inhale-15*spit
                r['back_arm_upper']=-12*inhale+15*spit;r['back_arm_lower']=10*inhale-10*spit
                burst=curve(t,[(0,.35),(.5,.35),(.6,1.1),(.7,.7),(duration,1.)])
                s['spit_fx']=[burst,burst]
                reset=curve(t,[(0,1),(.7,1),(.9,0),(duration,0)])
                p['spit_fx']=[(110-145*burst)*reset,10*burst*reset]
                for key,d in config['limbs'].items():targets[key]=[d['ankle'][0]-30*spit,d['ankle'][1]+4*max(spit,0)]
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(5/FPS,.8),(.3,-.18),(duration,0)])
                lag=curve(t,[(0,0),(4/FPS,1),(8/FPS,-.2),(duration,0)])
                p['pelvis']=[12*shock,-7*shock];p['body']=[8*shock,-4*shock]
                r['body']=-9*shock;r['neck']=-6*lag;r['head']=-11*lag
                r['front_arm_upper']=18*lag;r['front_arm_lower']=-12*lag;r['back_arm_upper']=-14*lag
                r['jaw']=5*lag;r['tongue_base']=5*lag;r['tongue_tip']=-6*lag
                s['sac']=[1-.04*shock,1+.035*shock]
            elif name=='dead':
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)])
                fall=curve(t,[(0,0),(.25,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)])
                impact=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['pelvis']=[-80*fall,-150*fall-14*buckle];r['pelvis']=80*fall
                r['body']=6*fall+5*buckle;r['neck']=-6*fall;r['head']=-5*fall-4*impact
                r['jaw']=2*fall;r['tongue_base']=18*fall+3*impact;r['tongue_tip']=14*fall-3*impact
                s['sac']=[1-.03*fall,1+.03*fall]
                r['front_arm_upper']=-25*fall;r['front_arm_lower']=35*fall;r['front_arm_paw']=-20*fall
                r['back_arm_upper']=30*fall;r['back_arm_lower']=-25*fall;r['back_arm_paw']=15*fall
                for key in config['limbs']:
                    r[key+'_upper']=-20*fall-22*buckle;r[key+'_lower']=45*buckle;r[key+'_paw']=-22*buckle
            if name!='dead':solve_paws(rig,config,r,p,s,targets,pitches)
            for bone in definitions:
                parent=definitions.get(definitions[bone].get('parent'))
                local=rotation(-parent['rotation'])@p[bone] if parent else p[bone]
                timelines[bone]['rotate'].append({'time':t,'value':rounded(r[bone])})
                timelines[bone]['translate'].append({'time':t,'x':rounded(local[0]),'y':rounded(local[1])})
                timelines[bone]['scale'].append({'time':t,'x':rounded(s[bone][0]),'y':rounded(s[bone][1])})
        for bone in list(timelines):
            for kind in list(timelines[bone]):
                keys=('value',) if kind=='rotate' else ('x','y');neutral=1 if kind=='scale' else 0
                if bone!='root' and all(all(k[key]==neutral for key in keys) for k in timelines[bone][kind]):del timelines[bone][kind]
            if not timelines[bone]:del timelines[bone]
        animation={'bones':timelines}
        animation['slots']={'spit_fx':{'attachment':[{'time':0,'name':None}]}}
        if name=='attack_1':
            animation['events']=[{'time':.5,'name':'melee_hit'}]
            animation['drawOrder']=[{'time':0},{'time':.2,'offsets':[{'slot':'zuobi','offset':1}]},{'time':1.1}]
        if name=='attack_2':
            animation['events']=[{'time':.5,'name':'spawn_bullet'}]
            animation['slots']['spit_fx']['attachment'] += [{'time':.5,'name':'spit_fx'},{'time':.7,'name':None}]
        if name=='hit':animation['events']=[{'time':3/FPS,'name':'impact'}]
        if name=='dead':animation['events']=[{'time':42/FPS,'name':'death_end'}]
        if name=='move':animation['events']=[{'time':duration/2,'name':'footstep','string':'back_leg'},
                                           {'time':duration,'name':'footstep','string':'front_leg'}]
        animations[name]=animation
    return animations


def render(skeleton,images,animation=None,t=0,size=SIZE,camera=CAMERA,background=None,rig=False):
    return shared_render(skeleton,images,animation,t,size,camera,background,rig)


def correct_ground(skeleton,images):
    track=skeleton['animations']['dead']['bones']['root']['translate']
    image=render(skeleton,images,'dead',DURATIONS['dead'],size=(2200,1700),camera=(1000,1200))
    correction=image.getbbox()[3]-1200
    for key in track:
        key['y']=rounded(correction*curve(key['time'],[(0,0),(.25,0),(.95,1),(DURATIONS['dead'],1)]))
    for key in track:
        image=render(skeleton,images,'dead',key['time'],size=(2200,1700),camera=(1000,1200))
        key['y']=rounded(key['y']+max(0,image.getbbox()[3]-1200))


def scaled_preview(skeleton,factor=.55):
    result=deepcopy(skeleton);result['bones'][0]['scaleX']=factor;result['bones'][0]['scaleY']=factor
    for animation in result['animations'].values():
        for k in animation.get('bones',{}).get('root',{}).get('translate',[]):k['x']*=factor;k['y']*=factor
    return result


def build_bullet(config):
    rig={'skeleton':{'spine':config['target_version'],'images':'./images/','fps':FPS,'x':-200,'y':-110,'width':400,'height':220},
         'bones':[{'name':'root'},{'name':'core','parent':'root','x':-90},{'name':'tail','parent':'root','x':65}],
         'slots':[{'name':'poison','bone':'core','attachment':'poison'}],'skins':[{'name':'default','attachments':{}}],'animations':{}}
    points,triangles,hull=grid(20,10);vertices=[];uvs=[]
    for col,row in points:
        u,v=col/20,row/10;uvs.extend([u,v]);point=np.array([(u-.5)*360,(.5-v)*180])
        tail=smoothstep(.35,.75,u);weights=[(1,1-tail,-90),(2,tail,65)];active=[(i,w,x) for i,w,x in weights if w>1e-8];vertices.append(len(active))
        for index,weight,offset in active:vertices.extend([index,rounded(point[0]-offset),rounded(point[1]),weight])
    rig['skins'][0]['attachments']={'poison':{'poison':{'type':'mesh','path':'fx/poison-flight','width':360,'height':180,'uvs':uvs,'vertices':vertices,'triangles':triangles,'hull':hull}}}
    tracks={n:{'rotate':[],'translate':[],'scale':[]} for n in ('root','core','tail')}
    for frame in range(19):
        t=frame/FPS;phase=t/.6*math.tau
        for n in tracks:
            angle=4*(math.sin(phase-.3)-math.sin(-.3)) if n=='tail' else 0
            x=5*math.sin(phase) if n=='tail' else 0;y=5*(math.sin(phase-.3)-math.sin(-.3)) if n=='tail' else 0
            sx=1+.055*math.sin(phase) if n=='core' else 1;sy=1-.08*math.sin(phase) if n=='core' else 1
            tracks[n]['rotate'].append({'time':t,'value':rounded(angle)})
            tracks[n]['translate'].append({'time':t,'x':rounded(x),'y':rounded(y)})
            tracks[n]['scale'].append({'time':t,'x':rounded(sx),'y':rounded(sy)})
    rig['animations']={'bullet':{'bones':tracks}}
    return rig


def make_previews(skeleton,images,directory,bullet=None):
    directory.mkdir(parents=True,exist_ok=True);factor=.55;stage=tuple(round(n*factor) for n in SIZE);camera=tuple(round(n*factor) for n in CAMERA)
    viewrig=scaled_preview(skeleton,factor);bg=Image.new('RGBA',stage,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,stage[0],18):
        for y in range(0,stage[1],18):
            if (x//18+y//18)%2:d.rectangle((x,y,x+17,y+17),fill='#2d3d47')
    d.line((0,camera[1],stage[0],camera[1]),fill='#6e9f90',width=1)
    sheet=Image.new('RGB',(1440,len(DURATIONS)*300),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;image=render(viewrig,images,name,t,size=stage,camera=camera,background=bg).convert('RGB')
            view=image.resize((840,630),Image.Resampling.LANCZOS);ImageDraw.Draw(view).text((12,12),f'{name} / {frame:02d} / {t:.2f}s',fill='#9cdedc');frames.append(view)
            phone=Image.new('RGB',(420,264),'#172027');phone.paste(image.resize((284,213),Image.Resampling.LANCZOS),(68,28))
            ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack_1','attack_2'))
        save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=[0,.2,.5,.7] if name in ('attack_1','attack_2') else [0,duration*.3,duration*.65,duration]
        for column,t in enumerate(times):sheet.paste(frames[min(round(t*FPS),len(frames)-1)].resize((360,270),Image.Resampling.LANCZOS),(column*360,row*300+25))
        labels.text((12,row*300+6),name,fill='#9cdedc')
    sheet.save(directory/'animation-contact.png');render(viewrig,images,rig=True,size=stage,camera=camera,background=bg).convert('RGB').save(directory/'rig-setup.png')
    if bullet is not None:
        frames=[]
        for frame in range(19):
            image=render(bullet,images,'bullet',frame/FPS,size=(640,360),camera=(320,180),background=Image.new('RGBA',(640,360),'#26343e')).convert('RGB')
            ImageDraw.Draw(image).text((12,12),'bullet / stationary root / LEFT-facing flight pose',fill='#9cdedc');frames.append(image)
        save_gif(frames,directory/'bullet.gif',True)
        frames[4].save(directory/'bullet-frame.png')


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text(encoding='utf8'))
    config=json.loads((BASE/'rig-config.json').read_text(encoding='utf8'))
    images=load_images(BASE/'images')
    for name,entries in source['skins']['default'].items():
        a=entries[name]
        if images[name].size!=(a['width'],a['height']):raise ValueError('Image size mismatch '+name)
    skeleton=build_rig(source,config);skeleton['animations']=make_animations(skeleton,config)
    correct_ground(skeleton,images);write_json(BASE/'zsx-4.3.json',skeleton)
    bullet=build_bullet(config);write_json(BASE/'poison-bullet-4.3.json',bullet)
    if previews:make_previews(skeleton,images,BASE/'previews',bullet)
    print('Built ZSX:',len(skeleton['bones']),'bones,',len(skeleton['slots']),'weighted meshes;',list(skeleton['animations']))
    return skeleton


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args();build(not args.no_previews)


if __name__=='__main__':main()
