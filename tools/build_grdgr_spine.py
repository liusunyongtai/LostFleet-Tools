"""Editable infected worker: expressive foreground-arm punch and in-place walk."""
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

BASE=Path(__file__).resolve().parents[1]/'examples'/'grdgr-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'attack':1.,'move':1.}
SIZE=(1100,720)
CAMERA=(440,590)


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
    if name=='shenti':
        legs=smoothstep(392,421,depth);pelvis=smoothstep(350,395,depth)
        weights={'body':1-pelvis,'pelvis':pelvis-legs};front=smoothstep(187,208,x)
        for key,side in [('front_leg',front),('back_leg',1-front)]:
            d=config['limbs'][key];lower=smoothstep(-d['knee'][1]-12,-d['knee'][1]+12,depth)
            paw=smoothstep(-d['ankle'][1]-10,-d['ankle'][1]+6,depth)
            weights.update({key+'_upper':legs*side*(1-lower),key+'_lower':legs*side*(lower-paw),key+'_paw':legs*side*paw})
        return weights
    if name=='chushou':
        chain=config['neck_tendril'];arc=nearest_arc(point,chain);a=math.dist(*chain[:2]);b=math.dist(*chain[1:3])
        first=1-smoothstep(a-11,a+11,arc);last=smoothstep(a+b-11,a+b+11,arc)
        return {'neck_tendril_0':first,'neck_tendril_1':1-first-last,'neck_tendril_2':last}
    if name=='tou':
        best=(0,None,0)
        for key,d in config['tendrils'].items():
            chain=d['points'];arc=nearest_arc(point,chain);distance=1e9
            for start,end in zip(chain,chain[1:]):
                start,end=np.array(start),np.array(end);v=end-start;u=np.clip(np.dot(point-start,v)/np.dot(v,v),0,1)
                distance=min(distance,np.linalg.norm(point-(start+u*v)))
            gate={'top':smoothstep(236,254,x)*(1-smoothstep(110,135,depth)),
                  'right':smoothstep(290,307,x)*smoothstep(170,190,depth),
                  'lower':smoothstep(242,262,x)*smoothstep(230,250,depth)}[key]
            score=(1-smoothstep(*d['radii'],distance))*gate
            if score>best[0]:best=(score,key,arc)
        score,key,arc=best
        if not key:return {'head':1.}
        length=math.dist(*config['tendrils'][key]['points'][:2]);tip=smoothstep(length-10,length+10,arc)
        return {'head':1-score,'tendril_'+key+'_base':score*(1-tip),'tendril_'+key+'_tip':score*tip}
    key,d=next((k,v) for k,v in config['arms'].items() if v['image']==name)
    chain=[d['hip'],d['knee'],d['ankle'],d['toe']];arc=nearest_arc(point,chain)
    upper=math.dist(*chain[:2]);lower=math.dist(*chain[1:3]);first=1-smoothstep(upper-12,upper+12,arc);paw=smoothstep(upper+lower-8,upper+lower+8,arc)
    return {key+'_upper':first,key+'_lower':max(0,1-first-paw),key+'_paw':paw}


def weighted_attachment(name,region,config):
    columns,rows={'tou':(24,24),'shenti':(20,26),'chushou':(16,12)}.get(name,(10,16))
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
                        'x':-240,'y':0,'width':480,'height':470},'bones':[],'slots':[],'skins':[],
            'events':{'punch_hit':{'string':'punch_point'},'impact':{},'footstep':{'string':'back_leg'},'death_end':{}},'animations':{}}
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
                breath=(1-math.cos(phase))/2
                p['pelvis'][1]=-2.5*breath;p['body'][1]=5*breath
                r['body']=2*math.sin(phase);r['neck']=-2.5*math.sin(phase);r['head']=7*math.sin(phase)
                r['back_arm_upper']=4*math.sin(phase);r['front_arm_upper']=-5*math.sin(phase)
                r['front_arm_lower']=3*breath
            elif name=='move':
                stride=math.sin(phase);bob=(math.sqrt(stride*stride+.04**2)-.04)/(math.sqrt(1+.04**2)-.04)
                p['pelvis'][1]=-12*bob;r['body']=4*stride;r['neck']=-2*stride;r['head']=-6*stride
                r['back_arm_upper']=19*stride;r['front_arm_upper']=-19*stride
                r['back_arm_lower']=-7*stride;r['front_arm_lower']=7*stride
                for key,d in config['limbs'].items():
                    step=math.sin(phase+d['phase']);lift=max(step,0)**2
                    targets[key]=[d['ankle'][0]-22*step,d['ankle'][1]+15*lift];pitches[key]=-7*lift
            elif name=='attack':
                wind=curve(t,[(0,0),(.2,1),(7/FPS,.95),(12/FPS,0),(duration,0)])
                reach=curve(t,[(0,0),(7/FPS,0),(12/FPS,1),(14/FPS,.96),(24/FPS,-.08),(duration,0)])
                recoil=curve(t,[(0,0),(.4,0),(17/FPS,1),(24/FPS,-.2),(duration,0)])
                p['pelvis']=[12*wind-26*reach,-7*wind-6*reach];p['body']=[5*wind-8*reach,-4*wind]
                r['body']=-8*wind+10*reach;r['neck']=4*wind-10*reach;r['head']=-9*wind-12*reach-5*recoil
                p['head']=[50*reach,3*recoil]
                s['front_arm_upper']=[1+.10*reach,1.];s['front_arm_lower']=[1+.08*reach,1.]
                # Foreground glove travels left below the helmet; the far arm counterbalances.
                r['front_arm_upper']=45*wind-105*reach+14*recoil
                r['front_arm_lower']=-48*wind+12*reach-16*recoil
                r['front_arm_paw']=3*wind-7*reach+5*recoil
                r['back_arm_upper']=-18*wind+30*reach;r['back_arm_lower']=25*wind-20*reach
                for key,d in config['limbs'].items():
                    targets[key]=[d['ankle'][0]-25*reach,d['ankle'][1]+4*max(reach,0)]
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(5/FPS,.8),(.3,-.18),(duration,0)])
                lag=curve(t,[(0,0),(4/FPS,1),(8/FPS,-.2),(duration,0)])
                p['pelvis']=[8*shock,-5*shock];p['body']=[4*shock,-3*shock]
                r['body']=-11*shock;r['neck']=-5*lag;r['head']=-12*lag
                for arm in config['arms']:r[arm+'_upper']=18*lag;r[arm+'_lower']=-12*lag
            elif name=='dead':
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)])
                fall=curve(t,[(0,0),(.25,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)])
                impact=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['pelvis']=[-65*fall,-80*fall-12*buckle];r['pelvis']=85*fall
                r['body']=4*fall+5*buckle;r['neck']=-6*fall;r['head']=-8*fall-5*impact
                p['head']=[-13*fall,-1.7*fall]
                r['front_arm_upper']=-38*fall;r['front_arm_lower']=85*fall;r['front_arm_paw']=15*fall
                r['back_arm_upper']=35*fall;r['back_arm_lower']=65*fall
                for key in config['limbs']:
                    r[key+'_upper']=-20*fall-25*buckle
                    r[key+'_lower']=50*buckle;r[key+'_paw']=-25*buckle
            for index,key in enumerate(config['tendrils']):
                delay=.25+index*.18
                if name in ('stand','move'):
                    amplitude=5 if name=='stand' else 13
                    r['tendril_'+key+'_base']=amplitude*(math.sin(phase-delay)-math.sin(-delay))
                    r['tendril_'+key+'_tip']=amplitude*1.4*(math.sin(phase-delay-.3)-math.sin(-delay-.3))
                elif name=='attack':
                    lag=curve(t,[(0,0),(10/FPS,0),(16/FPS,1),(24/FPS,-.3),(duration,0)])
                    r['tendril_'+key+'_base']=-10*reach+15*lag;r['tendril_'+key+'_tip']=-12*reach+20*lag
                elif name=='hit':
                    r['tendril_'+key+'_base']=14*lag;r['tendril_'+key+'_tip']=-18*lag
                elif name=='dead':
                    r['tendril_'+key+'_base']=7*fall+8*impact
                    r['tendril_'+key+'_tip']=10*fall-10*impact
            for index in range(3):
                bone='neck_tendril_'+str(index)
                if name in ('stand','move'):
                    delay=.3+index*.25;r[bone]=(9+index*4)*(math.sin(phase-delay)-math.sin(-delay))
                elif name=='attack':r[bone]=(45*reach if index==0 else 0)+(-8*reach+18*lag)*(1+index*.2)
                elif name=='hit':r[bone]=16*lag*(1+index*.2)
                elif name=='dead':r[bone]=12*fall+10*impact*(1+index*.2)
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
        if name=='attack':animation['events']=[{'time':12/FPS,'name':'punch_hit'}]
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


def make_previews(skeleton,images,directory):
    directory.mkdir(parents=True,exist_ok=True)
    bg=Image.new('RGBA',SIZE,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,SIZE[0],24):
        for y in range(0,SIZE[1],24):
            if (x//24+y//24)%2:d.rectangle((x,y,x+23,y+23),fill='#2d3d47')
    d.line((0,CAMERA[1],SIZE[0],CAMERA[1]),fill='#6e9f90',width=1)
    sheet=Image.new('RGB',(1440,len(DURATIONS)*265),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;image=render(skeleton,images,name,t,background=bg).convert('RGB')
            view=image.resize((770,504),Image.Resampling.LANCZOS)
            ImageDraw.Draw(view).text((12,12),f'{name} / {frame:02d} / {t:.2f}s',fill='#9cdedc');frames.append(view)
            phone=Image.new('RGB',(420,264),'#172027')
            phone.paste(image.resize((300,196),Image.Resampling.LANCZOS),(15,28))
            ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack'))
        save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=[0,6/FPS,12/FPS,16/FPS] if name=='attack' else [0,duration*.3,duration*.65,duration]
        for column,t in enumerate(times):
            image=frames[min(round(t*FPS),len(frames)-1)].resize((360,235),Image.Resampling.LANCZOS)
            sheet.paste(image,(column*360,row*265+25))
        labels.text((12,row*265+6),name,fill='#9cdedc')
    sheet.save(directory/'animation-contact.png')
    render(skeleton,images,rig=True,background=bg).convert('RGB').save(directory/'rig-setup.png')


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text(encoding='utf8'))
    config=json.loads((BASE/'rig-config.json').read_text(encoding='utf8'))
    images=load_images(BASE/'images')
    for name,entries in source['skins']['default'].items():
        a=entries[name]
        if images[name].size!=(a['width'],a['height']):raise ValueError('Image size mismatch '+name)
    skeleton=build_rig(source,config);skeleton['animations']=make_animations(skeleton,config)
    correct_ground(skeleton,images);write_json(BASE/'grdgr-4.3.json',skeleton)
    if previews:make_previews(skeleton,images,BASE/'previews')
    print('Built GRDGR:',len(skeleton['bones']),'bones,',len(skeleton['slots']),'weighted meshes;',list(skeleton['animations']))
    return skeleton


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args();build(not args.no_previews)


if __name__=='__main__':main()
