"""Editable Spine PRO quadruped rig: anticipation, lunge-bite and in-place trot."""
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

BASE=Path(__file__).resolve().parents[1]/'examples'/'btmg-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'attack':1.,'move':.8}
SIZE=(1280,760)
CAMERA=(570,620)


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
    x,y=point
    if name=='qugan':
        hips=smoothstep(250,430,x);return {'hips':hips,'chest':1-hips}
    if name=='tou':
        jaw=smoothstep(238,285,-y);near=smoothstep(73,103,x)
        return {'head':1-jaw,'jaw_near':jaw*near,'jaw_far':jaw*(1-near)}
    if name=='weiba':
        definitions={b['name']:b for b in config['bones']};a=np.array(definitions['tail_base']['position']);b=np.array(definitions['tail_tip']['position'])
        u=np.dot(point-a,b-a)/np.dot(b-a,b-a);tip=smoothstep(.25,.95,u)
        return {'tail_base':1-tip,'tail_tip':tip}
    limb=next((key,value) for key,value in config['limbs'].items() if value['image']==name)
    key,data=limb;chain=[data['hip'],data['knee'],data['ankle'],data['toe']]
    arc=nearest_arc(point,chain);upper=math.dist(chain[0],chain[1]);lower=math.dist(chain[1],chain[2])
    first=1-smoothstep(upper-22,upper+22,arc);paw=smoothstep(upper+lower-17,upper+lower+17,arc)
    return {key+'_upper':first,key+'_lower':max(0,1-first-paw),key+'_paw':paw}


def weighted_attachment(name,region,config):
    columns,rows=(18,12) if name=='qugan' else ((16,18) if name=='tou' else ((8,10) if name=='weiba' else (12,18)))
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
                        'x':-310,'y':0,'width':620,'height':470},'bones':[],'slots':[],'skins':[],
            'events':{'bite_hit':{'string':'bite_point'},'impact':{},'footstep':{'string':'diagonal_a'},'death_end':{}},'animations':{}}
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
                p['hips'][1]=-1.8*breath;p['chest'][1]=4.8*breath
                r['chest']=1.4*math.sin(phase)
                r['neck']=-3*math.sin(phase);r['head']=7.5*(math.sin(phase-.15)-math.sin(-.15))
                r['tail_base']=7*(math.sin(phase-.35)-math.sin(-.35));r['tail_tip']=11*(math.sin(phase-.65)-math.sin(-.65))
                r['jaw_near']=2*breath;r['jaw_far']=-1.5*breath
            elif name=='move':
                stride=math.sin(phase)
                bob=(math.sqrt(stride*stride+.04**2)-.04)/(math.sqrt(1+.04**2)-.04)
                p['hips'][1]=-18*bob;p['chest'][1]=3*math.sin(2*phase)
                r['hips']=1.2*math.sin(phase);r['chest']=-1.7*math.sin(phase)
                r['neck']=3*math.sin(2*phase);r['head']=-4.5*math.sin(2*phase)
                r['tail_base']=11*(math.sin(phase-.3)-math.sin(-.3));r['tail_tip']=16*(math.sin(phase-.6)-math.sin(-.6))
                for key,data in config['limbs'].items():
                    step=math.sin(phase+data['phase']);lift=max(0,step)**2
                    targets[key]=[data['ankle'][0]-30*step,data['ankle'][1]+22*lift]
                    pitches[key]=-6*lift
            elif name=='attack':
                wind=curve(t,[(0,0),(6/FPS,1),(8/FPS,.9),(11/FPS,0),(duration,0)])
                reach=curve(t,[(0,0),(7/FPS,0),(11/FPS,1),(13/FPS,.95),(23/FPS,-.06),(duration,0)])
                tear=curve(t,[(0,0),(12/FPS,0),(16/FPS,1),(20/FPS,-.2),(duration,0)])
                gape=curve(t,[(0,0),(6/FPS,1),(9/FPS,1),(12/FPS,0),(duration,0)])
                close=curve(t,[(0,0),(10/FPS,0),(12/FPS,1),(16/FPS,1),(23/FPS,0),(duration,0)])
                p['hips']=[22*wind-60*reach,-14*wind+8*reach]
                p['chest']=[8*wind-15*reach,-4*wind-5*reach]
                r['chest']=-4*wind+6*reach;r['neck']=-8*wind+7*reach
                r['head']=-6*wind-4*reach-10*tear;p['head']=[-20*reach+14*tear,5*tear]
                s['head']=[1-.02*wind+.025*reach,1+.015*wind-.015*reach]
                r['jaw_near']=22*gape-11*close;r['jaw_far']=-17*gape+10*close
                lag=curve(t,[(0,0),(10/FPS,0),(15/FPS,1),(21/FPS,-.25),(duration,0)])
                r['tail_base']=-10*wind+15*lag;r['tail_tip']=22*lag
                for key,data in config['limbs'].items():
                    front='front' in key
                    targets[key]=[data['ankle'][0]-(80 if front else 35)*reach,
                                  data['ankle'][1]+(14 if front else 20)*max(0,reach)]
                    pitches[key]=(-7 if front else 5)*max(0,reach)
            elif name=='hit':
                shock=curve(t,[(0,0),(3/FPS,1),(5/FPS,.85),(9/FPS,-.15),(duration,0)])
                lag=curve(t,[(0,0),(4/FPS,1),(8/FPS,-.18),(duration,0)])
                p['hips']=[10*shock,-6*shock];p['chest']=[8*shock,-3*shock]
                r['chest']=-8*shock;r['neck']=-8*lag;r['head']=-6*lag
                r['tail_base']=12*lag;r['tail_tip']=-16*lag
                r['jaw_near']=7*lag;r['jaw_far']=-5*lag
            elif name=='dead':
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)])
                fall=curve(t,[(0,0),(.25,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)])
                impact=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['hips']=[12*fall,-140*fall-12*buckle];p['chest']=[-8*fall,-4*fall-5*buckle]
                r['hips']=-6*fall;r['chest']=3*fall-4*buckle;r['neck']=-5*fall-8*buckle
                r['head']=fall-10*buckle+5*impact;p['head'][1]=-8*fall
                r['jaw_near']=8*fall;r['jaw_far']=-6*fall
                r['tail_base']=-25*fall+8*impact;r['tail_tip']=-18*fall-10*impact
                for key in config['limbs']:
                    if 'front' in key:
                        r[key+'_upper']=(65 if key=='near_front' else 75)*fall
                        r[key+'_lower']=-110*fall;r[key+'_paw']=(160 if key=='near_front' else 100)*fall
                    elif key=='near_hind':
                        r[key+'_upper']=-105*fall;r[key+'_lower']=145*fall;r[key+'_paw']=30*fall
                    else:
                        r[key+'_upper']=-70*fall;r[key+'_lower']=110*fall;r[key+'_paw']=35*fall
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
        if name=='attack':animation['events']=[{'time':12/FPS,'name':'bite_hit'}]
        if name=='hit':animation['events']=[{'time':3/FPS,'name':'impact'}]
        if name=='dead':animation['events']=[{'time':42/FPS,'name':'death_end'}]
        if name=='move':animation['events']=[{'time':duration/2,'name':'footstep','string':'diagonal_a'},
                                           {'time':duration,'name':'footstep','string':'diagonal_b'}]
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
    sheet=Image.new('RGB',(1440,len(DURATIONS)*245),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;image=render(skeleton,images,name,t,background=bg).convert('RGB')
            view=image.resize((768,456),Image.Resampling.LANCZOS)
            ImageDraw.Draw(view).text((12,12),f'{name} / {frame:02d} / {t:.2f}s',fill='#9cdedc');frames.append(view)
            phone=Image.new('RGB',(420,264),'#172027')
            phone.paste(image.resize((390,232),Image.Resampling.LANCZOS),(15,28))
            ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack'))
        save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=[0,6/FPS,12/FPS,16/FPS] if name=='attack' else [0,duration*.3,duration*.65,duration]
        for column,t in enumerate(times):
            image=frames[min(round(t*FPS),len(frames)-1)].resize((360,214),Image.Resampling.LANCZOS)
            sheet.paste(image,(column*360,row*245+25))
        labels.text((12,row*245+6),name,fill='#9cdedc')
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
    correct_ground(skeleton,images);write_json(BASE/'btmg-4.3.json',skeleton)
    if previews:make_previews(skeleton,images,BASE/'previews')
    print('Built BTMG:',len(skeleton['bones']),'bones,',len(skeleton['slots']),'weighted meshes;',list(skeleton['animations']))
    return skeleton


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args();build(not args.no_previews)


if __name__=='__main__':main()
