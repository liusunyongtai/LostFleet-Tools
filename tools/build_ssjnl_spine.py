"""Dual-head cyborg: forward claw grab and raised downward slash."""
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

BASE=Path(__file__).resolve().parents[1]/'examples'/'ssjnl-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'attack_1':1.2,'attack_2':1.4,'move':1.}
SIZE=(1200,800)
CAMERA=(680,700)


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
    rigid={'tou_1':'head_main','tou_2':'head_secondary','yanjing':'eye','beibao':'backpack','yaobao':'waist_bag','lianzi':'chain'}
    if name in rigid:return {rigid[name]:1.}
    if name=='shangshen':
        pelvis=smoothstep(270,310,depth);return {'body':1-pelvis,'pelvis':pelvis}
    if name=='youshoubi':
        upper=1-smoothstep(225,260,depth);paw=smoothstep(265,290,depth)
        return {'right_arm_upper':upper,'right_arm_lower':1-upper-paw,'right_arm_paw':paw}
    if name=='zuoshoubi':
        data=config['arms']['left_arm'];points=[data[k] for k in ('hip','knee','ankle','toe')]
        arc=nearest_arc(point,points);a=math.dist(*points[:2]);b=math.dist(*points[1:3])
        upper=1-smoothstep(a-14,a+14,arc);paw=smoothstep(a+b-8,a+b+8,arc)
        return {'left_arm_upper':upper,'left_arm_lower':1-upper-paw,'left_arm_paw':paw}
    for key,d in config['limbs'].items():
        if name==d['image']:
            lower=smoothstep(-d['knee'][1]-25,-d['knee'][1]+25,depth)
            paw=smoothstep(-d['ankle'][1]-12,-d['ankle'][1]+8,depth)
            return {key+'_upper':1-lower,key+'_lower':lower-paw,key+'_paw':paw}
    raise ValueError(name)


def weighted_attachment(name,region,config):
    columns,rows={'zuotui':(12,20),'youtui':(14,24),'shangshen':(10,12),'zuoshoubi':(10,14),'youshoubi':(16,28)}.get(name,(1,1))
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
                        'x':-550,'y':0,'width':900,'height':650},'bones':[],'slots':[],'skins':[],
            'events':{'claw_hit':{'string':'claw_tip'},'slash_hit':{'string':'claw_tip'},'impact':{},'footstep':{'string':'back_leg'},'death_end':{}},'animations':{}}
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
                breath=(1-math.cos(phase))/2;p['pelvis'][1]=-2.5*breath;p['body'][1]=5*breath
                r['body']=2*math.sin(phase);r['neck_main']=-1.5*math.sin(phase);r['head_main']=7.5*math.sin(phase)
                r['head_secondary']=-5*(math.sin(phase-.3)-math.sin(-.3));r['neck_secondary']=2*math.sin(phase)
                r['left_arm_upper']=4*math.sin(phase);r['left_arm_lower']=-5*math.sin(phase)
                r['right_arm_upper']=-5*math.sin(phase);r['right_arm_lower']=4*math.sin(phase)
                r['backpack']=3*(math.sin(phase-.3)-math.sin(-.3));r['waist_bag']=-3*math.sin(phase)
                r['chain']=7*(math.sin(phase-.4)-math.sin(-.4))
            elif name=='move':
                stride=math.sin(phase);bob=(math.sqrt(stride*stride+.04**2)-.04)/(math.sqrt(1+.04**2)-.04)
                p['pelvis'][1]=-10*bob;r['body']=4*stride;r['neck_main']=-2*stride;r['head_main']=-6*stride
                r['head_secondary']=5*(math.sin(phase-.25)-math.sin(-.25))
                r['left_arm_upper']=14*stride;r['left_arm_lower']=-8*stride
                r['right_arm_upper']=-12*stride;r['right_arm_lower']=7*stride;r['right_arm_paw']=5*stride
                r['backpack']=-5*(math.sin(phase-.3)-math.sin(-.3));r['waist_bag']=6*stride
                r['chain']=9*(math.sin(phase-.45)-math.sin(-.45))
                for key,d in config['limbs'].items():
                    step=math.sin(phase+d['phase']);lift=max(step,0)**2
                    targets[key]=[d['ankle'][0]-18*step,d['ankle'][1]+12*lift];pitches[key]=-6*lift
            elif name=='attack_1':
                wind=curve(t,[(0,0),(.2,1),(7/FPS,.95),(.5,0),(duration,0)])
                reach=curve(t,[(0,0),(7/FPS,0),(.5,1),(16/FPS,.96),(.95,-.06),(duration,0)])
                recoil=curve(t,[(0,0),(.5,0),(.7,1),(1.,-.15),(duration,0)])
                p['pelvis']=[9*wind-22*reach,-5*wind-4*reach];p['body']=[4*wind-6*reach,-3*wind]
                r['body']=-6*wind+10*reach;r['neck_main']=4*wind-6*reach;r['head_main']=-8*wind-6*reach-4*recoil
                r['head_secondary']=-6*wind+9*recoil;r['neck_secondary']=-3*wind+3*reach
                r['right_arm_upper']=35*wind-85*reach+14*recoil
                r['right_arm_lower']=-20*wind-35*reach+15*recoil;r['right_arm_paw']=15*reach-10*recoil
                r['left_arm_upper']=-15*wind+20*reach;r['left_arm_lower']=12*wind-10*reach
                r['backpack']=3*wind-6*reach+6*recoil;r['waist_bag']=-5*wind+6*recoil;r['chain']=-8*wind+10*recoil
                for key,d in config['limbs'].items():targets[key]=[d['ankle'][0]-20*reach,d['ankle'][1]+3*max(reach,0)]
            elif name=='attack_2':
                raise_arm=curve(t,[(0,0),(.3,1),(.4,1),(19/FPS,0),(duration,0)])
                slash=curve(t,[(0,0),(.4,0),(19/FPS,1),(.72,.96),(1.12,-.04),(duration,0)])
                recoil=curve(t,[(0,0),(19/FPS,0),(.8,1),(1.12,-.15),(duration,0)])
                r['right_arm_upper']=130*raise_arm-65*slash+10*recoil
                r['right_arm_lower']=40*raise_arm+10*slash-20*recoil;r['right_arm_paw']=35*raise_arm+10*slash+12*recoil
                p['pelvis']=[8*raise_arm-18*slash,4*raise_arm-10*slash]
                p['body']=[0,3*raise_arm-4*slash];r['body']=-8*raise_arm+12*slash
                r['neck_main']=4*raise_arm-7*slash;r['head_main']=4*raise_arm-5*slash+4*recoil
                r['neck_secondary']=-4*raise_arm+4*slash;r['head_secondary']=8*raise_arm-5*slash-5*recoil
                r['left_arm_upper']=-15*raise_arm+22*slash;r['left_arm_lower']=12*raise_arm-15*slash
                r['backpack']=5*raise_arm-7*slash+5*recoil;r['waist_bag']=-5*raise_arm+8*recoil;r['chain']=8*raise_arm-10*slash+8*recoil
                for key,d in config['limbs'].items():targets[key]=[d['ankle'][0]-18*slash,d['ankle'][1]+3*max(slash,0)]
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(5/FPS,.8),(.3,-.18),(duration,0)])
                lag=curve(t,[(0,0),(4/FPS,1),(8/FPS,-.2),(duration,0)])
                p['pelvis']=[7*shock,-4*shock];p['body']=[4*shock,-3*shock]
                r['body']=-9*shock;r['neck_main']=-4*lag;r['head_main']=-10*lag
                r['neck_secondary']=-3*shock;r['head_secondary']=-12*lag
                r['right_arm_upper']=16*lag;r['right_arm_lower']=-12*lag;r['right_arm_paw']=10*lag
                r['left_arm_upper']=-12*lag;r['left_arm_lower']=9*lag
                r['backpack']=7*lag;r['waist_bag']=-8*lag;r['chain']=-10*lag
            elif name=='dead':
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)])
                fall=curve(t,[(0,0),(.25,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)])
                impact=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['pelvis']=[-45*fall,-65*fall-8*buckle];r['pelvis']=75*fall
                r['body']=8*fall+4*buckle;r['neck_main']=-6*fall;r['head_main']=-8*fall-4*impact
                r['neck_secondary']=-5*fall;r['head_secondary']=-3*fall+5*impact
                p['head_secondary']=[-9*fall,-2*fall]
                r['left_arm_upper']=35*fall;r['left_arm_lower']=-15*fall;r['left_arm_paw']=10*fall
                r['right_arm_upper']=-25*fall;r['right_arm_lower']=25*fall;r['right_arm_paw']=-20*fall
                r['backpack']=-6*fall;r['waist_bag']=4*fall;r['chain']=20*fall-8*impact
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
        if name=='attack_1':animation['events']=[{'time':.5,'name':'claw_hit'}]
        if name=='attack_2':animation['events']=[{'time':19/FPS,'name':'slash_hit'}]
        if name in ('attack_1','attack_2'):
            animation['drawOrder']=[{'time':0},{'time':.2,'offsets':[{'slot':'youshoubi','offset':7}]},{'time':1.1}]
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
    sheet=Image.new('RGB',(1440,len(DURATIONS)*285),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;image=render(skeleton,images,name,t,background=bg).convert('RGB')
            view=image.resize((840,560),Image.Resampling.LANCZOS)
            ImageDraw.Draw(view).text((12,12),f'{name} / {frame:02d} / {t:.2f}s',fill='#9cdedc');frames.append(view)
            phone=Image.new('RGB',(420,300),'#172027')
            phone.paste(image.resize((384,256),Image.Resampling.LANCZOS),(18,28))
            ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack_1','attack_2'))
        save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=([0,.2,.5,.7] if name=='attack_1' else ([0,.35,19/FPS,.8] if name=='attack_2' else [0,duration*.3,duration*.65,duration]))
        for column,t in enumerate(times):
            image=frames[min(round(t*FPS),len(frames)-1)].resize((360,240),Image.Resampling.LANCZOS)
            sheet.paste(image,(column*360,row*285+25))
        labels.text((12,row*285+6),name,fill='#9cdedc')
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
    correct_ground(skeleton,images);write_json(BASE/'ssjnl-4.3.json',skeleton)
    if previews:make_previews(skeleton,images,BASE/'previews')
    print('Built SSJNL:',len(skeleton['bones']),'bones,',len(skeleton['slots']),'weighted meshes;',list(skeleton['animations']))
    return skeleton


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args();build(not args.no_previews)


if __name__=='__main__':main()
