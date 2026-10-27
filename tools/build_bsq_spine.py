"""Build BSQ's editable region-bone rig and an independent poison projectile."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tools.build_xqdz_spine import (rotation, rounded, interpolate_controls, pose,
                                   render as region_render, save_gif, load_images, attachment_matrix, visible_attachments)

BASE = Path(__file__).resolve().parents[1] / 'examples' / 'bsq-spine'
FPS = 30
DURATIONS = {'stand':2.4,'dead':1.6,'hit':.5,'attack_1':.9,'attack_2':1.1,'move':1.0}
SIZE = (1800,1200)
CAMERA = (500,1080)
SPIT_TIME = 12/FPS
BULLET_DURATION = .4


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def smoothstep(a,b,x):
    t=float(np.clip((x-a)/(b-a),0,1));return t*t*(3-2*t)


def make_body_mesh(source,config,columns=18,rows=22):
    region=source['skins']['default']['shenti']['shenti']
    definitions={b['name']:b for b in config['bones']}
    indices={b['name']:i for i,b in enumerate(config['bones'])}
    perimeter=([(x,0) for x in range(columns+1)] + [(columns,y) for y in range(1,rows+1)]
              + [(x,rows) for x in range(columns-1,-1,-1)] + [(0,y) for y in range(rows-1,0,-1)])
    order=perimeter+[(x,y) for y in range(1,rows) for x in range(1,columns)]
    lookup={point:i for i,point in enumerate(order)};uvs=[];vertices=[];triangles=[]
    for col,row in order:
        u,v=col/columns,row/rows;uvs.extend([u,v])
        wx=region['x']+(u-.5)*region['width'];wy=region['y']+(.5-v)*region['height'];depth=-wy
        torso=1-smoothstep(730,790,depth);back=1-smoothstep(430,480,wx)
        weights={'body':torso}
        for side,share in [('back',back),('front',1-back)]:
            knee=-definitions[side+'_shin']['position'][1];ankle=-definitions[side+'_foot']['position'][1]
            thigh=1-smoothstep(knee-38,knee+38,depth);foot=smoothstep(ankle-25,ankle+15,depth)
            for bone,weight in [(side+'_thigh',thigh),(side+'_shin',max(0,1-thigh-foot)),(side+'_foot',foot)]:
                weights[bone]=share*(1-torso)*weight
        active={name:weight for name,weight in weights.items() if weight>1e-8};total=sum(active.values())
        vertices.append(len(active))
        for name,weight in active.items():
            bone=definitions[name];local=rotation(-bone['rotation']) @ (np.array([wx,wy])-bone['position'])
            vertices.extend([indices[name],rounded(local[0]),rounded(local[1]),weight/total])
    for row in range(rows):
        for col in range(columns):
            a=lookup[(col,row)];b=lookup[(col+1,row)];c=lookup[(col+1,row+1)];d=lookup[(col,row+1)]
            triangles.extend([a,b,c,a,c,d])
    return {'type':'mesh','path':'shenti','width':region['width'],'height':region['height'],
            'uvs':uvs,'triangles':triangles,'vertices':vertices,'hull':len(perimeter)}


def mesh_world(skeleton,attachment,world):
    data=attachment['vertices'];cursor=0;points=[];bones=skeleton['bones']
    while cursor<len(data):
        count=int(data[cursor]);cursor+=1;point=np.zeros(2)
        for _ in range(count):
            index,x,y,weight=data[cursor:cursor+4];cursor+=4
            point+=(world[bones[int(index)]['name']] @ [x,y,1])[:2]*weight
        points.append(point)
    return np.array(points)


def build_rig(source,config,use_mesh=True):
    definitions={b['name']:b for b in config['bones']};origin=np.array(config['origin'],float)
    skeleton={'skeleton':{'spine':config['target_version'],'images':'./images/','fps':FPS,
                          'x':-450,'y':0,'width':900,'height':1000},
              'bones':[],'slots':[],'skins':[],
              'events':{'claw_hit':{'string':'claw_tip'},
                        'spawn_bullet':{'string':'bullet','float':config['projectile_spawn_offset']},
                        'impact':{},'death_end':{},'footstep':{'string':'front'}},'animations':{}}
    for b in config['bones']:
        parent=definitions.get(b.get('parent'))
        position=np.array(b['position'],float)
        local=rotation(-parent['rotation']) @ (position-parent['position']) if parent else position-origin
        entry={'name':b['name'],'x':rounded(local[0]),'y':rounded(local[1]),
               'rotation':rounded(b['rotation']-(parent['rotation'] if parent else 0)),'length':rounded(b['length'])}
        if parent:entry['parent']=parent['name']
        skeleton['bones'].append(entry)
    skin={}
    def add(name,bone,a):
        a=deepcopy(a);b=definitions[bone]
        local=rotation(-b['rotation']) @ (np.array([a['x'],a['y']])-b['position'])
        a.update(x=rounded(local[0]),y=rounded(local[1]),rotation=rounded(-b['rotation']))
        a.pop('bone',None);a.pop('crop_bounds',None)
        skeleton['slots'].append({'name':name,'bone':bone,'attachment':name})
        skin[name]={name:a}
    for slot in source['slots']:
        name=slot['name']
        if name=='shenti' and use_mesh:
            skeleton['slots'].append({'name':'shenti','bone':'body','attachment':'shenti'})
            skin['shenti']={'shenti':make_body_mesh(source,config)}
        else:
            a=deepcopy(source['skins']['default'][name][name]);a['path']=name
            add(name,'body' if name=='shenti' else config['bindings'][name],a)
    skeleton['slots'].append({'name':'spit_fx','bone':'mouth'})
    skin['spit_fx']={f'spit_{i}':{'path':f'fx/spit_{i}','x':70,'y':0,'width':180,'height':120} for i in range(1,4)}
    skeleton['skins']=[{'name':'default','attachments':skin}]
    return skeleton


def curve(t,points):return interpolate_controls(t,points)


def solve_leg(config,side,target,pelvis_offset):
    defs={b['name']:b for b in config['bones']}
    thigh=defs[side+'_thigh'];shin=defs[side+'_shin'];foot=defs[side+'_foot']
    hip=np.array(thigh['position'],float)+pelvis_offset
    vector=np.array(target)-hip;length=np.linalg.norm(vector)
    upper=thigh['length'];lower=shin['length']
    neutral_u=np.array(shin['position'])-thigh['position']
    neutral_v=np.array(foot['position'])-shin['position']
    sign=1 if neutral_u[0]*neutral_v[1]-neutral_u[1]*neutral_v[0]>=0 else -1
    distance=np.clip(length,abs(upper-lower)+1e-5,upper+lower-1e-5)
    bend=sign*math.acos(np.clip((distance**2-upper**2-lower**2)/(2*upper*lower),-1,1))
    angle=math.atan2(vector[1],vector[0])-math.atan2(lower*math.sin(bend),upper+lower*math.cos(bend))
    first=math.degrees(angle);second=first+math.degrees(bend)
    return first-thigh['rotation'], (second-first)-(shin['rotation']-thigh['rotation']), shin['rotation']-second


def make_animations(config,use_mesh=True):
    definitions={b['name']:b for b in config['bones']};animations={}
    for name,duration in DURATIONS.items():
        timelines={bone:{'rotate':[],'translate':[],'scale':[]} for bone in definitions}
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;phase=t/duration*math.tau
            r={bone:0. for bone in definitions};p={bone:[0.,0.] for bone in definitions};s={bone:[1.,1.] for bone in definitions}
            if name=='stand':
                breath=(1-math.cos(phase))/2
                p['body'][1]=12*breath;r['body']=2.5*math.sin(phase)
                r['head']=-8.5*(math.sin(phase-.15)-math.sin(-.15))
                r['sac']=8*(math.sin(phase-.5)-math.sin(-.5))
                s['sac']=[1+.025*breath,1+.035*breath]
                r['near_upper_arm']=4*math.sin(phase);r['near_forearm']=-6*math.sin(phase)
                r['far_arm']=-4*math.sin(phase-.35)+4*math.sin(-.35)
            elif name=='move':
                stride=math.sin(phase)
                drop=-50*(math.sqrt(stride*stride+.03**2)-.03)/(math.sqrt(1+.03**2)-.03)
                p['pelvis']=[0,drop]
                for side,offset in [('back',0),('front',math.pi)]:
                    step=math.sin(phase+offset);swing=max(0,step)**2;lift=36*swing
                    base=definitions[side+'_foot']['position']
                    target=[base[0]+45*step,base[1]+lift]
                    a,b,c=solve_leg(config,side,target,p['pelvis'])
                    r[side+'_thigh']=a;r[side+'_shin']=b;r[side+'_foot']=c+8*swing
                r['body']=3*stride;r['head']=-6*stride
                r['sac']=10*(math.sin(phase-.4)-math.sin(-.4))
                r['near_upper_arm']=6*stride;r['near_forearm']=-8*stride;r['far_arm']=-6*stride
                s['sac']=[1+.015*(1-math.cos(2*phase)),1+.02*(1-math.cos(2*phase))]
                if not use_mesh:
                    # Essentials fallback: preserve the complete source body image.
                    p['pelvis']=[0,0];p['body'][1]=5*(1-math.cos(2*phase))
                    r['body']=1.2*stride;r['head']=-4*stride
                    for side in ('back','front'):
                        r[side+'_thigh']=r[side+'_shin']=r[side+'_foot']=0
            elif name=='attack_1':
                wind=curve(t,[(0,0),(6/FPS,1),(8/FPS,.9),(11/FPS,0),(duration,0)])
                reach=curve(t,[(0,0),(7/FPS,0),(11/FPS,1),(13/FPS,.95),(20/FPS,-.08),(duration,0)])
                settle=curve(t,[(0,0),(11/FPS,0),(16/FPS,1),(duration,0)])
                r['body']=6*wind-10*reach+2*settle
                p['body']=[35*reach-8*wind,-7*wind]
                r['near_upper_arm']=-18*wind+111*reach
                r['near_forearm']=24*wind-119*reach
                s['near_upper_arm']=[1+.08*max(0,reach),1]
                s['near_forearm']=[1+.10*max(0,reach),1]
                r['far_arm']=-15*wind+38*reach
                r['head']=8*wind+13*reach;p['head'][0]=-15*reach
                r['sac']=-5*wind+13*curve(t,[(0,0),(10/FPS,0),(15/FPS,1),(duration,0)])
            elif name=='attack_2':
                charge=curve(t,[(0,0),(8/FPS,1),(10/FPS,1),(12/FPS,0),(duration,0)])
                release=curve(t,[(0,0),(10/FPS,0),(12/FPS,1),(15/FPS,.6),(22/FPS,-.12),(duration,0)])
                lag=curve(t,[(0,0),(12/FPS,0),(16/FPS,1),(duration,0)])
                r['body']=7*charge-8*release;p['body']=[18*release,-5*charge]
                r['head']=12*charge+8*release;p['head']=[32*release,-8*release]
                s['head']=[1+.035*charge-.02*release,1-.015*charge+.035*release]
                s['sac']=[1+.035*charge+.025*lag,1+.055*charge+.035*lag]
                r['sac']=-5*charge+9*lag
                r['near_upper_arm']=-8*charge+15*release;r['near_forearm']=8*charge-12*release
                r['far_arm']=-12*charge+9*release
            elif name=='hit':
                shock=curve(t,[(0,0),(3/FPS,1),(5/FPS,.85),(9/FPS,-.15),(duration,0)])
                lag=curve(t,[(0,0),(4/FPS,1),(8/FPS,-.15),(duration,0)])
                r['body']=15*shock;p['body']=[-16*shock,-7*shock]
                r['head']=12*lag;r['sac']=-13*lag
                r['near_upper_arm']=-20*shock;r['near_forearm']=18*lag;r['far_arm']=15*shock
                s['sac']=[1-.02*shock,1+.03*shock]
            elif name=='dead':
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)])
                fall=curve(t,[(0,0),(.25,0),(.95,1),(1.15,.96),(1.4,1),(duration,1)])
                impact=curve(t,[(0,0),(.85,0),(1.0,1),(1.2,-.2),(1.4,0),(duration,0)])
                p['pelvis']=[100*fall,-150*fall-12*buckle];r['pelvis']=-86*fall
                r['body']=10*buckle-5*fall;r['head']=12*buckle+9*impact+8*fall
                r['sac']=-8*buckle+12*fall-10*impact
                r['near_upper_arm']=20*fall;r['near_forearm']=-20*fall;r['far_arm']=-25*fall
                r['back_thigh']=18*fall;r['back_shin']=-10*fall
                r['front_thigh']=-16*fall;r['front_shin']=12*fall
            for bone in definitions:
                parent=definitions.get(definitions[bone].get('parent'))
                local=rotation(-parent['rotation']) @ p[bone] if parent else p[bone]
                timelines[bone]['rotate'].append({'time':t,'value':rounded(r[bone])})
                timelines[bone]['translate'].append({'time':t,'x':rounded(local[0]),'y':rounded(local[1])})
                timelines[bone]['scale'].append({'time':t,'x':rounded(s[bone][0]),'y':rounded(s[bone][1])})
        for bone in list(timelines):
            for kind in list(timelines[bone]):
                keys=('value',) if kind=='rotate' else ('x','y');neutral=1 if kind=='scale' else 0
                if bone!='root' and all(all(k[key]==neutral for key in keys) for k in timelines[bone][kind]):del timelines[bone][kind]
            if not timelines[bone]:del timelines[bone]
        fx=[{'time':0,'name':None}]
        if name=='attack_2':
            fx += [{'time':12/FPS,'name':'spit_1'},{'time':13/FPS,'name':'spit_2'},
                   {'time':14/FPS,'name':'spit_3'},{'time':16/FPS,'name':None}]
        animation={'bones':timelines,'slots':{'spit_fx':{'attachment':fx}}}
        if name=='attack_1':
            animation['events']=[{'time':11/FPS,'name':'claw_hit'}]
            animation['drawOrder']=[{'time':0},{'time':9/FPS,'offsets':[{'slot':'youshoubi','offset':1}]},{'time':18/FPS}]
        if name=='attack_2':animation['events']=[{'time':SPIT_TIME,'name':'spawn_bullet','string':'bullet','float':config['projectile_spawn_offset']}]
        if name=='hit':animation['events']=[{'time':3/FPS,'name':'impact'}]
        if name=='dead':animation['events']=[{'time':42/FPS,'name':'death_end'}]
        if name=='move':animation['events']=[{'time':.5,'name':'footstep','string':'back'},{'time':1.,'name':'footstep','string':'front'}]
        animations[name]=animation
    return animations


def make_poison(directory):
    directory.mkdir(parents=True,exist_ok=True)
    def save(name,size,draw):
        im=Image.new('RGBA',(size[0]*3,size[1]*3));d=ImageDraw.Draw(im)
        draw(d,3);im.resize(size,Image.Resampling.LANCZOS).save(directory/(name+'.png'))
    def core(d,s):
        d.polygon([(8*s,50*s),(65*s,22*s),(95*s,45*s),(65*s,78*s)],fill='#88a83c')
        d.ellipse((48*s,17*s,156*s,84*s),fill='#a7c84d',outline='#405b27',width=3*s)
        d.ellipse((84*s,25*s,142*s,65*s),fill='#c9df64')
        d.ellipse((102*s,27*s,134*s,40*s),fill='#e5ee9a')
        d.ellipse((67*s,52*s,80*s,65*s),fill='#6c982e')
    save('poison_core',(160,100),core)
    def tail(d,s):
        d.polygon([(0,29*s),(125*s,4*s),(145*s,28*s),(125*s,51*s)],fill='#7e9b36')
        d.polygon([(20*s,29*s),(128*s,14*s),(137*s,30*s),(126*s,42*s)],fill='#b8d456')
    save('poison_tail',(150,56),tail)
    def drop(d,s):
        d.ellipse((4*s,4*s,29*s,27*s),fill='#b5d555',outline='#4c6d2c',width=2*s)
        d.ellipse((14*s,7*s,24*s,13*s),fill='#e4ee9d')
    save('poison_drop',(32,32),drop)
    for i in range(1,4):
        def spit(d,s,i=i):
            d.polygon([(0,60*s),(45*s,43*s),(115*s,(28+i*5)*s),(175*s,62*s),
                       (115*s,92*s),(50*s,78*s)],fill=(137,174,59,210))
            d.ellipse((20*s,45*s,100*s,80*s),fill=(195,220,93,245))
            for x,y,r in [(120,25+i*4,9),(151,72-i*4,8),(82,96,7)]:
                d.ellipse(((x-r)*s,(y-r)*s,(x+r)*s,(y+r)*s),fill='#bbd95e')
        save('spit_'+str(i),(180,120),spit)


def build_bullet(version):
    result={'skeleton':{'spine':version,'images':'./images/','fps':FPS,'x':-160,'y':-80,'width':250,'height':160},
      'bones':[{'name':'root'},{'name':'core','parent':'root'},
               {'name':'tail','parent':'root','x':-75},{'name':'drop_top','parent':'root','x':-85,'y':25},
               {'name':'drop_bottom','parent':'root','x':-120,'y':-24}],
      'slots':[{'name':'tail','bone':'tail','attachment':'tail'},
               {'name':'drop_top','bone':'drop_top','attachment':'drop'},
               {'name':'drop_bottom','bone':'drop_bottom','attachment':'drop'},
               {'name':'core','bone':'core','attachment':'core'}],
      'skins':[{'name':'default','attachments':{
          'tail':{'tail':{'path':'fx/poison_tail','x':-20,'width':150,'height':56}},
          'drop_top':{'drop':{'path':'fx/poison_drop','width':22,'height':22}},
          'drop_bottom':{'drop':{'path':'fx/poison_drop','width':16,'height':16}},
          'core':{'core':{'path':'fx/poison_core','x':-10,'width':160,'height':100}}}}],
      'animations':{'bullet':{'bones':{}}}}
    tracks=result['animations']['bullet']['bones']
    for name in ('root','core','tail','drop_top','drop_bottom'):
        tracks[name]={'rotate':[],'translate':[],'scale':[]}
        for frame in range(13):
            t=frame/FPS;phase=t/BULLET_DURATION*math.tau
            angle=6*math.sin(phase) if name=='core' else (10*math.sin(phase) if name=='tail' else 0)
            x=8*math.sin(phase) if name.startswith('drop_') else 0
            y=(7 if name=='drop_top' else -7)*math.sin(phase) if name.startswith('drop_') else 0
            sx=1+.10*math.sin(phase) if name in ('core','tail') else 1
            sy=1-.07*math.sin(phase) if name=='core' else 1
            tracks[name]['rotate'].append({'time':t,'value':rounded(angle)})
            tracks[name]['translate'].append({'time':t,'x':rounded(x),'y':rounded(y)})
            tracks[name]['scale'].append({'time':t,'x':rounded(sx),'y':rounded(sy)})
    return result


def draw_order(skeleton,animation,t):
    slots=list(skeleton['slots'])
    selected={}
    for key in skeleton['animations'].get(animation,{}).get('drawOrder',[]):
        if key.get('time',0)<=t+1e-6:selected=key
    offsets=selected.get('offsets',[])
    if offsets:
        # Offsets refer to setup indices, including when several slots move together.
        order=[None]*len(slots);unchanged=[];cursor=0
        for item in offsets:
            index=next(i for i,s in enumerate(slots) if s['name']==item['slot'])
            unchanged.extend(slots[cursor:index])
            order[index+item['offset']]=slots[index];cursor=index+1
        unchanged.extend(slots[cursor:]);remaining=iter(unchanged)
        slots=[slot if slot is not None else next(remaining) for slot in order]
    return slots


def mesh_tile(skeleton,attachment,image,world,camera,size):
    points=mesh_world(skeleton,attachment,world)
    screen=np.column_stack((points[:,0]+camera[0],camera[1]-points[:,1]))
    x0,y0=np.floor(screen.min(axis=0)).astype(int);x1,y1=np.ceil(screen.max(axis=0)).astype(int)
    x0=max(0,x0);y0=max(0,y0);x1=min(size[0],x1);y1=min(size[1],y1)
    if x1<=x0 or y1<=y0:return None
    map_x=np.full((y1-y0,x1-x0),-100,dtype=np.float32);map_y=map_x.copy()
    source=np.array(attachment['uvs']).reshape(-1,2)*[image.width,image.height]
    for triangle in np.array(attachment['triangles']).reshape(-1,3):
        target=screen[triangle]-[x0,y0];texture=source[triangle]
        left,top=np.floor(target.min(axis=0)).astype(int);right,bottom=np.ceil(target.max(axis=0)).astype(int)+1
        left=max(0,left);top=max(0,top);right=min(x1-x0,right);bottom=min(y1-y0,bottom)
        if left>=right or top>=bottom:continue
        transform=cv2.getAffineTransform(target.astype(np.float32),texture.astype(np.float32))
        yy,xx=np.mgrid[top:bottom,left:right];xx=xx+.5;yy=yy+.5
        mask=np.zeros((bottom-top,right-left),np.uint8)
        cv2.fillConvexPoly(mask,np.rint((target-[left,top])*16).astype(np.int32),1,shift=4)
        u=transform[0,0]*xx+transform[0,1]*yy+transform[0,2]-.5
        v=transform[1,0]*xx+transform[1,1]*yy+transform[1,2]-.5
        map_x[top:bottom,left:right][mask>0]=u[mask>0]
        map_y[top:bottom,left:right][mask>0]=v[mask>0]
    tile=cv2.remap(np.asarray(image),map_x,map_y,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    return Image.fromarray(tile),(x0,y0)


def render(skeleton,images,animation=None,t=0,size=SIZE,camera=CAMERA,background=None,rig=False):
    out=background.copy() if background is not None else Image.new('RGBA',size)
    world=pose(skeleton,animation,t)
    view={**skeleton,'slots':draw_order(skeleton,animation,t)}
    screen=np.array([[1,0,camera[0]],[0,-1,camera[1]],[0,0,1.]])
    for slot,attachment in visible_attachments(view,animation,t):
        image=images[attachment.get('path',slot['name'])]
        if attachment.get('type')=='mesh':
            result=mesh_tile(skeleton,attachment,image,world,camera,size)
            if result:out.alpha_composite(*result)
        else:
            transform=screen @ attachment_matrix(world,slot,attachment,image)
            corners=transform @ np.array([[0,image.width,image.width,0],[0,0,image.height,image.height],[1,1,1,1]])
            x0,y0=np.floor(corners[:2].min(axis=1)).astype(int);x1,y1=np.ceil(corners[:2].max(axis=1)).astype(int)
            x0=max(0,x0);y0=max(0,y0);x1=min(size[0],x1);y1=min(size[1],y1)
            if x1<=x0 or y1<=y0:continue
            inverse=np.linalg.inv(transform);offset=np.array([[1,0,x0],[0,1,y0],[0,0,1]])
            tile=image.transform((x1-x0,y1-y0),Image.Transform.AFFINE,(inverse@offset)[:2].ravel(),Image.Resampling.BICUBIC)
            out.alpha_composite(tile,(x0,y0))
    if rig:
        draw=ImageDraw.Draw(out)
        for b in skeleton['bones']:
            point=screen@world[b['name']];x,y=point[:2,2]
            if b.get('parent'):
                p=screen@world[b['parent']];draw.line((p[0,2],p[1,2],x,y),fill='#5fffc2',width=2)
            draw.ellipse((x-4,y-4,x+4,y+4),fill='#ffcf75');draw.text((x+6,y-14),b['name'],fill='#ffcf75')
    return out


def correct_ground(skeleton,images):
    track=skeleton['animations']['dead']['bones']['root']['translate']
    frame=render(skeleton,images,'dead',DURATIONS['dead'],size=(3000,2400),camera=(1200,1800))
    correction=frame.getbbox()[3]-1800
    for key in track:key['y']=rounded(correction*curve(key['time'],[(0,0),(.25,0),(.95,1),(1.6,1)]))
    for key in track:
        frame=render(skeleton,images,'dead',key['time'],size=(3000,2400),camera=(1200,1800))
        key['y']=rounded(key['y']+max(0,frame.getbbox()[3]-1800))


def spawn_transform(skeleton,t,offset=96):
    matrix=pose(skeleton,'attack_2',t)['mouth']
    position=matrix @ [offset,0,1]
    angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
    return position[:2],angle


def projectile_in_scene(role,bullet,images,t):
    if t<SPIT_TIME:return None
    initial,angle=spawn_transform(role,SPIT_TIME)
    elapsed=t-SPIT_TIME
    velocity=rotation(angle) @ [900*elapsed,0]
    actor=deepcopy(bullet);actor['bones'][0].update(x=rounded(initial[0]+velocity[0]),y=rounded(initial[1]+velocity[1]),rotation=rounded(angle))
    return render(actor,images,'bullet',elapsed%BULLET_DURATION)


def make_previews(skeleton,bullet,images,output):
    output.mkdir(parents=True,exist_ok=True)
    bg=Image.new('RGBA',SIZE,'#28343c');d=ImageDraw.Draw(bg)
    for x in range(0,SIZE[0],32):
        for y in range(0,SIZE[1],32):
            if (x//32+y//32)%2:d.rectangle((x,y,x+31,y+31),fill='#2d3b43')
    d.line((0,CAMERA[1],SIZE[0],CAMERA[1]),fill='#67927c',width=2)
    sheet=Image.new('RGB',(1440,len(DURATIONS)*285),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS
            full=render(skeleton,images,name,t,background=bg)
            if name=='attack_2':
                projectile=projectile_in_scene(skeleton,bullet,images,t)
                if projectile:full.alpha_composite(projectile)
            image=full.convert('RGB').resize((720,480),Image.Resampling.LANCZOS)
            ImageDraw.Draw(image).text((12,12),f'{name} / {frame:02d} / {t:.2f}s',fill='#b4e0a3')
            frames.append(image)
            phone=Image.new('RGB',(320,206),'#172027')
            phone.paste(full.convert('RGB').resize((241,161),Image.Resampling.LANCZOS),(25,30))
            ImageDraw.Draw(phone).text((9,9),f'{name} / character ~128px',fill='#b4e0a3');mobile.append(phone)
        save_gif(frames,output/(name+'.gif'),name in ('stand','move','attack_1','attack_2'))
        save_gif(mobile,output/('mobile-'+name+'.gif'),True)
        times=[0,.2,.4,.9] if name=='attack_1' else ([0,.3,.4,.65] if name=='attack_2' else [0,duration*.3,duration*.65,duration])
        for column,t in enumerate(times):
            frame=frames[min(round(t*FPS),len(frames)-1)].resize((360,240),Image.Resampling.LANCZOS)
            sheet.paste(frame,(column*360,row*285+25))
        labels.text((12,row*285+6),name,fill='#b4e0a3')
    sheet.save(output/'animation-contact.png')
    render(skeleton,images,rig=True,background=bg).convert('RGB').save(output/'rig-setup.png')
    frames=[]
    for frame in range(25):
        # Presentation-only displacement; bullet JSON's root stays stationary.
        actor=deepcopy(bullet);actor['bones'][0]['x']=-200+frame*18
        frame_image=render(actor,images,'bullet',(frame/FPS)%BULLET_DURATION,size=(640,240),camera=(280,120),background=Image.new('RGBA',(640,240),'#25343a'))
        ImageDraw.Draw(frame_image).text((12,12),'bullet / flight demo (translation by game code)',fill='#b4e0a3')
        frames.append(frame_image.convert('RGB'))
    save_gif(frames,output/'bullet.gif',True)


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text(encoding='utf8'))
    config=json.loads((BASE/'rig-config.json').read_text(encoding='utf8'))
    directory=BASE/'images'
    for name,attachments in source['skins']['default'].items():
        image=Image.open(directory/(name+'.png'));a=attachments[name]
        if image.size!=(a['width'],a['height']):raise ValueError('Image dimensions mismatch: '+name)
    make_poison(directory/'fx')
    skeleton=build_rig(source,config);skeleton['animations']=make_animations(config)
    fallback=build_rig(source,config,False);fallback['animations']=make_animations(config,False)
    bullet=build_bullet(config['target_version']);images=load_images(directory)
    correct_ground(skeleton,images);correct_ground(fallback,images)
    write_json(BASE/'bsq-4.3.json',skeleton);write_json(BASE/'bullet-4.3.json',bullet);write_json(BASE/'bsq-regions-4.3.json',fallback)
    if previews:make_previews(skeleton,bullet,images,BASE/'previews')
    print('Built BSQ:',len(skeleton['bones']),'bones;',len(skeleton['slots']),'slots;',list(skeleton['animations']))
    print('Independent projectile:',list(bullet['animations']))
    return skeleton,bullet


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args();build(not args.no_previews)


if __name__=='__main__':main()
