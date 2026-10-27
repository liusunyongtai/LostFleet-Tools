"""Spine PRO hovering saw robot: projected disc spin and detachable fork throw."""
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
from tools.build_xqdz_spine import rotation,rounded,interpolate_controls,pose,load_images,save_gif,visible_attachments
from tools.build_bsq_spine import mesh_world,render as shared_render,write_json,draw_order
from tools.build_btmg_spine import grid
BASE=Path(__file__).resolve().parents[1]/'examples'/'xd-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'attack':1.2,'move':1.2}
SIZE=(1400,900)
CAMERA=(720,780)
RELEASE=15/FPS


def curve(t,points):return interpolate_controls(t,points)


def build_rig(source,config):
    definitions={b['name']:b for b in config['bones']};origin=np.array(config['origin'],float)
    rig={'skeleton':{'spine':config['target_version'],'images':'./images/','fps':FPS,'x':-650,'y':0,'width':1000,'height':750},
         'bones':[],'slots':[],'skins':[{'name':'default','attachments':{}}],
         'events':{'throw_fork':{'string':'fork_flight'},'impact':{},'death_end':{}},'animations':{}}
    for b in config['bones']:
        parent=definitions.get(b.get('parent'));position=np.array(b['position'],float)
        local=rotation(-parent['rotation'])@(position-parent['position']) if parent else position-origin
        entry={'name':b['name'],'x':rounded(local[0]),'y':rounded(local[1]),'rotation':rounded(b['rotation']-(parent['rotation'] if parent else 0)),'length':rounded(b['length'])}
        if parent:entry['parent']=parent['name']
        for key in ('scaleX','scaleY'):
            if key in b:entry[key]=b[key]
        rig['bones'].append(entry)
    rest=pose(rig);indices={b['name']:i for i,b in enumerate(rig['bones'])}
    def add(name,bone,a,visible=True,path=None):
        points,triangles,hull=grid(1,1);vertices=[];uvs=[];inverse=np.linalg.inv(rest[bone])
        for col,row in points:
            u,v=float(col),float(row);uvs.extend([u,v])
            source_point=np.array([a['x']+(u-.5)*a['width'],a['y']+(.5-v)*a['height']])-origin
            local=inverse@[source_point[0],source_point[1],1]
            vertices.extend([1,indices[bone],rounded(local[0]),rounded(local[1]),1.])
        attachment={'type':'mesh','path':path or a.get('path',name),'width':a['width'],'height':a['height'],'uvs':uvs,'vertices':vertices,'triangles':triangles,'hull':hull}
        slot={'name':name,'bone':bone}
        if visible:slot['attachment']=name
        rig['slots'].append(slot);rig['skins'][0]['attachments'][name]={name:attachment}
    for slot in source['slots']:
        name=slot['name'];a=source['skins']['default'][name][name]
        add(name,config['bindings'][name],a)
    add('fork_flying','fork_flight',config['projectile_region'],False)
    return rig


def make_animations(rig,config):
    definitions={b['name']:b for b in config['bones']};animations={}
    for name,duration in DURATIONS.items():
        timelines={b:{'rotate':[],'translate':[],'scale':[]} for b in definitions}
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;phase=t/duration*math.tau
            r={b:0. for b in definitions};p={b:[0.,0.] for b in definitions};s={b:[1.,1.] for b in definitions}
            if name=='stand':
                breath=(1-math.cos(phase))/2;p['hover'][1]=13*math.sin(phase)
                r['body']=2.5*math.sin(phase);r['head']=6*math.sin(phase-.15)-6*math.sin(-.15)
                s['body']=[1+.035*breath,1+.025*breath];s['upper_column'][0]=1+.045*breath
                r['back_upper_arm']=6*math.sin(phase);r['back_forearm']=-9*math.sin(phase)
                r['front_upper_arm']=-6*math.sin(phase);r['front_forearm']=9*math.sin(phase)
                r['disc_spin']=720*t/duration
            elif name=='move':
                p['hover'][1]=16*math.sin(phase);r['lower_column']=3;r['upper_column']=4
                r['body']=16+4*math.sin(phase);p['body'][0]=-6
                r['neck']=-5;r['head']=-6-4*math.sin(phase)
                r['back_upper_arm']=9*math.sin(phase);r['back_forearm']=-12*math.sin(phase)
                r['front_upper_arm']=-10*math.sin(phase);r['front_forearm']=12*math.sin(phase)
                r['disc_spin']=360*t/duration
            elif name=='attack':
                wind=curve(t,[(0,0),(.2,1),(7/FPS,.95),(RELEASE,0),(duration,0)])
                reach=curve(t,[(0,0),(7/FPS,0),(RELEASE,1),(16/FPS,.98),(.9,-.08),(duration,0)])
                recoil=curve(t,[(0,0),(RELEASE,0),(19/FPS,1),(.95,-.15),(duration,0)])
                p['hover']=[10*wind-15*reach,5*wind-8*reach]
                r['lower_column']=-3*wind;r['upper_column']=-5*wind+4*reach
                r['body']=-12*wind+16*reach;r['neck']=5*wind-7*reach;r['head']=4*wind-5*reach-5*recoil
                r['front_upper_arm']=45*wind-108*reach+12*recoil
                r['front_forearm']=-65*wind+164*reach-10*recoil;r['fork_mount']=10*wind
                r['back_upper_arm']=-25*wind+30*reach;r['back_forearm']=30*wind-20*reach
                r['disc_spin']=360*t/duration
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(5/FPS,.8),(.3,-.2),(duration,0)])
                lag=curve(t,[(0,0),(4/FPS,1),(8/FPS,-.2),(duration,0)])
                p['hover']=[12*shock,6*shock];r['upper_column']=-6*shock;r['body']=-10*shock;r['head']=-13*lag
                r['front_upper_arm']=22*lag;r['front_forearm']=-18*lag
                r['back_upper_arm']=-18*lag;r['back_forearm']=15*lag
                r['disc_spin']=360*t/duration
            else:
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)])
                fall=curve(t,[(0,0),(.25,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)])
                impact=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['hover']=[-20*fall,-75*fall-8*buckle]
                r['lower_column']=30*fall;r['upper_column']=35*fall;r['body']=15*fall+6*buckle
                r['neck']=-5*fall;r['head']=-10*fall-6*impact
                r['front_upper_arm']=-38*fall;r['front_forearm']=-65*fall
                r['back_upper_arm']=25*fall;r['back_forearm']=-35*fall
                r['disc_spin']=curve(t,[(0,0),(.25,75),(.6,150),(1.,185),(1.4,195),(duration,195)])
            for b in definitions:
                parent=definitions.get(definitions[b].get('parent'));local=rotation(-parent['rotation'])@p[b] if parent else p[b]
                timelines[b]['rotate'].append({'time':t,'value':rounded(r[b])})
                timelines[b]['translate'].append({'time':t,'x':rounded(local[0]),'y':rounded(local[1])})
                timelines[b]['scale'].append({'time':t,'x':rounded(s[b][0]),'y':rounded(s[b][1])})
        for b in list(timelines):
            for kind in list(timelines[b]):
                keys=('value',) if kind=='rotate' else ('x','y');neutral=1 if kind=='scale' else 0
                if b!='root' and all(all(k[key]==neutral for key in keys) for k in timelines[b][kind]):del timelines[b][kind]
            if not timelines[b]:del timelines[b]
        animation={'bones':timelines,'slots':{
            'youshou':{'attachment':[{'time':0,'name':'youshou'}]},
            'fork_flying':{'attachment':[{'time':0,'name':None}]}}}
        if name=='attack':
            animation['events']=[{'time':RELEASE,'name':'throw_fork'}]
            animation['slots']['youshou']['attachment'] += [{'time':RELEASE,'name':None},{'time':config['refill_time'],'name':'youshou'}]
            animation['slots']['fork_flying']['attachment'] += [{'time':RELEASE,'name':'fork_flying'},{'time':config['flight_end'],'name':None}]
            animation['drawOrder']=[{'time':0},{'time':7/FPS,'offsets':[{'slot':'youbi','offset':1},{'slot':'youshou','offset':1}]},{'time':1.}]
        if name=='hit':animation['events']=[{'time':.1,'name':'impact'}]
        if name=='dead':animation['events']=[{'time':1.4,'name':'death_end'}]
        animations[name]=animation
    return animations


def add_fork_flight(skeleton,config):
    world=pose(skeleton,'attack',RELEASE);bone=next(b for b in config['bones'] if b['name']=='fork_mount')
    region=config['projectile_region'];local=rotation(-bone['rotation'])@(np.array([region['x'],region['y']])-bone['position'])
    centre=(world['fork_mount']@[local[0],local[1],1])[:2]
    angle=math.degrees(math.atan2(world['fork_mount'][1,0],world['fork_mount'][0,0]))-bone['rotation']
    neutral=np.array([region['x'],region['y']])-config['origin']
    tracks={'rotate':[],'translate':[]}
    for frame in range(round(DURATIONS['attack']*FPS)+1):
        t=frame/FPS;u=float(np.clip((t-RELEASE)/(config['flight_end']-RELEASE),0,1))
        position=centre+[-550*u,90*math.sin(math.pi*u)-25*u]-neutral
        tracks['rotate'].append({'time':t,'value':rounded(angle+360*u)})
        tracks['translate'].append({'time':t,'x':rounded(position[0]),'y':rounded(position[1])})
    skeleton['animations']['attack']['bones']['fork_flight']=tracks


def render(skeleton,images,animation=None,t=0,size=SIZE,camera=CAMERA,background=None,rig=False):
    return shared_render(skeleton,images,animation,t,size,camera,background,rig)


def correct_ground(skeleton,images):
    track=skeleton['animations']['dead']['bones']['root']['translate']
    image=render(skeleton,images,'dead',DURATIONS['dead'],size=(2200,1700),camera=(1000,1200))
    correction=image.getbbox()[3]-1200
    for key in track:key['y']=rounded(correction*curve(key['time'],[(0,0),(.25,0),(.95,1),(1.6,1)]))
    for key in track:
        image=render(skeleton,images,'dead',key['time'],size=(2200,1700),camera=(1000,1200))
        key['y']=rounded(key['y']+max(0,image.getbbox()[3]-1200))


def make_previews(skeleton,images,directory):
    directory.mkdir(parents=True,exist_ok=True);bg=Image.new('RGBA',SIZE,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,SIZE[0],24):
        for y in range(0,SIZE[1],24):
            if (x//24+y//24)%2:d.rectangle((x,y,x+23,y+23),fill='#2d3d47')
    d.line((0,CAMERA[1],SIZE[0],CAMERA[1]),fill='#6e9f90',width=1)
    sheet=Image.new('RGB',(1440,len(DURATIONS)*265),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;image=render(skeleton,images,name,t,background=bg).convert('RGB')
            view=image.resize((840,540),Image.Resampling.LANCZOS)
            ImageDraw.Draw(view).text((12,12),f'{name} / {frame:02d} / {t:.2f}s',fill='#9cdedc');frames.append(view)
            phone=Image.new('RGB',(420,264),'#172027');phone.paste(image.resize((350,225),Image.Resampling.LANCZOS),(20,28))
            ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack'))
        save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=[0,.2,RELEASE,.7] if name=='attack' else [0,duration*.3,duration*.65,duration]
        for column,t in enumerate(times):
            image=frames[min(round(t*FPS),len(frames)-1)].resize((360,231),Image.Resampling.LANCZOS);sheet.paste(image,(column*360,row*265+25))
        labels.text((12,row*265+6),name,fill='#9cdedc')
    sheet.save(directory/'animation-contact.png');render(skeleton,images,rig=True,background=bg).convert('RGB').save(directory/'rig-setup.png')


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text());config=json.loads((BASE/'rig-config.json').read_text());images=load_images(BASE/'images')
    for n,a in source['skins']['default'].items():
        if images[n].size!=(a[n]['width'],a[n]['height']):raise ValueError('Image size mismatch '+n)
    rig=build_rig(source,config);rig['animations']=make_animations(rig,config);add_fork_flight(rig,config);correct_ground(rig,images)
    write_json(BASE/'xd-4.3.json',rig)
    if previews:make_previews(rig,images,BASE/'previews')
    print('Built XD:',len(rig['bones']),'bones,',len(rig['slots']),'mesh slots;',list(rig['animations']))
    return rig


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true');args=parser.parse_args();build(not args.no_previews)
