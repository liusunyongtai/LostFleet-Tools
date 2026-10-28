"""Bird112 winged creature: flowing wing/tail meshes, airborne idle, flight and persistent dive loop."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.build_xqdz_spine import rotation,rounded,pose,load_images,save_gif,interpolate_controls
from tools.build_bsq_spine import mesh_world,render as shared_render,smoothstep,write_json
from tools.build_jbjs_spine import grid,nearest_arc
from tools.spine_mesh_topology import perimeter_edges,editor_mesh_check
BASE=Path(__file__).resolve().parents[1]/'examples/bird112-spine'
FPS=30
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'move':.8,'attack':.8}
SIZE=(1800,1400)
CAMERA=(950,1250)


def curve(t,points):return interpolate_controls(t,points)


def weights(name,point,config):
    x,y=point
    if name=='shenti':
        if x<690:return {'body':1.}
        chain=config['tail_chain'];arc=nearest_arc(np.array(point),chain);knots=[0]
        for a,b in zip(chain,chain[1:]):knots.append(knots[-1]+math.dist(a,b))
        for i,(a,b) in enumerate(zip(knots,knots[1:])):
            if arc<=b:
                u=smoothstep(a,b,arc);out={}
                for n,w in [(config['tail_bones'][i],1-u),(config['tail_bones'][i+1],u)]:out[n]=out.get(n,0)+w
                blend=smoothstep(690,750,x);out={n:w*blend for n,w in out.items()};out['body']=out.get('body',0)+1-blend
                return out
        return {'tail_curl':1.}
    if name=='zuochibang':
        mid=smoothstep(620,805,x);tip=smoothstep(830,990,x)
        return {'front_wing':1-mid,'front_wing_elbow':mid-tip,'front_wing_tip':tip}
    if name=='youchibang':
        mid=smoothstep(20,170,425-x);tip=smoothstep(200,340,425-x)
        return {'back_wing':1-mid,'back_wing_elbow':mid-tip,'back_wing_tip':tip}
    if name in ('zuoxiae','youxiae'):
        n='front_jaw' if name=='zuoxiae' else 'back_jaw';tip=smoothstep(455,595,-y);return {n:1-tip,n+'_tip':tip}
    return {config['bindings'][name]:1.}


def build_rig(source,config):
    definitions={b['name']:b for b in config['bones']};origin=np.array(config['origin'],float)
    result={'skeleton':{'spine':'4.3.23','images':'./images/','fps':FPS,'x':-1000,'y':-80,'width':2000,'height':1500},'bones':[],'slots':[],'skins':[{'name':'default','attachments':{}}],'events':{'impact':{},'death_end':{}},'animations':{}}
    for b in config['bones']:
        parent=definitions.get(b.get('parent'));pos=np.array(b['position']);local=pos-parent['position'] if parent else pos-origin
        entry={'name':b['name'],'x':rounded(local[0]),'y':rounded(local[1]),'length':b['length']}
        if parent:entry['parent']=parent['name']
        result['bones'].append(entry)
    rest=pose(result);indices={b['name']:i for i,b in enumerate(result['bones'])}
    for slot in source['slots']:
        n=slot['name'];a=source['skins']['default'][n][n];cols,rows={'shenti':(24,12),'zuochibang':(20,12),'youchibang':(18,12),'zuoxiae':(10,20),'youxiae':(10,20)}.get(n,(1,1))
        points,triangles,hull=grid(cols,rows);vertices=[];uvs=[]
        for col,row in points:
            u,v=col/cols,row/rows;uvs.extend([u,v]);world=np.array([a['x']+(u-.5)*a['width'],a['y']+(.5-v)*a['height']]);active={k:w for k,w in weights(n,world,config).items() if w>1e-8};total=sum(active.values());vertices.append(len(active))
            for bone,w in active.items():
                local=world-definitions[bone]['position'];vertices.extend([indices[bone],rounded(local[0]),rounded(local[1]),w/total])
        attachment={'type':'mesh','path':n,'width':a['width'],'height':a['height'],'uvs':uvs,'vertices':vertices,'triangles':triangles,'hull':hull,'edges':perimeter_edges(hull)}
        errors=editor_mesh_check(attachment)
        if errors:raise ValueError(f'{n}: {errors}')
        result['slots'].append({'name':n,'bone':config['bindings'][n],'attachment':n});result['skins'][0]['attachments'][n]={n:attachment}
    return result


def make_animations(rig,config):
    names=[b['name'] for b in rig['bones']];animations={}
    for name,duration in DURATIONS.items():
        tracks={n:{'rotate':[],'translate':[],'scale':[]} for n in names}
        for f in range(round(duration*FPS)+1):
            t=f/FPS;phase=t/duration*math.tau;r={n:0. for n in names};p={n:[0.,0.] for n in names};s={n:[1.,1.] for n in names}
            if name=='stand':
                flap=phase*2;breath=(1-math.cos(phase))/2;p['hover'][1]=18*math.sin(flap);r['body']=2*math.sin(phase);r['head']=4*math.sin(phase)
                r['front_wing']=30*math.sin(flap);r['back_wing']=-30*math.sin(flap)
                r['front_wing_elbow']=-13*(math.sin(flap-.5)-math.sin(-.5));r['back_wing_elbow']=13*(math.sin(flap-.5)-math.sin(-.5))
                r['front_wing_tip']=-8*(math.sin(flap-.8)-math.sin(-.8));r['back_wing_tip']=8*(math.sin(flap-.8)-math.sin(-.8))
                s['body']=[1+.015*breath,1+.025*breath]
                for i,n in enumerate(('tail_base','tail_mid','tail_tip','tail_curl')):r[n]=(4+i)*(math.sin(phase-.3*i)-math.sin(-.3*i))
                r['front_jaw']=4*math.sin(phase);r['back_jaw']=-4*math.sin(phase);r['front_jaw_tip']=-3*math.sin(phase-.2)+3*math.sin(-.2);r['back_jaw_tip']=3*math.sin(phase-.2)-3*math.sin(-.2)
            elif name=='move':
                p['hover'][1]=15*math.sin(phase);r['body']=8+3*math.sin(phase);r['neck']=-2*math.sin(phase);r['head']=-4+3*math.sin(phase)
                r['front_wing']=40*math.sin(phase);r['back_wing']=-40*math.sin(phase)
                r['front_wing_elbow']=-17*math.sin(phase-.45);r['back_wing_elbow']=17*math.sin(phase-.45)
                r['front_wing_tip']=-10*math.sin(phase-.8);r['back_wing_tip']=10*math.sin(phase-.8)
                for i,n in enumerate(('tail_base','tail_mid','tail_tip','tail_curl')):r[n]=-(5+i)*math.sin(phase-.3*i)
                r['front_jaw']=5*math.sin(phase-.35);r['back_jaw']=-5*math.sin(phase-.35);r['front_jaw_tip']=-4*math.sin(phase-.6);r['back_jaw_tip']=4*math.sin(phase-.6)
            elif name=='attack':
                p['hover'][1]=6*math.sin(phase);r['body']=36+4*math.sin(phase);r['neck']=-2;r['head']=-4+2*math.sin(phase-.25)
                r['front_wing']=-18+9*math.sin(phase);r['back_wing']=20-9*math.sin(phase)
                r['front_wing_elbow']=5*math.sin(phase-.4);r['back_wing_elbow']=-5*math.sin(phase-.4)
                r['front_wing_tip']=4*math.sin(phase-.7);r['back_wing_tip']=-4*math.sin(phase-.7)
                s['front_wing']=[.62,.82];s['back_wing']=[.58,.82]
                for i,n in enumerate(('tail_base','tail_mid','tail_tip','tail_curl')):r[n]=-3-(4+i)*math.sin(phase-.3*i)
                r['front_jaw']=-8+3*math.sin(phase-.3);r['back_jaw']=8-3*math.sin(phase-.3);r['front_jaw_tip']=-3*math.sin(phase-.5);r['back_jaw_tip']=3*math.sin(phase-.5)
                r['front_fang']=-8;r['back_fang']=-8
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(.17,.8),(.32,-.2),(duration,0)]);lag=curve(t,[(0,0),(.15,1),(.3,-.25),(duration,0)])
                p['hover']=[30*shock,10*shock];r['body']=-14*shock;r['neck']=5*lag;r['head']=8*lag
                r['front_wing']=45*shock;r['back_wing']=-45*shock;r['front_wing_elbow']=-20*lag;r['back_wing_elbow']=20*lag
                for i,n in enumerate(('tail_base','tail_mid','tail_tip','tail_curl')):r[n]=-(8+i*2)*lag
                r['front_jaw']=14*lag;r['back_jaw']=-14*lag
            elif name=='dead':
                fall=curve(t,[(0,0),(.2,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)]);shock=curve(t,[(0,0),(.1,1),(.3,0),(duration,0)]);impact=curve(t,[(0,0),(.88,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['hover']=[25*fall,-520*fall+20*shock];r['body']=8*fall-10*shock+6*impact;r['neck']=-3*fall;r['head']=-5*fall-5*impact
                r['front_wing']=15*fall+20*shock;r['back_wing']=-15*fall-20*shock;r['front_wing_elbow']=30*fall;r['back_wing_elbow']=-30*fall;r['front_wing_tip']=15*fall;r['back_wing_tip']=-15*fall
                s['front_wing']=[1-.35*fall,1];s['back_wing']=[1-.45*fall,1]
                r['front_jaw']=125*fall;r['back_jaw']=130*fall;r['front_jaw_tip']=-10*fall;r['back_jaw_tip']=10*fall;r['front_fang']=95*fall;r['back_fang']=95*fall
                for i,n in enumerate(('tail_base','tail_mid','tail_tip','tail_curl')):r[n]=(-6+2*i)*fall+(6+i*2)*impact
            for i,n in enumerate(('spine_01','spine_02','spine_03')):
                if name in ('stand','move','attack'):r[n]=(4+i*2)*(math.sin(phase-.25*i)-math.sin(-.25*i))
                elif name=='hit':r[n]=-(6+i*2)*lag
                else:r[n]=(6+i*2)*impact
            for n in names:
                tracks[n]['rotate'].append({'time':t,'value':rounded(r[n])});tracks[n]['translate'].append({'time':t,'x':rounded(p[n][0]),'y':rounded(p[n][1])});tracks[n]['scale'].append({'time':t,'x':rounded(s[n][0]),'y':rounded(s[n][1])})
        for n in list(tracks):
            for kind in list(tracks[n]):
                fields=('value',) if kind=='rotate' else ('x','y');neutral=1 if kind=='scale' else 0
                if n!='root' and all(all(k[field]==neutral for field in fields) for k in tracks[n][kind]):del tracks[n][kind]
            if not tracks[n]:del tracks[n]
        animation={'bones':tracks}
        if name=='hit':animation['events']=[{'time':.1,'name':'impact'}]
        if name=='dead':animation['events']=[{'time':1.4,'name':'death_end'}]
        animations[name]=animation
    return animations


def render(skeleton,images,animation=None,t=0,size=SIZE,camera=CAMERA,background=None,rig=False):return shared_render(skeleton,images,animation,t,size,camera,background,rig)


def correct_ground(skeleton,images):
    track=skeleton['animations']['dead']['bones']['root']['translate'];size=(3400,2600);camera=(1700,2200)
    correction=render(skeleton,images,'dead',1.6,size=size,camera=camera).getbbox()[3]-camera[1]
    for k in track:k['y']=rounded(correction*curve(k['time'],[(0,0),(.2,0),(.95,1),(1.6,1)]))
    for k in track:
        bottom=render(skeleton,images,'dead',k['time'],size=size,camera=camera).getbbox()[3];k['y']=rounded(k['y']+max(0,bottom-camera[1]))


def scaled_preview(skeleton,factor=.55):
    result=deepcopy(skeleton);result['bones'][0]['scaleX']=factor;result['bones'][0]['scaleY']=factor
    for a in result['animations'].values():
        for k in a.get('bones',{}).get('root',{}).get('translate',[]):k['x']*=factor;k['y']*=factor
    return result


def make_previews(skeleton,images):
    directory=BASE/'previews';factor=.55;stage=tuple(round(v*factor) for v in SIZE);camera=tuple(round(v*factor) for v in CAMERA);view=scaled_preview(skeleton,factor);bg=Image.new('RGBA',stage,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,stage[0],18):
        for y in range(0,stage[1],18):
            if (x//18+y//18)%2:d.rectangle((x,y,x+17,y+17),fill='#2d3d47')
    d.line((0,camera[1],stage[0],camera[1]),fill='#6e9f90')
    sheet=Image.new('RGB',(1440,len(DURATIONS)*300),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for f in range(round(duration*FPS)+1):
            im=render(view,images,name,f/FPS,size=stage,camera=camera,background=bg).convert('RGB');frame=im.resize((840,653),Image.Resampling.LANCZOS);ImageDraw.Draw(frame).text((12,12),f'{name} / {f/FPS:.2f}s',fill='#9cdedc');frames.append(frame)
            phone=Image.new('RGB',(420,320),'#172027');phone.paste(im.resize((376,293),Image.Resampling.LANCZOS),(22,26));ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack'));save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=([0,.3,.9,1.2] if name=='stand' else [0,.4,.95,1.6] if name=='dead' else [0,.1,.3,.5] if name=='hit' else [0,duration*.25,duration*.5,duration*.75])
        for col,t in enumerate(times):sheet.paste(frames[round(t*FPS)].resize((360,280),Image.Resampling.LANCZOS),(col*360,row*300+20))
        labels.text((12,row*300+4),name,fill='#9cdedc')
    sheet.save(directory/'animation-contact.png');render(view,images,size=stage,camera=camera,background=bg,rig=True).convert('RGB').save(directory/'rig-setup.png')


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text());config=json.loads((BASE/'rig-config.json').read_text());images=load_images(BASE/'images')
    for n,e in source['skins']['default'].items():
        a=e[n]
        if images[n].size!=(a['width'],a['height']):raise ValueError('Image dimensions mismatch: '+n)
    actor=build_rig(source,config);actor['animations']=make_animations(actor,config);correct_ground(actor,images);write_json(BASE/'bird112-4.3.json',actor)
    compat=deepcopy(actor)
    for slot in compat['slots']:
        n=slot['name'];a=source['skins']['default'][n][n];bone=next(b for b in config['bones'] if b['name']==slot['bone'])
        compat['skins'][0]['attachments'][n]={n:{'path':n,'x':a['x']-bone['position'][0],'y':a['y']-bone['position'][1],'width':a['width'],'height':a['height']}}
    for k in compat['animations']['dead']['bones']['root']['translate']:k['y']=0
    correct_ground(compat,images)
    write_json(BASE/'bird112-regions-4.3.json',compat)
    if previews:make_previews(actor,images)
    print('Built Bird112:',len(actor['bones']),'bones;',len(actor['slots']),'meshes;',list(actor['animations']));return actor

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true');args=parser.parse_args();build(not args.no_previews)
