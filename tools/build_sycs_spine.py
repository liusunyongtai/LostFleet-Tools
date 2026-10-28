"""Abyss tentacle creature: creeping, weighted lash, glass-contained rising bubbles."""
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
BASE=Path(__file__).resolve().parents[1]/'examples/sycs-spine'
FPS=30
DURATIONS={'stand':3.2,'dead':1.6,'hit':.5,'move':1.6,'attack':1.6}
SIZE=(2500,1650)
CAMERA=(1450,1500)
BUBBLE_SLOTS=tuple(f'bubble_{i}' for i in range(8))


def curve(t,points):return interpolate_controls(t,points)


def chain_weights(d,point):
    arc=nearest_arc(np.array(point),d['points']);knots=[0]
    for a,b in zip(d['points'],d['points'][1:]):knots.append(knots[-1]+math.dist(a,b))
    transitions=[smoothstep(v-d['blend'],v+d['blend'],arc) for v in knots[1:-1]]
    values=[1,*transitions,0];return {n:values[i]-values[i+1] for i,n in enumerate(d['bones'])}


def weights(n,point,c):
    x,y=point;depth=-y
    for key,d in c['chains'].items():
        if n in d['images']:
            result=chain_weights(d,point)
            # Keep eyes, teeth and collar on each mouth rigid even where the
            # coiled tube passes close to the mouth in the flat illustration.
            if key=='front':mouth=max(smoothstep(965,1005,x)*(1-smoothstep(660,740,depth)),smoothstep(1020,1060,x)*(1-smoothstep(865,925,depth)))
            elif key=='back':mouth=1-smoothstep(610,720,depth)
            else:mouth=0
            result={bone:w*(1-mouth) for bone,w in result.items()};last=d['bones'][-1];result[last]=result.get(last,0)+mouth
            return result
    if n=='sehnti':
        hips=smoothstep(1000,1110,depth);feet=smoothstep(1090,1180,depth);front=smoothstep(570,640,x)
        return {'breath':1-hips,'pelvis':hips-feet,'foot_back':feet*(1-front),'foot_front':feet*front}
    if n=='toubu':
        tip=smoothstep(890,1000,depth)*smoothstep(580,670,x);return {'head':1-tip,'face_tip':tip}
    if n in ('maoziguanzi_01','maoziguanzi_02','maoziguanzi_03','maoziguanzi_04'):
        head=smoothstep(450,760,depth);return {'jar':1-head,'head':head}
    if n in ('beihouguanzi_02','beihouguanzi_03'):
        head=smoothstep(410,620,depth);return {'jar':1-head,'head':head}
    if n=='beihouguanzi_01':
        low=smoothstep(850,1050,depth);return {'head':1-low,'pelvis':low}
    return {c['bindings'][n]:1.}


def attachment(n,a,c,rig,uv_bounds=(0,1,0,1),rigid=None,layout=None):
    cols,rows={'zuochushou_01':(20,16),'youchushou_01':(16,20),'sehnti':(14,18),'toubu':(16,16)}.get(n,(10,16))
    if layout:cols,rows=layout
    if rigid:cols,rows=1,1
    points,triangles,hull=grid(cols,rows);rest=pose(rig);inverse={n:np.linalg.inv(m) for n,m in rest.items()};indices={b['name']:i for i,b in enumerate(rig['bones'])};origin=np.array(c['origin']);vertices=[];uvs=[]
    for col,row in points:
        u=uv_bounds[0]+(uv_bounds[1]-uv_bounds[0])*col/cols;v=uv_bounds[2]+(uv_bounds[3]-uv_bounds[2])*row/rows;uvs.extend([u,v]);point=np.array([a['x']+(u-.5)*a['width'],a['y']+(.5-v)*a['height']]);active=({rigid:1.} if rigid else {n:w for n,w in weights(n,point,c).items() if w>1e-8});total=sum(active.values());vertices.append(len(active))
        for bone,w in active.items():
            local=inverse[bone]@np.r_[point-origin,1];vertices.extend([indices[bone],rounded(local[0]),rounded(local[1]),w/total])
    result={'type':'mesh','path':n,'width':a['width'],'height':a['height'],'uvs':uvs,'triangles':triangles,'vertices':vertices,'hull':hull,'edges':perimeter_edges(hull)}
    errors=editor_mesh_check(result)
    if errors:raise ValueError(f'{n}: {errors}')
    return result


def bubble_point(d,t,c):
    phase=(t/c['bubble_cycle']+d['phase'])%1;u=d['lane'];bottom=np.array(c['bubble_bottom'][0])*(1-u)+np.array(c['bubble_bottom'][1])*u;top=np.array(c['bubble_top'][0])*(1-u)+np.array(c['bubble_top'][1])*u
    pixel=bottom*(1-phase)+top*phase;pixel[0]+=5*math.sin(t/c['bubble_cycle']*math.tau+int(d['name'].split('_')[-1]));return pixel,phase


def build_rig(source,c):
    defs={b['name']:b for b in c['bones']};origin=np.array(c['origin']);rig={'skeleton':{'spine':'4.3.23','images':'./images/','fps':FPS,'x':-1550,'y':-150,'width':2600,'height':1650},'bones':[],'slots':[],'skins':[{'name':'default','attachments':{}}],'events':{'whip_hit':{'string':'strike_tip'},'impact':{},'death_end':{}},'animations':{}}
    for b in c['bones']:
        parent=defs.get(b.get('parent'));pos=np.array(b['position']);local=rotation(-parent['rotation'])@(pos-parent['position']) if parent else pos-origin
        entry={'name':b['name'],'x':rounded(local[0]),'y':rounded(local[1]),'rotation':rounded(b['rotation']-(parent['rotation'] if parent else 0)),'length':rounded(b['length'])}
        if parent:entry['parent']=parent['name']
        rig['bones'].append(entry)
    skin=rig['skins'][0]['attachments']
    for slot in source['slots']:
        n=slot['name'];a=source['skins']['default'][n][n];bone=c['bindings'][n];rig['slots'].append({'name':n,'bone':bone,'attachment':n})
        if n=='maozi':
            p=np.array([a['x'],a['y']])-defs['jar']['position'];skin[n]={n:{'path':n,'x':float(p[0]),'y':float(p[1]),'width':a['width'],'height':a['height']}}
            for d in c['bubbles']:
                k=d['name'];rig['slots'].append({'name':k,'bone':k});skin[k]={'bubble':{'path':'fx/bubble','width':d['size'],'height':d['size']}}
        elif n=='zuochushou_01':
            split=(1005-(a['x']-a['width']/2))/a['width'];skin[n]={n:attachment(n,a,c,rig,(0,split,0,1))}
            outer='front_outer';rig['slots'].append({'name':outer,'bone':c['chains']['front']['bones'][2],'attachment':outer});skin[outer]={outer:attachment(n,a,c,rig,(split,1,10/16,1),layout=(20,6))}
            mouth='front_mouth';rig['slots'].append({'name':mouth,'bone':c['chains']['front']['bones'][-1],'attachment':mouth});skin[mouth]={mouth:attachment(n,a,c,rig,(split,1,0,10/16),c['chains']['front']['bones'][-1])}
        else:skin[n]={n:attachment(n,a,c,rig)}
    rig['slots'].append({'name':'whip_air','bone':'air_anchor','blend':'additive'});skin['whip_air']={}
    return rig


def world_controls(rig,c,r,p,s):
    defs={b['name']:b for b in c['bones']};world={}
    for bone in rig['bones']:
        n=bone['name'];par=bone.get('parent');offset=rotation(-defs[par]['rotation'])@p[n] if par else p[n];local=np.eye(3);local[:2,:2]=rotation(bone.get('rotation',0)+r[n])@np.diag(s[n]);local[:2,2]=[bone.get('x',0)+offset[0],bone.get('y',0)+offset[1]];world[n]=world[par]@local if par else local
    return world


def fabrik(points,lengths,target):
    points=np.array(points,float).copy();root=points[0].copy();target=np.array(target)
    if np.linalg.norm(target-root)>=sum(lengths):
        unit=(target-root)/max(np.linalg.norm(target-root),1e-9)
        for i,length in enumerate(lengths):points[i+1]=points[i]+unit*length
        return points
    for _ in range(150):
        points[-1]=target
        for i in range(len(lengths)-1,-1,-1):
            vec=points[i]-points[i+1];points[i]=points[i+1]+vec/max(np.linalg.norm(vec),1e-9)*lengths[i]
        points[0]=root
        for i,length in enumerate(lengths):
            vec=points[i+1]-points[i];points[i+1]=points[i]+vec/max(np.linalg.norm(vec),1e-9)*length
        if np.linalg.norm(points[-1]-target)<1e-6:break
    return points


def solve_whip(rig,c,r,p,s,wind,strike,recoil,previous):
    d=c['chains']['front'];defs={b['name']:b for b in c['bones']};origin=np.array(c['origin']);world=world_controls(rig,c,r,p,s);rest=pose(rig);transform=world[d['parent']]@np.linalg.inv(rest[d['parent']]);preferred=np.array([(transform@np.r_[np.array(v)-origin,1])[:2] for v in d['points']]);lengths=[math.dist(a,b) for a,b in zip(d['points'],d['points'][1:])]
    target=np.array([1170+90*wind-1250*max(strike,0),-755+230*wind-65*max(strike,0)])-origin
    heading=defs[d['bones'][-1]]['rotation']+45*wind-118*strike-10*recoil;mouth=target-lengths[-1]*np.array([math.cos(math.radians(heading)),math.sin(math.radians(heading))])
    seed=preferred[:-1].copy()
    if previous is not None:seed=.85*(previous['points'][:-1]+preferred[0]-previous['points'][0])+.15*seed
    solved=fabrik(seed,lengths[:-1],mouth);solved=np.vstack((solved,target));parent_angle=math.degrees(math.atan2(world[d['parent']][1,0],world[d['parent']][0,0]));strength=min(1,max(wind,abs(strike),.35*abs(recoil)))
    world_angles={};body_angle=parent_angle
    for i,n in enumerate(d['bones']):
        desired=heading if i==len(d['bones'])-1 else math.degrees(math.atan2(*(solved[i+1]-solved[i])[::-1]));baseline=defs[n]['rotation']+body_angle
        if previous is not None:
            prev=previous['worldAngles'][n];desired=prev+(desired-prev+180)%360-180
        else:desired=baseline+(desired-baseline+180)%360-180
        world_angles[n]=desired;mixed=baseline+strength*(desired-baseline);parent=d['parent'] if i==0 else d['bones'][i-1]
        r[n]=mixed-parent_angle-(defs[n]['rotation']-defs[parent]['rotation']);parent_angle=mixed
    actual=world_controls(rig,c,r,p,s);return {'points':np.array([actual[n][:2,2] for n in d['bones']]+[actual['strike_tip'][:2,2]]),'worldAngles':world_angles}


def make_animations(rig,c):
    names=[b['name'] for b in rig['bones']];defs={b['name']:b for b in c['bones']};animations={}
    for name,duration in DURATIONS.items():
        tracks={n:{'rotate':[],'translate':[],'scale':[]} for n in names};bubble_alphas={n:[] for n in BUBBLE_SLOTS};previous=None
        for f in range(round(duration*FPS)+1):
            t=f/FPS;phase=t/duration*math.tau;r={n:0. for n in names};p={n:[0.,0.] for n in names};s={n:[1.,1.] for n in names};breath=(1-math.cos(phase))/2
            if name in ('stand','move'):
                creep=name=='move';r['body']=(5 if creep else 0)+(2 if creep else 1.5)*math.sin(phase);p['pelvis']=[(8 if creep else 3)*math.sin(phase),-(12 if creep else 3)*breath];p['body'][1]=(-12 if creep else 4*breath)
                s['breath']=[1+.03*breath,1+.018*breath];r['head']=-2*math.sin(phase-.2)+2*math.sin(-.2);r['jar']=-1.5*(math.sin(phase-.45)-math.sin(-.45));r['face_tip']=6*math.sin(phase-.4)-6*math.sin(-.4)
                for k,d in c['chains'].items():
                    if k=='stump_front':continue
                    for i,n in enumerate(d['bones']):
                        amp=(4+i*.9)*(1.35 if creep else 1);lag=phase-i*.5+(0 if 'front' in k else .9);r[n]=amp*(math.sin(lag)-math.sin(-i*.5+(0 if 'front' in k else .9)))
                r['stump_front_0']=3*math.sin(phase);r['stump_front_1']=-5*math.sin(phase)
                if creep:
                    for n,sign in [('foot_back',1),('foot_front',-1)]:p[n]=[7*sign*math.sin(phase),6*max(sign*math.sin(phase),0)**2]
            elif name=='attack':
                wind=curve(t,[(0,0),(.3,1),(.38,1),(.6,0),(duration,0)]);strike=curve(t,[(0,0),(.38,0),(.6,1),(.7,.96),(1.15,-.06),(duration,0)]);recoil=curve(t,[(0,0),(.6,0),(.9,1),(1.2,-.15),(duration,0)])
                p['pelvis']=[18*wind-65*strike,-12*wind-20*strike];p['body']=[-35*strike,-10*strike];r['body']=-6*wind+10*strike;r['head']=5*wind-7*strike+5*recoil;r['jar']=-3*wind+4*strike-6*recoil;r['face_tip']=7*recoil
                s['breath']=[1+.025*wind,1-.01*wind]
                for k,d in c['chains'].items():
                    if k=='front':continue
                    for i,n in enumerate(d['bones']):r[n]=(5*wind-6*strike+5*recoil)*math.sin(i*.55+.8)
                previous=solve_whip(rig,c,r,p,s,wind,strike,recoil,previous)
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(.17,.8),(.32,-.2),(duration,0)]);lag=curve(t,[(0,0),(.15,1),(.3,-.25),(duration,0)]);p['pelvis']=[25*shock,-12*shock];r['body']=-10*shock;r['head']=-6*lag;r['jar']=9*lag;r['face_tip']=-10*lag
                for k,d in c['chains'].items():
                    for i,n in enumerate(d['bones']):r[n]=(8+i)*lag*math.cos(i*.6)
            else:
                fall=curve(t,[(0,0),(.2,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)]);impact=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)]);p['pelvis']=[-120*fall,-80*fall];r['pelvis']=82*fall;r['body']=7*fall;r['head']=-8*fall-5*impact;r['jar']=-7*fall+7*impact;r['face_tip']=15*fall
                for k,d in c['chains'].items():
                    for i,n in enumerate(d['bones']):r[n]=(12 if 'front' in k else -8)*fall*math.cos(i*.55)+(5+i)*impact
            if name=='dead':
                r['back_0']=-80*fall;r['floor_back_0']=-140*fall;r['stump_back_0']=55*fall
                world=world_controls(rig,c,r,p,s);rest=pose(rig);follow=world['pelvis']@np.linalg.inv(rest['pelvis'])
                for foot in ('foot_back','foot_front'):
                    target=(follow@np.r_[rest[foot][:2,2],1])[:2];p[foot]=(target-rest[foot][:2,2]).tolist();r[foot]=r['pelvis']
            for d in c['bubbles']:
                bt=0 if name in ('hit','dead') else t;pixel,u=bubble_point(d,bt,c);target=np.array([342+pixel[0],-23-pixel[1]]);p[d['name']]=(target-defs[d['name']]['position']).tolist();size=1+.12*math.sin(u*math.pi);s[d['name']]=[size,size]
                alpha=.72*max(0,min((u-.08)/.1,(.92-u)/.1,1))
                if name=='dead':alpha*=1-smoothstep(.1,.6,t)
                bubble_alphas[d['name']].append({'time':t,'value':rounded(alpha)})
            for n in names:
                parent=defs.get(defs[n].get('parent'));local=rotation(-parent['rotation'])@p[n] if parent else p[n];angle=r[n]
                if tracks[n]['rotate']:
                    prev=tracks[n]['rotate'][-1]['value'];angle=prev+(angle-prev+180)%360-180
                tracks[n]['rotate'].append({'time':t,'value':rounded(angle)});tracks[n]['translate'].append({'time':t,'x':rounded(local[0]),'y':rounded(local[1])});tracks[n]['scale'].append({'time':t,'x':rounded(s[n][0]),'y':rounded(s[n][1])})
        for n in list(tracks):
            for kind in list(tracks[n]):
                fields=('value',) if kind=='rotate' else ('x','y');neutral=1 if kind=='scale' else 0
                if n!='root' and all(all(k[q]==neutral for q in fields) for k in tracks[n][kind]):del tracks[n][kind]
            if not tracks[n]:del tracks[n]
        slots={n:{'attachment':[{'time':0,'name':'bubble'}]+([{'time':.65,'name':None}] if name=='dead' else []),'alpha':bubble_alphas[n]} for n in BUBBLE_SLOTS};slots['whip_air']={'attachment':[{'time':0,'name':None}],'alpha':[{'time':0,'value':0}]};a={'bones':tracks,'slots':slots}
        if name=='attack':
            a['events']=[{'time':.6,'name':'whip_hit'}];original=[v['name'] for v in rig['slots']];moving=['zuochushou_01','front_outer','front_mouth'];order=[n for n in original if n not in moving];index=order.index('whip_air');order[index:index]=moving;offsets=[{'slot':n,'offset':order.index(n)-i} for i,n in enumerate(original) if order.index(n)!=i];a['drawOrder']=[{'time':0},{'time':.3,'offsets':offsets},{'time':1.3}]
        if name=='hit':a['events']=[{'time':.1,'name':'impact'}]
        if name=='dead':a['events']=[{'time':1.4,'name':'death_end'}]
        animations[name]=a
    return animations


def smooth_attack_world(actor,c):
    # Solve readable key poses, then interpolate their WORLD angles. This avoids
    # elbow branch switches during the return while retaining the exact strike.
    bones=c['chains']['front']['bones'];defs={b['name']:b for b in c['bones']};times=[0,.3,.38,.6,.9,1.15,1.6];keys={n:[] for n in bones}
    for t in times:
        world=pose(actor,'attack',t)
        for n in bones:
            angle=math.degrees(math.atan2(world[n][1,0],world[n][0,0]))
            if keys[n]:
                prev=keys[n][-1][1];angle=prev+(angle-prev+180)%360-180
            keys[n].append((t,angle))
    tracks=actor['animations']['attack']['bones']
    for n in bones:tracks[n]['rotate']=[]
    for f in range(49):
        t=f/FPS;body=pose(actor,'attack',t)['body'];parent_angle=math.degrees(math.atan2(body[1,0],body[0,0]))
        for i,n in enumerate(bones):
            angle=curve(t,keys[n]);parent='body' if i==0 else bones[i-1];delta=angle-parent_angle-(defs[n]['rotation']-defs[parent]['rotation'])
            previous=tracks[n]['rotate']
            if previous:
                value=previous[-1]['value'];delta=value+(delta-value+180)%360-180
            previous.append({'time':t,'value':rounded(delta)});parent_angle=angle


def make_effects():
    directory=BASE/'images/fx';scale=4;im=Image.new('RGBA',(160,160));d=ImageDraw.Draw(im);d.ellipse((15,15,145,145),fill=(238,127,180,25),outline=(255,193,225,210),width=10);d.arc((31,28,126,124),195,280,fill=(255,238,247,245),width=12);im.resize((40,40),Image.Resampling.LANCZOS).save(directory/'bubble.png')
    yy,xx=np.mgrid[:96,:512];u=(xx+.5)/512;v=(yy+.5)/96;alpha=175*np.sin(np.pi*u)**.7*np.maximum(0,1-abs(v-.5)*2)**1.4;rgba=np.zeros((96,512,4),np.uint8);rgba[:,:,:3]=[225,225,250];rgba[:,:,3]=np.rint(alpha).astype('uint8');Image.fromarray(rgba).save(directory/'whip_air.png')


def add_airflow(actor):
    index=next(i for i,b in enumerate(actor['bones']) if b['name']=='air_anchor');skin=actor['skins'][0]['attachments']['whip_air'];keys=[{'time':0,'name':None}]
    for f in range(15,19):
        t=f/FPS;nodes,triangles,hull=grid(28,1);uvs=[];vertices=[]
        for col,row in nodes:
            u,v=col/28,float(row);tt=max(.4,t-.09)+(t-max(.4,t-.09))*u;point=pose(actor,'attack',tt)['strike_tip'][:2,2];before=pose(actor,'attack',tt-.002)['strike_tip'][:2,2];after=pose(actor,'attack',tt+.002)['strike_tip'][:2,2];tangent=after-before;normal=np.array([-tangent[1],tangent[0]])/max(np.linalg.norm(tangent),1e-8);half=55*(.12+.88*math.sin(math.pi*u)**.7);xy=point+normal*half*(1 if row==0 else -1);uvs.extend([u,v]);vertices.extend([1,index,float(xy[0]),float(xy[1]),1])
        a={'type':'mesh','path':'fx/whip_air','width':512,'height':96,'uvs':uvs,'triangles':triangles,'vertices':vertices,'hull':hull,'edges':perimeter_edges(hull)};errors=editor_mesh_check(a)
        if errors:raise ValueError(f'Whip air {f}: {errors}')
        n=f'air_{f}';skin[n]=a;keys.append({'time':t,'name':n})
    keys.append({'time':.7,'name':None});actor['animations']['attack']['slots']['whip_air']={'attachment':keys,'alpha':[{'time':0,'value':0},{'time':14/FPS,'value':0},{'time':.5,'value':.8},{'time':.6,'value':1},{'time':.7,'value':0}]}


def render(skeleton,images,animation=None,t=0,size=SIZE,camera=CAMERA,background=None,rig=False):return shared_render(skeleton,images,animation,t,size,camera,background,rig)


def correct_ground(actor,images):
    track=actor['animations']['dead']['bones']['root']['translate'];size=(4000,3000);camera=(2100,2500);correction=render(actor,images,'dead',1.6,size=size,camera=camera).getbbox()[3]-camera[1]
    for k in track:k['y']=rounded(correction*curve(k['time'],[(0,0),(.2,0),(.95,1),(1.6,1)]))
    for k in track:
        bottom=render(actor,images,'dead',k['time'],size=size,camera=camera).getbbox()[3];k['y']=rounded(k['y']+max(0,bottom-camera[1]))


def scaled_preview(actor,factor=.42):
    a=deepcopy(actor);a['bones'][0]['scaleX']=factor;a['bones'][0]['scaleY']=factor
    for animation in a['animations'].values():
        for k in animation['bones'].get('root',{}).get('translate',[]):k['x']*=factor;k['y']*=factor
    return a


def make_previews(actor,images):
    directory=BASE/'previews';factor=.42;stage=tuple(round(v*factor) for v in SIZE);camera=tuple(round(v*factor) for v in CAMERA);view=scaled_preview(actor,factor);bg=Image.new('RGBA',stage,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,stage[0],18):
        for y in range(0,stage[1],18):
            if (x//18+y//18)%2:d.rectangle((x,y,x+17,y+17),fill='#2d3d47')
    d.line((0,camera[1],stage[0],camera[1]),fill='#6e9f90');sheet=Image.new('RGB',(1440,1500),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for f in range(round(duration*FPS)+1):
            im=render(view,images,name,f/FPS,size=stage,camera=camera,background=bg).convert('RGB');frame=im.resize((840,554),Image.Resampling.LANCZOS);ImageDraw.Draw(frame).text((12,12),f'{name} / {f/FPS:.2f}s',fill='#e9b0d1');frames.append(frame);phone=Image.new('RGB',(420,264),'#172027');phone.paste(im.resize((274,181),Image.Resampling.LANCZOS),(73,40));ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#e9b0d1');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move'));save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=([0,.3,.6,.9] if name=='attack' else [0,.4,.95,1.6] if name=='dead' else [0,.1,.3,.5] if name=='hit' else [0,duration*.25,duration*.5,duration*.75])
        for col,t in enumerate(times):sheet.paste(frames[min(round(t*FPS),len(frames)-1)].resize((360,237),Image.Resampling.LANCZOS),(col*360,row*300+25))
        labels.text((12,row*300+5),name,fill='#e9b0d1')
    sheet.save(directory/'animation-contact.png');render(view,images,size=stage,camera=camera,background=bg,rig=True).convert('RGB').save(directory/'rig-setup.png')
    # Readable jar close-up makes the bubbles inspectable without enlarging the character in game.
    frames=[]
    for f in range(round(DURATIONS['stand']*FPS)+1):
        im=render(view,images,'stand',f/FPS,size=stage,camera=camera,background=bg).convert('RGB');frames.append(im.crop((420,85,770,400)).resize((600,540)))
    save_gif(frames,directory/'jar-bubbles.gif',True)


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text());c=json.loads((BASE/'rig-config.json').read_text());make_effects();images=load_images(BASE/'images');actor=build_rig(source,c);actor['animations']=make_animations(actor,c);smooth_attack_world(actor,c);add_airflow(actor);correct_ground(actor,images);write_json(BASE/'sycs-4.3.json',actor)
    if previews:make_previews(actor,images)
    print('Built SYCS:',len(actor['bones']),'bones;',len(actor['slots']),'slots;',list(actor['animations']));return actor

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--no-previews',action='store_true');args=p.parse_args();build(not args.no_previews)
