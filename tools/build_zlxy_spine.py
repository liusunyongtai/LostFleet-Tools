"""Armored rifle carrier: two-hand grip, weapon smash, muzzle flash, additive light, stationary bullet."""
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
from tools.build_xqdz_spine import rotation,rounded,interpolate_controls,pose,load_images,save_gif,make_flash
from tools.build_bsq_spine import mesh_world,render as shared_render,smoothstep,write_json

BASE=Path(__file__).resolve().parents[1]/'examples'/'zlxy-spine'
FPS=30
SHOT=.4
LIT_PARTS=('wuqi','youbi','zuobi','tou','qugan')
LIGHT_SLOTS=tuple('shot_light_'+n for n in LIT_PARTS)+('muzzle_glow',)
DURATIONS={'stand':2.4,'dead':1.6,'hit':.5,'attack_1':1.2,'attack_2':1.2,'move':1.}
SIZE=(1200,850)
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
    rigid={'tou':'head','beibao':'backpack','wuqi':'weapon','muzzle_glow':'muzzle','muzzle_flash':'muzzle'}
    if name in rigid:return {rigid[name]:1.}
    if name=='qugan':
        hips=smoothstep(337,380,depth);return {'body':1-hips,'pelvis':hips}
    if name=='guanzi':
        top=1-smoothstep(170,207,depth);bottom=smoothstep(215,250,depth)
        return {'hose_top':top,'hose_mid':1-top-bottom,'hose_bottom':bottom}
    for key,d in config['limbs'].items():
        if name==d['image']:
            lower=smoothstep(-d['knee'][1]-35,-d['knee'][1]+35,depth)
            paw=smoothstep(-d['ankle'][1]-12,-d['ankle'][1]+12,depth)
            return {key+'_upper':1-lower,key+'_lower':max(0,lower-paw),key+'_paw':paw}
    key,d=next((k,v) for k,v in config['arms'].items() if v['image']==name)
    points=[d[k] for k in ('hip','knee','ankle','toe')];arc=nearest_arc(point,points)
    a=math.dist(*points[:2]);b=math.dist(*points[1:3]);upper=1-smoothstep(a-12,a+12,arc);paw=smoothstep(a+b-8,a+b+8,arc)
    return {key+'_upper':upper,key+'_lower':max(0,1-upper-paw),key+'_paw':paw}


def weighted_attachment(name,region,config):
    columns,rows={'zuotui':(12,22),'youtui':(12,22),'qugan':(10,14),'zuobi':(14,16),'youbi':(16,16),'guanzi':(10,18)}.get(name,(1,1))
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
                        'x':-650,'y':0,'width':1100,'height':800},'bones':[],'slots':[],'skins':[],
            'events':{'melee_hit':{'string':'melee_tip'},'shoot':{'string':'projectile_spawn'},'impact':{},'footstep':{'string':'back_leg'},'death_end':{}},'animations':{}}
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
        if name in LIT_PARTS:
            light='shot_light_'+name;attachment=deepcopy(skin[name][name]);attachment['path']='fx/'+light
            skin[light]={light:attachment};result['slots'].append({'name':light,'bone':config['bindings'][name],'blend':'additive'})
    glow=weighted_attachment('muzzle_glow',{'x':63,'y':-281,'width':340,'height':260},config);glow['path']='fx/muzzle_glow'
    skin['muzzle_glow']={'muzzle_glow':glow};result['slots'].append({'name':'muzzle_glow','bone':'muzzle','blend':'additive'})
    flashes={}
    for i in range(1,4):
        attachment=weighted_attachment('muzzle_flash',{'x':-57,'y':-281,'width':240,'height':160},config)
        attachment['path']='fx/muzzle_flash_'+str(i)
        attachment['uvs']=[1-v if n%2==0 else v for n,v in enumerate(attachment['uvs'])]
        flashes['muzzle_flash_'+str(i)]=attachment
    skin['muzzle_flash']=flashes;result['slots'].append({'name':'muzzle_flash','bone':'muzzle','blend':'additive'})
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


def solve_grips(rig,config,r,p,s,release):
    free={n:r[n] for n in ('back_arm_upper','back_arm_lower','back_arm_paw')}
    world=controls_world(rig,config,r,p,s);rest=pose(rig)
    image=world['weapon']@np.linalg.inv(rest['weapon']);origin=np.array(config['origin'])
    targets={};pitches={}
    delta=math.degrees(math.atan2(world['weapon'][1,0],world['weapon'][0,0]))-180
    delta=(delta+180)%360-180
    for key,point in config['grips'].items():
        v=np.array(point)-origin;target=(image@[v[0],v[1],1])[:2]
        targets[key]=(target+origin).tolist();pitches[key]=delta
    solve_paws(rig,{**config,'limbs':config['arms']},r,p,s,targets,pitches)
    for n,value in free.items():
        # Blend IK to the released hand along the nearest angular branch.
        r[n]+=release*((value-r[n]+180)%360-180)


def make_animations(rig,config):
    definitions={b['name']:b for b in config['bones']};animations={}
    for name,duration in DURATIONS.items():
        timelines={b:{'rotate':[],'translate':[],'scale':[]} for b in definitions}
        for frame in range(round(duration*FPS)+1):
            t=frame/FPS;phase=t/duration*math.tau;r={b:0. for b in definitions};p={b:[0.,0.] for b in definitions};s={b:[1.,1.] for b in definitions}
            targets={key:list(d['ankle']) for key,d in config['limbs'].items()};pitches={key:0 for key in targets};release=0
            if name=='stand':
                breath=(1-math.cos(phase))/2;p['pelvis'][1]=-2*breath;p['body'][1]=4*breath
                r['body']=1.8*math.sin(phase);r['neck']=-1.2*math.sin(phase);r['head']=5*math.sin(phase)
                r['weapon']=-1.5*math.sin(phase);r['backpack']=2*(math.sin(phase-.3)-math.sin(-.3))
            elif name=='move':
                stride=math.sin(phase);bob=(math.sqrt(stride*stride+.04**2)-.04)/(math.sqrt(1+.04**2)-.04)
                p['pelvis'][1]=-10*bob;r['body']=3*stride;r['neck']=-2*stride;r['head']=-4*stride
                r['weapon']=-2*stride;r['backpack']=-4*(math.sin(phase-.3)-math.sin(-.3))
                for key,d in config['limbs'].items():
                    step=math.sin(phase+d['phase']);lift=max(step,0)**2
                    targets[key]=[d['ankle'][0]-18*step,d['ankle'][1]+12*lift];pitches[key]=-5*lift
            elif name=='attack_1':
                wind=curve(t,[(0,0),(.3,1),(.4,1),(.6,0),(duration,0)])
                strike=curve(t,[(0,0),(.4,0),(.6,1),(20/FPS,.96),(.98,-.06),(duration,0)])
                recoil=curve(t,[(0,0),(.6,0),(.8,1),(1.05,-.15),(duration,0)])
                p['pelvis']=[10*wind-25*strike,3*wind-8*strike];p['body']=[6*wind-7*strike,4*wind-4*strike]
                r['body']=-6*wind+10*strike;r['neck']=4*wind-6*strike;r['head']=4*wind-5*strike+4*recoil
                r['weapon']=-65*wind+40*strike-10*recoil;p['weapon']=[4*wind,90*wind]
                r['back_arm_upper']=-20*wind+25*strike;r['back_arm_lower']=30*wind-20*strike;r['back_arm_paw']=10*strike
                release=curve(t,[(0,0),(.1,1),(.95,1),(duration,0)])
                r['backpack']=4*wind-6*strike+4*recoil
                for key,d in config['limbs'].items():targets[key]=[d['ankle'][0]-23*strike,d['ankle'][1]+3*max(strike,0)]
            elif name=='attack_2':
                aim=curve(t,[(0,0),(.2,1),(.3,1),(SHOT,0),(duration,0)])
                recoil=curve(t,[(0,0),(11/FPS,0),(SHOT,1),(13/FPS,.82),(.6,-.16),(.9,0),(duration,0)])
                lag=curve(t,[(0,0),(SHOT,0),(13/FPS,1),(.65,-.15),(.95,0),(duration,0)])
                p['pelvis']=[2*aim+7*recoil,-2*aim];p['body']=[4*recoil,-2*aim]
                r['body']=-3*aim-4*recoil;r['neck']=2*aim-3*lag;r['head']=-3*aim-7*lag
                r['weapon']=-r['body']-8*recoil;p['weapon']=[-2*aim+12*recoil,0]
                r['backpack']=4*lag
            elif name=='hit':
                shock=curve(t,[(0,0),(.1,1),(5/FPS,.8),(.3,-.18),(duration,0)])
                lag=curve(t,[(0,0),(4/FPS,1),(8/FPS,-.2),(duration,0)])
                p['pelvis']=[7*shock,-4*shock];p['body']=[4*shock,-2*shock]
                r['body']=-8*shock;r['neck']=-4*lag;r['head']=-9*lag;r['weapon']=6*lag;r['backpack']=5*lag
            elif name=='dead':
                buckle=curve(t,[(0,0),(.2,1),(.5,0),(duration,0)])
                fall=curve(t,[(0,0),(.25,0),(.95,1),(1.12,.96),(1.4,1),(duration,1)])
                impact=curve(t,[(0,0),(.85,0),(1.,1),(1.2,-.15),(1.4,0),(duration,0)])
                p['pelvis']=[-45*fall,-90*fall-8*buckle];r['pelvis']=75*fall
                r['body']=5*fall+4*buckle;r['neck']=-5*fall;r['head']=-5*fall-4*impact
                r['weapon']=-80*fall;p['weapon']=[50*fall,0];r['backpack']=-4*fall
                r['back_arm_upper']=35*fall;r['back_arm_lower']=30*fall;r['back_arm_paw']=10*fall;release=curve(t,[(0,0),(.2,1),(duration,1)])
                for key in config['limbs']:
                    r[key+'_upper']=60*fall-22*buckle;r[key+'_lower']=45*buckle;r[key+'_paw']=-22*buckle
            if name!='dead':solve_paws(rig,config,r,p,s,targets,pitches)
            solve_grips(rig,config,r,p,s,release)
            for b in definitions:
                parent=definitions.get(definitions[b].get('parent'));local=rotation(-parent['rotation'])@p[b] if parent else p[b]
                angle=r[b]
                if timelines[b]['rotate']:
                    previous=timelines[b]['rotate'][-1]['value'];angle=previous+(angle-previous+180)%360-180
                timelines[b]['rotate'].append({'time':t,'value':rounded(angle)});timelines[b]['translate'].append({'time':t,'x':rounded(local[0]),'y':rounded(local[1])});timelines[b]['scale'].append({'time':t,'x':rounded(s[b][0]),'y':rounded(s[b][1])})
        for b in list(timelines):
            for kind in list(timelines[b]):
                keys=('value',) if kind=='rotate' else ('x','y');neutral=1 if kind=='scale' else 0
                if b!='root' and all(all(k[key]==neutral for key in keys) for k in timelines[b][kind]):del timelines[b][kind]
            if not timelines[b]:del timelines[b]
        slots={slot:{'attachment':[{'time':0,'name':None}],'alpha':[{'time':0,'value':0}]} for slot in (*LIGHT_SLOTS,'muzzle_flash')}
        if name=='attack_2':
            alpha=[{'time':0,'value':0},{'time':11/FPS,'value':0},{'time':SHOT,'value':1},{'time':13/FPS,'value':.78},{'time':14/FPS,'value':.35},{'time':.5,'value':.12},{'time':17/FPS,'value':0}]
            for slot in LIGHT_SLOTS:
                slots[slot]={'attachment':[{'time':0,'name':None},{'time':SHOT,'name':slot},{'time':17/FPS,'name':None}],'alpha':deepcopy(alpha)}
            slots['muzzle_flash']={'attachment':[{'time':0,'name':None}]+[{'time':(12+i)/FPS,'name':'muzzle_flash_'+str(i+1)} for i in range(3)]+[{'time':.5,'name':None}],'alpha':deepcopy(alpha)}
        animation={'bones':timelines,'slots':slots}
        if name=='attack_1':
            animation['events']=[{'time':.6,'name':'melee_hit'}]
            original=[q['name'] for q in rig['slots']];moving=['wuqi','shot_light_wuqi','youbi','shot_light_youbi']
            order=[q for q in original if q not in moving];index=order.index('shot_light_tou')+1;order[index:index]=moving
            offsets=[{'slot':q,'offset':order.index(q)-i} for i,q in enumerate(original) if order.index(q)!=i]
            animation['drawOrder']=[{'time':0},{'time':.1,'offsets':offsets},{'time':1.1}]
        if name=='attack_2':animation['events']=[{'time':SHOT,'name':'shoot'}]
        if name=='hit':animation['events']=[{'time':.1,'name':'impact'}]
        if name=='dead':animation['events']=[{'time':1.4,'name':'death_end'}]
        if name=='move':animation['events']=[{'time':.5,'name':'footstep','string':'back_leg'},{'time':1.,'name':'footstep','string':'front_leg'}]
        animations[name]=animation
    return animations


def make_effects(source,config):
    directory=BASE/'images/fx';make_flash(directory)
    yy,xx=np.mgrid[-1:1:256j,-1:1:256j];radial=np.maximum(0,(np.exp(-4*(xx*xx+yy*yy))-math.exp(-4))/(1-math.exp(-4)))
    glow=np.zeros((256,256,4),np.uint8);glow[:,:,:3]=[255,177,80];glow[:,:,3]=np.rint(80*radial).astype(np.uint8);Image.fromarray(glow).save(directory/'muzzle_glow.png')
    for name in LIT_PARTS:
        im=Image.open(BASE/'images'/(name+'.png')).convert('RGBA');a=source['skins']['default'][name][name]
        y,x=np.mgrid[:im.height,:im.width];wx=a['x']+(x+.5-im.width/2);wy=a['y']+(im.height/2-y-.5)
        strength=.68*np.exp(-((wx-63)**2+(wy+281)**2)/(2*180**2))
        light=np.zeros((im.height,im.width,4),np.uint8);light[:,:,:3]=[255,190,95];light[:,:,3]=np.rint(np.asarray(im)[:,:,3]*strength).astype(np.uint8)
        Image.fromarray(light).save(directory/('shot_light_'+name+'.png'))


def build_bullet(config):
    rig={'skeleton':{'spine':config['target_version'],'images':'./images/','fps':FPS,'x':-130,'y':-85,'width':260,'height':170},
         'bones':[{'name':'root'},{'name':'core','parent':'root','x':-45},{'name':'tail','parent':'root','x':45}],
         'slots':[{'name':'projectile','bone':'core','attachment':'projectile'}],'skins':[{'name':'default','attachments':{}}],'animations':{}}
    points,triangles,hull=grid(20,10);vertices=[];uvs=[]
    for col,row in points:
        u,v=col/20,row/10;uvs.extend([u,v]);point=np.array([(u-.5)*200,(.5-v)*100]);tail=smoothstep(.5,.78,u)
        active=[(i,w,x) for i,w,x in [(1,1-tail,-45),(2,tail,45)] if w>1e-8];vertices.append(len(active))
        for index,weight,offset in active:vertices.extend([index,rounded(point[0]-offset),rounded(point[1]),weight])
    rig['skins'][0]['attachments']={'projectile':{'projectile':{'type':'mesh','path':'fx/flying-bullet','width':200,'height':100,'uvs':uvs,'vertices':vertices,'triangles':triangles,'hull':hull}}}
    tracks={n:{'rotate':[],'translate':[],'scale':[]} for n in ('root','core','tail')}
    for f in range(13):
        t=f/FPS;phase=t/.4*math.tau
        for n in tracks:
            angle=3.5*(math.sin(phase-.3)-math.sin(-.3)) if n=='tail' else 0;x=3*math.sin(phase) if n=='tail' else 0;y=2*(math.sin(phase-.3)-math.sin(-.3)) if n=='tail' else 0
            sx=1+.07*math.sin(phase) if n=='tail' else 1;sy=1+.18*math.sin(phase) if n=='tail' else 1
            tracks[n]['rotate'].append({'time':t,'value':rounded(angle)});tracks[n]['translate'].append({'time':t,'x':rounded(x),'y':rounded(y)});tracks[n]['scale'].append({'time':t,'x':rounded(sx),'y':rounded(sy)})
    rig['animations']={'bullet':{'bones':tracks}};return rig


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


def make_previews(skeleton,images,directory,bullet=None):
    directory.mkdir(parents=True,exist_ok=True);bg=Image.new('RGBA',SIZE,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,SIZE[0],20):
        for y in range(0,SIZE[1],20):
            if (x//20+y//20)%2:d.rectangle((x,y,x+19,y+19),fill='#2d3d47')
    d.line((0,CAMERA[1],SIZE[0],CAMERA[1]),fill='#6e9f90')
    sheet=Image.new('RGB',(1440,len(DURATIONS)*285),'#172027');labels=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[];mobile=[]
        for f in range(round(duration*FPS)+1):
            t=f/FPS;image=render(skeleton,images,name,t,background=bg).convert('RGB');view=image.resize((840,595),Image.Resampling.LANCZOS)
            ImageDraw.Draw(view).text((12,12),f'{name} / {f:02d} / {t:.2f}s',fill='#9cdedc');frames.append(view)
            phone=Image.new('RGB',(420,300),'#172027');phone.paste(image.resize((338,239),Image.Resampling.LANCZOS),(41,28));ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack_1','attack_2'));save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        times=([0,.35,.6,.85] if name=='attack_1' else ([0,.2,SHOT,.5] if name=='attack_2' else [0,duration*.3,duration*.65,duration]))
        for col,t in enumerate(times):sheet.paste(frames[min(round(t*FPS),len(frames)-1)].resize((360,255),Image.Resampling.LANCZOS),(col*360,row*285+25))
        labels.text((12,row*285+6),name,fill='#9cdedc')
    sheet.save(directory/'animation-contact.png');render(skeleton,images,rig=True,background=bg).convert('RGB').save(directory/'rig-setup.png')
    if bullet is not None:
        view=deepcopy(bullet);view['bones'][0]['scaleX']=1.6;view['bones'][0]['scaleY']=1.6;frames=[]
        for f in range(13):
            im=render(view,images,'bullet',f/FPS,size=(640,360),camera=(320,180),background=Image.new('RGBA',(640,360),'#26343e')).convert('RGB');ImageDraw.Draw(im).text((12,12),'bullet / fixed root, rigid nose, animated flame trail',fill='#9cdedc');frames.append(im)
        save_gif(frames,directory/'bullet.gif',True);frames[3].save(directory/'bullet-frame.png')


def build(previews=True):
    source=json.loads((BASE/'source/photoshop.json').read_text(encoding='utf8'))
    config=json.loads((BASE/'rig-config.json').read_text(encoding='utf8'))
    make_effects(source,config)
    images=load_images(BASE/'images')
    for name,entries in source['skins']['default'].items():
        a=entries[name]
        if images[name].size!=(a['width'],a['height']):raise ValueError('Image size mismatch '+name)
    skeleton=build_rig(source,config);skeleton['animations']=make_animations(skeleton,config)
    correct_ground(skeleton,images);write_json(BASE/'zlxy-4.3.json',skeleton)
    bullet=build_bullet(config);write_json(BASE/'flying-bullet-4.3.json',bullet)
    if previews:make_previews(skeleton,images,BASE/'previews',bullet)
    print('Built ZLXY:',len(skeleton['bones']),'bones,',len(skeleton['slots']),'weighted meshes;',list(skeleton['animations']))
    return skeleton


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args();build(not args.no_previews)


if __name__=='__main__':main()
