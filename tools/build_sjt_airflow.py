"""Add attack_2 wind trails to the uploaded SJT export without editing its original curves."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json,math,sys,argparse
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.build_bsq_spine import render as shared_render,write_json
from tools.build_xqdz_spine import load_images,save_gif
BASE=Path(__file__).resolve().parents[1]/'examples/sjt-airflow-spine'
FX_SLOTS=('airflow_trail','impact_air')
START=.5
CONTACT=.6
END=.9


def report_samples():return json.loads((BASE/'source/original-runtime.json').read_text())['trajectory']


def matrices(item,factor=1):
    result={}
    for n,(a,b,c,d,x,y) in item['bones'].items():
        m=np.array([[a,b,x],[c,d,y],[0,0,1.]],float);m[:2,:]*=factor;result[n]=m
    return result


def sample_point(samples,t,bone='tool_tip'):
    # 240 Hz official-runtime tool trajectory, not the obsolete repository animation.
    if t<=samples[0]['time']:return np.array(samples[0]['point'])
    for a,b in zip(samples,samples[1:]):
        if t<=b['time']+1e-8:
            u=(t-a['time'])/(b['time']-a['time']);return np.array(a['point'])*(1-u)+np.array(b['point'])*u
    return np.array(samples[-1]['point'])


def make_effects():
    d=BASE/'images/fx';d.mkdir(exist_ok=True)
    # Tapered translucent air: white center, subtle cold blue edges, no flame imagery.
    y,x=np.mgrid[:128,:512];u=(x+.5)/512;v=(y+.5)/128
    longitudinal=np.sin(np.pi*u)**.75;cross=np.maximum(0,1-np.abs(v-.5)*2)**1.25
    alpha=longitudinal*cross*155;rgb=np.empty((128,512,3));rgb[:]=[174,222,245];rgb+=cross[:,:,None]*np.array([76,28,10])
    rgba=np.concatenate([np.clip(rgb,0,255),alpha[:,:,None]],axis=2).round().astype('uint8');Image.fromarray(rgba).save(d/'air_streak.png')
    # Open, broken pressure rings and outward wisps, rather than a solid white disk.
    scale=3;im=Image.new('RGBA',(512*scale,256*scale));draw=ImageDraw.Draw(im)
    for radius,height,width,alpha in [(180,65,8,220),(220,88,4,150),(128,40,5,170)]:
        box=((256-radius)*scale,(145-height)*scale,(256+radius)*scale,(145+height)*scale)
        for start,end in [(5,73),(90,165),(183,250),(270,344)]:draw.arc(box,start,end,fill=(199,235,250,alpha),width=width*scale)
    for sign in (-1,1):
        for i in range(3):
            pts=[(256+sign*(42+i*25),128-i*6),(256+sign*(85+i*31),70-i*9),(256+sign*(120+i*35),35-i*4)]
            draw.line([(x*scale,y*scale) for x,y in pts],fill=(218,241,250,190-i*25),width=(6-i)*scale)
    im.resize((512,256),Image.Resampling.LANCZOS).save(d/'impact_air.png')


def ribbon(samples,t,bone_index):
    vertices=[];uvs=[];triangles=[];segments=36
    # Three ribbons follow the real edited tool path over the previous 0.10 seconds.
    for band in (-1,0,1):
        offset=band*82;thickness=145 if band==0 else 55;base=len(uvs)//2
        for i in range(segments+1):
            u=i/segments;tt=max(.475,t-.10)+(t-max(.475,t-.10))*u;point=sample_point(samples,tt)
            tangent=sample_point(samples,tt+.004)-sample_point(samples,tt-.004);length=np.linalg.norm(tangent)
            normal=np.array([-tangent[1],tangent[0]])/max(length,1e-8)
            center=point+normal*offset;half=thickness*(.1+.9*math.sin(math.pi*u)**.6)/2
            for sign,v in [(-1,0),(1,1)]:
                xy=center+normal*half*sign;uvs.extend([u,v]);vertices.extend([1,bone_index,float(xy[0]),float(xy[1]),1])
        for i in range(segments):
            a=base+i*2;b=a+1;c=a+2;d=a+3;triangles.extend([a,b,d,a,d,c])
    return {'type':'mesh','path':'fx/air_streak','width':512,'height':128,'uvs':uvs,'triangles':triangles,'vertices':vertices}


def quad(bone_index,width,height):
    uv=[0,0,1,0,1,1,0,1];vertices=[]
    for x,y in [(-width/2,height/2),(width/2,height/2),(width/2,-height/2),(-width/2,-height/2)]:vertices.extend([1,bone_index,x,y,1])
    return {'type':'mesh','path':'fx/impact_air','width':width,'height':height,'uvs':uv,'vertices':vertices,'triangles':[0,1,2,0,2,3],'hull':4}


def pack_atlas():
    page=Image.new('RGBA',(1024,512));regions=[('air_streak',4,4),('impact_air',4,140)]
    text=(BASE/'source/original-atlas.txt').read_text().rstrip()+'\n\nsjt-airflow-fx.png\nsize:1024,512\nfilter:Linear,Linear\npma:true\n'
    for name,x,y in regions:
        im=Image.open(BASE/'images/fx'/f'{name}.png').convert('RGBA');a=np.asarray(im).copy();a[:,:,:3]=np.rint(a[:,:,:3].astype(float)*a[:,:,3:4]/255).astype('uint8');page.paste(Image.fromarray(a),(x,y))
        text+=f'fx/{name}\nbounds:{x},{y},{im.width},{im.height}\n'
    page.save(BASE/'sjt-airflow-fx.png');(BASE/'sjt-4.3.atlas.txt').write_text(text)


def build():
    actor=json.loads((BASE/'source/uploaded-skeleton.json').read_text());samples=report_samples();make_effects()
    actor['skeleton'].pop('hash',None);actor['skeleton'].update(x=-850,y=-80,width=1600,height=1400)
    trail_index=len(actor['bones']);impact_index=trail_index+1;contact=sample_point(samples,CONTACT)
    actor['bones'] += [{'name':'airflow_anchor','parent':'root'},{'name':'impact_anchor','parent':'root','x':round(float(contact[0]),5),'y':8}]
    actor['slots'].insert(0,{'name':'airflow_trail','bone':'airflow_anchor','blend':'additive'})
    actor['slots'].append({'name':'impact_air','bone':'impact_anchor','blend':'additive'})
    skin=actor['skins'][0]['attachments'];skin['airflow_trail']={};keys=[{'time':0,'name':None}]
    for frame in range(15,20):
        t=frame/30;name=f'airflow_{frame:02d}';skin['airflow_trail'][name]=ribbon(samples,t,trail_index);keys.append({'time':t,'name':name})
    keys.append({'time':20/30,'name':None});skin['impact_air']={'impact_air':quad(impact_index,520,260)}
    for name,a in actor['animations'].items():
        slots=a.setdefault('slots',{})
        for n in FX_SLOTS:slots[n]={'attachment':[{'time':0,'name':None}],'alpha':[{'time':0,'value':0}]}
        if name=='attack_2':
            slots['airflow_trail']={'attachment':keys,'alpha':[{'time':0,'value':0},{'time':14/30,'value':0},{'time':START,'value':.9},{'time':16/30,'value':1},{'time':.6,'value':.9},{'time':20/30,'value':0}]}
            slots['impact_air']={'attachment':[{'time':0,'name':None},{'time':CONTACT,'name':'impact_air'},{'time':END,'name':None}],'alpha':[{'time':0,'value':0},{'time':17/30,'value':0},{'time':CONTACT,'value':1},{'time':19/30,'value':.9},{'time':.75,'value':.45},{'time':END,'value':0}]}
            a['bones']['impact_anchor']={'scale':[{'time':0,'x':1,'y':1},{'time':CONTACT-.001,'x':.28,'y':.28},{'time':CONTACT,'x':.35,'y':.35},{'time':19/30,'x':.8,'y':.7},{'time':.75,'x':1.35,'y':.95},{'time':END,'x':1.85,'y':1.1}]}
    write_json(BASE/'sjt-4.3.json',actor);pack_atlas();print('Built airflow: original 17 bones/8 slots/6 action curves preserved; added 2 bones, 2 VFX slots, 5 ribbon meshes and pressure ring.')
    return actor


def runtime_render(actor,images,report,animation,t,size=(1200,900),camera=(650,780),factor=.65,background=None,hide_fx=False):
    item=min((v for v in report['samples'] if v['animation']==animation),key=lambda v:abs(v['time']-t));world=matrices(item,factor);view=actor
    if hide_fx:
        view=deepcopy(actor);view['slots']=[s for s in view['slots'] if s['name'] not in FX_SLOTS]
    # Authoritative official-runtime matrices preserve uploaded Bezier curves and omitted values.
    with patch('tools.build_bsq_spine.pose',lambda *args:world):
        return shared_render(view,images,animation,item['time'],size=size,camera=camera,background=background)


def previews():
    actor=json.loads((BASE/'sjt-4.3.json').read_text());report=json.loads((BASE/'runtime-validation.json').read_text());images=load_images(BASE/'images');directory=BASE/'previews'
    stage=(1000,840);camera=(520,775);factor=.58;bg=Image.new('RGBA',stage,'#27343e');d=ImageDraw.Draw(bg)
    for x in range(0,stage[0],20):
        for y in range(0,stage[1],20):
            if (x//20+y//20)%2:d.rectangle((x,y,x+19,y+19),fill='#2d3d47')
    d.line((0,camera[1],stage[0],camera[1]),fill='#6e9f90')
    for name,info in report['animations'].items():
        frames=[];mobile=[];compare=[];count=round(info['duration']*30)
        for f in range(count+1):
            t=f/30;im=runtime_render(actor,images,report,name,t,size=stage,camera=camera,factor=factor,background=bg).convert('RGB');frame=im.resize((840,706),Image.Resampling.LANCZOS);ImageDraw.Draw(frame).text((12,12),f'{name} / official-runtime pose / {t:.2f}s',fill='#9cdedc');frames.append(frame)
            phone=Image.new('RGB',(420,300),'#172027');phone.paste(im.resize((279,234),Image.Resampling.LANCZOS),(70,40));ImageDraw.Draw(phone).text((10,8),f'{name} / character ~128px high',fill='#9cdedc');mobile.append(phone)
            if name=='attack_2':
                before=runtime_render(actor,images,report,name,t,size=stage,camera=camera,factor=factor,background=bg,hide_fx=True).convert('RGB').resize((600,504));pair=Image.new('RGB',(1200,530),'#172027');pair.paste(before,(0,26));pair.paste(im.resize((600,504)),(600,26));pd=ImageDraw.Draw(pair);pd.text((12,8),'BEFORE / uploaded animation',fill='#9cdedc');pd.text((612,8),'AFTER / airflow + pressure wave',fill='#9cdedc');compare.append(pair)
        save_gif(frames,directory/(name+'.gif'),name in ('stand','move','attack_1','attack_2'));save_gif(mobile,directory/('mobile-'+name+'.gif'),True)
        if compare:save_gif(compare,directory/'attack_2-comparison.gif',True)
    sheet=Image.new('RGB',(1440,630),'#172027');sd=ImageDraw.Draw(sheet)
    for i,t in enumerate([14/30,.5,16/30,17/30,.6,19/30,.7,.8]):
        im=runtime_render(actor,images,report,'attack_2',t,size=stage,camera=camera,factor=factor,background=bg).convert('RGB').resize((360,302),Image.Resampling.LANCZOS);x=(i%4)*360;y=(i//4)*315;sheet.paste(im,(x,y+13));sd.text((x+8,y),f'attack_2 / {t:.3f}s',fill='#9cdedc')
    sheet.save(directory/'airflow-contact.png');print('Built previews from official Spine 4.3 matrices, including before/after and mobile.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--previews-only',action='store_true');args=parser.parse_args();previews() if args.previews_only else build()
