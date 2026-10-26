"""Build a portable, editable Spine 4.3 rigid-region animation draft.

Only the supplied nine original images are used; no hidden anatomy is generated.
Gun flash is an explicitly procedural placeholder effect, not an edited source PNG.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1] / 'examples' / 'xqdz-spine'
FPS = 30
DURATIONS = {'idle': 2.0, 'attack': 0.8, 'hit': 0.4, 'death': 1.4, 'move': 0.8}
SHOT = 8 / FPS


def rotation(degrees):
    r = math.radians(degrees)
    return np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])


def rounded(value):
    value = float(value)
    return round(value, 6) if abs(value) > 1e-7 else 0


def build_skeleton(source, config):
    """Inverse parent transforms preserve every source attachment's setup rectangle."""
    origin = np.array(config['origin'], dtype=float)
    definitions = {b['name']: b for b in config['bones']}
    result = {'skeleton': {'spine': config['target_version'], 'images': './images/',
                           'fps': FPS, 'x': -300, 'y': 0, 'width': 440, 'height': 460},
              'bones': [], 'slots': [], 'skins': [],
              'events': {'shoot': {'string': 'muzzle'}, 'footstep': {'string': 'left'},
                         'impact': {'string': 'body'}, 'death_end': {}}, 'animations': {}}
    for b in config['bones']:
        parent = definitions.get(b.get('parent'))
        position = np.array(b['position'], dtype=float)
        local = rotation(-parent['rotation']) @ (position - parent['position']) if parent else position - origin
        entry = {'name': b['name'], 'x': rounded(local[0]), 'y': rounded(local[1]),
                 'rotation': rounded(b['rotation'] - (parent['rotation'] if parent else 0)), 'length': b['length']}
        if parent:
            entry['parent'] = parent['name']
        result['bones'].append(entry)
    skin = {}
    for slot in source['slots']:
        name = slot['name']; bone_name = config['bindings'][name]; bone = definitions[bone_name]
        attachment = deepcopy(source['skins']['default'][name][name])
        local = rotation(-bone['rotation']) @ (np.array([attachment.get('x', 0), attachment.get('y', 0)]) - bone['position'])
        attachment.update(x=rounded(local[0]), y=rounded(local[1]), rotation=rounded(-bone['rotation']), path=name)
        result['slots'].append({'name': name, 'bone': bone_name, 'attachment': name})
        skin[name] = {name: attachment}
    # Hidden in setup and in every non-attack animation, to prevent flash leakage.
    result['slots'].append({'name': 'muzzle_flash', 'bone': 'muzzle', 'blend': 'additive'})
    skin['muzzle_flash'] = {f'muzzle_flash_{i}': {'path': f'fx/muzzle_flash_{i}', 'x': 40,
                                'y': 0, 'width': 80, 'height': 56} for i in range(1, 4)}
    result['skins'] = [{'name': 'default', 'attachments': skin}]
    return result


def interpolate_controls(t, controls):
    if t <= controls[0][0]: return controls[0][1]
    for (t0, a), (t1, b) in zip(controls, controls[1:]):
        if t <= t1:
            u = (t-t0)/(t1-t0)
            u = u*u*(3-2*u)
            return a+(b-a)*u
    return controls[-1][1]


def make_animations(config):
    definitions = {b['name']: b for b in config['bones']}
    animations = {}
    for name, duration in DURATIONS.items():
        timelines = {bone: {'rotate': [], 'translate': [], 'scale': []} for bone in definitions}
        for frame in range(round(duration*FPS)+1):
            t = frame/FPS; phase = t/duration*math.tau
            r = {bone: 0.0 for bone in definitions}
            p = {bone: [0.0, 0.0] for bone in definitions}
            s = {bone: [1.0, 1.0] for bone in definitions}
            if name == 'idle':
                breath = (1-math.cos(phase))/2
                p['body'][1] = 2.2*breath; r['body'] = 0.9*math.sin(phase)
                r['head'] = -1.8*math.sin(phase); r['aim'] = -0.7*math.sin(phase)
                s['body'] = [1+0.003*breath, 1+0.004*breath]
                r['antenna_01'] = 4*math.sin(phase)
                r['antenna_02'] = 3*(math.sin(phase-0.5)-math.sin(-0.5))
            elif name == 'move':
                stride = math.sin(phase); bob = 2.5*(1-math.cos(phase*2))
                p['pelvis'][1] = bob; r['pelvis'] = 1.5*stride
                r['right_leg'] = 15*stride; r['left_leg'] = -15*stride
                p['right_leg'] = [0, -bob + max(0, -stride)*6]
                p['left_leg'] = [0, -bob + max(0, stride)*6]
                r['body'] = 1.8*stride; r['head'] = -2*stride
                r['aim'] = -1.5*stride
                r['antenna_01'] = 7*stride; r['antenna_02'] = -5*stride
            elif name == 'attack':
                # Point left; recoil moves the weapon and both attached arms right.
                recoil = interpolate_controls(t, [(0,0),(6/FPS,0),(8/FPS,0),(9/FPS,1),(13/FPS,0),(duration,0)])
                aim = interpolate_controls(t, [(0,0),(5/FPS,-2),(8/FPS,-2),(9/FPS,-5),(14/FPS,0),(duration,0)])
                r['aim'] = aim; p['aim'][0] = 8*recoil
                r['body'] = -2*recoil; r['head'] = -1.5*recoil
                r['antenna_01'] = 6*recoil; r['antenna_02'] = -5*recoil
            elif name == 'hit':
                shock = interpolate_controls(t, [(0,0),(3/FPS,1),(7/FPS,0.3),(duration,0)])
                r['body'] = -9*shock; r['head'] = -8*shock
                p['body'] = [5*shock,-2*shock]; r['aim'] = 6*shock
                r['antenna_01'] = 16*shock; r['antenna_02'] = -13*shock
            elif name == 'death':
                fall = interpolate_controls(t, [(0,0),(0.2,0.05),(0.8,1),(1.0,0.96),(1.2,1),(duration,1)])
                p['pelvis'] = [85*fall,-82*fall]; r['pelvis'] = -82*fall
                r['body'] = -4*fall; r['head'] = 6*fall; r['aim'] = 28*fall
                r['right_leg'] = 16*fall; r['left_leg'] = -8*fall
                r['antenna_01'] = -15*fall; r['antenna_02'] = 10*fall
            for bone in definitions:
                parent = definitions.get(definitions[bone].get('parent'))
                local = rotation(-parent['rotation']) @ p[bone] if parent else p[bone]
                timelines[bone]['rotate'].append({'time': rounded(t), 'value': rounded(r[bone])})
                timelines[bone]['translate'].append({'time': rounded(t), 'x': rounded(local[0]), 'y': rounded(local[1])})
                timelines[bone]['scale'].append({'time': rounded(t), 'x': rounded(s[bone][0]), 'y': rounded(s[bone][1])})
        # Keep identity rotations/translations only on root for duration/reset;
        # other unused channels are omitted to keep editing manageable.
        for bone in list(timelines):
            for kind in list(timelines[bone]):
                channel = timelines[bone][kind]
                keys = ('value',) if kind == 'rotate' else ('x','y')
                neutral = 1 if kind == 'scale' else 0
                if bone != 'root' and all(all(k[key] == neutral for key in keys) for k in channel):
                    del timelines[bone][kind]
            if not timelines[bone]: del timelines[bone]
        flash = [{'time':0, 'name':None}]
        if name == 'attack':
            flash += [{'time':rounded((8+i)/FPS),'name':f'muzzle_flash_{i+1}'} for i in range(3)]
            flash += [{'time':rounded(11/FPS),'name':None}]
        animations[name] = {'bones': timelines, 'slots': {'muzzle_flash': {'attachment': flash}}}
        if name == 'attack': animations[name]['events'] = [{'time':rounded(SHOT),'name':'shoot'}]
        if name == 'move': animations[name]['events'] = [{'time':0.2,'name':'footstep','string':'left'}, {'time':0.6,'name':'footstep','string':'right'}]
        if name == 'hit': animations[name]['events'] = [{'time':0.1,'name':'impact'}]
        if name == 'death': animations[name]['events'] = [{'time':1.2,'name':'death_end'}]
    return animations


def sample(channel, t, key, default):
    if not channel: return default
    if t <= channel[0]['time']: return channel[0].get(key,default)
    for a,b in zip(channel,channel[1:]):
        if t <= b['time']:
            u=(t-a['time'])/(b['time']-a['time'])
            return a.get(key,default)+(b.get(key,default)-a.get(key,default))*u
    return channel[-1].get(key,default)


def pose(skeleton, animation=None, t=0):
    world={};tracks=skeleton['animations'].get(animation,{}).get('bones',{})
    for b in skeleton['bones']:
        channels=tracks.get(b['name'],{})
        x=b.get('x',0)+sample(channels.get('translate'),t,'x',0)
        y=b.get('y',0)+sample(channels.get('translate'),t,'y',0)
        angle=b.get('rotation',0)+sample(channels.get('rotate'),t,'value',0)
        scale=[sample(channels.get('scale'),t,'x',1),sample(channels.get('scale'),t,'y',1)]
        matrix=np.eye(3);matrix[:2,:2]=rotation(angle) @ np.diag(scale);matrix[:2,2]=[x,y]
        world[b['name']]=world[b['parent']] @ matrix if b.get('parent') else matrix
    return world


def visible_attachments(skeleton, animation=None, t=0):
    tracks=skeleton['animations'].get(animation,{}).get('slots',{})
    skin=skeleton['skins'][0]['attachments']
    for slot in skeleton['slots']:
        name=slot.get('attachment')
        for key in tracks.get(slot['name'],{}).get('attachment',[]):
            if key['time']<=t+1e-6:name=key.get('name')
        if name:
            yield slot,skin[slot['name']][name]


def attachment_matrix(world,slot,attachment,image):
    local=np.eye(3);local[:2,:2]=rotation(attachment.get('rotation',0))
    local[:2,2]=[attachment.get('x',0),attachment.get('y',0)]
    pixels=np.array([[attachment['width']/image.width,0,-attachment['width']/2],
                     [0,-attachment['height']/image.height,attachment['height']/2],[0,0,1.]])
    return world[slot['bone']] @ local @ pixels


def render(skeleton, images, animation=None, t=0, size=(960,640), camera=(360,540), rig=False):
    out=Image.new('RGBA',size)
    world=pose(skeleton,animation,t)
    screen=np.array([[1,0,camera[0]],[0,-1,camera[1]],[0,0,1.]])
    for slot,a in visible_attachments(skeleton,animation,t):
        image=images[a.get('path',slot['name'])]
        transform=screen @ attachment_matrix(world,slot,a,image)
        corners=transform @ np.array([[0,image.width,image.width,0],[0,0,image.height,image.height],[1,1,1,1]])
        x0,y0=np.floor(corners[:2].min(axis=1)).astype(int);x1,y1=np.ceil(corners[:2].max(axis=1)).astype(int)
        x0=max(0,x0);y0=max(0,y0);x1=min(size[0],x1);y1=min(size[1],y1)
        if x1<=x0 or y1<=y0:continue
        inverse=np.linalg.inv(transform);offset=np.array([[1,0,x0],[0,1,y0],[0,0,1]])
        coeff=(inverse @ offset)[:2].ravel()
        tile=image.transform((x1-x0,y1-y0),Image.Transform.AFFINE,coeff,Image.Resampling.BICUBIC)
        out.alpha_composite(tile,(x0,y0))
    if rig:
        draw=ImageDraw.Draw(out)
        for b in skeleton['bones']:
            m=screen @ world[b['name']];x,y=m[:2,2]
            if b.get('parent'):
                parent=screen @ world[b['parent']];draw.line((parent[0,2],parent[1,2],x,y),fill='#5fffc2',width=2)
            draw.ellipse((x-3,y-3,x+3,y+3),fill='#ffcb70');draw.text((x+5,y-12),b['name'],fill='#ffdf9d')
    return out


def make_flash(directory):
    directory.mkdir(parents=True,exist_ok=True)
    for i in range(1,4):
        image=Image.new('RGBA',(80,56));d=ImageDraw.Draw(image)
        tips=[[(0,28),(23,21),(15,5),(37,17),(75,10),(52,28),(80,43),(37,37),(18,53),(23,34)],
              [(0,28),(20,18),(26,2),(41,19),(80,25),(48,35),(56,51),(29,39),(11,46),(19,32)],
              [(0,28),(20,22),(30,12),(37,23),(65,28),(36,34),(26,44),(19,33)]][i-1]
        d.polygon(tips,fill=(255,119,24,220));d.polygon([(1,28),(24,22),(54,27),(29,34)],fill=(255,221,98,255))
        d.polygon([(0,28),(20,25),(37,28),(20,31)],fill=(255,255,227,255))
        image.save(directory/f'muzzle_flash_{i}.png')


def load_images(directory):
    return {p.relative_to(directory).with_suffix('').as_posix():Image.open(p).convert('RGBA') for p in directory.rglob('*.png')}


def make_previews(skeleton, images, output):
    output.mkdir(parents=True,exist_ok=True)
    background=Image.new('RGBA',(960,640),'#29333e');d=ImageDraw.Draw(background)
    for x in range(0,960,20):
        for y in range(0,640,20):
            if (x//20+y//20)%2:d.rectangle((x,y,x+19,y+19),fill='#2d3843')
    d.line((0,540,960,540),fill='#678777',width=1)
    framesheet=Image.new('RGB',(1200,5*280),'#202832');label=ImageDraw.Draw(framesheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[]
        for frame in range(round(duration*FPS)+1):
            composed=background.copy();composed.alpha_composite(render(skeleton,images,name,frame/FPS))
            composed=composed.convert('RGB').resize((600,400),Image.Resampling.LANCZOS)
            ImageDraw.Draw(composed).text((12,12),f'{name} / {frame:02d} / {frame/FPS:.2f}s',fill='#a8e9cb')
            frames.append(composed)
        # A shared palette keeps the checkerboard stable and makes GIFs smaller.
        palette_sample=Image.new('RGB',(600,800));palette_sample.paste(frames[0],(0,0))
        palette_sample.paste(frames[min(8,len(frames)-1)],(0,400))
        palette=palette_sample.quantize(colors=256)
        indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
        args={'save_all':True,'append_images':indexed[1:],'duration':[30 if i%3<2 else 40 for i in range(len(frames))],'optimize':True}
        if name in ('idle','move','attack'):args['loop']=0
        indexed[0].save(output/f'{name}.gif',**args)
        snapshots=[0,8,10,24] if name=='attack' else [round((len(frames)-1)*v) for v in (0,.3,.65,1)]
        for column,index in enumerate(snapshots):
            thumbnail=frames[min(index,len(frames)-1)].resize((300,200),Image.Resampling.LANCZOS)
            framesheet.paste(thumbnail,(column*300,row*280+25))
        label.text((12,row*280+6),name,fill='#a8e9cb')
    framesheet.save(output/'animation-contact.png')
    setup=background.copy();setup.alpha_composite(render(skeleton,images,rig=True));setup.convert('RGB').save(output/'rig-setup.png')


def build(source_path, images_path, config_path, output, previews=True):
    source=json.loads(source_path.read_text(encoding='utf8'));config=json.loads(config_path.read_text(encoding='utf8'))
    for name, entries in source['skins']['default'].items():
        image=Image.open(images_path/f'{name}.png')
        attachment=entries[name]
        if image.size != (attachment['width'],attachment['height']):raise ValueError(f'Image dimensions mismatch: {name}')
    skeleton=build_skeleton(source,config);skeleton['animations']=make_animations(config)
    make_flash(images_path/'fx');images=load_images(images_path)
    # Rigid collapse contact: adjust root Y so the final visible pixels meet ground.
    fall=render(skeleton,images,'death',DURATIONS['death'],size=(1600,1400),camera=(700,800))
    correction=fall.getbbox()[3]-800
    for key in skeleton['animations']['death']['bones']['root']['translate']:
        factor=interpolate_controls(key['time'],[(0,0),(.2,0),(.8,1),(DURATIONS['death'],1)])
        key['y']=rounded(correction*factor)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(skeleton,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if previews:make_previews(skeleton,images,output.parent/'previews')
    print(f'Built {output}: {len(skeleton["bones"])} bones, {len(skeleton["animations"])} animations; death contact correction {correction}px')
    return skeleton


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=BASE/'source'/'photoshop.json')
    parser.add_argument('--images',type=Path,default=BASE/'images')
    parser.add_argument('--config',type=Path,default=BASE/'rig-config.json')
    parser.add_argument('--output',type=Path,default=BASE/'xqdz-4.3.json')
    parser.add_argument('--no-previews',action='store_true')
    args=parser.parse_args();build(args.source,args.images,args.config,args.output,not args.no_previews)


if __name__=='__main__':main()
