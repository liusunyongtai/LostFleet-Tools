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
from PIL import Image, ImageDraw, ImageChops

BASE = Path(__file__).resolve().parents[1] / 'examples' / 'xqdz-spine'
FPS = 30
DURATIONS = {'idle': 2.0, 'attack': 0.8, 'hit': 0.4, 'death': 1.4, 'move': 0.8}
SHOT = 8 / FPS
LIT_PARTS = ('wuqi', 'zuoshoubni', 'youshoubi', 'toubvu', 'qugan')
LIGHT_SLOTS = tuple('shot_light_' + name for name in LIT_PARTS) + ('muzzle_glow',)
PREVIEW_SIZE = (1120, 640)
PREVIEW_CAMERA = (450, 540)


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
    for name in LIT_PARTS:
        light_name = 'shot_light_' + name
        result['slots'].append({'name': light_name, 'bone': config['bindings'][name], 'blend': 'additive'})
        attachment = deepcopy(skin[name][name]);attachment['path'] = 'fx/' + light_name
        skin[light_name] = {light_name: attachment}
    result['slots'].append({'name': 'muzzle_glow', 'bone': 'muzzle', 'blend': 'additive'})
    skin['muzzle_glow'] = {'muzzle_glow': {'path': 'fx/muzzle_glow', 'x': -15,
                                        'y': 0, 'width': 360, 'height': 280}}
    result['slots'].append({'name': 'muzzle_flash', 'bone': 'muzzle', 'blend': 'additive'})
    # Twice v1's width and height, still anchored at the muzzle rather than its center.
    skin['muzzle_flash'] = {f'muzzle_flash_{i}': {'path': f'fx/muzzle_flash_{i}', 'x': 80,
                                'y': 0, 'width': 160, 'height': 112} for i in range(1, 4)}
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
                p['body'] = [2.5*math.sin(phase),6.5*breath]
                r['body'] = 3.6*math.sin(phase)
                r['head'] = -9.5*(math.sin(phase-.12)-math.sin(-.12))
                r['aim'] = -2.6*math.sin(phase)
                s['body'] = [1+0.018*breath,1-0.010*breath]
                r['antenna_01'] = 14*(math.sin(phase-.35)-math.sin(-.35))
                r['antenna_02'] = 11*(math.sin(phase-.65)-math.sin(-.65))
            elif name == 'move':
                stride = math.sin(phase)+.12*math.sin(2*phase)
                bob = 4.8*(1-math.cos(phase*2))
                p['pelvis'][1] = bob; r['pelvis'] = 3.2*math.sin(phase)
                r['right_leg'] = 26*stride; r['left_leg'] = -26*stride
                p['right_leg'] = [0,-bob+max(0,-math.sin(phase))*13]
                p['left_leg'] = [0,-bob+max(0,math.sin(phase))*13]
                r['body'] = 3.8*math.sin(phase)+2.5*(1-math.cos(2*phase))
                r['head'] = -5.5*math.sin(phase)-1.5*(1-math.cos(2*phase))
                r['aim'] = -3.5*math.sin(phase)
                r['antenna_01'] = 18*(math.sin(phase-.35)-math.sin(-.35))
                r['antenna_02'] = 14*(math.sin(phase-.65)-math.sin(-.65))
            elif name == 'attack':
                # Anticipation, a fast recoil accent, an overshoot, then recovery.
                recoil = interpolate_controls(t,[(0,0),(8/FPS,0),(9/FPS,1),(11/FPS,.55),(15/FPS,-.18),(duration,0)])
                brace = interpolate_controls(t,[(0,0),(5/FPS,1),(8/FPS,.8),(9/FPS,0),(14/FPS,0),(duration,0)])
                r['body'] = 4*brace-8*recoil
                r['aim'] = -4*brace-5*recoil
                p['aim'][0] = 6*recoil
                p['body'] = [4*recoil,-3*recoil-2*brace]
                lag = interpolate_controls(t,[(0,0),(8/FPS,0),(10/FPS,1),(13/FPS,-.25),(duration,0)])
                r['head'] = -8*lag
                r['antenna_01'] = 23*lag; r['antenna_02'] = -18*lag
                s['body'] = [1-.018*recoil,1+.012*recoil]
            elif name == 'hit':
                shock = interpolate_controls(t,[(0,0),(3/FPS,1),(5/FPS,.85),(8/FPS,-.18),(duration,0)])
                lag = interpolate_controls(t,[(0,0),(4/FPS,1),(7/FPS,-.22),(duration,0)])
                r['body'] = -20*shock; r['head'] = -15*lag
                p['body'] = [12*shock,-5*shock]; r['aim'] = 14*shock
                r['antenna_01'] = 30*lag; r['antenna_02'] = -24*lag
                s['body'] = [1-.035*max(0,shock),1+.022*max(0,shock)]
            elif name == 'death':
                fall = interpolate_controls(t,[(0,0),(.2,0),(.8,1),(1.0,.965),(1.2,1),(duration,1)])
                buckle = interpolate_controls(t,[(0,0),(.2,1),(.45,0),(duration,0)])
                impact = interpolate_controls(t,[(0,0),(.72,0),(.85,1),(1.05,-.2),(1.2,0),(duration,0)])
                p['pelvis'] = [95*fall,-84*fall-7*buckle]; r['pelvis'] = -85*fall+5*buckle
                r['body'] = -6*fall-14*buckle-4*impact
                r['head'] = 10*fall-12*buckle+10*impact
                r['aim'] = 35*fall-8*buckle
                r['right_leg'] = 23*fall; r['left_leg'] = -14*fall
                r['antenna_01'] = -23*fall+20*impact
                r['antenna_02'] = 16*fall-18*impact
            for bone in definitions:
                parent = definitions.get(definitions[bone].get('parent'))
                local = rotation(-parent['rotation']) @ p[bone] if parent else p[bone]
                timelines[bone]['rotate'].append({'time': t, 'value': rounded(r[bone])})
                timelines[bone]['translate'].append({'time': t, 'x': rounded(local[0]), 'y': rounded(local[1])})
                timelines[bone]['scale'].append({'time': t, 'x': rounded(s[bone][0]), 'y': rounded(s[bone][1])})
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
            flash += [{'time':(8+i)/FPS,'name':f'muzzle_flash_{i+1}'} for i in range(3)]
            flash += [{'time':11/FPS,'name':None}]
        slot_tracks = {'muzzle_flash': {'attachment': flash}}
        for slot_name in LIGHT_SLOTS:
            keys=[{'time':0,'name':None}]
            alpha=[{'time':0,'value':0}]
            if name=='attack':
                keys += [{'time':SHOT,'name':slot_name}, {'time':14/FPS,'name':None}]
                # Keep the lead-in black, so interpolated alpha never lights before firing.
                alpha += [{'time':7/FPS,'value':0}, {'time':SHOT,'value':1},
                          {'time':9/FPS,'value':.60}, {'time':10/FPS,'value':.30},
                          {'time':12/FPS,'value':.08}, {'time':14/FPS,'value':0}]
            slot_tracks[slot_name]={'attachment':keys,'alpha':alpha}
        animations[name] = {'bones': timelines, 'slots': slot_tracks}
        if name == 'attack': animations[name]['events'] = [{'time':SHOT,'name':'shoot'}]
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


def additive_over(destination, source):
    """Preview additive light on a straight-alpha surface (exact RGB for opaque backgrounds)."""
    dst=np.asarray(destination,dtype=np.float32)/255
    src=np.asarray(source,dtype=np.float32)/255
    alpha=src[:,:,3:4]+dst[:,:,3:4]*(1-src[:,:,3:4])
    rgb=np.minimum(1,dst[:,:,:3]*dst[:,:,3:4]+src[:,:,:3]*src[:,:,3:4])
    rgb=np.divide(rgb,alpha,out=np.zeros_like(rgb),where=alpha>0)
    return Image.fromarray(np.rint(np.clip(np.concatenate((rgb,alpha),axis=2),0,1)*255).astype(np.uint8))


def render(skeleton, images, animation=None, t=0, size=PREVIEW_SIZE, camera=PREVIEW_CAMERA, rig=False, background=None):
    out=background.copy() if background is not None else Image.new('RGBA',size)
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
        slot_track=skeleton['animations'].get(animation,{}).get('slots',{}).get(slot['name'],{})
        alpha=sample(slot_track.get('alpha'),t,'value',1)
        if alpha<1:
            tile.putalpha(tile.getchannel('A').point(lambda value:round(value*max(0,alpha))))
        if slot.get('blend')=='additive':
            out.paste(additive_over(out.crop((x0,y0,x1,y1)),tile),(x0,y0))
        else:
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
        image=Image.new('RGBA',(320,224));d=ImageDraw.Draw(image)
        tips=[[(0,28),(23,21),(15,5),(37,17),(75,10),(52,28),(80,43),(37,37),(18,53),(23,34)],
              [(0,28),(20,18),(26,2),(41,19),(80,25),(48,35),(56,51),(29,39),(11,46),(19,32)],
              [(0,28),(20,22),(30,12),(37,23),(65,28),(36,34),(26,44),(19,33)]][i-1]
        enlarge=lambda points:[(x*4,y*4) for x,y in points]
        d.polygon(enlarge(tips),fill=(255,119,24,220))
        d.polygon(enlarge([(1,28),(24,22),(54,27),(29,34)]),fill=(255,221,98,255))
        d.polygon(enlarge([(0,28),(20,25),(37,28),(20,31)]),fill=(255,255,227,255))
        image.resize((160,112),Image.Resampling.LANCZOS).save(directory/f'muzzle_flash_{i}.png')


def make_lighting(images_path, source, config):
    """Separate warm alpha masks; never edit any of the nine original PNG files."""
    directory=images_path/'fx';directory.mkdir(parents=True,exist_ok=True)
    y,x=np.mgrid[-1:1:256j,-1:1:256j]
    radial=np.maximum(0,(np.exp(-4*(x*x+y*y))-math.exp(-4))/(1-math.exp(-4)))
    glow=np.zeros((256,256,4),dtype=np.uint8);glow[:,:,:3]=[255,181,91]
    glow[:,:,3]=np.rint(52*radial).astype(np.uint8)
    Image.fromarray(glow).save(directory/'muzzle_glow.png')
    muzzle=next(b['position'] for b in config['bones'] if b['name']=='muzzle')
    for name in LIT_PARTS:
        original=Image.open(images_path/f'{name}.png').convert('RGBA')
        a=source['skins']['default'][name][name]
        py,px=np.mgrid[:original.height,:original.width]
        wx=a['x']+(px+.5-original.width/2)*a['width']/original.width
        wy=a['y']+(original.height/2-py-.5)*a['height']/original.height
        strength=.42*np.exp(-((wx-muzzle[0])**2+(wy-muzzle[1])**2)/(2*145**2))
        light=np.zeros((original.height,original.width,4),dtype=np.uint8);light[:,:,:3]=[255,198,110]
        light[:,:,3]=np.rint(np.asarray(original)[:,:,3]*strength).astype(np.uint8)
        Image.fromarray(light).save(directory/f'shot_light_{name}.png')


def load_images(directory):
    return {p.relative_to(directory).with_suffix('').as_posix():Image.open(p).convert('RGBA') for p in directory.rglob('*.png')}


def save_gif(frames, filename, loop):
    width,height=frames[0].size
    sample_image=Image.new('RGB',(width,height*2));sample_image.paste(frames[0],(0,0))
    sample_image.paste(frames[min(8,len(frames)-1)],(0,height))
    palette=sample_image.quantize(colors=256)
    indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    kwargs={'save_all':True,'append_images':indexed[1:],
            'duration':[30 if i%3<2 else 40 for i in range(len(frames))],'optimize':True}
    if loop:kwargs['loop']=0
    indexed[0].save(filename,**kwargs)


def make_mobile_previews(skeleton, baseline, images, background, output):
    """Two nominal 128px-high characters side-by-side, without auto enlarging them."""
    sheet=Image.new('RGB',(640,5*216),'#172027');draw=ImageDraw.Draw(sheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[]
        for frame in range(round(duration*FPS)+1):
            canvas=Image.new('RGB',(640,216),'#172027')
            label=ImageDraw.Draw(canvas)
            for column,data in enumerate((baseline,skeleton)):
                image=render(data,images,name,frame/FPS,background=background)
                image=image.convert('RGB').resize((320,183),Image.Resampling.LANCZOS)
                canvas.paste(image,(column*320,29))
                label.text((column*320+10,9),f'{"V1" if column==0 else "V2"} / {name} / character ~128px',fill='#a8e9cb')
            frames.append(canvas)
        save_gif(frames,output/f'mobile-compare-{name}.gif',True)
        selected=9 if name=='attack' else (3 if name=='hit' else (6 if name=='death' else round(duration*FPS*.3)))
        sheet.paste(frames[selected],(0,row*216))
    sheet.save(output/'mobile-comparison.png')


def make_previews(skeleton, images, output):
    output.mkdir(parents=True,exist_ok=True)
    background=Image.new('RGBA',PREVIEW_SIZE,'#29333e');d=ImageDraw.Draw(background)
    for x in range(0,PREVIEW_SIZE[0],20):
        for y in range(0,640,20):
            if (x//20+y//20)%2:d.rectangle((x,y,x+19,y+19),fill='#2d3843')
    d.line((0,540,PREVIEW_SIZE[0],540),fill='#678777',width=1)
    framesheet=Image.new('RGB',(1200,5*280),'#202832');label=ImageDraw.Draw(framesheet)
    for row,(name,duration) in enumerate(DURATIONS.items()):
        frames=[]
        for frame in range(round(duration*FPS)+1):
            composed=render(skeleton,images,name,frame/FPS,background=background)
            composed=composed.convert('RGB').resize((700,400),Image.Resampling.LANCZOS)
            ImageDraw.Draw(composed).text((12,12),f'{name} / {frame:02d} / {frame/FPS:.2f}s',fill='#a8e9cb')
            frames.append(composed)
        save_gif(frames,output/f'{name}.gif',name in ('idle','move','attack'))
        snapshots=[0,8,10,24] if name=='attack' else [round((len(frames)-1)*v) for v in (0,.3,.65,1)]
        for column,index in enumerate(snapshots):
            thumbnail=frames[min(index,len(frames)-1)].resize((300,171),Image.Resampling.LANCZOS)
            framesheet.paste(thumbnail,(column*300,row*280+25))
        label.text((12,row*280+6),name,fill='#a8e9cb')
    framesheet.save(output/'animation-contact.png')
    setup=render(skeleton,images,rig=True,background=background);setup.convert('RGB').save(output/'rig-setup.png')
    baseline_file=output.parent/'source/v1-animation-reference.json'
    if baseline_file.exists():
        baseline=json.loads(baseline_file.read_text(encoding='utf8'))
        make_mobile_previews(skeleton,baseline,images,background,output)


def build(source_path, images_path, config_path, output, previews=True):
    source=json.loads(source_path.read_text(encoding='utf8'));config=json.loads(config_path.read_text(encoding='utf8'))
    for name, entries in source['skins']['default'].items():
        image=Image.open(images_path/f'{name}.png')
        attachment=entries[name]
        if image.size != (attachment['width'],attachment['height']):raise ValueError(f'Image dimensions mismatch: {name}')
    skeleton=build_skeleton(source,config);skeleton['animations']=make_animations(config)
    make_flash(images_path/'fx');make_lighting(images_path,source,config);images=load_images(images_path)
    # Rigid collapse contact: adjust root Y so the final visible pixels meet ground.
    fall=render(skeleton,images,'death',DURATIONS['death'],size=(1600,1400),camera=(700,800))
    correction=fall.getbbox()[3]-800
    for key in skeleton['animations']['death']['bones']['root']['translate']:
        factor=interpolate_controls(key['time'],[(0,0),(.2,0),(.8,1),(DURATIONS['death'],1)])
        key['y']=rounded(correction*factor)
    # Larger anticipation/rotation must not push the silhouette below the floor.
    # This is a sampled rigid-body height correction, not an IK foot lock.
    for key in skeleton['animations']['death']['bones']['root']['translate']:
        frame=render(skeleton,images,'death',key['time'],size=(1600,1400),camera=(700,800))
        penetration=max(0,frame.getbbox()[3]-800)
        key['y']=rounded(key['y']+penetration)
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
