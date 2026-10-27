"""Independent checks for ZLXY source fidelity, contacts, lighting and runtime compatibility."""
from copy import deepcopy
from hashlib import sha256
import json
import math
import unittest
import numpy as np
from PIL import Image
from tools.build_zlxy_spine import BASE,DURATIONS,LIT_PARTS,pose,mesh_world,load_images,render
from tools.build_bsq_spine import draw_order
from tools.build_xqdz_spine import visible_attachments,sample

class ZlxySpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor=json.loads((BASE/'zlxy-4.3.json').read_text());cls.bullet=json.loads((BASE/'flying-bullet-4.3.json').read_text())
        cls.source=json.loads((BASE/'source/photoshop.json').read_text());cls.config=json.loads((BASE/'rig-config.json').read_text())
        cls.images=load_images(BASE/'images')

    def test_names_bones_slots_and_version(self):
        self.assertEqual(set(self.actor['animations']),set(DURATIONS));self.assertEqual(len(self.actor['bones']),25);self.assertEqual(len(self.actor['slots']),16)
        for a in (self.actor,self.bullet):self.assertEqual(a['skeleton']['spine'],'4.3.23');self.assertEqual(a['skeleton']['images'],'./images/')
        self.assertEqual([s['name'] for s in self.actor['slots'] if s.get('attachment')],[s['name'] for s in self.source['slots']])

    def test_nine_original_png_hashes_and_transparency(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),9)
        for name,h in hashes.items():
            p=BASE/'images'/name;self.assertEqual(sha256(p.read_bytes()).hexdigest(),h)
            im=Image.open(p);self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255))

    def test_setup_meshes_match_photoshop_positions(self):
        world=pose(self.actor);skin=self.actor['skins'][0]['attachments'];origin=np.array(self.config['origin'])
        for n in self.source['skins']['default']:
            a=skin[n][n];uv=np.array(a['uvs']).reshape(-1,2);r=self.source['skins']['default'][n][n]
            expected=np.column_stack((r['x']+(uv[:,0]-.5)*r['width'],r['y']+(.5-uv[:,1])*r['height']))-origin
            np.testing.assert_allclose(mesh_world(self.actor,a,world),expected,atol=.0001)
            data=a['vertices'];i=0
            while i<len(data):
                count=int(data[i]);i+=1;weights=[data[i+j*4+3] for j in range(count)];self.assertAlmostEqual(sum(weights),1,places=8);self.assertTrue(all(w>0 for w in weights));i+=count*4

    def test_setup_render_matches_original_assembly(self):
        size=(900,800);camera=(450,650);reference=Image.new('RGBA',size);ox,oy=self.config['origin']
        for s in self.source['slots']:
            n=s['name'];r=self.source['skins']['default'][n][n]
            reference.alpha_composite(self.images[n],(round(r['x']-r['width']/2-ox+camera[0]),round(-r['y']-r['height']/2+oy+camera[1])))
        actual=np.asarray(render(self.actor,self.images,size=size,camera=camera)).astype(float);expected=np.asarray(reference).astype(float);mask=(actual[:,:,3]>0)|(expected[:,:,3]>0)
        self.assertLess(np.abs(actual-expected)[mask].mean(),.3)

    def test_loops_and_action_recovery(self):
        for n in ('stand','move','hit','attack_1','attack_2'):
            a=pose(self.actor,n,0);b=pose(self.actor,n,DURATIONS[n])
            for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.00005)
        for k in self.actor['animations']['move']['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])

    def test_walk_foot_paths(self):
        for f in range(31):
            world=pose(self.actor,'move',f/30)
            for n,d in self.config['limbs'].items():
                step=math.sin(f/30*math.tau+d['phase']);target=np.array([d['ankle'][0]-18*step,d['ankle'][1]+12*max(step,0)**2])-self.config['origin']
                np.testing.assert_allclose(world[n+'_paw'][:2,2],target,atol=.003)

    def test_grips_follow_weapon_and_offhand_releases_for_melee(self):
        rest=pose(self.actor);origin=np.array(self.config['origin'])
        for name,duration in DURATIONS.items():
            if name=='dead':continue
            for f in range(round(duration*30)+1):
                t=f/30;world=pose(self.actor,name,t);matrix=world['weapon']@np.linalg.inv(rest['weapon'])
                for arm,point in self.config['grips'].items():
                    if arm=='back_arm' and name=='attack_1' and .0<t<duration:continue
                    target=(matrix@np.r_[np.array(point)-origin,1])[:2]
                    np.testing.assert_allclose(world[arm+'_paw'][:2,2],target,atol=.003,err_msg=f'{name} {t} {arm}')
        w=pose(self.actor,'attack_1',.35);target=(w['weapon']@np.linalg.inv(rest['weapon'])@np.r_[np.array(self.config['grips']['back_arm'])-origin,1])[:2]
        self.assertGreater(np.linalg.norm(target-w['back_arm_paw'][:2,2]),50)

    def test_melee_large_raise_forward_smash_and_layering(self):
        a=pose(self.actor,'attack_1',.35);b=pose(self.actor,'attack_1',.6)
        self.assertGreater(a['melee_tip'][1,2]-b['melee_tip'][1,2],150)
        self.assertLess(b['melee_tip'][0,2],a['melee_tip'][0,2]-100)
        self.assertEqual(self.actor['animations']['attack_1']['events'],[{'time':.6,'name':'melee_hit'}])
        names=[s['name'] for s in draw_order(self.actor,'attack_1',.35)]
        self.assertGreater(names.index('wuqi'),names.index('tou'));self.assertGreater(names.index('youbi'),names.index('wuqi'))

    def test_shoot_kick_and_event(self):
        a=pose(self.actor,'attack_2',11/30);b=pose(self.actor,'attack_2',.4)
        self.assertGreater(b['weapon'][0,2],a['weapon'][0,2]+10)
        self.assertEqual(self.actor['animations']['attack_2']['events'],[{'time':.4,'name':'shoot'}])
        self.assertEqual(self.actor['events']['shoot']['string'],'projectile_spawn')

    def test_flash_attachment_and_alpha_window(self):
        for name,duration in DURATIONS.items():
            for t in (0,duration/2,duration):
                if name=='attack_2':continue
                self.assertFalse(any(s['name'].startswith(('muzzle_','shot_light_')) for s,a in visible_attachments(self.actor,name,t)))
        for t,expected in [(0,False),(.39,False),(.4,True),(.46,True),(.6,False)]:
            self.assertEqual('muzzle_flash' in {s['name'] for s,a in visible_attachments(self.actor,'attack_2',t)},expected)
        track=self.actor['animations']['attack_2']['slots']['shot_light_tou']['alpha']
        self.assertAlmostEqual(sample(track,.4,'value',1),1);self.assertLess(sample(track,.5,'value',1),.2)

    def test_warm_surface_lighting_actually_brightens_armor_pixels(self):
        plain=deepcopy(self.actor)
        for s in plain['slots']:
            if s.get('blend')=='additive':s.pop('attachment',None)
        plain['animations']['attack_2']['slots']={}
        a=np.asarray(render(self.actor,self.images,'attack_2',.4)).astype(float);b=np.asarray(render(plain,self.images,'attack_2',.4)).astype(float)
        mask=b[:,:,3]>240;diff=(a[:,:,:3]-b[:,:,:3])[mask]
        self.assertGreater(diff[:,0].mean(),15);self.assertGreater(np.count_nonzero(diff[:,0]>50),1500)
        np.testing.assert_array_equal(np.asarray(render(self.actor,self.images,'attack_2',.8)),np.asarray(render(plain,self.images,'attack_2',.8)))

    def test_light_masks_have_same_mesh_and_occlude_with_surface(self):
        skin=self.actor['skins'][0]['attachments'];names=[s['name'] for s in self.actor['slots']]
        for n in LIT_PARTS:
            light='shot_light_'+n;self.assertEqual(names.index(light),names.index(n)+1)
            for field in ('vertices','uvs','triangles'):self.assertEqual(skin[n][n][field],skin[light][light][field])
            self.assertEqual(next(s for s in self.actor['slots'] if s['name']==light)['blend'],'additive')

    def test_rigid_weapon_and_attached_muzzle(self):
        skin=self.actor['skins'][0]['attachments'];rest=pose(self.actor);origin=np.array(self.config['origin'])
        original=mesh_world(self.actor,skin['wuqi']['wuqi'],rest);dist=np.linalg.norm(original[0]-original[2])
        for name,t in [('attack_1',.35),('attack_1',.6),('attack_2',.4),('dead',1.6)]:
            w=pose(self.actor,name,t);points=mesh_world(self.actor,skin['wuqi']['wuqi'],w)
            self.assertAlmostEqual(np.linalg.norm(points[0]-points[2]),dist,places=4)
            target=w['weapon']@np.linalg.inv(rest['weapon'])@np.r_[np.array([63,-281])-origin,1]
            np.testing.assert_allclose(w['muzzle'][:2,2],target[:2],atol=.0001)

    def test_death_is_grounded_and_settles(self):
        a=pose(self.actor,'dead',1.4);b=pose(self.actor,'dead',1.6)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.00005)
        head=deepcopy(self.actor);head['slots']=[s for s in head['slots'] if s['name']=='tou']
        self.assertLessEqual(abs(render(head,self.images,'dead',1.6).getbbox()[3]-700),5)
        im=render(self.actor,self.images,'dead',1.6,size=(1200,850),camera=(680,700));self.assertLessEqual(abs(im.getbbox()[3]-700),1)

    def test_no_adjacent_rotation_jumps(self):
        for name,a in self.actor['animations'].items():
            for bone,tracks in a['bones'].items():
                keys=tracks.get('rotate',[])
                for x,y in zip(keys,keys[1:]):self.assertLess(abs(x['value']-y['value']),45,(name,bone))

    def compare_runtime(self,actor,filename,reportname,count):
        report=json.loads((BASE/reportname).read_text());self.assertEqual(report['skeletonSHA256'],sha256((BASE/filename).read_bytes()).hexdigest())
        self.assertFalse(report['desktopEditorTested']);self.assertEqual(len(report['samples']),count);skin=actor['skins'][0]['attachments']
        for item in report['samples']:
            name,t=item['animation'],item['time'];world=pose(actor,name,t)
            for n,v in item['bones'].items():
                m=world[n];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],v,atol=.005)
            self.assertEqual([s['name'] for s in draw_order(actor,name,t)],item['drawOrder'])
            self.assertEqual({s['name'] for s,a in visible_attachments(actor,name,t)},{s for s,a in item['attachments'].items() if a})
            for n,r in item['meshes'].items():
                a=skin[n][item['attachments'][n]];points=mesh_world(actor,a,world)
                np.testing.assert_allclose(points[r['indices']].ravel(),r['vertices'],atol=.005)
            for n,alpha in item['alpha'].items():
                track=actor['animations'][name].get('slots',{}).get(n,{}).get('alpha',[])
                self.assertAlmostEqual(sample(track,t,'value',1),alpha,places=5)

    def test_official_runtime_character_480_poses(self):self.compare_runtime(self.actor,'zlxy-4.3.json','runtime-validation.json',480)
    def test_official_runtime_bullet_25_poses(self):self.compare_runtime(self.bullet,'flying-bullet-4.3.json','bullet-runtime-validation.json',25)

    def test_projectile_provenance_transparency_and_independent_animation(self):
        p=json.loads((BASE/'source/projectile-image.json').read_text());im=Image.open(BASE/p['path'])
        self.assertEqual(sha256((BASE/p['path']).read_bytes()).hexdigest(),p['sha256']);self.assertIn('AI',p['kind']);self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255))
        self.assertEqual(set(self.bullet['animations']),{'bullet'});self.assertEqual(len(self.bullet['bones']),3)

    def test_projectile_loop_root_stationary_metal_rigid_tail_animated(self):
        for k in self.bullet['animations']['bullet']['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])
        a=pose(self.bullet,'bullet',0);b=pose(self.bullet,'bullet',.4);quarter=pose(self.bullet,'bullet',.1)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.000001)
        np.testing.assert_allclose(a['core'],quarter['core'],atol=.000001);self.assertGreater(np.linalg.norm(a['tail']-quarter['tail']),1)
        np.testing.assert_array_equal(np.asarray(render(self.bullet,self.images,'bullet',0)),np.asarray(render(self.bullet,self.images,'bullet',.4)))

if __name__=='__main__':unittest.main(verbosity=2)
