"""Source fidelity, joint contacts, VFX visibility, loops, and official-runtime agreement."""
from copy import deepcopy
from hashlib import sha256
import json
import math
import unittest
import cv2
import numpy as np
from PIL import Image
from tools.build_zhshz_spine import BASE,DURATIONS,curve,pose,render,mesh_world,load_images,scaled_preview
from tools.build_bsq_spine import draw_order
from tools.build_xqdz_spine import visible_attachments,sample

class ZhshzSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor=json.loads((BASE/'zhshz-4.3.json').read_text());cls.bullet=json.loads((BASE/'poison-bullet-4.3.json').read_text())
        cls.source=json.loads((BASE/'source/photoshop.json').read_text());cls.config=json.loads((BASE/'rig-config.json').read_text());cls.images=load_images(BASE/'images')

    def test_names_version_bones_slots_and_original_layer_order(self):
        self.assertEqual(set(self.actor['animations']),{'stand','dead','hit','move','attack_1','attack_2'})
        self.assertEqual(len(self.actor['bones']),21);self.assertEqual(len(self.actor['slots']),16)
        self.assertEqual([s['name'] for s in self.actor['slots'] if s.get('attachment')],[s['name'] for s in self.source['slots']])
        for a in (self.actor,self.bullet):self.assertEqual(a['skeleton']['spine'],'4.3.23');self.assertEqual(a['skeleton']['images'],'./images/')

    def test_fourteen_original_pngs_are_byte_identical_and_transparent(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),14)
        for name,h in hashes.items():
            self.assertEqual(sha256((BASE/'images'/name).read_bytes()).hexdigest(),h);im=Image.open(BASE/'images'/name)
            self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255))

    def test_setup_mesh_coordinates_preserve_every_source_layer(self):
        world=pose(self.actor);origin=np.array(self.config['origin']);skin=self.actor['skins'][0]['attachments']
        for name in self.source['skins']['default']:
            a=skin[name][name];r=self.source['skins']['default'][name][name];uv=np.array(a['uvs']).reshape(-1,2)
            expected=np.column_stack((r['x']+(uv[:,0]-.5)*r['width'],r['y']+(.5-uv[:,1])*r['height']))-origin
            np.testing.assert_allclose(mesh_world(self.actor,a,world),expected,atol=.0001)
            data=a['vertices'];i=0
            while i<len(data):
                count=int(data[i]);i+=1;weights=[data[i+j*4+3] for j in range(count)]
                self.assertAlmostEqual(sum(weights),1,places=8);self.assertTrue(all(0<w<=1 for w in weights));i+=count*4

    def test_rendered_setup_preserves_photoshop_assembly(self):
        size=(1500,1450);camera=(900,1300);ox,oy=self.config['origin'];reference=Image.new('RGBA',size)
        for slot in self.source['slots']:
            n=slot['name'];a=self.source['skins']['default'][n][n]
            reference.alpha_composite(self.images[n],(round(a['x']-a['width']/2-ox+camera[0]),round(-a['y']-a['height']/2+oy+camera[1])))
        a=np.asarray(render(self.actor,self.images,size=size,camera=camera)).astype(float);b=np.asarray(reference).astype(float);mask=(a[:,:,3]>0)|(b[:,:,3]>0)
        self.assertLess(np.abs(a-b)[mask].mean(),.3)

    def test_stand_move_loops_and_action_recovery(self):
        for name in ('stand','move','attack_1','attack_2','hit'):
            start=pose(self.actor,name,0);end=pose(self.actor,name,DURATIONS[name])
            for n in start:np.testing.assert_allclose(start[n],end[n],atol=.0001)
        for k in self.actor['animations']['move']['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])

    def test_walk_feet_hit_unclamped_paths(self):
        for f in range(31):
            t=f/30;w=pose(self.actor,'move',t)
            for n,d in self.config['limbs'].items():
                step=math.sin(t*math.tau+d['phase']);target=np.array([d['ankle'][0]-50*step,d['ankle'][1]+35*max(step,0)**2])-self.config['origin']
                np.testing.assert_allclose(w[n+'_paw'][:2,2],target,atol=.003)

    def test_claw_arm_reaches_without_ik_clamping_at_every_key(self):
        for f in range(37):
            t=f/30;wind=curve(t,[(0,0),(.2,1),(.23,1),(.5,0),(1.2,0)]);reach=curve(t,[(0,0),(.23,0),(.5,1),(16/30,.97),(.95,-.08),(1.2,0)]);rake=curve(t,[(0,0),(.5,0),(.7,1),(.95,-.15),(1.2,0)])
            target=np.array([968+32*wind-523*max(reach,0),-920+160*wind+130*max(reach,0)-40*rake-(90*math.sin(math.pi*max(0,min(1,reach))) if t<=.5 else 0)])-self.config['origin']
            np.testing.assert_allclose(pose(self.actor,'attack_1',t)['front_arm_paw'][:2,2],target,atol=.003)

    def test_melee_wind_reach_rake_and_event(self):
        rest=pose(self.actor,'attack_1',0);wind=pose(self.actor,'attack_1',.23);strike=pose(self.actor,'attack_1',.5);rake=pose(self.actor,'attack_1',.7)
        self.assertGreater(wind['front_arm_paw'][0,2],rest['front_arm_paw'][0,2]+25)
        self.assertLess(strike['front_arm_paw'][0,2],wind['front_arm_paw'][0,2]-500)
        self.assertLess(rake['front_arm_paw'][1,2],strike['front_arm_paw'][1,2]-15)
        self.assertEqual(self.actor['animations']['attack_1']['events'],[{'time':.5,'name':'claw_hit'}])
        names=[s['name'] for s in draw_order(self.actor,'attack_1',.5)]
        self.assertGreater(names.index('zuoshou'),names.index('tou'))

    def test_spit_anticipation_head_forward_and_event(self):
        wind=pose(self.actor,'attack_2',.3);spit=pose(self.actor,'attack_2',.5)
        self.assertLess(spit['mouth'][0,2],wind['mouth'][0,2]-250)
        self.assertGreater(np.linalg.det(wind['body'][:2,:2]),np.linalg.det(spit['body'][:2,:2]))
        self.assertEqual(self.actor['animations']['attack_2']['events'],[{'time':.5,'name':'spawn_bullet'}])
        self.assertEqual(self.actor['events']['spawn_bullet']['string'],'projectile_spawn')

    def test_fx_hidden_in_setup_and_other_actions(self):
        for name,t in [(None,0),('stand',1),('move',.5),('hit',.1),('dead',1)]:
            visible={s['name'] for s,a in visible_attachments(self.actor,name,t)};self.assertFalse(visible&{'claw_fx','spit_fx'})
        for name,fx,times in [('attack_1','claw_fx',[(.49,False),(.5,True),(.7,True),(.8,False)]),('attack_2','spit_fx',[(.49,False),(.5,True),(.6,True),(.7,False)])]:
            for t,on in times:self.assertEqual(fx in {s['name'] for s,a in visible_attachments(self.actor,name,t)},on)

    def test_red_slashes_have_true_alpha_and_visible_red_pixels(self):
        im=self.images['fx/claw_fx'];self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema()[0],0)
        arr=np.asarray(im);red=(arr[:,:,0]>180)&(arr[:,:,0]>arr[:,:,1]*1.5)&(arr[:,:,3]>200);self.assertGreater(red.sum(),5000)
        a=np.asarray(render(scaled_preview(self.actor,.15),self.images,'attack_1',.5,size=(400,300),camera=(260,245)))
        red=(a[:,:,0].astype(float)>a[:,:,1]*1.8)&(a[:,:,0]>150)&(a[:,:,3]>100);self.assertGreater(red.sum(),40)

    def test_spit_jet_edge_is_attached_to_mouth(self):
        a=self.actor['skins'][0]['attachments']['spit_fx']['spit_fx'];uv=np.array(a['uvs']).reshape(-1,2)
        for t in (.5,.55,.6,.65):
            world=pose(self.actor,'attack_2',t);points=mesh_world(self.actor,a,world);right=points[uv[:,0]>.99].mean(axis=0)
            np.testing.assert_allclose(right,world['mouth'][:2,2],atol=.0001)

    def test_head_crown_delays_and_body_breathes(self):
        animation=self.actor['animations']['stand']['bones'];self.assertGreater(max(k['value'] for k in animation['crown']['rotate'])-min(k['value'] for k in animation['crown']['rotate']),9)
        self.assertGreater(max(k['y'] for k in animation['body']['scale']),1)
        a=pose(self.actor,'move',.25);b=pose(self.actor,'move',.75);self.assertGreater(np.linalg.norm(a['front_arm_paw'][:2,2]-b['front_arm_paw'][:2,2]),100)

    def test_death_grounded_with_head_contact_and_settled(self):
        a=pose(self.actor,'dead',1.4);b=pose(self.actor,'dead',1.6)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.0001)
        im=render(self.actor,self.images,'dead',1.6);self.assertLessEqual(abs(im.getbbox()[3]-1330),1)
        s=deepcopy(self.actor);s['slots']=[slot for slot in s['slots'] if slot['name']=='tou']
        self.assertLessEqual(abs(render(s,self.images,'dead',1.6).getbbox()[3]-1330),5)

    def test_no_full_turn_adjacent_rotation_jumps(self):
        for name,a in self.actor['animations'].items():
            for n,tracks in a['bones'].items():
                keys=tracks.get('rotate',[])
                for x,y in zip(keys,keys[1:]):self.assertLess(abs(x['value']-y['value']),45,(name,n))

    def test_scaled_preview_is_uniform_and_does_not_modify_source(self):
        s=scaled_preview(self.actor,.4)
        for name,t in [('stand',1),('attack_1',.5),('attack_2',.5),('dead',1.6)]:
            a=pose(self.actor,name,t);b=pose(s,name,t)
            for n in a:np.testing.assert_allclose(a[n][:2,:]*.4,b[n][:2,:],atol=.00001)
        self.assertNotIn('scaleX',self.actor['bones'][0])

    def compare_runtime(self,actor,filename,reportname,count):
        report=json.loads((BASE/reportname).read_text());self.assertEqual(report['skeletonSHA256'],sha256((BASE/filename).read_bytes()).hexdigest());self.assertFalse(report['desktopEditorTested']);self.assertEqual(len(report['samples']),count)
        skin=actor['skins'][0]['attachments']
        for item in report['samples']:
            name,t=item['animation'],item['time'];world=pose(actor,name,t)
            for n,v in item['bones'].items():
                m=world[n];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],v,atol=.005)
            self.assertEqual([s['name'] for s in draw_order(actor,name,t)],item['drawOrder'])
            self.assertEqual({s['name'] for s,a in visible_attachments(actor,name,t)},{s for s,a in item['attachments'].items() if a})
            for n,r in item['meshes'].items():
                points=mesh_world(actor,skin[n][item['attachments'][n]],world);np.testing.assert_allclose(points[r['indices']].ravel(),r['vertices'],atol=.005)
            for n,alpha in item['alpha'].items():self.assertAlmostEqual(sample(actor['animations'][name].get('slots',{}).get(n,{}).get('alpha',[]),t,'value',1),alpha,places=5)

    def test_official_runtime_character_492_poses(self):self.compare_runtime(self.actor,'zhshz-4.3.json','runtime-validation.json',492)
    def test_official_runtime_bullet_37_poses(self):self.compare_runtime(self.bullet,'poison-bullet-4.3.json','bullet-runtime-validation.json',37)

    def test_projectile_source_hash_and_alpha(self):
        p=json.loads((BASE/'source/projectile-image.json').read_text());self.assertEqual(sha256((BASE/p['path']).read_bytes()).hexdigest(),p['sha256']);self.assertIn('Reused',p['kind'])
        im=Image.open(BASE/p['path']);self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255));self.assertEqual(im.getpixel((0,0))[3],0)

    def test_projectile_stays_in_place_and_loops(self):
        self.assertEqual(set(self.bullet['animations']),{'bullet'})
        for k in self.bullet['animations']['bullet']['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])
        a=pose(self.bullet,'bullet',0);b=pose(self.bullet,'bullet',.6)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.000001)
        np.testing.assert_array_equal(np.asarray(render(self.bullet,self.images,'bullet',0)),np.asarray(render(self.bullet,self.images,'bullet',.6)))

    def test_projectile_changes_shape_and_is_one_main_glob(self):
        a=np.asarray(render(self.bullet,self.images,'bullet',0,size=(640,360),camera=(320,180)));b=np.asarray(render(self.bullet,self.images,'bullet',.15,size=(640,360),camera=(320,180)));self.assertGreater(np.count_nonzero(a!=b),1000)
        for t in (0,.15,.3,.45,.6):
            mask=(np.asarray(render(self.bullet,self.images,'bullet',t,size=(640,360),camera=(320,180)))[:,:,3]>16).astype('uint8');_,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
            self.assertEqual(int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>40)),1)

if __name__=='__main__':unittest.main(verbosity=2)
