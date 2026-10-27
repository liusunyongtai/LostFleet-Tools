from hashlib import sha256
import json
import math
import unittest
import cv2
import numpy as np
from PIL import Image
from tools.build_zsx_spine import (BASE,DURATIONS,pose,render,mesh_world,load_images,
                                   scaled_preview)
from tools.build_bsq_spine import draw_order
from tools.build_xqdz_spine import visible_attachments


class ZsxSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor=json.loads((BASE/'zsx-4.3.json').read_text())
        cls.bullet=json.loads((BASE/'poison-bullet-4.3.json').read_text())
        cls.source=json.loads((BASE/'source/photoshop.json').read_text())
        cls.config=json.loads((BASE/'rig-config.json').read_text())
        cls.images=load_images(BASE/'images');cls.view=scaled_preview(cls.actor,.4)
        cls.report=json.loads((BASE/'runtime-validation.json').read_text())
        cls.bullet_report=json.loads((BASE/'bullet-runtime-validation.json').read_text())

    def actor_image(self,name=None,t=0,skeleton=None):
        return render(skeleton or self.view,self.images,name,t,size=(960,720),camera=(560,600))

    def bullet_image(self,t):
        return render(self.bullet,self.images,'bullet',t,size=(640,360),camera=(320,180))

    def test_exact_six_actor_names_and_editable_version(self):
        self.assertEqual(set(self.actor['animations']),{'stand','dead','hit','attack_1','attack_2','move'})
        self.assertEqual(self.actor['skeleton']['spine'],'4.3.23');self.assertEqual(self.actor['skeleton']['images'],'./images/')
        self.assertEqual(len(self.actor['bones']),26);self.assertEqual(len(self.actor['slots']),5)
        self.assertEqual([s['name'] for s in self.actor['slots'] if s['name']!='spit_fx'],[s['name'] for s in self.source['slots']])

    def test_four_original_pngs_remain_unchanged_and_keep_alpha(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),4)
        for n,digest in hashes.items():
            p=BASE/'images'/n;self.assertEqual(sha256(p.read_bytes()).hexdigest(),digest)
            im=Image.open(p);self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255))

    def test_setup_mesh_positions_and_normalized_weights(self):
        world=pose(self.actor)
        for name,entries in self.actor['skins'][0]['attachments'].items():
            a=entries[name];uv=np.array(a['uvs']).reshape(-1,2)
            source=({'x':300,'y':-550,'width':300,'height':150} if name=='spit_fx' else self.source['skins']['default'][name][name])
            expected=np.column_stack((source['x']+(uv[:,0]-.5)*source['width']-733,source['y']+(.5-uv[:,1])*source['height']+1209))
            np.testing.assert_allclose(mesh_world(self.actor,a,world),expected,atol=.0001)
            data=a['vertices'];cursor=0;count=0
            while cursor<len(data):
                n=int(data[cursor]);cursor+=1;weights=[data[cursor+i*4+3] for i in range(n)]
                self.assertAlmostEqual(sum(weights),1,places=9);self.assertTrue(all(0<w<=1 for w in weights))
                cursor+=n*4;count+=1
            self.assertEqual(count,len(uv))

    def test_original_ps_assembly_is_preserved(self):
        size=(1500,1450);camera=(800,1300);reference=Image.new('RGBA',size)
        for slot in self.source['slots']:
            n=slot['name'];a=self.source['skins']['default'][n][n]
            reference.alpha_composite(self.images[n],(round(a['x']-a['width']/2+camera[0]-733),round(-a['y']-a['height']/2+camera[1]-1209)))
        actual=np.asarray(render(self.actor,self.images,size=size,camera=camera)).astype(np.int16);expected=np.asarray(reference).astype(np.int16)
        mask=(actual[:,:,3]>0)|(expected[:,:,3]>0);self.assertLess(np.mean(np.abs(actual-expected)[mask]),.3)

    def test_stand_move_loops_and_move_root_stays_in_place(self):
        for name in ('stand','move'):
            a=pose(self.actor,name,0);b=pose(self.actor,name,DURATIONS[name])
            for n in a:np.testing.assert_allclose(a[n],b[n],atol=.00004)
        for k in self.actor['animations']['move']['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])

    def test_feet_follow_the_original_scale_walk_paths(self):
        for f in range(31):
            t=f/30;world=pose(self.actor,'move',t)
            for n,d in self.config['limbs'].items():
                step=math.sin(t*math.tau+d['phase']);lift=max(step,0)**2
                target=np.array([d['ankle'][0]-32*step,d['ankle'][1]+18*lift])-self.config['origin']
                np.testing.assert_allclose(world[n+'_paw'][:2,2],target,atol=.003)

    def test_melee_retracts_then_reaches_forward(self):
        a=pose(self.actor,'attack_1',0);wind=pose(self.actor,'attack_1',.2);hit=pose(self.actor,'attack_1',.5)
        self.assertGreater(wind['melee_tip'][0,2]-a['melee_tip'][0,2],120)
        self.assertLess(hit['melee_tip'][0,2]-a['melee_tip'][0,2],-650)
        self.assertEqual(self.actor['animations']['attack_1']['events'],[{'time':.5,'name':'melee_hit'}])

    def test_spit_inflates_then_compresses_belly_and_pushes_mouth_forward(self):
        wind=pose(self.actor,'attack_2',.3);spit=pose(self.actor,'attack_2',.5)
        self.assertGreater(np.linalg.det(wind['sac'][:2,:2]),1.15)
        self.assertLess(np.linalg.det(spit['sac'][:2,:2]),1.)
        self.assertLess(spit['mouth'][0,2],wind['mouth'][0,2]-240)
        self.assertEqual(self.actor['animations']['attack_2']['events'],[{'time':.5,'name':'spawn_bullet'}])
        self.assertEqual(self.actor['events']['spawn_bullet']['string'],'projectile_spawn')

    def test_spit_fx_is_visible_only_during_emission(self):
        for t,expected in [(0,False),(.49,False),(.5,True),(.6,True),(.7,False),(1.4,False)]:
            names={s['name'] for s,a in visible_attachments(self.actor,'attack_2',t)};self.assertEqual('spit_fx' in names,expected)
        for name in ('stand','dead','hit','attack_1','move'):
            self.assertNotIn('spit_fx',{s['name'] for s,a in visible_attachments(self.actor,name,DURATIONS[name]/2)})

    def test_spit_trailing_edge_stays_at_mouth_while_liquid_grows(self):
        for t in (.5,.55,.6,.65):
            world=pose(self.actor,'attack_2',t)
            tail=(world['spit_fx']@[145,-10,1])[:2]
            np.testing.assert_allclose(tail,world['mouth'][:2,2],atol=.001)
        setup=pose(self.actor)
        self.assertAlmostEqual(setup['mouth'][0,2]-setup['projectile_spawn'][0,2],175,places=4)

    def test_body_and_head_silhouettes_remain_connected_at_key_poses(self):
        animations={k:{x:y for x,y in a.items() if x!='drawOrder'} for k,a in self.view['animations'].items()}
        for layer in ('shenti','tou'):
            q={**self.view,'animations':animations,'slots':[s for s in self.view['slots'] if s['name']==layer]}
            for name,t in [('move',.25),('move',.75),('attack_1',.5),('attack_2',.3),('attack_2',.5),('dead',1.6)]:
                mask=(np.asarray(self.actor_image(name,t,q))[:,:,3]>16).astype(np.uint8)
                _,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
                self.assertEqual(int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>30)),1,(layer,name,t))

    def test_death_is_grounded_without_relying_only_on_tongue(self):
        for t in (0,.2,.4,.6,.8,1.,1.2,1.4,1.6):self.assertLessEqual(self.actor_image('dead',t).getbbox()[3],601)
        a=pose(self.actor,'dead',1.4);b=pose(self.actor,'dead',1.6)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.00004)
        self.assertLess(abs(b['face_contact'][1,2]),35)

    def test_attacks_hit_recover_and_no_adjacent_key_full_turn(self):
        setup=pose(self.actor)
        for name in ('attack_1','attack_2','hit'):
            end=pose(self.actor,name,DURATIONS[name])
            for n in setup:np.testing.assert_allclose(setup[n],end[n],atol=.00004)
        for name,animation in self.actor['animations'].items():
            for bone,tracks in animation['bones'].items():
                keys=tracks.get('rotate',[])
                for a,b in zip(keys,keys[1:]):self.assertLess(abs(a['value']-b['value']),45,(name,bone))

    def test_preview_scale_is_uniform_and_does_not_change_animation(self):
        for name,t in [('stand',.6),('attack_1',.5),('attack_2',.6),('dead',1.6)]:
            a=pose(self.actor,name,t);b=pose(self.view,name,t)
            for n in a:
                np.testing.assert_allclose(b[n][:2,:],a[n][:2,:]*.4,atol=.00001)

    def compare_runtime(self,skeleton,report,filename,expected_count):
        self.assertEqual(report['skeletonSHA256'],sha256((BASE/filename).read_bytes()).hexdigest())
        self.assertFalse(report['desktopEditorTested']);self.assertEqual(report['sampleHz'],60);self.assertEqual(len(report['samples']),expected_count)
        skin=skeleton['skins'][0]['attachments']
        for sample in report['samples']:
            name,t=sample['animation'],sample['time'];world=pose(skeleton,name,t)
            for n,values in sample['bones'].items():
                m=world[n];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],values,atol=.005)
            self.assertEqual([s['name'] for s in draw_order(skeleton,name,t)],sample['drawOrder'])
            self.assertEqual({s['name'] for s,a in visible_attachments(skeleton,name,t)},{n for n,a in sample['attachments'].items() if a})
            for n,r in sample['meshes'].items():
                points=mesh_world(skeleton,skin[n][n],world);np.testing.assert_allclose(points[r['indices']].ravel(),r['vertices'],atol=.005)

    def test_official_runtime_matches_all_character_poses(self):
        self.compare_runtime(self.actor,self.report,'zsx-4.3.json',492)

    def test_projectile_is_independent_with_one_bullet_animation(self):
        self.assertEqual(set(self.bullet['animations']),{'bullet'});self.assertEqual(self.bullet['skeleton']['spine'],'4.3.23')
        self.assertEqual(len(self.bullet['bones']),3);self.assertEqual(len(self.bullet['slots']),1)
        self.assertEqual(self.bullet['skins'][0]['attachments']['poison']['poison']['path'],'fx/poison-flight')

    def test_bullet_root_never_moves_and_loop_seam_is_identical(self):
        for k in self.bullet['animations']['bullet']['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])
        a=pose(self.bullet,'bullet',0);b=pose(self.bullet,'bullet',.6)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.000001)
        np.testing.assert_array_equal(np.asarray(self.bullet_image(0)),np.asarray(self.bullet_image(.6)))

    def test_new_projectile_png_is_transparent_and_identified_as_generated(self):
        provenance=json.loads((BASE/'source/projectile-image.json').read_text());p=BASE/provenance['path']
        self.assertEqual(sha256(p.read_bytes()).hexdigest(),provenance['sha256'])
        image=Image.open(p);self.assertEqual(image.mode,'RGBA');self.assertEqual(image.getchannel('A').getextrema(),(0,255))
        self.assertEqual(image.getpixel((0,0))[3],0)
        alpha=np.asarray(image)[:,:,3];self.assertGreater(alpha[:,:image.width//3].sum(),alpha[:,-image.width//3:].sum()*1.5)

    def test_bullet_changes_shape_while_staying_a_single_slime_glob(self):
        start=np.asarray(self.bullet_image(0));quarter=np.asarray(self.bullet_image(.15));self.assertGreater(np.count_nonzero(start!=quarter),1000)
        for t in (0,.15,.3,.45,.6):
            mask=(np.asarray(self.bullet_image(t))[:,:,3]>16).astype(np.uint8);_,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
            self.assertEqual(int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>40)),1,t)

    def test_official_runtime_matches_all_bullet_poses(self):
        self.compare_runtime(self.bullet,self.bullet_report,'poison-bullet-4.3.json',37)


if __name__=='__main__':unittest.main(verbosity=2)
