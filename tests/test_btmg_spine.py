from hashlib import sha256
import json
import math
import unittest

import cv2
import numpy as np
from PIL import Image
from tools.build_btmg_spine import (BASE,CAMERA,DURATIONS,SIZE,mesh_world,pose,render,
                                    rotation,load_images)


class BtmgSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=json.loads((BASE/'btmg-4.3.json').read_text())
        cls.config=json.loads((BASE/'rig-config.json').read_text())
        cls.source=json.loads((BASE/'source/photoshop.json').read_text())
        cls.images=load_images(BASE/'images')
        cls.report=json.loads((BASE/'runtime-validation.json').read_text())

    def test_exact_names_version_and_source_layers(self):
        self.assertEqual(set(self.skeleton['animations']),{'stand','dead','hit','attack','move'})
        self.assertEqual(self.skeleton['skeleton']['spine'],'4.3.23')
        self.assertEqual(len(self.skeleton['bones']),22)
        self.assertEqual([s['name'] for s in self.skeleton['slots']],[s['name'] for s in self.source['slots']])
        self.assertEqual(self.skeleton['skeleton']['images'],'./images/')

    def test_all_seven_original_pngs_are_unchanged(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text())
        self.assertEqual(len(hashes),7)
        for name,digest in hashes.items():
            file=BASE/'images'/name
            self.assertEqual(sha256(file.read_bytes()).hexdigest(),digest)
            im=Image.open(file);self.assertEqual(im.mode,'RGBA')
            low,high=im.getchannel('A').getextrema();self.assertEqual(low,0);self.assertGreater(high,200)

    def test_setup_vertices_preserve_ps_image_positions_and_weights_sum_one(self):
        world=pose(self.skeleton)
        for name,entries in self.skeleton['skins'][0]['attachments'].items():
            a=entries[name];self.assertEqual(a['type'],'mesh')
            source=self.source['skins']['default'][name][name];uv=np.array(a['uvs']).reshape(-1,2)
            expected=np.column_stack((source['x']+(uv[:,0]-.5)*source['width']-310,
                                      source['y']+(.5-uv[:,1])*source['height']+440))
            np.testing.assert_allclose(mesh_world(self.skeleton,a,world),expected,atol=.0001)
            data=a['vertices'];cursor=0;count=0
            while cursor<len(data):
                n=int(data[cursor]);cursor+=1
                weights=[data[cursor+i*4+3] for i in range(n)]
                self.assertAlmostEqual(sum(weights),1,places=9)
                self.assertTrue(all(0<w<=1 for w in weights));cursor+=n*4;count+=1
            self.assertEqual(count,len(uv));self.assertLess(max(a['triangles']),count)

    def test_setup_render_keeps_original_assembly(self):
        reference=Image.new('RGBA',SIZE)
        for slot in self.source['slots']:
            n=slot['name'];a=self.source['skins']['default'][n][n]
            xy=(round(a['x']-a['width']/2+CAMERA[0]-310),round(-a['y']-a['height']/2+CAMERA[1]-440))
            reference.alpha_composite(self.images[n],xy)
        a=np.asarray(render(self.skeleton,self.images)).astype(float);b=np.asarray(reference).astype(float)
        mask=(a[:,:,3]>0)|(b[:,:,3]>0)
        self.assertLess(np.mean(np.abs(a-b)[mask]),.3)

    def test_loops_match_and_move_root_has_no_displacement(self):
        for name in ('stand','move'):
            a=pose(self.skeleton,name,0);b=pose(self.skeleton,name,DURATIONS[name])
            for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.00004)
        for key in self.skeleton['animations']['move']['bones']['root']['translate']:
            self.assertEqual([key['x'],key['y']],[0,0])

    def test_four_paws_follow_reachable_trot_paths(self):
        for frame in range(25):
            t=frame/30;phase=t/.8*math.tau;world=pose(self.skeleton,'move',t)
            for name,limb in self.config['limbs'].items():
                step=math.sin(phase+limb['phase']);lift=max(0,step)**2
                expected=np.array([limb['ankle'][0]-30*step,limb['ankle'][1]+22*lift])-self.config['origin']
                np.testing.assert_allclose(world[name+'_paw'][:2,2],expected,atol=.003)

    def test_diagonal_pairs_lift_together(self):
        for t,raised in [(.2,{'near_front','far_hind'}),(.6,{'far_front','near_hind'})]:
            world=pose(self.skeleton,'move',t)
            for name,limb in self.config['limbs'].items():
                lift=world[name+'_paw'][1,2]-(limb['ankle'][1]+440)
                self.assertAlmostEqual(lift,22 if name in raised else 0,places=3)

    def test_interpolated_rotations_cannot_make_full_turns_between_keys(self):
        for name,animation in self.skeleton['animations'].items():
            for bone,tracks in animation['bones'].items():
                keys=tracks.get('rotate',[])
                for first,second in zip(keys,keys[1:]):
                    self.assertLess(abs(second['value']-first['value']),45,(name,bone))

    def test_attack_has_backwards_anticipation_and_forward_lunge(self):
        start=pose(self.skeleton,'attack',0);wind=pose(self.skeleton,'attack',6/30);strike=pose(self.skeleton,'attack',12/30)
        self.assertGreater(wind['hips'][0,2]-start['hips'][0,2],15)
        self.assertLess(strike['hips'][0,2]-start['hips'][0,2],-50)
        self.assertLess(strike['bite_point'][0,2]-start['bite_point'][0,2],-65)
        self.assertEqual(self.skeleton['animations']['attack']['events'],[{'time':.4,'name':'bite_hit'}])

    def test_jaws_open_during_windup_and_close_at_impact(self):
        definitions={b['name']:b for b in self.config['bones']}
        def separation(t):
            world=pose(self.skeleton,'attack',t);tips=[]
            for name,tip in [('jaw_near',[109,-320]),('jaw_far',[60,-302])]:
                bone=definitions[name];local=rotation(-bone['rotation'])@(np.array(tip)-bone['position'])
                tips.append((world[name]@[local[0],local[1],1])[:2])
            return np.linalg.norm(tips[0]-tips[1])
        self.assertGreater(separation(6/30),separation(12/30)*1.5)

    def test_attack_and_hit_recover(self):
        setup=pose(self.skeleton)
        for name in ('attack','hit'):
            end=pose(self.skeleton,name,DURATIONS[name])
            for bone in setup:np.testing.assert_allclose(setup[bone],end[bone],atol=.00004)

    def test_death_is_grounded_and_stable_at_end(self):
        for t in (0,.2,.4,.6,.8,1.,1.2,1.4,1.6):
            im=render(self.skeleton,self.images,'dead',t,size=(2200,1700),camera=(1000,1200))
            self.assertLessEqual(im.getbbox()[3],1201)
        final=pose(self.skeleton,'dead',1.6);settled=pose(self.skeleton,'dead',1.4)
        for bone in final:np.testing.assert_allclose(final[bone],settled[bone],atol=.00004)

    def test_leg_silhouettes_remain_connected_at_trot_extremes(self):
        for name in self.config['limbs']:
            image_name=self.config['limbs'][name]['image'];single={**self.skeleton,'slots':[s for s in self.skeleton['slots'] if s['name']==image_name]}
            for t in (.2,.6):
                image=render(single,self.images,'move',t)
                mask=(np.asarray(image)[:,:,3]>16).astype(np.uint8)
                _,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
                self.assertEqual(int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>50)),1,(name,t))

    def test_runtime_bones_and_sampled_mesh_match_preview_even_between_keys(self):
        self.assertEqual(self.report['skeletonSHA256'],sha256((BASE/'btmg-4.3.json').read_bytes()).hexdigest())
        self.assertFalse(self.report['desktopEditorTested']);self.assertEqual(len(self.report['samples']),383)
        self.assertEqual(self.report['sampleHz'],60)
        skin=self.skeleton['skins'][0]['attachments']
        for sample in self.report['samples']:
            world=pose(self.skeleton,sample['animation'],sample['time'])
            for name,values in sample['bones'].items():
                m=world[name];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],values,atol=.005)
            for name,record in sample['meshes'].items():
                points=mesh_world(self.skeleton,skin[name][name],world)
                self.assertEqual(len(points),record['count'])
                np.testing.assert_allclose(points[record['indices']].ravel(),record['vertices'],atol=.005)


if __name__=='__main__':unittest.main(verbosity=2)
