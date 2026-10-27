from hashlib import sha256
import json
import math
import unittest

import cv2
import numpy as np
from PIL import Image
from tools.build_jbjs_spine import (BASE,CAMERA,DURATIONS,SIZE,mesh_world,pose,render,
                                    rotation,load_images)
from tools.build_bsq_spine import draw_order


class JbjsSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=json.loads((BASE/'jbjs-4.3.json').read_text())
        cls.config=json.loads((BASE/'rig-config.json').read_text())
        cls.source=json.loads((BASE/'source/photoshop.json').read_text())
        cls.images=load_images(BASE/'images')
        cls.report=json.loads((BASE/'runtime-validation.json').read_text())

    def test_exact_names_version_and_source_layers(self):
        self.assertEqual(set(self.skeleton['animations']),{'stand','dead','hit','attack','move'})
        self.assertEqual(self.skeleton['skeleton']['spine'],'4.3.23')
        self.assertEqual(len(self.skeleton['bones']),15)
        self.assertEqual([s['name'] for s in self.skeleton['slots']],[s['name'] for s in self.source['slots']])
        self.assertEqual(self.skeleton['skeleton']['images'],'./images/')

    def test_all_eight_original_pngs_are_unchanged(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text())
        self.assertEqual(len(hashes),8)
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
            expected=np.column_stack((source['x']+(uv[:,0]-.5)*source['width']-380,
                                      source['y']+(.5-uv[:,1])*source['height']+884))
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
            xy=(round(a['x']-a['width']/2+CAMERA[0]-380),round(-a['y']-a['height']/2+CAMERA[1]-884))
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

    def test_boots_follow_in_place_walk_paths(self):
        for frame in range(31):
            t=frame/30;world=pose(self.skeleton,'move',t)
            for name,d in self.config['limbs'].items():
                step=math.sin(t*math.tau+d['phase']);lift=max(step,0)**2
                expected=np.array([d['ankle'][0]-28*step,d['ankle'][1]+22*lift])-self.config['origin']
                np.testing.assert_allclose(world[name+'_paw'][:2,2],expected,atol=.003)

    def test_interpolated_rotations_cannot_make_full_turns_between_keys(self):
        for name,animation in self.skeleton['animations'].items():
            for bone,tracks in animation['bones'].items():
                keys=tracks.get('rotate',[])
                for first,second in zip(keys,keys[1:]):
                    self.assertLess(abs(second['value']-first['value']),45,(name,bone))

    def test_single_arm_claw_winds_back_then_reaches_forward_and_rakes(self):
        start=pose(self.skeleton,'attack',0);wind=pose(self.skeleton,'attack',.2);strike=pose(self.skeleton,'attack',.5);rake=pose(self.skeleton,'attack',.7)
        self.assertGreater(wind['claw_tip'][0,2]-start['claw_tip'][0,2],70)
        self.assertLess(strike['claw_tip'][0,2]-start['claw_tip'][0,2],-180)
        self.assertGreater(strike['claw_tip'][1,2]-start['claw_tip'][1,2],200)
        self.assertLess(rake['claw_tip'][1,2],strike['claw_tip'][1,2]-100)
        self.assertEqual(self.skeleton['animations']['attack']['events'],[{'time':.5,'name':'claw_hit'}])
        self.assertEqual([b['name'] for b in self.skeleton['bones'] if b['name'].endswith('_arm_upper')],['attack_arm_upper'])

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

    def test_leg_meshes_remain_connected_at_stride_extremes(self):
        for name in ('zuotui','youtui'):
            single={**self.skeleton,'slots':[s for s in self.skeleton['slots'] if s['name']==name]}
            for t in (.25,.75):
                mask=(np.asarray(render(single,self.images,'move',t))[:,:,3]>16).astype(np.uint8)
                _,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
                self.assertEqual(int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>50)),1,(name,t))

    def test_runtime_bones_and_sampled_mesh_match_preview_even_between_keys(self):
        self.assertEqual(self.report['skeletonSHA256'],sha256((BASE/'jbjs-4.3.json').read_bytes()).hexdigest())
        self.assertFalse(self.report['desktopEditorTested']);self.assertEqual(len(self.report['samples']),407)
        self.assertEqual(self.report['sampleHz'],60)
        skin=self.skeleton['skins'][0]['attachments']
        for sample in self.report['samples']:
            world=pose(self.skeleton,sample['animation'],sample['time'])
            for name,values in sample['bones'].items():
                m=world[name];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],values,atol=.005)
            self.assertEqual([s['name'] for s in draw_order(self.skeleton,sample['animation'],sample['time'])],sample['drawOrder'])
            for name,record in sample['meshes'].items():
                points=mesh_world(self.skeleton,skin[name][name],world)
                self.assertEqual(len(points),record['count'])
                np.testing.assert_allclose(points[record['indices']].ravel(),record['vertices'],atol=.005)


if __name__=='__main__':unittest.main(verbosity=2)
