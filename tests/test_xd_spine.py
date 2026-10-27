from hashlib import sha256
import json
import math
import unittest
import numpy as np
from PIL import Image
from tools.build_xd_spine import (BASE,CAMERA,DURATIONS,SIZE,RELEASE,mesh_world,pose,render,
                                  load_images,visible_attachments,draw_order)


class XdSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton=json.loads((BASE/'xd-4.3.json').read_text())
        cls.config=json.loads((BASE/'rig-config.json').read_text())
        cls.source=json.loads((BASE/'source/photoshop.json').read_text())
        cls.images=load_images(BASE/'images')
        cls.report=json.loads((BASE/'runtime-validation.json').read_text())

    def test_names_version_and_source_layer_order(self):
        self.assertEqual(set(self.skeleton['animations']),{'stand','dead','hit','attack','move'})
        self.assertEqual(self.skeleton['skeleton']['spine'],'4.3.23')
        self.assertEqual(len(self.skeleton['bones']),16)
        self.assertEqual(len(self.skeleton['slots']),10)
        self.assertEqual(self.skeleton['skeleton']['images'],'./images/')
        self.assertEqual([s['name'] for s in self.skeleton['slots'] if s['name']!='fork_flying'],[s['name'] for s in self.source['slots']])

    def test_all_nine_original_pngs_are_unchanged_and_transparent(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),9)
        for name,digest in hashes.items():
            file=BASE/'images'/name;self.assertEqual(sha256(file.read_bytes()).hexdigest(),digest)
            image=Image.open(file);self.assertEqual(image.mode,'RGBA')
            low,high=image.getchannel('A').getextrema();self.assertEqual(low,0);self.assertGreaterEqual(high,250)

    def test_setup_vertices_preserve_original_ps_positions(self):
        world=pose(self.skeleton)
        for name,entries in self.skeleton['skins'][0]['attachments'].items():
            a=entries[name];source_name='youshou' if name=='fork_flying' else name
            source=self.source['skins']['default'][source_name][source_name];uv=np.array(a['uvs']).reshape(-1,2)
            expected=np.column_stack((source['x']+(uv[:,0]-.5)*source['width']-320,
                                      source['y']+(.5-uv[:,1])*source['height']+680))
            np.testing.assert_allclose(mesh_world(self.skeleton,a,world),expected,atol=.0001)
            self.assertEqual(a['path'],source_name)
            self.assertEqual(a['vertices'][::5],[1]*len(uv))
            self.assertEqual(a['vertices'][4::5],[1.]*len(uv))

    def test_setup_render_matches_original_complete_assembly(self):
        reference=Image.new('RGBA',SIZE)
        for slot in self.source['slots']:
            n=slot['name'];a=self.source['skins']['default'][n][n]
            reference.alpha_composite(self.images[n],(round(a['x']-a['width']/2+CAMERA[0]-320),round(-a['y']-a['height']/2+CAMERA[1]-680)))
        actual=np.asarray(render(self.skeleton,self.images)).astype(float);expected=np.asarray(reference).astype(float)
        mask=(actual[:,:,3]>0)|(expected[:,:,3]>0)
        self.assertLess(np.mean(np.abs(actual-expected)[mask]),.3)

    def test_stand_and_move_loop_seamlessly_without_root_displacement(self):
        for name in ('stand','move'):
            first=pose(self.skeleton,name,0);last=pose(self.skeleton,name,DURATIONS[name])
            for b in first:np.testing.assert_allclose(first[b],last[b],atol=.00004)
            for key in self.skeleton['animations'][name]['bones']['root']['translate']:
                self.assertEqual([key['x'],key['y']],[0,0])

    def test_hover_does_not_touch_ground_in_stand_or_move(self):
        for name in ('stand','move'):
            for frame in range(round(DURATIONS[name]*30)+1):
                image=render(self.skeleton,self.images,name,frame/30)
                self.assertGreaterEqual(CAMERA[1]-image.getbbox()[3],40,(name,frame))

    def test_breathing_changes_body_height_and_oscillates_hover(self):
        low=pose(self.skeleton,'stand',0);high=pose(self.skeleton,'stand',1.2)
        self.assertGreater(np.linalg.norm(high['body'][:2,0]),np.linalg.norm(low['body'][:2,0])+.03)
        a=pose(self.skeleton,'stand',.6);b=pose(self.skeleton,'stand',1.8)
        self.assertGreater(a['hover'][1,2]-b['hover'][1,2],25)

    def test_disc_spins_in_projected_plane_and_keeps_hub_fixed(self):
        single={**self.skeleton,'slots':[s for s in self.skeleton['slots'] if s['name']=='dipan']}
        for frame in range(25):
            t=frame/30;world=pose(self.skeleton,'stand',t)
            np.testing.assert_allclose(world['disc_spin'][:2,2],world['disc_plane'][:2,2],atol=.00001)
            self.assertAlmostEqual(abs(np.linalg.det(world['disc_spin'][:2,:2])),.56,places=5)
            bounds=render(single,self.images,'stand',t).getbbox()
            self.assertGreater((bounds[2]-bounds[0])/(bounds[3]-bounds[1]),1.3)
        keys=self.skeleton['animations']['stand']['bones']['disc_spin']['rotate']
        self.assertEqual(keys[0]['value'],0);self.assertEqual(keys[-1]['value'],720)
        self.assertTrue(all(a['value']<b['value'] for a,b in zip(keys,keys[1:])))

    def test_move_body_leans_forward_while_hovering_in_place(self):
        for t in (0,.3,.6,.9,1.2):
            world=pose(self.skeleton,'move',t);body=world['body']
            self.assertGreater(math.degrees(math.atan2(body[1,0],body[0,0])),105)

    def test_throw_has_backwards_windup_and_large_forward_reach(self):
        start=pose(self.skeleton,'attack',0);wind=pose(self.skeleton,'attack',.2);release=pose(self.skeleton,'attack',RELEASE)
        self.assertGreater(wind['throw_point'][0,2]-start['throw_point'][0,2],70)
        self.assertLess(release['throw_point'][0,2]-start['throw_point'][0,2],-250)
        self.assertEqual(self.skeleton['animations']['attack']['events'],[{'time':RELEASE,'name':'throw_fork'}])

    def test_entire_fork_detaches_once_and_rearms_after_flight(self):
        for t,held,flying in [(0,True,False),(RELEASE-1/60,True,False),(RELEASE,False,True),(.7,False,True),(.95,False,False),(1.1,True,False),(1.2,True,False)]:
            names={s['name'] for s,a in visible_attachments(self.skeleton,'attack',t)}
            self.assertEqual('youshou' in names,held);self.assertEqual('fork_flying' in names,flying)
        skin=self.skeleton['skins'][0]['attachments']
        self.assertEqual(skin['fork_flying']['fork_flying']['path'],'youshou')
        self.assertEqual(skin['fork_flying']['fork_flying']['height'],169)

    def test_held_fork_does_not_sweep_through_saw_during_throw(self):
        animations={**self.skeleton['animations'],'attack':{k:v for k,v in self.skeleton['animations']['attack'].items() if k!='drawOrder'}}
        views={n:{**self.skeleton,'animations':animations,'slots':[s for s in self.skeleton['slots'] if s['name']==n]} for n in ('youshou','dipan')}
        for frame in range(round(RELEASE*30)):
            a=np.asarray(render(views['youshou'],self.images,'attack',frame/30))[:,:,3]
            b=np.asarray(render(views['dipan'],self.images,'attack',frame/30))[:,:,3]
            # Compare visible silhouettes, not transparent corners of sprite bounds.
            self.assertEqual(int(np.count_nonzero((a>32)&(b>32))),0,frame)

    def test_fork_handoff_has_no_position_scale_or_rotation_pop(self):
        world=pose(self.skeleton,'attack',RELEASE);skin=self.skeleton['skins'][0]['attachments']
        held=mesh_world(self.skeleton,skin['youshou']['youshou'],world)
        flying=mesh_world(self.skeleton,skin['fork_flying']['fork_flying'],world)
        np.testing.assert_allclose(held,flying,atol=.0001)
        later=pose(self.skeleton,'attack',.8)
        self.assertLess(later['fork_flight'][0,2],world['fork_flight'][0,2]-400)
        self.assertGreater(later['fork_flight'][1,2],world['fork_flight'][1,2])

    def test_group_draw_order_keeps_throwing_arm_in_front_of_head(self):
        order=[s['name'] for s in draw_order(self.skeleton,'attack',RELEASE)]
        self.assertLess(order.index('tou'),order.index('youbi'))
        self.assertLess(order.index('youbi'),order.index('youshou'))

    def test_attack_hit_recover_and_death_lands_and_settles(self):
        setup=pose(self.skeleton)
        for name in ('attack','hit'):
            end=pose(self.skeleton,name,DURATIONS[name])
            for b in setup:
                if b!='fork_flight':np.testing.assert_allclose(setup[b],end[b],atol=.00004)
        for t in (0,.2,.4,.6,.8,1.,1.2,1.4,1.6):
            im=render(self.skeleton,self.images,'dead',t,size=(2200,1700),camera=(1000,1200));self.assertLessEqual(im.getbbox()[3],1201)
        final=render(self.skeleton,self.images,'dead',1.6);self.assertLessEqual(abs(final.getbbox()[3]-CAMERA[1]),1)
        a=pose(self.skeleton,'dead',1.4);b=pose(self.skeleton,'dead',1.6)
        for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.00004)

    def test_rotations_do_not_spin_unintentionally_between_keyframes(self):
        for name,animation in self.skeleton['animations'].items():
            for b,tracks in animation['bones'].items():
                keys=tracks.get('rotate',[])
                for first,second in zip(keys,keys[1:]):self.assertLess(abs(second['value']-first['value']),45,(name,b))

    def test_official_runtime_matches_bones_meshes_visibility_and_order_at_60hz(self):
        self.assertEqual(self.report['skeletonSHA256'],sha256((BASE/'xd-4.3.json').read_bytes()).hexdigest())
        self.assertFalse(self.report['desktopEditorTested']);self.assertEqual(self.report['sampleHz'],60);self.assertEqual(len(self.report['samples']),419)
        skin=self.skeleton['skins'][0]['attachments']
        for sample in self.report['samples']:
            name,t=sample['animation'],sample['time'];world=pose(self.skeleton,name,t)
            for b,v in sample['bones'].items():
                m=world[b];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],v,atol=.005)
            visible={s['name']:a for s,a in visible_attachments(self.skeleton,name,t)}
            self.assertEqual(set(visible),{s for s,a in sample['attachments'].items() if a})
            self.assertEqual([s['name'] for s in draw_order(self.skeleton,name,t)],sample['drawOrder'])
            for b,record in sample['meshes'].items():
                points=mesh_world(self.skeleton,skin[b][b],world)
                np.testing.assert_allclose(points[record['indices']].ravel(),record['vertices'],atol=.005)


if __name__=='__main__':unittest.main(verbosity=2)
