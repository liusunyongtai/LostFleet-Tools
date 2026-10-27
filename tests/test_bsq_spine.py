from hashlib import sha256
import json
import unittest

import cv2
import numpy as np
from PIL import Image
from tools.build_bsq_spine import (BASE,DURATIONS,SIZE,CAMERA,SPIT_TIME,draw_order,
                                  mesh_world,render,spawn_transform)
from tools.build_xqdz_spine import load_images,pose,visible_attachments


class BsqSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.role=json.loads((BASE/'bsq-4.3.json').read_text())
        cls.fallback=json.loads((BASE/'bsq-regions-4.3.json').read_text())
        cls.bullet=json.loads((BASE/'bullet-4.3.json').read_text())
        cls.source=json.loads((BASE/'source/photoshop.json').read_text())
        cls.config=json.loads((BASE/'rig-config.json').read_text())
        cls.images=load_images(BASE/'images')
        cls.report=json.loads((BASE/'runtime-validation.json').read_text())

    def test_exact_animation_names_and_independent_bullet(self):
        expected={'stand','dead','hit','attack_1','attack_2','move'}
        self.assertEqual(set(self.role['animations']),expected)
        self.assertEqual(set(self.fallback['animations']),expected)
        self.assertEqual(set(self.bullet['animations']),{'bullet'})
        self.assertEqual(len(self.role['bones']),16)

    def test_six_original_images_keep_bytes_and_alpha(self):
        manifest=json.loads((BASE/'source/image-sha256.json').read_text())
        self.assertEqual(len(manifest),6)
        for name,digest in manifest.items():
            path=BASE/'images'/name
            self.assertEqual(sha256(path.read_bytes()).hexdigest(),digest)
            image=Image.open(path)
            self.assertEqual(image.mode,'RGBA')
            self.assertEqual(image.getchannel('A').getextrema(),(0,255))

    def test_weighted_mesh_setup_preserves_every_original_uv_position(self):
        mesh=self.role['skins'][0]['attachments']['shenti']['shenti']
        self.assertEqual(mesh['type'],'mesh')
        actual=mesh_world(self.role,mesh,pose(self.role))
        uv=np.array(mesh['uvs']).reshape(-1,2)
        a=self.source['skins']['default']['shenti']['shenti']
        expected=np.column_stack((a['x']+(uv[:,0]-.5)*a['width']-450,
                                  a['y']+(.5-uv[:,1])*a['height']+992))
        np.testing.assert_allclose(actual,expected,atol=.0001)
        vertices=mesh['vertices'];cursor=0;count=0
        while cursor<len(vertices):
            influences=int(vertices[cursor]);cursor+=1
            weights=[vertices[cursor+i*4+3] for i in range(influences)]
            self.assertAlmostEqual(sum(weights),1,places=9)
            self.assertTrue(all(0<w<=1 for w in weights))
            cursor+=influences*4;count+=1
        self.assertEqual(count,len(uv))

    def test_setup_render_preserves_original_assembly(self):
        reference=Image.new('RGBA',SIZE)
        for slot in self.source['slots']:
            name=slot['name'];a=self.source['skins']['default'][name][name]
            xy=(round(a['x']-a['width']/2+50),round(-a['y']-a['height']/2+88))
            reference.alpha_composite(self.images[name],xy)
        rendered=np.asarray(render(self.role,self.images)).astype(float)
        original=np.asarray(reference).astype(float)
        visible=(rendered[:,:,3]>0)|(original[:,:,3]>0)
        self.assertLess(np.mean(np.abs(rendered-original)[visible]),.2)

    def test_move_body_remains_continuous_at_both_deep_bends(self):
        body={**self.role,'slots':[s for s in self.role['slots'] if s['name']=='shenti']}
        for frame in (7,8,22,23):
            image=render(body,self.images,'move',frame/30)
            mask=(np.asarray(image)[:,:,3]>16).astype(np.uint8)
            _,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
            substantial=stats[1:,cv2.CC_STAT_AREA]>100
            self.assertEqual(int(np.count_nonzero(substantial)),1,f'frame {frame}')

    def test_original_place_loops_and_bullet_loop_close(self):
        for skeleton,name,duration in [(self.role,'stand',2.4),(self.role,'move',1.),(self.bullet,'bullet',.4)]:
            a=pose(skeleton,name,0);b=pose(skeleton,name,duration)
            for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.00002)
        for skeleton,name in [(self.role,'move'),(self.bullet,'bullet')]:
            for key in skeleton['animations'][name]['bones']['root']['translate']:
                self.assertEqual([key['x'],key['y']],[0,0])

    def test_move_ankles_follow_the_baked_foot_path(self):
        defs={b['name']:b for b in self.config['bones']}
        for frame in range(31):
            phase=frame/30*np.pi*2;worlds=pose(self.role,'move',frame/30)
            for side,offset in [('back',0),('front',np.pi)]:
                step=np.sin(phase+offset);base=defs[side+'_foot']['position']
                expected=[base[0]+45*step-450,base[1]+36*max(0,step)**2+992]
                np.testing.assert_allclose(worlds[side+'_foot'][:2,2],expected,atol=.002)

    def test_grab_extends_forward_and_is_drawn_in_front_of_head(self):
        initial=pose(self.role,'attack_1',0)['claw_tip'][0,2]
        impact=pose(self.role,'attack_1',11/30)['claw_tip'][0,2]
        self.assertGreater(impact-initial,200)
        order=[s['name'] for s in draw_order(self.role,'attack_1',11/30)]
        self.assertGreater(order.index('youshoubi'),order.index('tou'))
        event=self.role['animations']['attack_1']['events'][0]
        self.assertEqual(event['name'],'claw_hit');self.assertEqual(event['time'],11/30)

    def test_spit_timing_and_independent_projectile_spawn(self):
        event=self.role['animations']['attack_2']['events'][0]
        self.assertEqual(event,{'time':SPIT_TIME,'name':'spawn_bullet','string':'bullet','float':96})
        for name,duration in DURATIONS.items():
            frames=[]
            for frame in range(round(duration*30)+1):
                if any(s['name']=='spit_fx' for s,a in visible_attachments(self.role,name,frame/30)):
                    frames.append(frame)
            self.assertEqual(frames,[12,13,14,15] if name=='attack_2' else [])
        position,angle=spawn_transform(self.role,SPIT_TIME)
        matrix=pose(self.role,'attack_2',SPIT_TIME)['mouth']
        np.testing.assert_allclose(position,(matrix@[96,0,1])[:2])
        self.assertLess(abs(angle),.01)
        for name in ('poison_core','poison_tail','poison_drop'):
            self.assertEqual(self.images['fx/'+name].getchannel('A').getextrema(),(0,255))

    def test_reaction_and_attacks_recover_to_setup(self):
        setup=pose(self.role)
        for name in ('hit','attack_1','attack_2'):
            end=pose(self.role,name,DURATIONS[name])
            for bone in setup:np.testing.assert_allclose(setup[bone],end[bone],atol=.00002)

    def test_dead_is_grounded_and_holds_last_pose(self):
        for t in (.0,.2,.4,.6,.8,1.,1.2,1.4,1.6):
            image=render(self.role,self.images,'dead',t,size=(3000,2400),camera=(1200,1800))
            self.assertLessEqual(image.getbbox()[3],1801)
        a=pose(self.role,'dead',1.5);b=pose(self.role,'dead',1.6)
        for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.00002)

    def test_fallback_has_no_mesh_dependencies(self):
        attachments=self.fallback['skins'][0]['attachments']
        for slot in attachments.values():
            for a in slot.values():self.assertNotEqual(a.get('type'),'mesh')

    def test_official_runtime_report_matches_current_files_bones_mesh_and_order(self):
        skeletons={'role':self.role,'regions':self.fallback,'bullet':self.bullet}
        for filename,data in self.report['files'].items():
            self.assertEqual(data['sha256'],sha256((BASE/filename).read_bytes()).hexdigest())
        self.assertFalse(self.report['desktopEditorTested']);self.assertEqual(len(self.report['samples']),475)
        for sample in self.report['samples']:
            skeleton=skeletons[sample['kind']];worlds=pose(skeleton,sample['animation'],sample['time'])
            for name,values in sample['bones'].items():
                m=worlds[name]
                np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],values,atol=.004)
            self.assertEqual([s['name'] for s in draw_order(skeleton,sample['animation'],sample['time'])],sample['drawOrder'])
            if sample['kind']=='role':
                mesh=skeleton['skins'][0]['attachments']['shenti']['shenti']
                np.testing.assert_allclose(mesh_world(skeleton,mesh,worlds).ravel(),sample['mesh'],atol=.004)


if __name__=='__main__':unittest.main(verbosity=2)
