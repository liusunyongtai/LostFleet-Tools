from hashlib import sha256
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from tools.build_xqdz_spine import (BASE, DURATIONS, LIGHT_SLOTS, LIT_PARTS,
                                    PREVIEW_CAMERA, attachment_matrix, build_skeleton,
                                    load_images, pose, render, sample, visible_attachments)


class XqdzSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skeleton = json.loads((BASE/'xqdz-4.3.json').read_text())
        cls.source = json.loads((BASE/'source/photoshop.json').read_text())
        cls.config = json.loads((BASE/'rig-config.json').read_text())
        cls.images = load_images(BASE/'images')

    def test_original_pngs_are_unchanged_and_keep_alpha(self):
        manifest=json.loads((BASE/'source/image-sha256.json').read_text())
        self.assertEqual(len(manifest),9)
        for name,digest in manifest.items():
            with self.subTest(name=name):
                path=BASE/'images'/name
                self.assertEqual(sha256(path.read_bytes()).hexdigest(),digest)
                image=Image.open(path)
                self.assertEqual(image.mode,'RGBA')
                self.assertEqual(image.getchannel('A').getextrema(),(0,255))

    def test_setup_attachment_corners_preserve_original_assembly(self):
        world=pose(self.skeleton)
        origin=np.array(self.config['origin'])
        for slot,a in visible_attachments(self.skeleton):
            name=slot['name'];image=self.images[name]
            matrix=attachment_matrix(world,slot,a,image)
            corners=matrix @ np.array([[0,image.width,image.width,0],
                         [0,0,image.height,image.height],[1,1,1,1]])
            source=self.source['skins']['default'][name][name]
            x,y=source['x']-origin[0],source['y']-origin[1]
            w,h=source['width'],source['height']
            expected=np.array([[x-w/2,x+w/2,x+w/2,x-w/2],
                               [y+h/2,y+h/2,y-h/2,y-h/2]])
            np.testing.assert_allclose(corners[:2],expected,atol=2e-5)
        self.assertEqual(len(list(visible_attachments(self.skeleton))),9)

    def test_all_references_and_relative_paths_are_resolved(self):
        ordered=set()
        for bone in self.skeleton['bones']:
            if 'parent' in bone:self.assertIn(bone['parent'],ordered)
            self.assertNotIn(bone['name'],ordered);ordered.add(bone['name'])
        self.assertEqual(len(ordered),13)
        self.assertEqual(self.skeleton['skeleton']['images'],'./images/')
        skin=self.skeleton['skins'][0]['attachments']
        for slot in self.skeleton['slots']:
            self.assertIn(slot['bone'],ordered)
            for attachment in skin[slot['name']].values():
                self.assertIn(attachment['path'],self.images)

    def test_loops_match_start_and_end_world_transforms(self):
        for name in ('idle','move'):
            start=pose(self.skeleton,name,0);end=pose(self.skeleton,name,DURATIONS[name])
            for bone in start:
                np.testing.assert_allclose(start[bone],end[bone],atol=1e-5)

    def test_flash_and_shoot_event_are_synchronized(self):
        event=self.skeleton['animations']['attack']['events'][0]
        self.assertEqual(event['name'],'shoot')
        self.assertAlmostEqual(event['time'],8/30,places=12)
        for name,duration in DURATIONS.items():
            visible=[]
            for frame in range(round(duration*30)+1):
                names=[slot['name'] for slot,_ in visible_attachments(self.skeleton,name,frame/30)]
                if 'muzzle_flash' in names:visible.append(frame)
            self.assertEqual(visible,[8,9,10] if name=='attack' else [])
        for i in range(1,4):
            self.assertEqual(self.images[f'fx/muzzle_flash_{i}'].getchannel('A').getextrema(),(0,255))

    def test_recoil_moves_muzzle_backwards_for_left_facing_gun(self):
        at_shot=pose(self.skeleton,'attack',8/30)['muzzle']
        recoil=pose(self.skeleton,'attack',9/30)['muzzle']
        self.assertGreater(recoil[0,2],at_shot[0,2])

    def test_death_ends_at_ground_and_remains_still(self):
        frame=render(self.skeleton,self.images,'death',1.4)
        self.assertLessEqual(abs(frame.getbbox()[3]-540),1)
        end=pose(self.skeleton,'death',1.4);settled=pose(self.skeleton,'death',1.3)
        for bone in end:np.testing.assert_allclose(end[bone],settled[bone],atol=1e-5)

    def test_death_preview_keeps_the_entire_fallen_character(self):
        complete=render(self.skeleton,self.images,'death',1.4,size=(1600,1400),camera=(700,800))
        expected=[v-((700-PREVIEW_CAMERA[0]) if i%2==0 else (800-PREVIEW_CAMERA[1]))
                  for i,v in enumerate(complete.getbbox())]
        cropped=render(self.skeleton,self.images,'death',1.4)
        self.assertEqual(list(cropped.getbbox()),expected)

    def test_death_does_not_sink_below_ground_during_the_fall(self):
        for frame in range(43):
            image=render(self.skeleton,self.images,'death',frame/30,size=(1600,1400),camera=(700,800))
            self.assertLessEqual(image.getbbox()[3],801,f'frame {frame}')

    def test_runtime_pose_report_is_current_and_matches_preview_math(self):
        report=json.loads((BASE/'runtime-validation.json').read_text())
        self.assertEqual(report['skeletonSHA256'],sha256((BASE/'xqdz-4.3.json').read_bytes()).hexdigest())
        self.assertFalse(report['desktopEditorTested'])
        self.assertEqual(len(report['samples']),167)
        for entry in report['samples']:
            worlds=pose(self.skeleton,entry['animation'],entry['time'])
            for name,values in entry['bones'].items():
                m=worlds[name]
                actual=[m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]]
                np.testing.assert_allclose(actual,values,atol=0.002,
                                          err_msg=f'{entry["animation"]} {entry["time"]} {name}')

    def test_generated_setup_matches_configuration(self):
        rebuilt=build_skeleton(self.source,self.config)
        for field in ('bones','slots','skins','skeleton'):
            self.assertEqual(rebuilt[field],self.skeleton[field])

    def test_flash_is_twice_v1_in_both_dimensions_and_starts_at_muzzle(self):
        flash=self.skeleton['skins'][0]['attachments']['muzzle_flash']
        for attachment in flash.values():
            self.assertEqual((attachment['width'],attachment['height']),(160,112))
            self.assertEqual(attachment['x']-attachment['width']/2,0)

    def test_each_action_has_a_stronger_readable_accent_than_v1(self):
        baseline=json.loads((BASE/'source/v1-animation-reference.json').read_text())
        for animation,bone in [('idle','head'),('move','right_leg'),('attack','body'),
                               ('hit','body'),('death','body')]:
            old=max(abs(k['value']) for k in baseline['animations'][animation]['bones'][bone]['rotate'])
            new=max(abs(k['value']) for k in self.skeleton['animations'][animation]['bones'][bone]['rotate'])
            self.assertGreaterEqual(new,old*1.6,(animation,bone))

    def test_light_masks_are_clipped_to_original_alpha(self):
        for name in LIT_PARTS:
            original=np.asarray(self.images[name])[:,:,3]
            mask=np.asarray(self.images['fx/shot_light_'+name])[:,:,3]
            self.assertTrue(np.all(mask<=original))
            self.assertTrue(np.any(mask>0))

    def test_lights_are_attack_only_and_match_official_runtime_fade(self):
        report=json.loads((BASE/'runtime-validation.json').read_text())
        for entry in report['samples']:
            visible={slot['name'] for slot,_ in visible_attachments(self.skeleton,entry['animation'],entry['time'])}
            for name in LIGHT_SLOTS:
                official=entry['lights'][name]
                self.assertEqual(bool(official['attachment']),name in visible)
                track=self.skeleton['animations'][entry['animation']]['slots'][name]['alpha']
                alpha=sample(track,entry['time'],'value',1)
                self.assertAlmostEqual(alpha,official['alpha'],places=5)
                if entry['animation']!='attack':self.assertFalse(official['attachment'])

    def test_shot_light_actually_brightens_the_gun_and_turns_off(self):
        from copy import deepcopy
        unlit=deepcopy(self.skeleton)
        for animation in unlit['animations'].values():
            for name in LIGHT_SLOTS:animation['slots'][name]={'attachment':[{'time':0,'name':None}]}
        background=Image.new('RGBA',(1120,640),'#26323e')
        before=render(unlit,self.images,'attack',8/30,background=background)
        after=render(self.skeleton,self.images,'attack',8/30,background=background)
        slot=next(slot for slot in self.skeleton['slots'] if slot['name']=='wuqi')
        a=self.skeleton['skins'][0]['attachments']['wuqi']['wuqi']
        world=attachment_matrix(pose(self.skeleton,'attack',8/30),slot,a,self.images['wuqi']) @ [60,50,1]
        x,y=round(world[0]+PREVIEW_CAMERA[0]),round(PREVIEW_CAMERA[1]-world[1])
        self.assertGreater(after.getpixel((x,y))[0],before.getpixel((x,y))[0]+20)
        np.testing.assert_array_equal(np.asarray(render(unlit,self.images,'attack',.6,background=background)),
                                      np.asarray(render(self.skeleton,self.images,'attack',.6,background=background)))


if __name__=='__main__':unittest.main(verbosity=2)
