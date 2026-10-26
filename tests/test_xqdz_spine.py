from hashlib import sha256
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from tools.build_xqdz_spine import (BASE, DURATIONS, attachment_matrix, build_skeleton,
                                    load_images, pose, render, visible_attachments)


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
        self.assertEqual(self.skeleton['animations']['attack']['events'],
                         [{'time':0.266667,'name':'shoot'}])
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
        expected=[v-(340 if i%2==0 else 260) for i,v in enumerate(complete.getbbox())]
        cropped=render(self.skeleton,self.images,'death',1.4)
        self.assertEqual(list(cropped.getbbox()),expected)

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


if __name__=='__main__':unittest.main(verbosity=2)
