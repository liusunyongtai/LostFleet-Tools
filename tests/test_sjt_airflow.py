"""Regression for the actual uploaded Spine export, plus airflow/atlas correctness."""
from copy import deepcopy
from hashlib import sha256
import json
import unittest
import numpy as np
from PIL import Image
from tools.build_sjt_airflow import BASE,FX_SLOTS,TRAIL_SLOTS,runtime_render,matrices,sample_point,report_samples
from tools.build_xqdz_spine import load_images,visible_attachments,sample
from tools.build_bsq_spine import mesh_world,draw_order

class SjtAirflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor=json.loads((BASE/'sjt-4.3.json').read_text());cls.source=json.loads((BASE/'source/uploaded-skeleton.json').read_text())
        cls.report=json.loads((BASE/'runtime-validation.json').read_text());cls.old_report=json.loads((BASE/'source/original-runtime.json').read_text());cls.images=load_images(BASE/'images')

    def test_uploaded_bones_slots_and_meshes_are_preserved(self):
        self.assertEqual(self.actor['bones'][:17],self.source['bones']);self.assertEqual([s for s in self.actor['slots'] if s['name'] not in FX_SLOTS],self.source['slots'])
        self.assertEqual(len(self.actor['bones']),19);self.assertEqual(len(self.actor['slots']),12)
        for n,entries in self.source['skins'][0]['attachments'].items():self.assertEqual(self.actor['skins'][0]['attachments'][n],entries)

    def test_all_original_action_curves_and_events_remain_exact(self):
        self.assertEqual(set(self.actor['animations']),set(self.source['animations']));self.assertEqual(self.actor['events'],self.source['events'])
        for n,a in self.actor['animations'].items():
            a=deepcopy(a);a['bones'].pop('impact_anchor',None)
            for slot in FX_SLOTS:a.get('slots',{}).pop(slot,None)
            if not a.get('slots'):a.pop('slots',None)
            self.assertEqual(a,self.source['animations'][n],n)

    def test_original_pngs_and_project_backup_hashes(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),8)
        for n,h in hashes.items():self.assertEqual(sha256((BASE/'images'/n).read_bytes()).hexdigest(),h)
        hashes=json.loads((BASE/'source/upload-provenance.json').read_text())
        for key,p in [('sjt-4.3.json','source/uploaded-skeleton.json'),('sjt.spine','source/original-project/sjt.spine'),('sjt-4.3.png','sjt-4.3.png'),('sjt-4.3.atlas.txt','source/original-atlas.txt')]:self.assertEqual(sha256((BASE/p).read_bytes()).hexdigest(),hashes[key])

    def test_real_atlas_loads_both_pages_and_all_regions(self):
        self.assertEqual(self.report['atlasPages'],['sjt-4.3.png','sjt-airflow-fx.png'])
        self.assertEqual(set(self.report['atlasRegions']),set(self.source['skins'][0]['attachments'])|{'fx/air_streak','fx/impact_air'})
        text=(BASE/'sjt-4.3.atlas.txt').read_text();self.assertTrue(text.startswith((BASE/'source/original-atlas.txt').read_text().rstrip()+'\n\n'))
        self.assertEqual(text.count('pma:true'),2)

    def test_fx_atlas_pixels_are_correctly_premultiplied(self):
        page=np.asarray(Image.open(BASE/'sjt-airflow-fx.png'));self.assertEqual(page.shape,(512,1024,4));self.assertTrue(np.all(page[:,:,:3]<=page[:,:,3:4]))
        for n,x,y in [('air_streak',4,4),('impact_air',4,140)]:
            raw=np.asarray(self.images['fx/'+n]);expected=raw.copy();expected[:,:,:3]=np.rint(raw[:,:,:3].astype(float)*raw[:,:,3:4]/255).astype('uint8')
            np.testing.assert_array_equal(page[y:y+raw.shape[0],x:x+raw.shape[1]],expected)

    def test_both_effect_textures_are_transparent_with_soft_alpha(self):
        for n in ('air_streak','impact_air'):
            im=self.images['fx/'+n];self.assertEqual(im.mode,'RGBA');a=np.asarray(im)[:,:,3]
            self.assertEqual(int(a.min()),0);self.assertGreater(int(a.max()),100);self.assertGreater(np.count_nonzero((a>0)&(a<200)),3000)

    def test_original_world_poses_are_unchanged_in_official_runtime(self):
        self.assertEqual(len(self.old_report['samples']),448);self.assertEqual(len(self.report['samples']),448)
        for old,new in zip(self.old_report['samples'],self.report['samples']):
            self.assertEqual((old['animation'],old['time']),(new['animation'],new['time']))
            for n,m in old['bones'].items():np.testing.assert_allclose(new['bones'][n],m,atol=1e-7)
            for n,m in old['meshes'].items():np.testing.assert_allclose(new['meshes'][n]['vertices'],m['vertices'],atol=1e-7)

    def test_reports_match_current_source_hashes(self):
        for name,report in [('sjt-4.3.json',self.report),('source/uploaded-skeleton.json',self.old_report)]:self.assertEqual(sha256((BASE/name).read_bytes()).hexdigest(),report['skeletonSHA256'])
        self.assertFalse(self.report['desktopEditorTested']);self.assertEqual(self.report['sampleHz'],60)

    def test_fx_hidden_outside_attack_2_and_when_it_finishes(self):
        for name in self.actor['animations']:
            if name=='attack_2':continue
            for t in (0,.5,1.):self.assertFalse({s['name'] for s,a in visible_attachments(self.actor,name,t)}&set(FX_SLOTS))
        for t in (0,.4,.49,.91,1.2,1.4):self.assertFalse({s['name'] for s,a in visible_attachments(self.actor,'attack_2',t)}&set(FX_SLOTS))

    def test_airflow_and_impact_windows(self):
        for t,wanted in [(.5,set(TRAIL_SLOTS)),(.55,set(TRAIL_SLOTS)),(.6,set(FX_SLOTS)),(.65,set(FX_SLOTS)),(.7,{'impact_air'}),(.8,{'impact_air'})]:
            visible={s['name'] for s,a in visible_attachments(self.actor,'attack_2',t)}&set(FX_SLOTS);self.assertEqual(visible,wanted)

    def test_ribbon_heads_follow_actual_uploaded_tool_trajectory(self):
        skin=self.actor['skins'][0]['attachments']['airflow_trail'];trajectory=report_samples()
        for f in range(15,20):
            t=f/30;name=f'airflow_{f:02d}';a=skin[name];points=mesh_world(self.actor,a,matrices(self.report['samples'][0]));uv=np.array(a['uvs']).reshape(-1,2)
            # The center ribbon has its own nonoverlapping UV disk.
            head=points[uv[:,0]>.99].mean(axis=0)
            np.testing.assert_allclose(head,sample_point(trajectory,t),atol=.001)
            self.assertEqual(len(a['triangles']),36*6)

    def test_added_meshes_agree_with_official_runtime(self):
        skin=self.actor['skins'][0]['attachments']
        for item in self.report['samples']:
            world=matrices(item)
            for n in FX_SLOTS:
                if n not in item['meshes']:continue
                r=item['meshes'][n];a=skin[n][item['attachments'][n]];p=mesh_world(self.actor,a,world)
                np.testing.assert_allclose(p[r['indices']].ravel(),r['vertices'],atol=.003)
                track=self.actor['animations'][item['animation']]['slots'][n]['alpha'];self.assertAlmostEqual(sample(track,item['time'],'value',1),item['alpha'][n],places=5)

    def test_trail_is_behind_character_and_pressure_wave_in_front(self):
        self.assertEqual(self.actor['slots'][0]['name'],'airflow_trail');self.assertEqual(self.actor['slots'][-1]['name'],'impact_air')
        for item in self.report['samples']:
            self.assertEqual([n for n in item['drawOrder'] if n not in FX_SLOTS],[s['name'] for s in self.source['slots']])

    def test_pressure_wave_expands_and_fades(self):
        track=self.actor['animations']['attack_2']['bones']['impact_anchor']['scale'];alpha=self.actor['animations']['attack_2']['slots']['impact_air']['alpha']
        self.assertLess(sample(track,.6,'x',1),sample(track,.75,'x',1));self.assertGreater(sample(alpha,.6,'value',1),sample(alpha,.8,'value',1))
        self.assertEqual(sample(alpha,.9,'value',1),0)

    def test_actual_render_has_clear_mobile_airflow_and_no_late_leak(self):
        kwargs=dict(size=(480,360),camera=(260,330),factor=.25,background=Image.new('RGBA',(480,360),'#27343e'))
        for t,should_differ in [(.4,False),(17/30,True),(.7,True),(1.2,False)]:
            a=np.asarray(runtime_render(self.actor,self.images,self.report,'attack_2',t,**kwargs));b=np.asarray(runtime_render(self.actor,self.images,self.report,'attack_2',t,hide_fx=True,**kwargs))
            changed=np.count_nonzero(np.max(np.abs(a.astype(float)-b.astype(float)),axis=2)>10)
            if should_differ:self.assertGreater(changed,100)
            else:self.assertEqual(changed,0)

if __name__=='__main__':unittest.main(verbosity=2)
