"""Editor topology regressions prompted by the actual desktop import log.

This checks explicit editor metadata/UV disks, not the proprietary editor itself.
"""
from copy import deepcopy
from hashlib import sha256
import json,unittest
from pathlib import Path
import numpy as np
from tools.spine_mesh_topology import editor_mesh_check,perimeter_edges
from tools.build_bird112_spine import BASE,pose,render,load_images


def quad():return {'uvs':[0,0,1,0,1,1,0,1],'triangles':[0,1,2,0,2,3],'hull':4,'edges':perimeter_edges(4)}


class EditorTopologyTests(unittest.TestCase):
    def test_valid_single_uv_disk(self):self.assertEqual(editor_mesh_check(quad()),[])

    def test_reversed_triangle_is_rejected(self):
        a=quad();a['triangles']=[0,2,1,0,3,2];self.assertIn('degenerate or reversed UV triangle',editor_mesh_check(a))

    def test_out_of_range_index_is_rejected(self):
        a=quad();a['triangles'][0]=99;self.assertIn('triangle vertex outside mesh',editor_mesh_check(a))

    def test_degenerate_triangle_is_rejected(self):
        a=quad();a['triangles'][0]=1;self.assertIn('degenerate or reversed UV triangle',editor_mesh_check(a))

    def test_duplicate_uv_is_rejected(self):
        a=quad();a['uvs'] += [0,0];self.assertIn('duplicate UV vertices',editor_mesh_check(a))

    def test_disconnected_ribbons_are_rejected(self):
        a=quad();a['uvs'] += [2,0,3,0,3,1,2,1];a['triangles'] += [4,5,6,4,6,7]
        self.assertIn('disconnected mesh islands',editor_mesh_check(a))

    def test_missing_nonessential_edges_is_flagged(self):
        a=quad();a.pop('edges');self.assertIn('missing editor edges',editor_mesh_check(a))

    def test_edge_indices_are_coordinate_offsets_not_vertex_indices(self):
        a=quad();a['edges']=[0,1,1,2,2,3,3,0];self.assertIn('invalid editor edge coordinates',editor_mesh_check(a))

    def test_wrong_hull_is_rejected(self):
        a=quad();a['hull']=3;self.assertIn('triangle boundary does not match ordered hull',editor_mesh_check(a))

    def test_bird_meshes_include_valid_editor_perimeters(self):
        a=json.loads((BASE/'bird112-4.3.json').read_text());count=0
        for slot,entries in a['skins'][0]['attachments'].items():
            for n,m in entries.items():count+=1;self.assertEqual(editor_mesh_check(m),[],(slot,n))
        self.assertEqual(count,11)

    def test_body_density_is_reduced_conservatively(self):
        a=json.loads((BASE/'bird112-4.3.json').read_text())['skins'][0]['attachments']['shenti']['shenti']
        self.assertEqual(len(a['uvs'])//2,325)  # Budget choice, not a claimed Spine format limit.

    def test_sjt_all_meshes_pass_topology_checks(self):
        p=BASE.parent/'sjt-airflow-spine/sjt-4.3.json';a=json.loads(p.read_text());count=0
        for slot,entries in a['skins'][0]['attachments'].items():
            for n,m in entries.items():count+=1;self.assertEqual(editor_mesh_check(m),[],(slot,n))
        self.assertEqual(count,24)

    def test_each_sjt_airflow_layer_has_unique_uvs_and_one_hull(self):
        a=json.loads((BASE.parent/'sjt-airflow-spine/sjt-4.3.json').read_text())
        for n in ('airflow_trail','airflow_inner','airflow_outer'):
            self.assertEqual(len(a['skins'][0]['attachments'][n]),5)
            for m in a['skins'][0]['attachments'][n].values():
                self.assertEqual(m['hull'],74);uv=list(zip(m['uvs'][::2],m['uvs'][1::2]));self.assertEqual(len(uv),len(set(uv)))

    def test_airflow_setup_triangles_have_export_style_world_winding(self):
        from tools.build_bsq_spine import mesh_world
        a=json.loads((BASE.parent/'sjt-airflow-spine/sjt-4.3.json').read_text());world=pose(a)
        for slot in ('airflow_trail','airflow_inner','airflow_outer'):
            for m in a['skins'][0]['attachments'][slot].values():
                p=mesh_world(a,m,world);tri=p[np.array(m['triangles']).reshape(-1,3)];v=tri[:,1]-tri[:,0];w=tri[:,2]-tri[:,0]
                self.assertTrue(np.all(v[:,0]*w[:,1]-v[:,1]*w[:,0]<-1e-8))

    def test_bird_full_motion_curves_and_bones_unchanged(self):
        a=json.loads((BASE/'bird112-4.3.json').read_text());hashes=json.loads((BASE/'source/animation-sha256.json').read_text())
        for k,h in hashes.items():self.assertEqual(sha256(json.dumps(a[k],sort_keys=True,separators=(',',':')).encode()).hexdigest(),h)


class RegionCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=json.loads((BASE/'bird112-regions-4.3.json').read_text());cls.full=json.loads((BASE/'bird112-4.3.json').read_text());cls.images=load_images(BASE/'images')

    def test_no_mesh_fields_are_present(self):
        count=0
        for entries in self.a['skins'][0]['attachments'].values():
            for m in entries.values():
                count+=1;self.assertEqual(m.get('type','region'),'region');self.assertFalse({'vertices','triangles','uvs','hull','edges'}&set(m))
        self.assertEqual(count,11)

    def test_compatibility_retains_five_actions_and_main_bones(self):
        self.assertEqual(set(self.a['animations']),{'stand','dead','hit','move','attack'});self.assertEqual(self.a['bones'],self.full['bones'])
        for n,a in self.a['animations'].items():
            for bone,tr in a['bones'].items():
                if n=='dead' and bone=='root':continue  # Rigid images get a separate floor correction.
                self.assertEqual(tr,self.full['animations'][n]['bones'][bone])

    def test_compatible_death_is_grounded(self):
        im=render(self.a,self.images,'dead',1.6);self.assertLessEqual(abs(im.getbbox()[3]-1250),1)

    def test_compatibility_loops_and_stationary_roots(self):
        for n,duration in [('stand',2.4),('move',.8),('attack',.8)]:
            a=pose(self.a,n,0);b=pose(self.a,n,duration)
            for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.0001)
            for k in self.a['animations'][n]['bones']['root']['translate']:self.assertEqual((k['x'],k['y']),(0,0))

    def test_region_runtime_geometry_matches_independent_transforms(self):
        r=json.loads((BASE/'regions-runtime-validation.json').read_text());self.assertEqual(len(r['samples']),371);self.assertEqual(r['skeletonSHA256'],sha256((BASE/'bird112-regions-4.3.json').read_bytes()).hexdigest());self.assertFalse(r['desktopEditorTested'])
        skin=self.a['skins'][0]['attachments'];slots={s['name']:s for s in self.a['slots']}
        for item in r['samples']:
            world=pose(self.a,item['animation'],item['time'])
            for n,rec in item['meshes'].items():
                m=skin[n][n];w,h=m['width'],m['height'];corners=np.array([[-w/2,-h/2],[-w/2,h/2],[w/2,h/2],[w/2,-h/2]])+[m['x'],m['y']]
                result=(world[slots[n]['bone']]@np.column_stack((corners,np.ones(4))).T).T[:,:2]
                np.testing.assert_allclose(result[rec['indices']].ravel(),rec['vertices'],atol=.005)

if __name__=='__main__':unittest.main(verbosity=2)
