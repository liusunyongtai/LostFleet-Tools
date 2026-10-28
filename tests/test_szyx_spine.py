"""Source fidelity, quadruped support, action distinction and official runtime parity."""
import hashlib,json,math,unittest
from copy import deepcopy
import cv2
import numpy as np
from PIL import Image
from tools.build_szyx_spine import BASE,DURATIONS,controls,pose,render,mesh_world,load_images,scaled_preview
from tools.spine_mesh_topology import editor_mesh_check
class SzyxSpineTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((BASE/'szyx-4.3.json').read_text());cls.c=json.loads((BASE/'rig-config.json').read_text());cls.src=json.loads((BASE/'source/photoshop.json').read_text());cls.images=load_images(BASE/'images')
 def test_exact_names_target_and_rig_counts(self):
  self.assertEqual(set(self.a['animations']),{'stand','dead','hit','move','attack','attack_2'});self.assertEqual(self.a['skeleton']['spine'],'4.3.23');self.assertEqual(self.a['skeleton']['images'],'./images/');self.assertEqual(len(self.a['bones']),51);self.assertEqual(len(self.a['slots']),16)
 def test_source_sixteen_pngs_byte_identical_and_transparent(self):
  hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),16)
  for n,h in hashes.items():
   self.assertEqual(hashlib.sha256((BASE/'images'/n).read_bytes()).hexdigest(),h);im=self.images[n[:-4]];self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255))
 def test_original_layer_order_preserved(self):self.assertEqual([s['name'] for s in self.a['slots']],[s['name'] for s in self.src['slots']])
 def test_editor_topology_valid_and_metadata_complete(self):
  for entries in self.a['skins'][0]['attachments'].values():
   for a in entries.values():self.assertEqual(editor_mesh_check(a),[]);self.assertLess(len(a['uvs'])//2,400);self.assertGreater(a['width'],0);self.assertGreater(a['height'],0)
 def test_inverse_bind_source_coordinates_and_positive_normalized_weights(self):
  w=pose(self.a)
  for n,e in self.a['skins'][0]['attachments'].items():
   a=e[n];r=self.src['skins']['default'][n][n];uv=np.array(a['uvs']).reshape(-1,2);expected=np.column_stack((r['x']+(uv[:,0]-.5)*r['width'],r['y']+(.5-uv[:,1])*r['height']))-self.c['origin'];np.testing.assert_allclose(mesh_world(self.a,a,w),expected,atol=.00001)
   i=0;v=a['vertices']
   while i<len(v):
    count=int(v[i]);i+=1;ws=[v[i+j*4+3] for j in range(count)];self.assertTrue(all(0<x<=1 for x in ws));self.assertAlmostEqual(sum(ws),1,places=8);i+=4*count
 def test_setup_assembly_matches_photoshop(self):
  size=(1700,1450);cam=(850,1250);ref=Image.new('RGBA',size);ox,oy=self.c['origin']
  for s in self.src['slots']:
   n=s['name'];r=self.src['skins']['default'][n][n];ref.alpha_composite(self.images[n],(round(r['x']-r['width']/2-ox+cam[0]),round(-r['y']-r['height']/2+oy+cam[1])))
  a=np.asarray(render(self.a,self.images,size=size,camera=cam)).astype(float);b=np.asarray(ref).astype(float);mask=(a[:,:,3]>0)|(b[:,:,3]>0);self.assertLess(np.abs(a-b)[mask].mean(),.4)
 def test_idle_and_movement_loop_endpoints(self):
  for n in ('stand','move'):
   a=pose(self.a,n,0);b=pose(self.a,n,DURATIONS[n])
   for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.0001)
 def test_root_stays_in_place_all_living_actions(self):
  for n in ('stand','move','hit','attack','attack_2'):
   for k in self.a['animations'][n]['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])
 def test_exact_ankle_support_targets_no_unreachable_clamping(self):
  for n,d in DURATIONS.items():
   if n=='dead':continue # Root ground correction is intentionally added after local IK.
   for f in range(round(d*30)+1):
    t=f/30;w=pose(self.a,n,t);targets=controls(n,t,self.c)[3]
    for k,p in targets.items():np.testing.assert_allclose(w[k+'_paw'][:2,2]+self.c['origin'],p,atol=.003,err_msg=f'{n} {t} {k}')
 def test_diagonal_trot_phasing_and_lift(self):
  a=controls('move',.3,self.c)[3];b=controls('move',.9,self.c)[3];rest={k:v['ankle'] for k,v in self.c['limbs'].items()}
  for k in ('near_front','far_hind'):self.assertGreater(a[k][1]-rest[k][1],60);self.assertAlmostEqual(b[k][1],rest[k][1])
  for k in ('far_front','near_hind'):self.assertGreater(b[k][1]-rest[k][1],60);self.assertAlmostEqual(a[k][1],rest[k][1])
 def test_stance_feet_push_back_monotonically_and_swing_forward(self):
  for k,l in self.c['limbs'].items():
   xs=[];stance=[]
   for f in range(181):
    t=f/180*DURATIONS['move'];p=controls('move',t,self.c)[3][k];xs.append(p[0]);stance.append(math.sin(t/DURATIONS['move']*math.tau+l['phase'])<=0)
   for i in range(1,len(xs)):
    if stance[i] and stance[i-1]:self.assertGreaterEqual(xs[i]-xs[i-1],-.00001)
    if not stance[i] and not stance[i-1]:self.assertLessEqual(xs[i]-xs[i-1],.00001)
 def test_tentacle_wriggling_has_delayed_nonidentical_tracks(self):
  for n in ('stand','move'):
   for img,d in self.c['chains'].items():
    if img=='weiba':continue
    tracks=self.a['animations'][n]['bones'];first=tracks[d['bones'][0]]['rotate'];last=tracks[d['bones'][-1]]['rotate'];self.assertNotEqual(first,last);self.assertGreater(np.ptp([k['value'] for k in last]),15)
 def test_head_teeth_and_jaw_are_rigid(self):
  for n in ('tou','xiaba','touci'):
   a=self.a['skins'][0]['attachments'][n][n];self.assertEqual(len(a['uvs']),8)
  for n,t in [('stand',1.2),('attack',.3),('attack',.53),('dead',1.6)]:
   w=pose(self.a,n,t);self.assertAlmostEqual(np.linalg.det(w['head'][:2,:2]),1,places=5);self.assertAlmostEqual(np.linalg.det(w['jaw'][:2,:2]),1,places=5)
 def test_bite_windup_lunge_jaw_snap_and_event(self):
  rest=pose(self.a);wind=pose(self.a,'attack',.3);strike=pose(self.a,'attack',.53)
  self.assertGreater(wind['head'][0,2],rest['head'][0,2]+35);self.assertLess(strike['head'][0,2],rest['head'][0,2]-120)
  tr=self.a['animations']['attack']['bones']['jaw']['rotate'];self.assertGreater(max(k['value'] for k in tr),20);self.assertLess(min(k['value'] for k in tr),-5);self.assertEqual(self.a['animations']['attack']['events'],[{'time':.53,'name':'bite_hit'}])
 def test_claw_has_distinct_high_windup_and_forward_rake(self):
  rest=pose(self.a);wind=pose(self.a,'attack_2',.3);strike=pose(self.a,'attack_2',.5)
  self.assertGreater(wind['near_front_paw'][1,2],rest['near_front_paw'][1,2]+120);self.assertGreater(wind['near_front_paw'][0,2],rest['near_front_paw'][0,2]+90);self.assertLess(strike['near_front_paw'][0,2],rest['near_front_paw'][0,2]-340);self.assertEqual(self.a['animations']['attack_2']['events'],[{'time':.5,'name':'claw_hit'}])
 def test_actions_recover_to_setup_pose(self):
  rest=pose(self.a)
  for n in ('hit','attack','attack_2'):
   end=pose(self.a,n,DURATIONS[n])
   for b in rest:np.testing.assert_allclose(rest[b],end[b],atol=.0001)
 def test_no_world_bone_rotation_flip(self):
  for n,d in DURATIONS.items():
   previous=None
   for f in range(round(d*30)+1):
    angles={k:math.degrees(math.atan2(m[1,0],m[0,0])) for k,m in pose(self.a,n,f/30).items()}
    if previous:
     for b,v in angles.items():self.assertLess(abs((v-previous[b]+180)%360-180),45,(n,b,f))
    previous=angles
 def test_dead_settles_grounded_and_body_drops(self):
  a=pose(self.a,'dead',1.4);b=pose(self.a,'dead',1.6)
  for n in a:np.testing.assert_allclose(a[n],b[n],atol=.0001)
  self.assertLess(a['hips'][1,2],pose(self.a)['hips'][1,2]-120)
  im=render(self.a,self.images,'dead',1.6,size=(2600,2000),camera=(1300,1700));self.assertLessEqual(abs(im.getbbox()[3]-1700),1)
 def test_death_does_not_penetrate_floor_at_key_poses(self):
  a=scaled_preview(self.a,.3)
  for t in (0,.2,.4,.6,.8,1.,1.2,1.4,1.6):
   im=render(a,self.images,'dead',t,size=(900,650),camera=(470,570));self.assertLessEqual(im.getbbox()[3],571,(t,im.getbbox()))
 def test_source_meshes_do_not_disconnect_at_action_extremes(self):
  a=scaled_preview(self.a,.3)
  for n,t in [('stand',1.2),('move',.3),('move',.9),('attack',.3),('attack',.53),('attack_2',.3),('attack_2',.5),('dead',1.6)]:
   im=np.asarray(render(a,self.images,n,t,size=(950,700),camera=(520,620)));count,labels,stats,centers=cv2.connectedComponentsWithStats((im[:,:,3]>64).astype('uint8'));large=[x for x in stats[1:,cv2.CC_STAT_AREA] if x>80];self.assertEqual(len(large),1,(n,t,large))
 def test_uniform_preview_scaling_including_dead_root(self):
  a=scaled_preview(self.a,.3)
  for n,t in [('stand',1.2),('attack',.53),('dead',1.6)]:
   w=pose(self.a,n,t);v=pose(a,n,t)
   for b in w:np.testing.assert_allclose(w[b][:2,:]*.3,v[b][:2,:],atol=.00001)
 def test_official_runtime_498_poses_match_independent_geometry(self):
  r=json.loads((BASE/'runtime-validation.json').read_text());self.assertEqual(len(r['samples']),498);self.assertFalse(r['desktopEditorTested']);self.assertEqual(r['skeletonSHA256'],hashlib.sha256((BASE/'szyx-4.3.json').read_bytes()).hexdigest());skin=self.a['skins'][0]['attachments']
  for item in r['samples']:
   world=pose(self.a,item['animation'],item['time'])
   for n,v in item['bones'].items():
    m=world[n];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],v,atol=.005)
   for n,rec in item['meshes'].items():np.testing.assert_allclose(mesh_world(self.a,skin[n][n],world)[rec['indices']].ravel(),rec['vertices'],atol=.005)
if __name__=='__main__':unittest.main(verbosity=2)
