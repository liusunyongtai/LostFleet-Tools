"""SYCS source fidelity, continuous tentacles, glass-contained bubbles and safe topology."""
from copy import deepcopy
from hashlib import sha256
import json,math,unittest
import cv2
import numpy as np
from PIL import Image
from tools.build_sycs_spine import BASE,DURATIONS,BUBBLE_SLOTS,pose,render,mesh_world,load_images,scaled_preview
from tools.build_bsq_spine import draw_order
from tools.build_xqdz_spine import visible_attachments,sample
from tools.spine_mesh_topology import editor_mesh_check

class SycsSpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=json.loads((BASE/'sycs-4.3.json').read_text());cls.source=json.loads((BASE/'source/photoshop.json').read_text());cls.c=json.loads((BASE/'rig-config.json').read_text());cls.images=load_images(BASE/'images');cls.report=json.loads((BASE/'runtime-validation.json').read_text())

    def test_exact_five_actions_version_and_bone_counts(self):
        self.assertEqual(set(self.a['animations']),{'stand','dead','hit','move','attack'});self.assertEqual(self.a['skeleton']['spine'],'4.3.23');self.assertEqual(self.a['skeleton']['images'],'./images/');self.assertEqual(len(self.a['bones']),42);self.assertEqual(len(self.a['slots']),28)

    def test_original_seventeen_pngs_remain_byte_identical(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),17)
        for n,h in hashes.items():
            self.assertEqual(sha256((BASE/'images'/n).read_bytes()).hexdigest(),h);im=Image.open(BASE/'images'/n);self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255))

    def test_original_layer_order_preserved_in_setup(self):
        extra=set(BUBBLE_SLOTS)|{'front_mouth','front_outer','whip_air'};self.assertEqual([s['name'] for s in self.a['slots'] if s['name'] not in extra],[s['name'] for s in self.source['slots']])

    def test_all_meshes_have_editor_metadata_and_valid_topology(self):
        count=0
        for n,entries in self.a['skins'][0]['attachments'].items():
            for k,a in entries.items():
                if a.get('type')=='mesh':count+=1;self.assertEqual(editor_mesh_check(a),[],(n,k));self.assertLess(len(a['uvs'])//2,400)
        self.assertEqual(count,22)

    def test_source_meshes_setup_positions_and_positive_weights(self):
        world=pose(self.a);origin=np.array(self.c['origin'])
        for n,entries in self.a['skins'][0]['attachments'].items():
            if n=='whip_air' or n in BUBBLE_SLOTS:continue
            a=next(iter(entries.values()))
            if a.get('type')!='mesh':continue
            r=self.source['skins']['default'][a['path']][a['path']];uv=np.array(a['uvs']).reshape(-1,2);expected=np.column_stack((r['x']+(uv[:,0]-.5)*r['width'],r['y']+(.5-uv[:,1])*r['height']))-origin
            np.testing.assert_allclose(mesh_world(self.a,a,world),expected,atol=.0001)
            data=a['vertices'];i=0
            while i<len(data):
                count=int(data[i]);i+=1;weights=[data[i+j*4+3] for j in range(count)];self.assertAlmostEqual(sum(weights),1,places=8);self.assertTrue(all(0<w<=1 for w in weights));i+=count*4

    def test_setup_render_preserves_original_assembly(self):
        size=(1900,1500);cam=(1000,1350);ox,oy=self.c['origin'];ref=Image.new('RGBA',size)
        for slot in self.source['slots']:
            n=slot['name'];a=self.source['skins']['default'][n][n];ref.alpha_composite(self.images[n],(round(a['x']-a['width']/2-ox+cam[0]),round(-a['y']-a['height']/2+oy+cam[1])))
        a=np.asarray(render(self.a,self.images,size=size,camera=cam)).astype(float);b=np.asarray(ref).astype(float);mask=(a[:,:,3]>0)|(b[:,:,3]>0);self.assertLess(np.abs(a-b)[mask].mean(),.4)

    def test_stand_move_loops_include_bubble_transforms_and_alpha(self):
        for n in ('stand','move'):
            a=pose(self.a,n,0);b=pose(self.a,n,DURATIONS[n])
            for bone in a:np.testing.assert_allclose(a[bone],b[bone],atol=.0001)
            for slot in BUBBLE_SLOTS:
                ch=self.a['animations'][n]['slots'][slot]['alpha'];self.assertAlmostEqual(sample(ch,0,'value',0),sample(ch,DURATIONS[n],'value',0),places=6)

    def test_root_stays_fixed_in_idle_creep_and_attack(self):
        for n in ('stand','move','attack'):
            for k in self.a['animations'][n]['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])

    def test_bubbles_rise_and_reset_only_while_invisible(self):
        animation=self.a['animations']['stand']
        for n in BUBBLE_SLOTS:
            keys=animation['bones'][n]['translate'];alpha=animation['slots'][n]['alpha'];visible=[];wraps=0
            for a,b in zip(keys,keys[1:]):
                aa=sample(alpha,a['time'],'value',0);ab=sample(alpha,b['time'],'value',0)
                if b['y']<a['y']-100:
                    wraps+=1;self.assertEqual(aa,0);self.assertEqual(ab,0)
                elif aa>.1 and ab>.1:self.assertGreater(b['y'],a['y'])
                if aa>.1:visible.append(a['y'])
            self.assertEqual(wraps,2);self.assertGreater(max(visible)-min(visible),100)

    def test_bubble_circles_remain_inside_the_jar_glass(self):
        poly=np.array(self.c['glass_polygon'],np.float32)
        for name in ('stand','move','attack','hit'):
            duration=DURATIONS[name]
            for f in range(round(duration*30)+1):
                t=f/30;world=pose(self.a,name,t);inverse=np.linalg.inv(world['jar'])
                for d in self.c['bubbles']:
                    alpha=sample(self.a['animations'][name]['slots'][d['name']]['alpha'],t,'value',0)
                    if alpha<.02:continue
                    local=inverse@np.r_[world[d['name']][:2,2],1];pixel=(float(local[0]+262),float(667-local[1]));scale=np.linalg.norm((inverse@world[d['name']])[:2,0]);radius=.43*d['size']*scale
                    self.assertGreater(cv2.pointPolygonTest(poly,pixel,True),radius,(name,t,d['name']))

    def test_jar_is_rigid_region_not_a_deforming_mesh(self):
        a=self.a['skins'][0]['attachments']['maozi']['maozi'];self.assertNotIn('vertices',a);self.assertEqual(a['path'],'maozi')
        for n,t in [('stand',.8),('move',.4),('attack',.6),('dead',1.6)]:self.assertAlmostEqual(np.linalg.det(pose(self.a,n,t)['jar'][:2,:2]),1,places=5)

    def test_bubbles_stop_and_hide_during_death(self):
        for n in BUBBLE_SLOTS:
            slots=self.a['animations']['dead']['slots'][n];self.assertEqual(sample(slots['alpha'],.6,'value',1),0)
        names={s['name'] for s,a in visible_attachments(self.a,'dead',.7)};self.assertFalse(names&set(BUBBLE_SLOTS))

    def test_move_has_broad_tentacle_wriggling(self):
        p=[]
        for f in range(49):p.append(pose(self.a,'move',f/30)['strike_tip'][:2,2])
        p=np.array(p);self.assertGreater(np.ptp(p[:,0]),80);self.assertGreater(np.ptp(p[:,1]),80)
        tracks=self.a['animations']['move']['bones'];self.assertNotEqual(tracks['front_0']['rotate'],tracks['front_3']['rotate'])

    def test_whip_has_visible_back_windup_and_large_forward_strike(self):
        rest=pose(self.a,'attack',0);wind=pose(self.a,'attack',.3);hit=pose(self.a,'attack',.6)
        self.assertGreater(wind['strike_tip'][0,2],rest['strike_tip'][0,2]+80);self.assertGreater(wind['strike_tip'][1,2],rest['strike_tip'][1,2]+200)
        self.assertLess(hit['strike_tip'][0,2],rest['strike_tip'][0,2]-1200);np.testing.assert_allclose(hit['strike_tip'][:2,2]+self.c['origin'],[-80,-820],atol=.01)
        self.assertEqual(self.a['animations']['attack']['events'],[{'time':.6,'name':'whip_hit'}])

    def test_striking_mouth_aims_forward(self):
        m=pose(self.a,'attack',.6)['front_5'];angle=math.degrees(math.atan2(m[1,0],m[0,0]));self.assertLess(abs(abs(angle)-180),3)

    def test_tentacle_chain_lengths_are_preserved(self):
        d=self.c['chains']['front'];lengths=[math.dist(a,b) for a,b in zip(d['points'],d['points'][1:])]
        for f in range(49):
            w=pose(self.a,'attack',f/30);points=[w[n][:2,2] for n in d['bones']]+[w['strike_tip'][:2,2]]
            np.testing.assert_allclose([np.linalg.norm(a-b) for a,b in zip(points,points[1:])],lengths,atol=.001)

    def test_mouth_overlay_is_rigid_and_reuses_original_png(self):
        a=self.a['skins'][0]['attachments']['front_mouth']['front_mouth'];self.assertEqual(a['path'],'zuochushou_01');self.assertEqual(len(a['uvs']),8);rest=mesh_world(self.a,a,pose(self.a));distance=np.linalg.norm(rest[0]-rest[2])
        for name,t in [('stand',.8),('move',.4),('attack',.5),('attack',.6),('attack',.9)]:
            points=mesh_world(self.a,a,pose(self.a,name,t));self.assertAlmostEqual(np.linalg.norm(points[0]-points[2]),distance,places=4)

    def test_air_is_visible_only_during_whip_stroke(self):
        for n in ('stand','dead','hit','move'):self.assertNotIn('whip_air',{s['name'] for s,a in visible_attachments(self.a,n,.4)})
        for t,on in [(.4,False),(.5,True),(.6,True),(.7,False),(1.2,False)]:self.assertEqual('whip_air' in {s['name'] for s,a in visible_attachments(self.a,'attack',t)},on)

    def test_airflow_leading_edge_matches_actual_tip(self):
        skin=self.a['skins'][0]['attachments']['whip_air'];world=pose(self.a)
        for f in range(15,19):
            a=skin[f'air_{f}'];points=mesh_world(self.a,a,world);uv=np.array(a['uvs']).reshape(-1,2);head=points[uv[:,0]>.99].mean(axis=0);target=pose(self.a,'attack',f/30)['strike_tip'][:2,2];np.testing.assert_allclose(head,target,atol=.001)

    def test_air_triangles_have_consistent_setup_winding(self):
        for a in self.a['skins'][0]['attachments']['whip_air'].values():
            p=mesh_world(self.a,a,pose(self.a));tri=p[np.array(a['triangles']).reshape(-1,3)];v=tri[:,1]-tri[:,0];w=tri[:,2]-tri[:,0];self.assertTrue(np.all(v[:,0]*w[:,1]-v[:,1]*w[:,0]<-1e-8))

    def test_actual_character_stays_connected_at_key_poses(self):
        actor=deepcopy(self.a);actor['slots']=[s for s in actor['slots'] if s['name'] not in BUBBLE_SLOTS and s['name']!='whip_air']
        for a in actor['animations'].values():a.pop('drawOrder',None)
        view=scaled_preview(actor,.3)
        for n,t in [('stand',0),('stand',.8),('move',.4),('move',1.2),('attack',.3),('attack',.5),('attack',.6),('attack',.9),('dead',1.6)]:
            mask=(np.asarray(render(view,self.images,n,t,size=(750,495),camera=(435,450)))[:,:,3]>32).astype('uint8');_,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8);self.assertEqual(int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>80)),1,(n,t))

    def test_dead_is_grounded_settled_and_feet_follow_collapse(self):
        a=pose(self.a,'dead',1.4);b=pose(self.a,'dead',1.6)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.0001)
        self.assertLessEqual(abs(render(self.a,self.images,'dead',1.6).getbbox()[3]-1500),1);rest=pose(self.a)
        for f in (0,10,20,30,42,48):
            w=pose(self.a,'dead',f/30)
            for n in ('foot_back','foot_front'):np.testing.assert_allclose(np.linalg.inv(w['pelvis'])@w[n],np.linalg.inv(rest['pelvis'])@rest[n],atol=.0001)

    def test_world_rotation_tracks_have_no_single_frame_flip(self):
        for n,d in DURATIONS.items():
            previous=None
            for f in range(round(d*30)+1):
                w=pose(self.a,n,f/30);angles={k:math.degrees(math.atan2(m[1,0],m[0,0])) for k,m in w.items()}
                if previous:
                    for bone,angle in angles.items():self.assertLess(abs((angle-previous[bone]+180)%360-180),45,(n,bone,f))
                previous=angles

    def test_attack_hit_return_to_neutral_main_pose(self):
        rest=pose(self.a)
        for n in ('attack','hit'):
            end=pose(self.a,n,DURATIONS[n])
            for bone in rest:
                if bone in BUBBLE_SLOTS:continue
                np.testing.assert_allclose(end[bone],rest[bone],atol=.0001)

    def test_official_runtime_515_poses_match_independent_geometry(self):
        r=self.report;self.assertEqual(len(r['samples']),515);self.assertFalse(r['desktopEditorTested']);self.assertEqual(r['skeletonSHA256'],sha256((BASE/'sycs-4.3.json').read_bytes()).hexdigest());skin=self.a['skins'][0]['attachments'];slots={s['name']:s for s in self.a['slots']}
        for item in r['samples']:
            name,t=item['animation'],item['time'];world=pose(self.a,name,t)
            for n,v in item['bones'].items():
                m=world[n];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],v,atol=.005)
            self.assertEqual([s['name'] for s in draw_order(self.a,name,t)],item['drawOrder']);self.assertEqual({s['name'] for s,a in visible_attachments(self.a,name,t)},{n for n,a in item['attachments'].items() if a})
            for n,rec in item['meshes'].items():
                a=skin[n][item['attachments'][n]]
                if a.get('type')=='mesh':points=mesh_world(self.a,a,world)
                else:
                    w,h=a['width'],a['height'];corners=np.array([[-w/2,-h/2],[-w/2,h/2],[w/2,h/2],[w/2,-h/2]])+[a.get('x',0),a.get('y',0)];points=(world[slots[n]['bone']]@np.column_stack((corners,np.ones(4))).T).T[:,:2]
                np.testing.assert_allclose(points[rec['indices']].ravel(),rec['vertices'],atol=.005)
            for n,alpha in item['alpha'].items():self.assertAlmostEqual(sample(self.a['animations'][name].get('slots',{}).get(n,{}).get('alpha',[]),t,'value',1),alpha,places=5)

    def test_bubble_effect_changes_real_pixels_inside_jar(self):
        view=scaled_preview(self.a,.42);hidden=deepcopy(view);hidden['slots']=[s for s in hidden['slots'] if s['name'] not in BUBBLE_SLOTS];kwargs=dict(size=(1050,693),camera=(609,630))
        a=np.asarray(render(view,self.images,'stand',.4,**kwargs)).astype(float);b=np.asarray(render(hidden,self.images,'stand',.4,**kwargs)).astype(float);self.assertGreater(np.count_nonzero(np.max(np.abs(a-b),axis=2)>8),150)

    def test_preview_scale_is_uniform(self):
        view=scaled_preview(self.a,.42)
        for n,t in [('stand',.8),('attack',.6),('dead',1.6)]:
            a=pose(self.a,n,t);b=pose(view,n,t)
            for bone in a:np.testing.assert_allclose(a[bone][:2,:]*.42,b[bone][:2,:],atol=.00001)

if __name__=='__main__':unittest.main(verbosity=2)
