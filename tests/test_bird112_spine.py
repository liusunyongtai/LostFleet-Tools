"""Bird112: source fidelity, readable flying loops, grounded death, official-runtime agreement."""
from copy import deepcopy
from hashlib import sha256
import json,math,unittest
import cv2
import numpy as np
from PIL import Image
from tools.build_bird112_spine import BASE,DURATIONS,pose,render,mesh_world,load_images,scaled_preview
from tools.build_bsq_spine import draw_order
from tools.build_xqdz_spine import visible_attachments

class Bird112SpineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor=json.loads((BASE/'bird112-4.3.json').read_text());cls.source=json.loads((BASE/'source/photoshop.json').read_text());cls.config=json.loads((BASE/'rig-config.json').read_text());cls.images=load_images(BASE/'images');cls.report=json.loads((BASE/'runtime-validation.json').read_text())

    def test_five_names_version_bones_and_original_layers(self):
        self.assertEqual(set(self.actor['animations']),{'stand','dead','hit','move','attack'});self.assertEqual(self.actor['skeleton']['spine'],'4.3.23');self.assertEqual(self.actor['skeleton']['images'],'./images/')
        self.assertEqual(len(self.actor['bones']),25);self.assertEqual(len(self.actor['slots']),11);self.assertEqual([s['name'] for s in self.actor['slots']],[s['name'] for s in self.source['slots']])

    def test_eleven_pngs_are_unchanged_and_transparent(self):
        hashes=json.loads((BASE/'source/image-sha256.json').read_text());self.assertEqual(len(hashes),11)
        for n,h in hashes.items():
            self.assertEqual(sha256((BASE/'images'/n).read_bytes()).hexdigest(),h);im=Image.open(BASE/'images'/n);self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getchannel('A').getextrema(),(0,255))

    def test_setup_coordinates_and_normalized_positive_weights(self):
        world=pose(self.actor);origin=np.array(self.config['origin'])
        for n,entries in self.actor['skins'][0]['attachments'].items():
            a=entries[n];uv=np.array(a['uvs']).reshape(-1,2);r=self.source['skins']['default'][n][n];expected=np.column_stack((r['x']+(uv[:,0]-.5)*r['width'],r['y']+(.5-uv[:,1])*r['height']))-origin
            np.testing.assert_allclose(mesh_world(self.actor,a,world),expected,atol=.0001)
            data=a['vertices'];i=0
            while i<len(data):
                count=int(data[i]);i+=1;weights=[data[i+j*4+3] for j in range(count)];self.assertAlmostEqual(sum(weights),1,places=8);self.assertTrue(all(0<w<=1 for w in weights));i+=count*4

    def test_setup_render_matches_photoshop_assembly(self):
        size=(1700,1500);camera=(850,1300);ox,oy=self.config['origin'];ref=Image.new('RGBA',size)
        for slot in self.source['slots']:
            n=slot['name'];r=self.source['skins']['default'][n][n];ref.alpha_composite(self.images[n],(round(r['x']-r['width']/2-ox+camera[0]),round(-r['y']-r['height']/2+oy+camera[1])))
        a=np.asarray(render(self.actor,self.images,size=size,camera=camera)).astype(float);b=np.asarray(ref).astype(float);mask=(a[:,:,3]>0)|(b[:,:,3]>0);self.assertLess(np.abs(a-b)[mask].mean(),.3)

    def test_three_flight_loops_have_matching_endpoints(self):
        for name in ('stand','move','attack'):
            a=pose(self.actor,name,0);b=pose(self.actor,name,DURATIONS[name])
            for n in a:np.testing.assert_allclose(a[n],b[n],atol=.0001)

    def test_flight_and_dive_root_never_move(self):
        for name in ('stand','move','attack'):
            for k in self.actor['animations'][name]['bones']['root']['translate']:self.assertEqual([k['x'],k['y']],[0,0])

    def test_dive_stays_head_down_for_entire_loop(self):
        for i in range(49):
            w=pose(self.actor,'attack',i/60);m=w['body'];angle=math.degrees(math.atan2(m[1,0],m[0,0]));self.assertGreaterEqual(angle,31.9);self.assertLessEqual(angle,40.1)
            self.assertLess(w['head'][1,2],w['body'][1,2]-60);self.assertGreater(w['tail_mid'][1,2],w['body'][1,2]+50)

    def test_dive_wings_have_reduced_span(self):
        rest=pose(self.actor)
        for t in (0,.2,.4,.6,.8):
            w=pose(self.actor,'attack',t)
            for n in ('front_wing','back_wing'):
                span=np.linalg.norm(w[n+'_tip'][:2,2]-w[n][:2,2]);restspan=np.linalg.norm(rest[n+'_tip'][:2,2]-rest[n][:2,2]);self.assertLess(span/restspan,.73)

    def test_move_has_large_wingtip_excursions(self):
        a=pose(self.actor,'move',.2);b=pose(self.actor,'move',.6)
        for n in ('front_wing_tip','back_wing_tip'):self.assertGreater(abs(a[n][1,2]-b[n][1,2]),180)

    def test_wing_elbow_and_tip_lag_the_primary_stroke(self):
        tracks=self.actor['animations']['move']['bones'];root=max(tracks['front_wing']['rotate'],key=lambda k:k['value'])['time'];elbow=min(tracks['front_wing_elbow']['rotate'],key=lambda k:k['value'])['time'];tip=min(tracks['front_wing_tip']['rotate'],key=lambda k:k['value'])['time']
        self.assertLess(root,elbow);self.assertLess(elbow,tip)

    def test_tail_chain_has_delayed_follow_through(self):
        tracks=self.actor['animations']['move']['bones'];peaks=[min(tracks[n]['rotate'],key=lambda k:k['value'])['time'] for n in ('tail_base','tail_mid','tail_tip','tail_curl')]
        self.assertTrue(all(a<b for a,b in zip(peaks,peaks[1:])))

    def test_wings_jaws_and_tail_stay_connected_at_key_poses(self):
        actor=scaled_preview(self.actor,.55)
        for name,t in [('stand',0),('stand',.3),('stand',.9),('move',.2),('move',.6),('attack',0),('attack',.2),('attack',.6),('hit',.1),('dead',1.6)]:
            mask=(np.asarray(render(actor,self.images,name,t,size=(990,770),camera=(522,688)))[:,:,3]>32).astype('uint8');_,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
            self.assertEqual(int(np.count_nonzero(stats[1:,cv2.CC_STAT_AREA]>80)),1,(name,t))

    def test_flight_clearance_and_preview_bounds(self):
        for name in ('stand','move','attack'):
            for t in (0,DURATIONS[name]/4,DURATIONS[name]/2,DURATIONS[name]*.75):
                bounds=render(self.actor,self.images,name,t).getbbox();self.assertGreater(bounds[0],0);self.assertGreater(bounds[1],0);self.assertLess(bounds[2],1800);self.assertLess(bounds[3],1250-80)

    def test_hit_recoils_and_recovers_without_reset_pop(self):
        start=pose(self.actor,'hit',0);hit=pose(self.actor,'hit',.1);end=pose(self.actor,'hit',.5)
        self.assertGreater(np.linalg.norm(hit['front_wing_tip'][:2,2]-start['front_wing_tip'][:2,2]),100)
        for n in start:np.testing.assert_allclose(start[n],end[n],atol=.0001)

    def test_death_ground_contact_and_settled_end(self):
        a=pose(self.actor,'dead',1.4);b=pose(self.actor,'dead',1.6)
        for n in a:np.testing.assert_allclose(a[n],b[n],atol=.0001)
        self.assertLessEqual(abs(render(self.actor,self.images,'dead',1.6).getbbox()[3]-1250),1)
        head=deepcopy(self.actor);head['slots']=[s for s in head['slots'] if s['name']=='toubu'];self.assertLessEqual(abs(render(head,self.images,'dead',1.6).getbbox()[3]-1250),15)

    def test_no_full_turn_adjacent_key_jumps(self):
        for name,a in self.actor['animations'].items():
            for n,tr in a['bones'].items():
                keys=tr.get('rotate',[])
                for x,y in zip(keys,keys[1:]):self.assertLess(abs(x['value']-y['value']),45,(name,n))

    def test_head_and_fangs_keep_rigid_shape_during_flight_dive_hit(self):
        rest=pose(self.actor);skin=self.actor['skins'][0]['attachments']
        for n in ('toubu','zuoya','youya'):
            a=skin[n][n];p=mesh_world(self.actor,a,rest);distance=np.linalg.norm(p[0]-p[2])
            for name,t in [('move',.2),('attack',.2),('hit',.1)]:
                p=mesh_world(self.actor,a,pose(self.actor,name,t));self.assertAlmostEqual(np.linalg.norm(p[0]-p[2]),distance,places=4)

    def test_preview_scale_is_uniform_and_does_not_modify_source(self):
        view=scaled_preview(self.actor,.55)
        for name,t in [('stand',.3),('attack',.2),('dead',1.6)]:
            a=pose(self.actor,name,t);b=pose(view,name,t)
            for n in a:np.testing.assert_allclose(a[n][:2,:]*.55,b[n][:2,:],atol=.00001)
        self.assertNotIn('scaleX',self.actor['bones'][0])

    def test_official_runtime_371_poses_match_independent_world_and_mesh(self):
        r=self.report;self.assertEqual(len(r['samples']),371);self.assertFalse(r['desktopEditorTested']);self.assertEqual(r['sampleHz'],60);self.assertEqual(r['skeletonSHA256'],sha256((BASE/'bird112-4.3.json').read_bytes()).hexdigest());skin=self.actor['skins'][0]['attachments']
        for item in r['samples']:
            n,t=item['animation'],item['time'];world=pose(self.actor,n,t)
            for bone,vals in item['bones'].items():
                m=world[bone];np.testing.assert_allclose([m[0,0],m[0,1],m[1,0],m[1,1],m[0,2],m[1,2]],vals,atol=.005)
            self.assertEqual([s['name'] for s in draw_order(self.actor,n,t)],item['drawOrder'])
            self.assertEqual({s['name'] for s,a in visible_attachments(self.actor,n,t)},{s for s,a in item['attachments'].items() if a})
            for slot,record in item['meshes'].items():
                p=mesh_world(self.actor,skin[slot][item['attachments'][slot]],world);np.testing.assert_allclose(p[record['indices']].ravel(),record['vertices'],atol=.005)

    def test_dive_loop_does_not_emit_unrequested_damage_events(self):
        self.assertNotIn('events',self.actor['animations']['attack']);self.assertEqual(self.actor['animations']['hit']['events'],[{'time':.1,'name':'impact'}]);self.assertEqual(self.actor['animations']['dead']['events'],[{'time':1.4,'name':'death_end'}])

if __name__=='__main__':unittest.main(verbosity=2)
