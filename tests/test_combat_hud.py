"""Combat HUD tests: no future outcomes, real launches, settlement, seeking and copying."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.training.combat_hud import CombatTimeline, annotations, augment_mcap, image_size

EPOCH=1_000_000_000_000_000_000
MS=1_000_000


def fixture(*, timeout=False, zero=False, before=False):
    events=[]
    def event(at,kind,**data):
        events.append(dict(event_id=len(events)+1,round_id=1,time_ns=at*MS,kind=kind,data=data))
    event(0,'round_started')
    if before:
        event(50,'shot_fired',projectile_id=2,request_id=2,robot_id=1)
        event(75,'damage_applied',projectile_id=2,request_id=2,shooter=1,target=2,actual=20)
        event(75,'projectile_ended',projectile_id=2,request_id=2,reason='armor')
    if not zero:event(150,'shot_fired',projectile_id=1,request_id=1,robot_id=1)
    event(300,'evaluation_window_closed',window_end_ns=300*MS)
    if not zero and not timeout:
        event(350,'damage_applied',projectile_id=1,request_id=1,shooter=1,target=2,actual=20)
        event(350,'projectile_ended',projectile_id=1,request_id=1,reason='armor')
    samples={};previous=-1
    for at in (0,100,200,300,350,400):
        shots=sum(e['kind']=='shot_fired' and e['time_ns']<=at*MS for e in events)
        damage=sum(e['data']['actual'] for e in events if e['kind']=='damage_applied' and e['time_ns']<=at*MS)
        own=dict(robot_id=1,hp=350,actual_shots=shots,damage_dealt=damage,heat_locked_s=.2 if at>=300 else 0.,
            resources=dict(heat=82.,heat_limit=88.,ordinary_heat_lock_count=int(at>=300),cooling_locked=at>=300))
        target=dict(robot_id=2,hp=(40 if before else 20)-damage,resources={})
        samples[EPOCH+at*MS]=dict(round_id=1,sim_time_ns=at*MS,evaluation=dict(robots=[target,own]),
            events=[e for e in events if previous<e['time_ns']<=at*MS])
        previous=at*MS
    expected=dict(status='incomplete' if timeout else 'complete',official_damage=None if timeout else 0 if zero else 20,
        eligible_shots=0 if zero else 1,damaging_projectiles=0 if zero or timeout else 1,
        eligible_damage_observed=0 if zero or timeout else 20,eligible_unresolved=int(timeout and not zero),
        excluded_own_shots=int(before),excluded_own_damage=20 if before else 0,
        own_physical_damage=(20 if before else 0)+(0 if zero or timeout else 20),event_count=len(events),closed=True)
    info=dict(complete=True,phase='complete',model=dict(kind='checkpoint',num_timesteps=1000),
        scene=dict(stage_id='C8-T2',case=16,motion_type='lateral',scenario=dict(duel=dict(target=dict(profile='standard')))),
        summary=dict(start_ns=100*MS,end_ns=300*MS,settlement=expected))
    return samples,info


class CombatHudTests(unittest.TestCase):
    def test_no_future_hit_then_tail_hit_then_final_score(self):
        samples,info=fixture();timeline=CombatTimeline(samples,info,EPOCH+400*MS)
        warm=timeline.snapshot(EPOCH+50*MS);self.assertIsNone(warm['shots']);self.assertEqual(warm['phase'],'warming')
        early=timeline.snapshot(EPOCH+200*MS)
        self.assertEqual((early['shots'],early['hits'],early['damage'],early['pending']),(1,0,0,1))
        self.assertIsNone(early['official_damage'])
        late=timeline.snapshot(EPOCH+350*MS)
        self.assertEqual((late['phase'],late['hits'],late['damage'],late['pending']),('settling',1,20,0))
        self.assertIsNone(late['official_damage'])
        final=timeline.snapshot(EPOCH+400*MS)
        self.assertEqual((final['phase'],final['official_damage'],final['kill_time_s']),('complete',20,.25))
        text=annotations(final,info,1280,720,EPOCH+290*MS,epoch=EPOCH,start_ns=100*MS)
        self.assertIn('最终成绩',text['texts'][4]['text']);self.assertIn('定格',text['texts'][4]['text'])

    def test_seeking_backwards_rebuilds_snapshot(self):
        samples,info=fixture();timeline=CombatTimeline(samples,info,EPOCH+400*MS)
        final=timeline.snapshot(EPOCH+400*MS);early=timeline.snapshot(EPOCH+200*MS)
        self.assertEqual(early['hits'],0);self.assertIsNone(early['official_damage'])
        self.assertEqual(final,timeline.snapshot(EPOCH+400*MS))

    def test_retransmitted_events_are_not_double_counted(self):
        samples,info=fixture();samples[EPOCH+400*MS]['events']=copy.deepcopy(samples[EPOCH]['events'])
        self.assertEqual(CombatTimeline(samples,info,EPOCH+400*MS).snapshot(EPOCH+400*MS)['damage'],20)
        samples[EPOCH+400*MS]['events'][0]['kind']='conflicting'
        with self.assertRaisesRegex(ValueError,'conflicting duplicate'):CombatTimeline(samples,info,EPOCH+400*MS)

    def test_window_statistics_do_not_use_total_physical_damage(self):
        samples,info=fixture(before=True);snapshot=CombatTimeline(samples,info,EPOCH+400*MS).snapshot(EPOCH+400*MS)
        self.assertEqual(snapshot['damage'],20);self.assertEqual(snapshot['shots'],1)
        self.assertEqual(samples[EPOCH+400*MS]['evaluation']['robots'][1]['damage_dealt'],40)

    def test_zero_launch_is_not_displayed_as_zero_percent(self):
        samples,info=fixture(zero=True);snapshot=CombatTimeline(samples,info,EPOCH+400*MS).snapshot(EPOCH+400*MS)
        text=annotations(snapshot,info,1280,720,EPOCH+400*MS)
        self.assertIn('—（0/0）',text['texts'][1]['text']);self.assertNotIn('0.00%',text['texts'][1]['text'])

    def test_timeout_never_claims_final_score(self):
        samples,info=fixture(timeout=True);snapshot=CombatTimeline(samples,info,EPOCH+400*MS).snapshot(EPOCH+400*MS)
        self.assertEqual(snapshot['phase'],'incomplete');self.assertIsNone(snapshot['official_damage'])
        text=annotations(snapshot,info,1280,720,EPOCH+290*MS,epoch=EPOCH,start_ns=100*MS)
        self.assertIn('结算不完整',text['texts'][4]['text']);self.assertIn('暂计伤害',text['texts'][2]['text'])

    def test_warmup_failure_is_not_revealed_early(self):
        samples,info=fixture(zero=True);info['summary']=dict(status='warmup_timed_out')
        timeline=CombatTimeline(samples,info,EPOCH+400*MS)
        self.assertEqual(timeline.snapshot(EPOCH+100*MS)['phase'],'warming')
        self.assertEqual(timeline.snapshot(EPOCH+400*MS)['phase'],'warmup_failed')

    def test_missing_hp_and_heat_are_unavailable_not_zero(self):
        samples,info=fixture()
        for row in samples.values():
            for robot in row['evaluation']['robots']:robot.pop('hp',None);robot['resources']={}
        snapshot=CombatTimeline(samples,info,EPOCH+400*MS).snapshot(EPOCH+400*MS)
        self.assertIsNone(snapshot['heat']);self.assertIsNone(snapshot['own_hp'])
        text=annotations(snapshot,info,1280,720,EPOCH+400*MS)
        self.assertIn('己方HP 不可用',text['texts'][2]['text']);self.assertIn('热量 不可用',text['texts'][3]['text'])

    def test_conflicting_final_score_fails_instead_of_showing_green(self):
        samples,info=fixture();info['summary']['settlement']['eligible_shots']=2
        with self.assertRaisesRegex(ValueError,'final score mismatch'):CombatTimeline(samples,info,EPOCH+400*MS)

    def test_dimensions_and_scaling(self):
        jpeg=b'\xff\xd8\xff\xc0'+struct.pack('>H',11)+bytes([8])+struct.pack('>HH',720,1280)+bytes([1,1,0x11,0])+b'\xff\xd9'
        self.assertEqual(image_size(jpeg),(1280,720))
        samples,info=fixture();snapshot=CombatTimeline(samples,info,EPOCH+400*MS).snapshot(EPOCH+400*MS)
        text=annotations(snapshot,info,640,360,EPOCH+400*MS)
        self.assertEqual(len(text['texts']),5);self.assertEqual(text['texts'][0]['font_size'],11)

    def test_mcap_preserves_messages_and_only_adds_available_view(self):
        from mcap.writer import Writer
        from mcap.reader import make_reader
        samples,info=fixture()
        jpeg=b'\xff\xd8\xff\xc0'+struct.pack('>H',11)+bytes([8])+struct.pack('>HH',720,1280)+bytes([1,1,0x11,0])+b'\xff\xd9'
        import base64
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.mcap';dest=Path(folder)/'hud.mcap'
            with source.open('wb') as f:
                writer=Writer(f);writer.start()
                schema=writer.register_schema('json','jsonschema',b'{"type":"object"}')
                physical=writer.register_channel('/replay/physical','json',schema)
                identity=writer.register_channel('/replay/info','json',schema)
                image=writer.register_channel('/vision/camera/image','json',schema)
                for at,row in samples.items():writer.add_message(physical,at,json.dumps(row).encode(),at)
                for at in (EPOCH,EPOCH+290*MS):writer.add_message(image,at,json.dumps(dict(data=base64.b64encode(jpeg).decode())).encode(),at)
                writer.add_message(identity,EPOCH+400*MS,json.dumps(info).encode(),EPOCH+400*MS)
                writer.add_metadata('original',dict(value='preserve me'));writer.finish()
            original=source.read_bytes();result=augment_mcap(source,dest)
            self.assertTrue(result['passed']);self.assertEqual(source.read_bytes(),original)
            self.assertEqual(set(result['hud_messages']),{'/replay/hud/first_person'})
            self.assertGreater(result['held_updates'],0)
            with dest.open('rb') as f:
                reader=make_reader(f);metadata=list(reader.iter_metadata())
            self.assertTrue(any(r.name=='original' for r in metadata))
            again=Path(folder)/'again.mcap';repeated=augment_mcap(dest,again)
            self.assertEqual(repeated['hud_messages'],result['hud_messages'])
            with self.assertRaises(FileExistsError):augment_mcap(source,source)


if __name__=='__main__':unittest.main()
