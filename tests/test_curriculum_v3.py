import copy
import math
import unittest
from collections import Counter
import numpy as np
from src.training import curriculum_v2 as old
from src.training import curriculum_v3 as c

ROLE="infantry_health_cooling"

def rows(course):
    return [dict(course=course,group="translation",profile=d["scenario"]["duel"]["target"]["profile"],
        motion_type=d["motion_type"],case=d["case"],status="complete",actual_shots=20,
        damaging_hits=10,end_reason="target_destroyed") for d in c.bank(ROLE,c.DEFAULT)[course]]

def calibration():
    return c.build_calibration(ROLE,{str(i):rows(i) for i in c.TRANSLATION},{"test":"fixture"})

class CurriculumV3Tests(unittest.TestCase):
    def test_frozen_legacy_distribution_and_balanced_splits(self):
        seen=set()
        for split in ("calibration","validation","test"):
            bank=c.bank(ROLE,c.DEFAULT,split=split)
            legacy=old.bank(ROLE,old.DEFAULT,test=split=="test")
            for course in range(12):
                if course not in c.TRANSLATION:
                    if split!="calibration":
                        self.assertEqual([d["scenario"] for d in bank[course]],[d["scenario"] for d in legacy[course]])
                    continue
                self.assertEqual(len(bank[course]),96)
                self.assertEqual(set(Counter((d["scenario"]["duel"]["target"]["profile"],d["motion_type"]) for d in bank[course]).values()),{8})
                for d in bank[course]:
                    self.assertNotIn(d["seed"],seen);seen.add(d["seed"])
                    self.assertEqual(c.classify(d["translation_angle_deg"]),d["motion_type"])
                    lo,hi=(1.5,2.) if course==7 else (1.,1.5)
                    self.assertTrue(lo<=d["scenario"]["duel"]["target"]["motion"]["amplitude_m"]<=hi)
    def test_direction_does_not_change_armor_orientation(self):
        a=c.scene(8,17,ROLE,strata={"motion_type":"radial"})
        b=c.scene(8,17,ROLE,strata={"motion_type":"lateral"})
        self.assertEqual(a["scenario"]["duel"]["target"]["yaw_rad"],b["scenario"]["duel"]["target"]["yaw_rad"])
        for d in (a,b):
            duel=d["scenario"]["duel"];v=np.asarray(duel["controlled"]["position_xz_m"])-duel["target"]["position_xz_m"]
            heading=duel["target"]["motion"]["translation_heading_rad"]
            axis=np.array([math.cos(heading),-math.sin(heading)])
            angle=math.degrees(math.acos(np.clip(abs(v@axis)/np.linalg.norm(v),0,1)))
            self.assertAlmostEqual(angle,d["translation_angle_deg"])
    def test_angle_boundaries(self):
        self.assertEqual([c.classify(x) for x in (0,19.99,20,59.99,60,90)],['radial','radial','oblique','oblique','lateral','lateral'])
    def test_calibration_failure_and_tamper(self):
        self.assertEqual(calibration()["thresholds"]["8"]["standard/lateral"]["hit_efficiency"],.45)
        for field,value in (("actual_shots",0),("damaging_hits",0),("end_reason","time_limit"),("status","settlement_timed_out")):
            data={str(i):rows(i) for i in c.TRANSLATION}
            for r in data['8']:
                if r['profile']=='folding' and r['motion_type']=='lateral':r[field]=value
            with self.assertRaises(ValueError):c.build_calibration(ROLE,data,{})
        bad=calibration();bad['thresholds']['8']['standard/lateral']['kill_rate']=0
        with self.assertRaises(ValueError):c.validate_calibration(bad)
    def test_easy_cases_cannot_hide_lateral_failure(self):
        data=rows(8)
        self.assertTrue(c.assess(8,data,calibration())["passed"])
        for r in data:
            if r['profile']=='folding' and r['motion_type']=='lateral':r['damaging_hits']=1
        self.assertFalse(c.assess(8,data,calibration())["passed"])
        data=rows(8);data[0]['status']='settlement_timed_out'
        self.assertFalse(c.assess(8,data,calibration())["passed"])
        self.assertFalse(c.assess(8,rows(8),None)["passed"])
    def test_two_of_three_latest_and_restoration(self):
        for sequence,expected in (([True,True],False),([True,False,True],True),([True,True,False],False),([False,False,True],False)):
            value=c.Curriculum(c.DEFAULT);value.stage=8
            for passed in sequence:value.observe(dict(passed=passed,deficit=0 if passed else 1))
            self.assertEqual(value.promotion_ready(),expected)
            restored=c.Curriculum(c.DEFAULT,state=value.state())
            self.assertEqual(restored.promotion_ready(),expected)
        value=c.Curriculum(c.DEFAULT)
        for passed in (True,False,True):value.observe(dict(passed=passed,deficit=0))
        self.assertFalse(value.promotion_ready())
    def test_release_revision_two_balances_32_without_changing_first_cases(self):
        old_bank=c.bank(ROLE,c.DEFAULT)
        expanded=c.bank(ROLE,c.REVISION_2)
        for course in c.TRANSLATION:
            self.assertEqual(len(expanded[course]),384)
            counts=Counter((d["scenario"]["duel"]["target"]["profile"],d["motion_type"]) for d in expanded[course])
            self.assertEqual(set(counts.values()),{32})
            self.assertEqual([d["scenario"] for d in old_bank[course]],[d["scenario"] for d in expanded[course][:96]])
            self.assertTrue(all(d["release_revision"]==2 for d in expanded[course]))
        value=c.Curriculum(c.REVISION_2);value.stage=8
        self.assertEqual(value.next_scene(ROLE,0)["release_revision"],2)
        self.assertEqual(c.Curriculum(c.REVISION_2,state=value.state()).state(),value.state())

    def test_revision_three_requires_clear_paths_for_every_course(self):
        from src.training.scene_admission import difficulty_signature
        previous=c.bank(ROLE,c.REVISION_2)
        current=c.bank(ROLE,c.CURRENT)
        for course,scenes in enumerate(current):
            for before,after in zip(previous[course],scenes):
                self.assertEqual(after['release_revision'],3)
                self.assertEqual(after['scenario']['visibility_admission'],'full_path_clear')
                expected=copy.deepcopy(before)
                expected['release_revision']=3
                expected['scenario']['visibility_admission']='full_path_clear'
                self.assertEqual(expected,after)
                retry=c.scene(course,after['seed'],ROLE,after['case'],attempt=7,
                    strata=after['strata'],release_revision=3)
                self.assertEqual(difficulty_signature(after),difficulty_signature(retry))
        with self.assertRaises(ValueError):c.validate(dict(c.CURRENT,calibration=calibration()))
        with self.assertRaises(ValueError):c.validate(dict(c.CURRENT,release_revision=4))

    def test_32_case_granularity_keeps_original_threshold_formula(self):
        samples={}
        for course in c.TRANSLATION:
            values=[]
            counts=Counter()
            for d in c.bank(ROLE,c.CURRENT)[course]:
                key=(d["scenario"]["duel"]["target"]["profile"],d["motion_type"])
                counts[key]+=1
                values.append(dict(course=course,group="translation",profile=key[0],motion_type=key[1],
                    status="complete",actual_shots=20,damaging_hits=10,
                    end_reason="target_destroyed" if counts[key]<=28 else "time_limit"))
            samples[str(course)]=values
        artifact=c.build_calibration(ROLE,samples,{},cases_per_bucket=32)
        c.validate_calibration(artifact)
        self.assertAlmostEqual(artifact["thresholds"]["8"]["standard/lateral"]["kill_rate"],.775)
        values=copy.deepcopy(samples['8']);counts=Counter()
        for row in values:
            key=(row['profile'],row['motion_type']);counts[key]+=1
            row['end_reason']='target_destroyed' if counts[key]<=25 else 'time_limit'
        self.assertTrue(c.assess(8,values,artifact)['passed'])
        next(r for r in reversed(values) if r['profile']=='standard' and r['motion_type']=='lateral' and r['end_reason']=='target_destroyed')['end_reason']='time_limit'
        self.assertFalse(c.assess(8,values,artifact)['passed'])

    def test_expanded_splits_are_disjoint_and_balance_roles_and_directions(self):
        for role in (ROLE, 'sentry'):
            seen=set()
            for split in ('calibration','validation','test'):
                bank=c.bank(role,c.CURRENT,split=split)
                for course in c.TRANSLATION:
                    groups={}
                    for descriptor in bank[course]:
                        self.assertNotIn(descriptor['seed'],seen)
                        seen.add(descriptor['seed'])
                        strata=descriptor['strata']
                        key=(strata['profile'],strata['motion_type'])
                        groups.setdefault(key,Counter())[(strata['target_role'],strata['direction'],strata['lateral_side'])]+=1
                    self.assertEqual(len(groups),12)
                    for counts in groups.values():
                        self.assertEqual(len(counts),8)
                        self.assertEqual(set(counts.values()),{4})

    def test_visibility_release_certificate_binds_geometry_and_runtime(self):
        from src.training.scene_admission import validate_visibility_certificate
        d=c.scene(8,17,ROLE,release_revision=3)
        d['admitted']=True
        proof=dict(version=1,policy='full_path_clear',duration_ms=45000,max_interval_ms=10,
            min_interval_ms=1,min_visible_armor_count=1,algorithm_fingerprint_fnv64='fixture',policy_inputs=False)
        d['visibility_certificate']=dict(proof=proof,scenario_sha256=c.digest(d['scenario']),
            native_sha256='native',simulator_config_sha256='config')
        validate_visibility_certificate(d,'native','config')
        for mutation in ('native','config','pose','proof','missing'):
            value=copy.deepcopy(d)
            if mutation=='native':value['visibility_certificate']['native_sha256']='changed'
            if mutation=='config':value['visibility_certificate']['simulator_config_sha256']='changed'
            if mutation=='pose':value['scenario']['duel']['target']['position_xz_m'][0]+=0.01
            if mutation=='proof':value['visibility_certificate']['proof']['duration_ms']=30000
            if mutation=='missing':value.pop('visibility_certificate')
            with self.assertRaises(ValueError):validate_visibility_certificate(value,'native','config')

    def test_sampling_weights(self):
        value=c.Curriculum(c.DEFAULT);value.stage=8
        self.assertAlmostEqual(value.weights()[8],.7)
        sample=[value.next_scene(ROLE,0) for _ in range(5000)]
        current=[s for s in sample if s['course']==8]
        counts=Counter(s['motion_type'] for s in current)
        for key,target in zip(c.MOTIONS,c.MOTION_WEIGHTS):self.assertAlmostEqual(counts[key]/len(current),target,delta=.03)

    def test_revision_three_random_training_uses_clearance_proposals_without_rng_drift(self):
        from src.training.curriculum import descriptor_scene
        from src.training.scene_admission import difficulty_signature
        previous=c.Curriculum(c.REVISION_2)
        current=c.Curriculum(c.CURRENT)
        previous.stage=current.stage=8
        for _ in range(12):
            before=previous.next_scene(ROLE,0)
            after=current.next_scene(ROLE,0)
            expected=copy.deepcopy(before)
            expected['release_revision']=3
            expected['scenario']['visibility_admission']='full_path_clear'
            self.assertEqual(difficulty_signature(expected),difficulty_signature(after))
            self.assertEqual('placement_search' in after,after['course'] in c.TRANSLATION)
            candidate=descriptor_scene(after,ROLE,0)
            expected_candidate=copy.deepcopy(after)
            if 'placement_search' not in after:expected_candidate.pop('episode_index')
            self.assertEqual(difficulty_signature(expected_candidate),difficulty_signature(candidate))
            self.assertEqual(current.rng.bit_generator.state,previous.rng.bit_generator.state)
        for split in ('calibration','validation','test'):
            self.assertTrue(all('placement_search' not in d for group in c.bank(ROLE,c.CURRENT,split=split) for d in group))

if __name__ == "__main__":unittest.main()
