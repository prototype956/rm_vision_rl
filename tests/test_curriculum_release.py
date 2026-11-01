"""Release evidence, qualified validation coverage and frozen dependencies."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch
from src.training.curriculum_v3 import assess, build_calibration, PROFILES, MOTIONS
from src.training.curriculum_v2 import digest
from tools.training import curriculum_v3_release as gate
from pathlib import Path
import tempfile
from tools.training.curriculum_v3_release import audit_qualification
from tools.training.curriculum_v3_release import bridge_libraries,verify_bridge_libraries,sha


def fixture():
    artifacts={};reports={}
    for role in ('infantry','sentry'):
        rows={str(c):[dict(group='translation',profile=p,motion_type=m,status='complete',actual_shots=20,
            damaging_hits=16,end_reason='target_destroyed') for p in PROFILES for m in MOTIONS
            for _ in range(32)] for c in (7,8)}
        artifact=build_calibration(role,rows,{},cases_per_bucket=32)
        artifacts[role]=artifact
        validation=copy.deepcopy(rows)
        if role=='infantry':
            for row in validation['8']:
                if row['profile']=='wheel_leg' and row['motion_type']=='oblique':row['damaging_hits']=14
        stages={c:assess(int(c),r,artifact) for c,r in validation.items()}
        reports[role]=dict(passed=all(s['passed'] for s in stages.values()),validation=stages)
    original=json.dumps(dict(passed=False,reports=reports))
    approval=dict(approved=True,scope=gate.SCOPE,authorization_text='explicit human approval',
        authorization_source='unit-test fixture',gate_report_sha256=hashlib.sha256(original.encode()).hexdigest(),
        calibration_sha256={r:a['sha256'] for r,a in artifacts.items()})
    approval['sha256']=digest(approval)
    return dict(status='approved_exception',roles=reports,original_gate_json=original,approval=approval),artifacts


class BaselineExceptionTests(unittest.TestCase):
    def setUp(self):
        self.report,self.artifacts=fixture()
        for name,value in [('AUTHORIZED_GATE_SHA256',self.report['approval']['gate_report_sha256']),
                           ('AUTHORIZED_CALIBRATION_SHA256',self.artifacts['infantry']['sha256'])]:
            patcher=patch.object(gate,name,value);patcher.start();self.addCleanup(patcher.stop)

    def test_single_release_exception_keeps_failed_scores_and_model_requirements(self):
        before=copy.deepcopy((self.report,self.artifacts))
        gate.validate_exception_report(self.report,self.artifacts)
        self.assertEqual((self.report,self.artifacts),before)
        value=self.report['roles']['infantry']['validation']['8']['metrics']['wheel_leg/oblique']
        self.assertFalse(value['passed'])
        self.assertEqual(value['required_hit_efficiency'],.9*value['baseline']['hit_efficiency'])
        self.assertEqual(value['required_kill_rate'],.8)

    def test_missing_or_wrong_authorization_and_unrelated_gate_are_rejected(self):
        for field,value in [('approved',False),('authorization_text',''),('gate_report_sha256','other')]:
            report=copy.deepcopy(self.report);report['approval'][field]=value
            report['approval']['sha256']=digest({k:v for k,v in report['approval'].items() if k!='sha256'})
            with self.subTest(field=field),self.assertRaises(ValueError):
                gate.validate_exception_report(report,self.artifacts)

    def test_cannot_conceal_failure_or_change_any_promotion_threshold(self):
        for field,value in [('passed',True),('required_hit_efficiency',.5),('invalid_cases',1),('cases',31)]:
            report=copy.deepcopy(self.report)
            report['roles']['infantry']['validation']['8']['metrics']['wheel_leg/oblique'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                gate.validate_exception_report(report,self.artifacts)

    def test_even_bound_report_cannot_add_a_second_exception(self):
        report=copy.deepcopy(self.report)
        value=report['roles']['sentry']['validation']['8']['metrics']['standard/radial']
        value.update(passed=False,hit_efficiency=.1)
        report['original_gate_json']=json.dumps(dict(passed=False,reports=report['roles']))
        sha=hashlib.sha256(report['original_gate_json'].encode()).hexdigest()
        report['approval']['gate_report_sha256']=sha
        report['approval']['sha256']=digest({k:v for k,v in report['approval'].items() if k!='sha256'})
        with patch.object(gate,'AUTHORIZED_GATE_SHA256',sha),self.assertRaisesRegex(ValueError,'single authorized'):
            gate.validate_exception_report(report,self.artifacts)


class QualificationAuditTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.frozen=dict(validation=[[dict(case=0,seed=10),dict(case=1,seed=11)]])
        for update in (108,109):
            path=self.root/'curriculum'/f'update-{update:04d}'/'C0';path.mkdir(parents=True)
            (path/'summary.json').write_text(json.dumps(dict(course=0,update=update,cases=2,
                passed=False,environment_errors=0,incomplete_cases=0,tracking_initialization_failures=0)))
            (path/'episodes.jsonl').write_text('\n'.join(json.dumps(dict(case=i,seed=10+i,status='complete')) for i in range(2)))

    def test_performance_failure_is_allowed_but_all_cases_must_be_complete(self):
        result=audit_qualification(self.root,107,109,self.frozen)
        self.assertTrue(result['validation_coverage_verified'])
        self.assertEqual(result['validation_cases_checked'],4)

    def test_missing_update_or_duplicate_case_is_rejected(self):
        path=self.root/'curriculum/update-0109/C0/episodes.jsonl'
        path.write_text('\n'.join([json.dumps(dict(case=0,seed=10,status='complete'))]*2))
        with self.assertRaises(ValueError):audit_qualification(self.root,107,109,self.frozen)
        with self.assertRaises(ValueError):audit_qualification(self.root,107,110,self.frozen)

    def test_green_summary_cannot_hide_an_environment_error(self):
        path=self.root/'curriculum/update-0109/C0/episodes.jsonl'
        path.write_text('\n'.join(json.dumps(dict(case=i,seed=10+i,status='complete',error='runtime collision')) for i in range(2)))
        with self.assertRaisesRegex(ValueError,'incomplete or invalid'):audit_qualification(self.root,107,109,self.frozen)


class ReleaseLibrariesTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.old=self.root/'old/libfoxglove.so';self.old.parent.mkdir();self.old.write_bytes(b'sdk')
        self.new=self.root/'frozen/libfoxglove.so';self.new.parent.mkdir();self.new.write_bytes(b'sdk')

    def test_non_system_dependency_is_identified_and_bundled_path_verified(self):
        with patch('subprocess.check_output',return_value=f'libfoxglove.so => {self.new} (0x123)\nlibc.so.6 => /lib/libc.so.6 (0x456)'):
            self.assertEqual(bridge_libraries(['bridge']),{'libfoxglove.so':self.new})
            verify_bridge_libraries(['bridge'],self.new.parent,{'libfoxglove.so':sha(self.new)})

    def test_same_bytes_in_an_old_temporary_path_are_not_sufficient(self):
        with patch('subprocess.check_output',return_value=f'libfoxglove.so => {self.old} (0x123)'):
            with self.assertRaisesRegex(ValueError,'outside'):
                verify_bridge_libraries(['bridge'],self.new.parent,{'libfoxglove.so':sha(self.new)})

    def test_missing_dependency_cannot_be_frozen(self):
        with patch('subprocess.check_output',return_value='libfoxglove.so => not found'):
            with self.assertRaisesRegex(ValueError,'missing'):bridge_libraries(['bridge'])


class CurriculumReportTests(unittest.TestCase):
    def test_role_group_preserves_independent_completion_gates(self):
        from tools.training.curriculum_report import generate_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for role, version in (("infantry", 2), ("sentry", 3)):
                directory = root/role
                directory.mkdir()
                (directory/"run.json").write_text(json.dumps(dict(status="running",
                    requested_timesteps=100, last_completed_timesteps=20,
                    config={"curriculum": {"version": version}},
                    training_budget_complete=False, curriculum_passed=True, thermal_passed=False)))
            records = generate_report(root)
            self.assertEqual(records["sentry"]["curriculum_version"], 3)
            self.assertFalse(records["sentry"]["budget_complete"])
            self.assertTrue(records["sentry"]["combat_passed"])
            self.assertFalse(records["sentry"]["thermal_passed"])
            self.assertTrue((root/"RESULTS.md").is_file())
            self.assertEqual(json.loads((root/"report.json").read_text()), records)

    def test_single_role_and_empty_directory(self):
        from tools.training.curriculum_report import generate_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "no curriculum"):
                generate_report(root)
            (root/"run.json").write_text(json.dumps(dict(status="paused", requested_timesteps=100,
                config={"curriculum": {"version": 3}})))
            records = generate_report(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[root.name]["curriculum_version"], 3)
            self.assertFalse(records[root.name]["combat_passed"])


if __name__ == "__main__":
    unittest.main()
