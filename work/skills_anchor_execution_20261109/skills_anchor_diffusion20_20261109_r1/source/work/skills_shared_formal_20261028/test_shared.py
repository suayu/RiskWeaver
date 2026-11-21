"""真实原派发循环、探测前置与资源收据的回归；不执行CUDA或HTTP。"""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import pytest
import launch_shared as q
import audit_shared as a


def test_busy_shared_gpu_admitted_only_when_no_own_active_arm(monkeypatch):
    monkeypatch.setattr(q.subprocess,'check_output',lambda *args,**kwargs:'0, 32510, 18300, 99\n1, 32510, 30000, 1\n')
    assert q.shared_idle(set())==['0']
    assert q.shared_idle({'0'})==[]
    monkeypatch.setattr(q.subprocess,'check_output',lambda *args,**kwargs:'0, 32510, 17000, 99\n')
    assert q.shared_idle(set())==[]


def setup_probe(tmp_path,passed):
    protocol=tmp_path/'protocol';protocol.mkdir();(protocol/'source').mkdir()
    probe=tmp_path/'probe';probe.mkdir()
    (probe/'launcher_result.json').write_text(json.dumps(dict(status='resource_probe_passed' if passed else 'resource_probe_failed')))
    (probe/'resource_acceptance.json').write_text(json.dumps(dict(passed=passed)))
    (probe/'probe_pid.json').write_text(json.dumps(dict(pid=999999999)))
    arms={n:{} for n in ['legacy_shared_recovery','catalog_no_knowledge','skills_no_expansion','skills_full_shared_recovery']}
    metadata=dict(required_smoke_receipts=[],model_source={},frozen_model_pool=['approved_test_only'],
        shared_resource=dict(probe_experiment='probe',gpu='0',max_active_arms=1))
    (protocol/'protocol.json').write_text(json.dumps(metadata));(protocol/'source_sha256.json').write_text('{}')
    (protocol/'development10.json').write_text(json.dumps(dict(scenario_count=10,original_scenario_indices=list(range(10)))))
    (protocol/'prepared_arms.json').write_text(json.dumps(arms))
    return protocol,arms


def test_actual_dispatch_loop_retains40_attempts_isolates_failed_arm_and_caps_concurrency(tmp_path,monkeypatch):
    (tmp_path/'work').mkdir();protocol,arms=setup_probe(tmp_path,True);out=tmp_path/'out'
    previous=q.previous.previous;launcher=previous.launcher
    monkeypatch.setattr(q,'ROOT',tmp_path);monkeypatch.setattr(launcher,'ROOT',tmp_path)
    monkeypatch.setattr(launcher,'verify_sources',lambda *args:dict(verified=True))
    calls=[]
    def shared(reserved):
        calls.append(set(reserved));return [] if reserved else ['0']
    monkeypatch.setattr(q,'shared_idle',shared)
    monkeypatch.setattr(launcher,'child_command',lambda prepared,manifest,output:['fake',output.name])
    monkeypatch.setattr(q,'audit_shared',lambda root:dict(development_gate=dict(passed=False)))
    seen=[]
    def popen(command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='0';seen.append(command[-1])
        return SimpleNamespace(pid=100+len(seen),wait=lambda:1 if command[-1]=='catalog_no_knowledge' else 0)
    monkeypatch.setattr(launcher.subprocess,'Popen',popen)
    monkeypatch.setattr(sys,'argv',['launch','--protocol',str(protocol),'--stage','development','--output-dir',str(out),'--free-protection-confirmed'])
    oldidle=launcher.idle_gpus;oldsave=launcher.save;oldaudit=previous.audit_four_arm
    q.main()
    assert launcher.idle_gpus is oldidle and launcher.save is oldsave and previous.audit_four_arm is oldaudit
    assert list(arms)==seen and len(seen)==4 and all(len(v)<=1 for v in calls)
    launch=json.loads((out/'launch.json').read_text());final=json.loads((out/'launcher_result.json').read_text())
    assert launch['planned_attempts']==40 and launch['planned_scenes_per_arm']==10 and launch['resource_admission_concurrency']==1
    assert final['status']=='completed_with_group_failures' and len(final['groups'])==4
    assert final['acceptance_passed'] is False
    assert all(json.loads((out/(arm+'_pid.json')).read_text())['gpu']=='0' for arm in arms)


def test_probe_failure_blocks_before_original_loop_or_output(tmp_path,monkeypatch):
    protocol,_=setup_probe(tmp_path,False);monkeypatch.setattr(q,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['launch','--protocol',str(protocol),'--stage','development'])
    monkeypatch.setattr(q.previous,'main',lambda:pytest.fail('must not enter dispatch'))
    with pytest.raises(ValueError,match='shared_resource_probe_not_passed'):q.main()
    assert not (tmp_path/'out').exists()


def receipts():
    common=dict(version='common_provider_control_20261028_v1',semantic_model_policy='service_only',
        thinking_required_exact_models=['glm-5.3','qwen3.8-2.4t-a95b'],true_service_fallback_unchanged=True,
        permanent_disable_only_404_or_explicit_quota=True,transport_limit_unchanged=True,
        skills_and_diffusion_and_safety_unchanged=True,extra_requests=0)
    resource=dict(version='shared_GPU_fence_v1_20261028',cap_gib=14,resource_fixture='off',
        source_solver_and_samples_unchanged=True,sampling_synchronization_for_timing=False,
        only_own_torch_allocator_capped=True,error_type=None)
    command=['python','/work/skills_shared_resource_20261028/run_capped.py','--semantic-model-policy','service_only',
        '--resource-cap-gib','14','--resource-reserve-gib','3','--resource-fixture','off',
        '--resource-observe-sampling','off','--steps','400']
    return command,common,resource


@pytest.mark.parametrize('flag,value',[('--resource-fixture','on'),('--resource-observe-sampling','on'),('--steps','40'),('--semantic-model-policy','legacy'),('--resource-cap-gib','15')])
def test_wrong_actual_command_cannot_claim_formal_shared_protocol(flag,value):
    command,common,resource=receipts();assert a.shared_receipts_valid(command,common,resource,'0')
    command[command.index(flag)+1]=value;assert not a.shared_receipts_valid(command,common,resource,'0')


def test_missing_receipts_wrong_gpu_or_fixture_receipt_rejected():
    command,common,resource=receipts()
    assert not a.shared_receipts_valid(command,common,{},'0')
    assert not a.shared_receipts_valid(command,common,resource,'1')
    assert not a.shared_receipts_valid(command,{},resource,'0')
    resource['resource_fixture']='on';assert not a.shared_receipts_valid(command,common,resource,'0')


def test_extra_resource_check_cannot_turn_original_failed_gate_into_pass(tmp_path,monkeypatch):
    base=dict(groups={'legacy_shared_recovery':dict(group_integrity_pass=True)},
        development_gate=dict(passed=False,all_four_groups_integrity=True))
    monkeypatch.setattr(a,'audit_four_arm',lambda root:copy.deepcopy(base))
    monkeypatch.setattr(a,'shared_receipts_valid',lambda *args:True)
    result=a.audit_shared(tmp_path)
    assert result['development_gate']['passed'] is False

