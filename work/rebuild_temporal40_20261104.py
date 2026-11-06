"""真实时序工厂重建40份目录与最终提示，传输边界前中止，零API。"""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import socket
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
OUT=ROOT/'work/skills_temporal_factory40_20261104_r1'
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_catalog250_20261102'),
    str(ROOT/'work/skills_v2_compact_dev_20261020'),str(ROOT/'work/skills_candidate_local_runtime_20261030')]
from audit import initial_environment
from assembly import install
import temporal_planner
import numpy as np
from omegaconf import OmegaConf
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.intent_validation import validate_world_intent


class CapturedBeforeTransport(BaseException):
    """明确的非API边界中止，不伪造模型回复或已接收计划。"""


def forbidden(*args,**kwargs):
    raise AssertionError('real_network_or_simulation_prohibited')


def main():
    OUT.mkdir(exist_ok=False)
    socket.socket.connect=forbidden;socket.create_connection=forbidden
    protocol=ROOT/'work/skills_realtime_dynamic_smoke074_protocol_20261104_r1'
    hashes=json.loads((protocol/'source_sha256.json').read_text())
    def unchanged():
        return all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    assert unchanged()
    inputs=ROOT/'work/skills_stage_a_verification_20261023/original40_inputs.json'
    original_digest=hashlib.sha256(inputs.read_bytes()).hexdigest()
    records=json.loads(inputs.read_text())['records'];assert len(records)==40
    manifest=json.loads((ROOT/'work/skills_omni_dynamic_pair_protocol_20261104_r1/manifest.json').read_text())
    lookup=dict(zip(manifest['original_scenario_indices'],zip(manifest['scenario_files'],manifest['scenario_sha256'])))
    cfg=OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    cfg.sim.llm.skills.grounding_validation_mode='candidate_local'
    cfg.sim.llm.skills.candidate_expansion_enabled=True
    cfg.sim.ego_profile.candidate_expansion_enabled=True
    cfg.sim.llm.skills.source_fact_contract_enabled=False
    (OUT/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg),encoding='utf-8')
    rows=[]
    with install('candidate_local',True):
        for item in records:
            scene=int(item['scene'].rsplit('_',1)[1]);name,digest=lookup[scene]
            raw=Path(name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
            geometry=pickle.loads(raw);geometry['_audit_filename']=Path(name).name
            env=initial_environment(geometry,cfg);frame=item['frame'];state=item['state']
            env.ego_state=np.asarray(frame['ego'],float);env.data_dict['agent']=[np.asarray(frame['agents'],float)]
            env.agent_active=np.asarray(frame['active'],bool);env.t=item['step'];env.step=forbidden
            env.get_state_for_planning=lambda state=state:state
            planner=LLMAdversarialPlanner(provider='dashscope',model_name='offline-audit',model_names=['offline-audit'],client=object())
            planner.use_obstacles=False
            engine=temporal_planner.TemporalCandidatePlanner(cfg.sim,4,extra_recipes=True,rolling_intent=True,
                explicit_contract=True,deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
            row=dict(scene=scene,step=item['step'],transport_boundary_reached=False,context=None)
            build=engine.build_context
            def captured_build(*args,**kwargs):
                context=build(*args,**kwargs)
                checks=[validate_world_intent(planner,state,c) for c in context['candidates']]
                assert all(c['accepted'] for c in checks)
                row['context']=copy.deepcopy(context)
                return context
            engine.build_context=captured_build
            def capture(prompt,scene_image=None):
                row['transport_boundary_reached']=True
                payload=json.loads(prompt.rsplit('\n',1)[1])
                assert payload['required_identity_copy_exactly']=={k:payload[k] for k in ['request_id','state_hash','catalog_hash']}
                for key in ['rolling_intent','temporal_candidate_evidence','comparison_facts']:
                    assert key in payload['context'],key
                row['final_prompt']=prompt
                raise CapturedBeforeTransport()
            planner._request_attack_plan=capture
            try:
                result=engine.plan(planner,env,state,['Only dynamic attacks; no executed outcome is known.'])
                row['returned_without_transport']=result
            except CapturedBeforeTransport:
                pass
            assert row['context'] is not None
            if row['context']['candidates']:
                assert row['transport_boundary_reached'],planner.last_trace
            row['factory_receipt']=engine.highlevel_factory_assembly
            rows.append(row)
            with (OUT/'records.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    assert unchanged() and hashlib.sha256(inputs.read_bytes()).hexdigest()==original_digest
    summary=dict(inputs=len(rows),nonempty=sum(bool(r['context']['candidates']) for r in rows),
        nonempty_scenes=len({r['scene'] for r in rows if r['context']['candidates']}),
        intercepted_prompt_count=sum(r['transport_boundary_reached'] for r in rows),
        checked_current_runtime_source_files=len(hashes),input_sha256=original_digest,
        source_and_input_unchanged=True,physical_API_calls=0,execution_frames=0,D=None,
        scope='actual configured temporal factory with final request rendering; intercepted before transport; no response acceptance or correctness certified',
        previous_memory='empty per historical input; no prior accepted intent invented; continuation lifecycle requires real closed loop')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
