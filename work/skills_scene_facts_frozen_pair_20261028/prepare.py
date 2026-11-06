"""从已归档源时刻冻结四个诊断输入；不读取之后的执行表现用于生成提示。"""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'work/skills_provider_controlled_protocol_20261028_r1'
OUT=ROOT/'work/skills_scene_facts_frozen4_protocol_20261028_r1'
SOURCE=ROOT/'experiments/skills_four_arm_incremental_development10_20261026_r2/skills_full_shared_recovery'
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from policies.intent_validation import canonical_hash
from launch_pool_pair import verify_sources
from model_pool_protocol import inspect_model_source


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    if OUT.exists():raise FileExistsError('preserve_frozen_diagnostic')
    old=json.loads((OLD/'protocol.json').read_text());hashes=json.loads((OLD/'source_sha256.json').read_text())
    if not verify_sources(old,hashes)['verified']:raise ValueError('queued_runtime_changed')
    selected=[(56,22),(140,0),(140,44),(74,11)]
    samples=[];trace_hashes={}
    for scene,step in selected:
        p=SOURCE/('movies/scenario_%03d/execution_trace.jsonl'%scene)
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        state=next(r['state'] for r in rows if r.get('kind')=='llm_input' and r['step']==step)
        trace=next(r['planner_trace'] for r in rows if r.get('kind')=='llm_output' and r['step']==step)
        prompt=trace['requests'][0]['prompt'];payload=json.loads(prompt.rsplit('\n',1)[1])
        if payload['repair_feedback'] is not None:raise ValueError('old_repair_feedback_would_leak_old_answer')
        if canonical_hash(state)!=payload['state_hash'] or canonical_hash(payload['context'])!=payload['catalog_hash']:
            raise ValueError('historical_snapshot_identity_mismatch')
        # 只保留请求的状态、目录、知识和当时指令，不复制历史回答、执行或危险度结果。
        payload=copy.deepcopy(payload);payload['request_id']='replaced_at_each_new_trial'
        samples.append(dict(case_id=f'scene{scene:03d}_step{step:03d}',scenario_index=scene,source_step=step,
            state=state,payload=payload,historical_trace=str(p),historical_trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        trace_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    metadata=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    pool=[m for m in old['frozen_model_pool'] if m not in metadata['disabled_models']]
    if not pool or pool[0]!='kimi-k3':raise ValueError('primary_control_model_unavailable')
    OUT.mkdir();save(OUT/'inputs.json',samples)
    protocol=dict(version='frozen_four_state_typed_fact_pair_20261028_v1',cases=selected,planned_attempts=8,
        max_repairs=1,max_physical_requests_per_attempt=4,max_physical_requests_total=32,
        frozen_model_pool=pool,model_source=metadata,free_protection_receipt=old['free_protection_receipt'],
        inputs_sha256=hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest(),historical_trace_sha256=trace_hashes,
        selection_basis='three_preidentified_fact_error_states_plus_one_diagnostic_control; not_random_or_unseen_test',
        unchanged_candidates_per_paired_state=True,randomized_requests_not_replayed_outputs=True,
        candidate_source='same_raw_historical_request_catalog_for_both_arms',
        interpretation='typed_contract_and_selected_reason_diagnostic_only; no_executed_attack_or_generalization_claim',
        no_GPU=True,new_execution_frames=0,D=None,goal_complete=False)
    save(OUT/'protocol.json',protocol)
    names=set(hashes)
    for directory in [HERE,ROOT/'work/skills_scene_facts_prototype_20261028']:
        names.update(p.relative_to(ROOT).as_posix() for p in directory.glob('*') if p.is_file() and p.suffix in {'.py','.md','.json'})
    names.update((OUT/n).relative_to(ROOT).as_posix() for n in ['inputs.json','protocol.json'])
    frozen={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    for n in frozen:
        p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    save(OUT/'source_sha256.json',frozen)
    print(json.dumps(dict(protocol=str(OUT),source_files=len(frozen),states=len(samples),planned_attempts=8,new_API_calls=0)))


if __name__=='__main__':main()
