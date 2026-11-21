"""冻结10个初态的两种技能目录和原始状态；不调用API，不混入历史响应。"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from policies.intent_validation import canonical_hash
from model_pool_protocol import inspect_model_source

SCENES=[53,56,74,80,103,140,181,182,207,220]
POOL=['deepseek-v4.1-flash-1017','deepseek-v4.1-flash','glm-5.3','deepseek-v4-pro-0813','qwen3.8-2.4t-a95b','qwen3.8-max','qwen3.7-flash']


def samples():
    run=ROOT/'experiments/skills_temporal_development10_paired_20261029_r1'
    replay=HERE/'development10_initial_contexts_20261030_r1';rows=[]
    summary=json.loads((replay/'results.json').read_text())
    if not summary['source_unchanged'] or len(summary['records'])!=20:raise ValueError('offline_context_replay_incomplete')
    for scene in SCENES:
        trace=run/'legacy_shared_recovery/movies'/f'scenario_{scene:03d}/execution_trace.jsonl'
        data=[json.loads(s) for s in trace.read_text().splitlines()]
        inputs=[r for r in data if r.get('kind')=='llm_input' and r['step']==0]
        if len(inputs)!=1:raise ValueError('expected_exactly_one_initial_planning_input')
        state=inputs[0]['state'];frame=next(r for r in data if r.get('kind')=='state' and r['step']==0)
        modes={m:json.loads((replay/f'scene{scene:03d}_{m}.json').read_text()) for m in ['temporal_original','full_grid']}
        digest=canonical_hash({k:state[k] for k in ('ego_state','agents','route','current_step')})
        if any(r['state_hash']!=digest for r in modes.values()):raise ValueError('frozen_state_mismatch')
        rows.append(dict(scene=scene,scene_id=f'frozen-development-{scene}',source_step=0,state=state,
            frame={k:frame[k] for k in ('ego','agents','active')},source_trace=str(trace),source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            contexts={m:r['context'] for m,r in modes.items()},event_grid_audit=modes['full_grid']['event_grid_audit']))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve_frozen_protocol')
    rows=samples();run=ROOT/'experiments/skills_temporal_development10_paired_20261029_r1'
    files=set(json.loads((run/'source_sha256.json').read_text()))
    files.update(f.relative_to(ROOT).as_posix() for f in HERE.glob('*.py'))
    files.add((HERE/'SKILL.md').relative_to(ROOT).as_posix())
    files.update(['work/skills_modelpool_budget_dev_20261022/launch_pool_pair.py','work/skills_modelpool_budget_dev_20261022/model_pool_protocol.py',
        'work/skills_modelpool_budget_dev_20261022/physical_call_budget.py','work/skills_provider_compat_20261028/provider_control.py'])
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(files)}
    a.output.mkdir();inputs=a.output/'inputs.json';inputs.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    receipt='work/bailian_pool_protection_confirmed_20261023.json'
    if not (ROOT/receipt).exists():raise ValueError('confirmed_free_protection_receipt_missing')
    protocol=dict(version='initial_highlevel_temporal_event_grid_pair_20261030_r1',scenes=SCENES,planned_state_arms=20,
        arms=['temporal_original','full_grid'],scope='same_development_initial_states_only; both_arms_skills_on; candidate_recipe_ablation_not_legacy_off_comparison',
        frozen_model_pool=POOL,model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),free_protection_receipt=receipt,
        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),physical_request_limit_per_state_arm=4,
        maximum_logical_initial_and_repair_requests=2,maximum_total_physical_requests=80,
        alternating_arm_order=True,service_only_fallback=True,no_actual_execution=True,D=None,
        engineering_plan_coverage_target=6,denominator=10,correct_plan_coverage_requires_independent_semantic_review=True,
        new_model_mixing_must_be_disclosed=True,all_failures_and_unattempted_preserved=True,majority_correct_not_automatically_certified=True)
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    shutil.copyfile(run/'skills_full_shared_recovery/resolved_config.yaml',a.output/'resolved_config.yaml')
    for n in hashes:
        dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    print(json.dumps(dict(frozen_sources=len(hashes),scenes=SCENES,planned_state_arms=20,API_calls=0)))


if __name__=='__main__':main()
