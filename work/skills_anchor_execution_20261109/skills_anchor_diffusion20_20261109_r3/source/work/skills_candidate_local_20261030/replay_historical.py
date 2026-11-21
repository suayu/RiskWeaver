"""只读重审74号原回复：保留全局拒绝，另报告逐候选结果，不改写历史成功。"""
from pathlib import Path
import copy,hashlib,json,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_temporal_candidates_20261029')]
from temporal_planner import TemporalCandidatePlanner
from policies.skills_planner import SkillsAttackPlanner
from policies.candidate_grounding import validate_candidate_local
from omegaconf import OmegaConf


def main():
    run=ROOT/'experiments/skills_temporal_development10_paired_20261029_r1'
    trace=run/'skills_full_shared_recovery/movies/scenario_074/execution_trace.jsonl'
    row=next(r for r in map(json.loads,trace.read_text().splitlines()) if r.get('kind')=='llm_output' and r['step']==11)
    cfg=OmegaConf.load(run/'skills_full_shared_recovery/resolved_config.yaml')
    records=[]
    for request in row['planner_trace']['requests']:
        payload=json.loads(request['prompt'].rsplit('\n',1)[1]);context=payload['context']
        response=request['response'];before=copy.deepcopy(response)
        engine=TemporalCandidatePlanner(cfg.sim,4,extra_recipes=True,rolling_intent=True,explicit_contract=True,
            deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
        engine.priority_evidence=context['priority_evidence'];engine.incremental_cards=context['incremental_cards']
        catalog={c['candidate_id']:c for c in context['candidates']}
        args=[response,payload['request_id'],payload['state_hash'],payload['catalog_hash'],catalog,False]
        try:engine.validate_response(*args);global_reason=None
        except (ValueError,TypeError) as error:global_reason=str(error)
        try:local=validate_candidate_local(*args,engine.validate_response,SkillsAttackPlanner.validate_response)
        except (ValueError,TypeError) as error:local=dict(error=str(error))
        assert response==before
        records.append(dict(repair=request['repair'],original_global_rejection=global_reason,local_validation=local,
            first_model_ranked_id=response['ranking'][0],raw_response_unchanged=True))
    result=dict(source_trace=str(trace),source_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),records=records,
        new_API_calls=0,new_execution_frames=0,D=None,does_not_reclassify_original_failed_experiment=True,
        scope='original_grounding_validation_only; full_compilation_is_separately_tested; no_prose_or_threat_certificate')
    output=ROOT/'work/skills_candidate_local_20261030/historical074_validation_replay.json'
    if output.exists():raise FileExistsError('preserve_previous_replay')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
