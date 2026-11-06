"""新模型池的独立配对入口：原生成器可回退，公共恢复/真实HTTP预算一致。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'work/skills_v2_compact_dev_20261020'))
sys.path.insert(0,str(ROOT/'work/skills_v3_candidate_recovery_fast_20261020'))
from model_pool_protocol import inspect_model_source,verify_pool_transition


def explicit_pool(arguments):
    """命令行与提供方启动脚本必须使用同一冻结池，不继承旧单模型设置。"""
    model=arguments[arguments.index('--model')+1]
    start=arguments.index('--models')+1;pool=[]
    for item in arguments[start:]:
        if item.startswith('--'):break
        pool.append(item)
    if len(pool)==1:pool=pool[0].split(',')
    if not pool or len(pool)!=len(set(pool)) or model not in pool:raise ValueError('invalid explicit frozen pool')
    if os.getenv('LLM_MODEL_NAME') not in (None,'',model):raise ValueError('environment model disagrees with protocol')
    env_pool=os.getenv('LLM_MODEL_NAMES')
    if env_pool and env_pool.split(',')!=pool:raise ValueError('environment pool disagrees with protocol')
    return model,pool


def runner_arguments(arguments,pool):
    """公共E1运行器的--models是单个CSV参数，不能把十个独立词直接透传。"""
    values=list(arguments);start=values.index('--models')+1;end=start
    while end<len(values) and not values[end].startswith('--'):end+=1
    values[start:end]=[','.join(pool)]
    return values


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--skills',choices=('off','on'),required=True)
    parser.add_argument('--compact-view',choices=('off','on'),default='off')
    parser.add_argument('--event-budget',choices=('off','on'),default='off')
    parser.add_argument('--reachability-preflight',choices=('off','on'),default='off')
    parser.add_argument('--joint-candidate-recovery',choices=('off','on'),default='off')
    parser.add_argument('--max-screening-queries',type=int,default=40)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--transport-limit',type=int,default=4)
    parser.add_argument('--free-protection-confirmed',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    options,remaining=parser.parse_known_args()
    if not 1<=options.transport_limit<=30 or not 1<=options.max_screening_queries<=80:raise ValueError('invalid budgets')
    if options.skills=='off' and any(getattr(options,key)=='on' for key in ['compact_view','event_budget','reachability_preflight']):
        raise ValueError('skills-off baseline cannot enable skills-only features')
    if options.reachability_preflight=='on' and options.compact_view!='on':raise ValueError('preflight requires compact development planner')
    model,pool=explicit_pool(remaining)
    remaining=runner_arguments(remaining,pool)
    output=Path(remaining[remaining.index('--output-dir')+1])
    if output.exists():raise FileExistsError(output)
    source=ROOT/'policies/llm_adversarial_planner.py';before=inspect_model_source(source)
    selected=[x for x in pool if x not in before['disabled_models']]
    if not selected:raise ValueError('frozen pool already fully disabled')
    protocol=dict(version='skills-modelpool-physical-budget-20261022',skills=options.skills,
        compact_view=options.compact_view,event_budget=options.event_budget,reachability_preflight=options.reachability_preflight,
        joint_candidate_recovery=options.joint_candidate_recovery,max_screening_queries=options.max_screening_queries,
        transport_limit=options.transport_limit,transport_basis='physical SDK create attempts including failed fallback and schema repair',
        requested_model=model,frozen_model_pool=pool,free_protection_operator_confirmed=options.free_protection_confirmed,
        runner_arguments=remaining,
        model_source_before=before,device=options.device,prepare_only=options.prepare_only,
        scope='full pipeline package comparison; model mixtures and shared continuous recovery require separate attribution')
    if options.prepare_only:
        print(json.dumps(protocol,ensure_ascii=False));return
    if not options.free_protection_confirmed:raise ValueError('free_only_protection_confirmation_required_before_new_inference')
    if options.device=='cpu':
        if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU run requires no visible GPU')
        import torch
        torch.set_num_threads(2)
    from experiments.comparison import run_e1_group as runner
    from experiments.skills.run_group import main as run_group
    from experiments.skills.run_instrumented import install_api_audit
    from run_closed_loop import install_variant
    from joint_candidate_recovery import install_recovery
    from physical_call_budget import install_physical_budget
    restores=[]
    try:
        restores.append(install_variant(options,runner))
        restores.append(install_api_audit(output))
        restores.append(install_physical_budget(output,options.transport_limit))
        if options.joint_candidate_recovery=='on':restores.append(install_recovery(max_candidates=7,max_seconds=30.))
        sys.argv=[sys.argv[0],'--skills',options.skills]+remaining
        run_group()
    finally:
        for restore in reversed(restores):restore()
        if output.exists():
            after=inspect_model_source(source)
            protocol.update(model_source_after=after,model_source_integrity=verify_pool_transition(before,after),
                entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (output/'modelpool_budget_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
