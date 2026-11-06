"""冻结仅skills无RAG/有RAG两组；只做真实入口prepare，不发API或执行场景。"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from model_pool_protocol import inspect_model_source


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--scene-indices',type=int,nargs='+',default=[74,182]);p.add_argument('--profile',choices=('off','on'),default='off')
    p.add_argument('--inherit',action='store_true');p.add_argument('--knowledge-source',type=Path)
    p.add_argument('--models',nargs='+',default=None)
    a=p.parse_args()
    if a.output.exists() or a.experiment_root.exists():raise FileExistsError('preserve_existing_RAG_protocol_and_results')
    if a.inherit!=(a.knowledge_source is not None):raise ValueError('inherit_requires_explicit_source_and_vice_versa')
    if len(set(a.scene_indices))!=len(a.scene_indices):raise ValueError('duplicate_scene_indices')
    a.experiment_root=a.experiment_root.resolve()
    if a.experiment_root.parent!=ROOT/'experiments':raise ValueError('experiment_root_must_be_new_workspace_experiment')
    base=ROOT/'work/skills_candidate_local_actual074_protocol_20261030_r3'
    prior=json.loads((base/'protocol.json').read_text());source_manifest=json.loads((ROOT/'experiments/skills_temporal_development10_paired_20261029_r1/manifest.json').read_text())
    model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    approved=['qwen3.8-max','qwen3.7-flash','glm-5.3','qwen3.8-2.4t-a95b','deepseek-v4.1-flash-1017','deepseek-v4-pro-0813']
    if a.models and (not set(a.models)<=set(approved) or not set(a.models)<=set(model_source['active_models'])):
        raise ValueError('model_not_approved_or_already_disabled_do_not_restore')
    a.models=a.models or [m for m in approved if m in model_source['active_models']]
    if not a.models:raise ValueError('approved_free_model_pool_empty')
    indices=source_manifest['original_scenario_indices'];selection=[indices.index(i) for i in a.scene_indices]
    manifest=dict(source_manifest,scenario_count=len(selection),comparison_scene_count=len(selection),original_scenario_indices=a.scene_indices,
        scenario_files=[source_manifest['scenario_files'][i] for i in selection],scenario_sha256=[source_manifest['scenario_sha256'][i] for i in selection])
    if any(sha(f)!=h for f,h in zip(manifest['scenario_files'],manifest['scenario_sha256'])):raise ValueError('frozen_scene_changed')
    a.output.mkdir(parents=True);(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # 两组从同一只读种子副本各自复制。启动后不再读取可能继续增长的旧库。
    seed=None
    if a.inherit:
        from policies.rag_store import read_inherited
        raw=a.knowledge_source.read_bytes();read_inherited(raw)
        seed=a.output/('inherit_seed'+a.knowledge_source.suffix);seed.write_bytes(raw)
        if sha(a.knowledge_source)!=sha(seed):raise ValueError('inherit_source_changed_during_freeze')
    commands={};prepared={}
    for arm,enabled in [('skills_no_rag','off'),('skills_rag','on')]:
        command=list(prior['command']);command[1]=str(HERE/'run_rag_worker.py')
        for key,value in [('--manifest',a.output/'manifest.json'),('--output-dir',a.experiment_root/arm),('--group','riskweaver_profile_'+a.profile),('--model',a.models[0])]:
            command[command.index(key)+1]=str(value)
        start=command.index('--models')+1;end=start
        while end<len(command) and not command[end].startswith('--'):end+=1
        command[start:end]=a.models;command+=['--rag',enabled]
        if seed:command+=['--rag-inherit','--rag-source',str(seed)]
        result=subprocess.run(command+['--prepare-only'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',
            LLM_MODEL_NAME=a.models[0],LLM_MODEL_NAMES=','.join(a.models)),capture_output=True,text=True,timeout=100)
        if result.returncode:
            (a.output/(arm+'_prepare_failure.log')).write_text(result.stdout+'\n'+result.stderr);raise RuntimeError(arm+'_prepare_failed')
        commands[arm]=command;prepared[arm]=json.loads(result.stdout)
    names=set(json.loads((base/'source_sha256.json').read_text()))
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    names.update(p.relative_to(ROOT).as_posix() for p in (HERE/'source_after').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    # 把公共实现的真实位置（而非仅暂存件）纳入冻结。
    names.update(p.relative_to(HERE/'source_after').as_posix() for p in (HERE/'source_after').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes={name:sha(ROOT/name) for name in sorted(names)}
    protocol=dict(version='skills_rag_pair_v1',arms=list(commands),commands=commands,planned_scenes=a.scene_indices,
        planned_attempts=2*len(selection),profile=a.profile,policy='idm',seed=42,maximum_frames=400,
        maximum_physical_SDK_per_scene=4,models=a.models,model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),
        free_protection_receipt=prior['free_protection_receipt'],experiment_root=str(a.experiment_root),
        inherited_seed=str(seed) if seed else None,inherited_seed_sha256=sha(seed) if seed else None,
        original_inherit_source=str(a.knowledge_source) if a.knowledge_source else None,
        manifest_sha256=sha(a.output/'manifest.json'),two_independent_knowledge_files=True,
        same_fixed_rules_candidates_solver_safety_and_budget=True,run_diffusion_refinement=False,
        minimum_queue_free_gib=18,CUDA_startup_headroom_gib=1,worker_cap_gib=14,worker_reserve_gib=3,
        observed_benefit_not_causal_uplift=True,goal_complete=False)
    for file,value in [('protocol.json',protocol),('source_sha256.json',hashes),('prepared_arms.json',prepared)]:
        (a.output/file).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    for name in hashes:
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print(json.dumps(dict(arms=list(commands),planned_attempts=protocol['planned_attempts'],frozen_sources=len(hashes),prepare_passed=True,API_requests=0,executed_frames=0)))


if __name__=='__main__':main()
