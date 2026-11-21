"""只准备配对协议和源码快照；不调用LLM、不创建实验目录、不启动进程。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
from model_pool_protocol import inspect_model_source

OUT=ROOT/'work/skills_modelpool_paired_protocol_20261022_r2'


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    OUT.mkdir(exist_ok=False)
    previous=ROOT/'work/skills_validation_protocol_20261020_v1_waitfix'
    for stage in ['development','validation']:
        manifest=json.loads((previous/f'{stage}10.json').read_text())
        assert manifest['scenario_count']==10
        for name,digest in zip(manifest['scenario_files'],manifest['scenario_sha256']):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
        dump(OUT/f'{stage}10.json',manifest)
    metadata=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    models=metadata['active_models'];assert len(models)==10
    common=['--group','riskweaver_profile_off','--model',models[0],'--models',*models,
            '--policy','idm','--seed','42','--steps','400','--transport-limit','4',
            '--joint-candidate-recovery','on','--device','cuda']
    arms={
      'legacy_shared_recovery':['--skills','off','--compact-view','off','--event-budget','off','--reachability-preflight','off'],
      'skills_full_shared_recovery':['--skills','on','--skill-level','full','--candidate-expansion','on',
        '--compact-view','on','--event-budget','on','--reachability-preflight','on','--max-screening-queries','40'],
    }
    entry=ROOT/'work/skills_modelpool_budget_dev_20261022/run_pool_validation.py'
    prepared={}
    for arm,flags in arms.items():
        command=[sys.executable,str(entry),*flags,*common,'--manifest',str(OUT/'development10.json'),
                 '--output-dir',str(ROOT/'experiments/skills_modelpool_development10_paired_20261022_r2'/arm)]
        result=subprocess.run(command+['--prepare-only'],cwd=ROOT,capture_output=True,text=True,check=True)
        prepared[arm]=dict(launch_command_pending_confirmation=command,
                          prepare_result=json.loads(result.stdout),started=False)
    dump(OUT/'prepared_arms.json',prepared)
    sources=[ROOT/'scenario_generator.py',ROOT/'simulator.py',ROOT/'run_simulation.py']
    for dirname,pattern in [('policies','*.py'),('cfgs','*.yaml'),('experiments/comparison','*.py'),
        ('experiments/skills','*.py'),('skills/riskweaver','*.md'),
        ('work/skills_v2_compact_dev_20261020','*.py'),('work/skills_v3_candidate_recovery_fast_20261020','*.py'),
        ('work/skills_modelpool_budget_dev_20261022','*.py')]:
        sources.extend((ROOT/dirname).rglob(pattern))
    sources.extend([ROOT/'skills/riskweaver/manifest.json',ROOT/'work/skills_v2_compact_dev_20261020/skills/manifest.json'])
    for p in (ROOT/'work/skills_v2_compact_dev_20261020/skills').rglob('*.md'):sources.append(p)
    hashes={}
    for src in sorted(set(sources)):
        if '__pycache__' in src.parts:continue
        relative=src.relative_to(ROOT);target=OUT/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(src.read_bytes());hashes[relative.as_posix()]=hashlib.sha256(src.read_bytes()).hexdigest()
    dump(OUT/'source_sha256.json',hashes)
    dump(OUT/'protocol.json',dict(version='prospective-modelpool-paired-20261022-r2',status='prepared_not_started',
        provider='dashscope',frozen_model_pool=models,model_source=metadata,free_protection_confirmation='pending_user',
        scenario_count_per_arm=10,planned_attempts_per_stage=20,policy='idm',seed=42,max_frames=400,
        physical_requests_per_scene=4,physical_requests_per_stage_max=80,
        counts_failed_fallback_and_schema_repairs=True,
        solver=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2),guidance_weights=[1.5,1,2,4],attack_duration_frames=10,
        common_continuous_recovery=dict(enabled=True,max_extra_candidates=7,acceptance_budget_seconds=30,
            extra_diffusion_samples=0,projection_deviation_m=.25,per_vehicle_projection_seconds=2.),
        stages=['development10 paired','validation10 paired only after development acceptance'],
        denominator='all ten planned scenes including generation/service/runtime failures and unattempted scenes',
        qualified_scene='complete natural/max400 episode, actual guided execution, valid D, no generation/runtime/service failure, no background collision, complete video/trace and <=4 physical requests',
        majority_threshold=6,required_background_collisions=0,required_service_or_runtime_failures=0,
        group_integrity='all planned scenes attempted with immutable input/source verification and complete artifacts',
        generation_failure_policy='each failed scene remains in denominator and cannot count qualified; does not erase valid outcomes from other scenes',
        superiority_scope='package performance; model mixture not a causal skill-text effect; no general non-inferiority guarantee from ten scenes',
        proceed_to_validation='at least6/10 qualified skill scenes, coverage not lower than paired legacy, no integrity/service/runtime/background failure; retain all failures',
        resource_policy='queue for an idle GPU, no shared busy GPU, two-thread CPU support only if separately selected',
        comparison_fields=['logical_queries','physical_requests','model_distribution','latency','tokens','accepted_plans',
                           'executed_plans','qualified_attack_coverage','generation_failure','TTC','EA','D','D_coverage',
                           'anchor_error','trajectory_realism','background_safety','completion','progress','all_videos']))
    print(json.dumps(dict(protocol=str(OUT),source_files=len(hashes),started=False,API_calls=0),ensure_ascii=False))


if __name__=='__main__':main()
