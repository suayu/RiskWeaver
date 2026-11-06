"""冻结两场×两组完整闭环；保留旧准备器，不发API，不自动启动实验。"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
P=ROOT/'work/skills_hybrid_closed_loop2_protocol_20261102_r1'
E=ROOT/'experiments/skills_hybrid_closed_loop2_paired_20261102_r1'


def main():
    old=ROOT/'work/rag_integration_20261030'
    source=(old/'prepare_pair.py').read_text()
    source=source.replace("ROOT/'experiments/skills_temporal_development10_paired_20261029_r1/manifest.json'",
        "ROOT/'work/skills_candidate_local_holdout_protocol_20261030_r1/validation10.json'")
    target=HERE/'prepare_pair.py'
    if target.exists():raise FileExistsError('preserve_preparer')
    target.write_text(source,encoding='utf-8')
    names=set(json.loads((ROOT/'work/skills_hybrid_rag_highlevel_review10_protocol_20261101_r1/source_sha256.json').read_text()))
    public=[p.relative_to(old/'source_after') for p in (old/'source_after').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    public+=[Path(n) for n in ['policies/skills_source_facts.py','policies/hybrid_skills_planner.py',
        'policies/static_attack_catalog.py','skills/riskweaver/rw-source-facts/SKILL.md','skills/riskweaver/rw-static-fallback/SKILL.md']]
    for name in public:
        p=HERE/'source_after'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    command=[sys.executable,str(target),'--output',str(P),'--experiment-root',str(E),
        '--scene-indices','6','45','--profile','off','--models','qwen3.8-2.4t-a95b']
    result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src'),capture_output=True,text=True,timeout=180)
    (HERE/'prepare.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:raise RuntimeError('prepare_failed_see_log')
    protocol=json.loads((P/'protocol.json').read_text());hashes=json.loads((P/'source_sha256.json').read_text())
    names.update(hashes);names.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    names.add('work/rag_integration_20261030/run_pair.py')
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    protocol.update(version='hybrid_closed_loop2_v1_20261102',common_transport=dict(timeout_seconds=180,max_tokens=8192,reasoning_effort='low'),
        static_fallback_enabled=True,source_fact_contract_enabled=True,attack_mode='joint',
        diffusion_execution=True,new_proximal_correction_enabled=False,
        prepare_scope='argument_and_factory_setup_only; actual_runtime_configuration_must_be_audited',
        actual_runtime_configuration_verified=False,same_frozen_model=True,
        cold_KB_per_arm=True,source_highlevel_review='skills_hybrid_rag_highlevel_review10_20261101_r1',goal_complete=False)
    (P/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (P/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    for name in hashes:
        p=P/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    print(json.dumps(dict(status='prepared_not_started',planned_attempts=4,scenes=[6,45],models=protocol['models'],frozen_sources=len(hashes),API_requests=0)))


if __name__=='__main__':main()
