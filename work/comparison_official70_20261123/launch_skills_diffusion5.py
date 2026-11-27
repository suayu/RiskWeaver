
from pathlib import Path
import os,sys,subprocess,json,hashlib,concurrent.futures
ROOT=Path.cwd(); HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/skills_natural_rollback_20261121'))
import campaign as c
OUT=ROOT/'experiments/comparison_skills_diffusion5_20261124_r1'
ARMS={'skills':'3'}
manifest=c.load(HERE/'manifest.json')
previous=c.load(ROOT/'experiments/skills_scene018_identity_paused_20261123_r1/source_sha256.json')
template=c.load(ROOT/'experiments/skills_scene018_identity_paused_20261123_r1/protocol.json')['template']
names=set(previous)
for folder in ('experiments/comparison','work/dual_track_20261105'):
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).glob('*.py'))
names.update(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))
names.update(str(x.relative_to(ROOT)) for x in (ROOT/'work/comparison_scoring_coverage_20261123').glob('*.py'))
hashes={n:c.sha(ROOT/n) for n in sorted(names)}
def run(arm,index,preflight=False):
    c.verify_sources(hashes)
    scenario=manifest['original_scenario_indices'][index]
    case=OUT/('preflight' if preflight else 'runs')/arm/f'scenario_{scenario:03d}'
    case.mkdir(parents=True,exist_ok=False)
    file=case/'manifest.json'; out=case/'run'
    m=dict(manifest,scenario_count=1)
    for key in ('scenario_files','scenario_sha256','original_scenario_indices'):
        m[key]=[manifest[key][index]]
    c.dump(file,m);c.dump(case/'source_sha256.json',hashes)
    worker=str(ROOT/'work/comparison_scoring_coverage_20261123/worker_skills_diffusion.py')
    if arm=='skills':
        cmd=[c.PY,worker,*template[2:]]
        # 显式选择原锚点引导扩散，移除插值专用 CPU 入口。
        cmd[cmd.index('--anchor-backend')+1]='diffusion'
        cmd.remove('--interpolation-cpu')
        cmd[cmd.index('--device')+1]='cuda'
        cmd[cmd.index('--manifest')+1]=str(file)
        cmd[cmd.index('--output-dir')+1]=str(out)
    else:
        group='riskweaver_profile_off' if arm=='no_skills' else arm
        cmd=[c.PY,worker,'--group',group,'--manifest',str(file),'--output-dir',str(out),
             '--policy','idm','--seed','42','--steps','400','--model','deepseek-flash','--models','deepseek-flash']
    env=c.environment()
    env.update(RW_COMPARISON_ARM=arm,CUDA_VISIBLE_DEVICES=ARMS[arm],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    if preflight:env['RW_COMPARISON_PREFLIGHT']='1'
    if arm in ('skills','no_skills'):cmd=['bash','scripts/run_with_llm_provider.sh','deepseek',*cmd]
    with (case/'worker.log').open('x') as log:
        try:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1500 if not preflight else 240)
            code=result.returncode
        except subprocess.TimeoutExpired:code=124
    result_file=out/'movies'/f'scenario_{scenario:03d}'/'attempt_result.json'
    receipt=out/'comparison_entry.json'
    result=c.load(result_file) if result_file.exists() else {}
    row={k:result.get(k) for k in ('executed_steps','scene_challenge_score','outcome_severity_score','scene_challenge','danger_coverage','collision','runtime_failure_reason','generation_failure_reason','reachability_source','excluded_from_effectiveness_statistics')}
    if not result: row['excluded_from_effectiveness_statistics']=True
    row.update(arm=arm,scenario_index=scenario,exit_code=code,entry_verified=receipt.exists(),preflight=preflight)
    c.dump(case/'receipt.json',row)
    return row
def run_arm(arm):
    rows=[]
    for i in range(5):
        try:row=run(arm,i)
        except Exception as e:row=dict(arm=arm,scenario_index=manifest['original_scenario_indices'][i],excluded_from_effectiveness_statistics=True,error=str(e))
        rows.append(row)
        c.dump(OUT/f'{arm}_partial.json',rows)
    c.dump(OUT/f'{arm}_final.json',rows)
    return rows
OUT.mkdir(exist_ok=False)
c.dump(OUT/'source_sha256.json',hashes)
c.dump(OUT/'protocol.json',dict(planned=5,groups=list(ARMS),frame_cap=None,seed=42,
    riskweaver_backend='diffusion',nl=False,rag=False,scenario_dreamer_tilt=-10,
    scoring='common DualObserver with reported method-specific forecast source',
    watchdog_seconds=1500,manifest=manifest))
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    check=list(ex.map(lambda arm:run(arm,0,True),ARMS))
    c.dump(OUT/'preflight.json',check)
    if not all(r['exit_code']==0 and r['entry_verified'] for r in check):
        c.dump(OUT/'launcher_result.json',dict(status='preflight_failed',planned=5,attempted=0))
        raise SystemExit(1)
    c.dump(OUT/'dispatch.json',dict(pid=os.getpid(),planned=5,arms=ARMS))
    rows=[r for group in ex.map(run_arm,ARMS) for r in group]
c.dump(OUT/'launcher_result.json',dict(status='terminal',planned=5,rows=rows))
