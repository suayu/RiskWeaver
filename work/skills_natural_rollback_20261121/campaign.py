# 官方场景固定10+50，固定前10验收后才允许扩展；不修改公共算法。
from pathlib import Path
import concurrent.futures,hashlib,json,os,random,shutil,signal,subprocess,sys,time
ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/skills_natural_10plus50_20261121_r2'
PY='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.new');q.write_text(json.dumps(v,ensure_ascii=False,indent=2));q.replace(p)
def environment():
    e=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',LLM_MODEL_NAME='deepseek-flash',LLM_MODEL_NAMES='deepseek-flash',RW_NL_PREFERENCE_ENABLED='0')
    e.pop('RW_NL_INSTRUCTION',None);return e

def structured_keys(messages):
    # 规则正文可提及RAG字段名称；仅检查实际发送的结构化上下文，避免把说明文字当检索结果。
    keys=set();parsed=0
    def visit(v):
        if isinstance(v,dict):
            keys.update(v)
            for item in v.values():visit(item)
        elif isinstance(v,list):
            for item in v:visit(item)
    for message in messages:
        content=message.get('content','')
        if not isinstance(content,str):continue
        for i,ch in enumerate(content):
            if ch!='{':continue
            try:value=json.loads(content[i:])
            except json.JSONDecodeError:continue
            visit(value);parsed+=1;break
    return keys,parsed

def prepare():
    assert not OUT.exists()
    pool=load(ROOT/'experiments/scenario_dreamer_ldm_expand250_20260928_r1/riskweaver_frozen_250_scenes.json')
    prefix='metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles'
    upstream=subprocess.check_output(['git','rev-parse','upstream/main'],cwd=ROOT,text=True).strip()
    inventory={}
    for line in subprocess.check_output(['git','ls-tree','-r',upstream,prefix],cwd=ROOT,text=True).splitlines():
        meta,name=line.split('\t',1)
        if name.endswith('.pkl'):inventory[str(ROOT/name)]=meta.split()[-1]
    assert len(inventory)==75
    official=[]
    for i,(p,h) in enumerate(zip(pool['scenario_files'],pool['scenario_sha256'])):
        if p not in inventory:continue
        raw=Path(p).read_bytes();assert hashlib.sha256(raw).hexdigest()==h
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==inventory[p]
        official.append(dict(index=i,path=p,sha256=h,upstream_blob=inventory[p]))
    assert len(official)==75
    selected=random.Random(20261121).sample(sorted(official,key=lambda x:x['index']),60)
    assert len({x['sha256']for x in selected})==60
    protocol=HERE/'protocol';protocol.mkdir(exist_ok=False)
    dump(protocol/'selection60.json',dict(selection_seed=20261121,upstream_ref=upstream,inventory_count=75,
        selection='uniform deterministic sample before new outcomes; first10 then remaining50; no outcome filtering',
        prior_exposure='official existing development scenes, not project-unseen holdout',scenes=selected))
    # 以本次生产代码与基线源码清单冻结输入，排除已回退的偏好原型。
    names=set(load(HERE/'source_paths.json'))
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.iterdir() if p.suffix in ('.py','.json'))
    hashes={n:sha(ROOT/n)for n in sorted(names)}
    for n in names:
        assert not n.endswith('.env');target=protocol/'source_snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,target)
    dump(protocol/'source_sha256.json',hashes)
    template=load(HERE/'template.json')
    template[1]=str(HERE/'natural_worker.py')
    dump(protocol/'protocol.json',dict(version='natural_10plus50_v1',planned=60,phase10=10,phase50=50,
        expansion_gate=dict(total_danger_at_least=8,dynamic_danger_at_least=6),
        seed=42,nl_guidance=False,rag=False,profile=False,attack_mode='joint',dynamic_first=True,static_fallback=True,
        model='deepseek-flash',physical_call_limit_per_scene=4,frame_cap=None,scene_wall_watchdog_s=3600,
        watchdog_outcome='runtime resource fault, never natural success',cpu_parallel_workers=2,
        danger='accepted attributed actual execution/exposure >=10 frames or contact, and OBB TTC<=3s/contact',
        background_collisions='record only, no rejection/no ego-collision credit',
        template=template,source_count=len(hashes),all_videos_required=True))
    check=list(template);check[check.index('--manifest')+1]=str(protocol/'smoke_manifest.json')
    check[check.index('--output-dir')+1]=str(HERE/'prepare_only_r2')
    one=selected[0];dump(protocol/'smoke_manifest.json',dict(scenario_count=1,scenario_files=[one['path']],scenario_sha256=[one['sha256']],original_scenario_indices=[one['index']]))
    with (HERE/'prepare_r2.log').open('x')as log:subprocess.run(['bash','scripts/run_with_llm_provider.sh','deepseek',*check,'--prepare-only'],cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    print(json.dumps(dict(prepared=True,first10=[x['index']for x in selected[:10]],source_count=len(hashes))))

def verify_sources(hashes):
    assert all(sha(ROOT/n)==h for n,h in hashes.items()),'frozen source changed'
    assert shutil.disk_usage(ROOT).free>5.5*2**30,'disk free below 5.5GiB'

def execute(entry,phase,protocol,hashes):
    from audit import audit
    verify_sources(hashes)
    case=OUT/phase/'cases'/f"scenario_{entry['index']:03d}";case.mkdir(parents=True,exist_ok=False)
    dump(case/'source_sha256.json',hashes)
    manifest=dict(scenario_count=1,comparison_scene_count=1,seed=42,scenario_files=[entry['path']],scenario_sha256=[entry['sha256']],original_scenario_indices=[entry['index']],provenance=[entry])
    dump(case/'manifest.json',manifest)
    cmd=list(protocol['template']);cmd[cmd.index('--manifest')+1]=str(case/'manifest.json');cmd[cmd.index('--output-dir')+1]=str(case/'skills')
    with (case/'worker.log').open('x')as log:
        p=subprocess.Popen(['bash','scripts/run_with_llm_provider.sh','deepseek',*cmd],cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(case/'pid.json',dict(pid=p.pid,command=cmd,started_at=time.time()))
        timeout=False
        try:code=p.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    if timeout:result=dict(scenario_index=entry['index'],danger=False,dynamic_danger=False,reason='wall_time_resource_fault',effect_statistics_eligible=False)
    else:
        try:result=audit(case)
        except Exception as e:result=dict(scenario_index=entry['index'],danger=False,dynamic_danger=False,reason='audit_error_'+type(e).__name__,effect_statistics_eligible=False)
    control=load(case/'skills/generation_mode.json') if (case/'skills/generation_mode.json').exists()else{}
    journal=case/'skills_api_calls.jsonl';calls=[json.loads(l)for l in journal.read_text().splitlines()]if journal.exists()else[]
    integrity=control.get('extra_language_guidance')is False and len(calls)<=4
    for call in calls:
        integrity &= call.get('model_requested')=='deepseek-flash' and call.get('model_returned') in [None,'deepseek-flash']
        keys,parsed=structured_keys(call.get('messages')or[]);integrity &= parsed>0 and not ({'natural_language_preference','retrieved_attack_evidence'}&keys) and 'rw-nl-preference/SKILL.md'not in str(call.get('messages'))
    if not integrity:result.update(danger=False,dynamic_danger=False,effect_statistics_eligible=False,reason='actual_control_or_budget_integrity_failure')
    result.update(scenario_index=entry['index'],exit_code=code,timed_out=timeout,control_integrity=bool(integrity),physical_calls=len(calls),tokens=sum((x.get('usage')or{}).get('total_tokens',0)or 0 for x in calls))
    dump(case/'independent_audit.json',result);return result

def run():
    prepared=HERE/'protocol';hashes=load(prepared/'source_sha256.json');verify_sources(hashes)
    assert load(HERE/'preflight_verified.json')['verified']
    OUT.mkdir(exist_ok=False)
    for name in ['selection60.json','protocol.json','source_sha256.json']:shutil.copy2(prepared/name,OUT/name)
    shutil.copytree(prepared/'source_snapshot',OUT/'source_snapshot')
    protocol=load(OUT/'protocol.json');selected=load(OUT/'selection60.json')['scenes'];reports=[]
    result=dict(controller_pid=os.getpid(),planned=60,started_at=time.time(),phases={})
    for phase,entries in [('phase10',selected[:10]),('phase50',selected[10:])]:
        if phase=='phase50' and not result.get('gate_passed'):break
        phase_rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
            futures={pool.submit(execute,e,phase,protocol,hashes):e for e in entries}
            for f in concurrent.futures.as_completed(futures):
                try:row=f.result()
                except Exception as e:row=dict(scenario_index=futures[f]['index'],danger=False,dynamic_danger=False,reason='controller_task_'+type(e).__name__,effect_statistics_eligible=False)
                phase_rows.append(row);reports.append(row);dump(OUT/'progress.json',dict(controller_pid=os.getpid(),phase=phase,completed=len(reports),planned=60,rows=reports))
        phase_result=dict(planned=len(entries),attempted=len(phase_rows),danger=sum(x['danger']for x in phase_rows),dynamic_danger=sum(x['dynamic_danger']for x in phase_rows),rows=phase_rows)
        result['phases'][phase]=phase_result
        verify_sources(hashes)
        if phase=='phase10':result['gate_passed']=phase_result['danger']>=8 and phase_result['dynamic_danger']>=6
        dump(OUT/'launcher_result.partial.json',result)
    result.update(status='terminal',finished_at=time.time(),phase50_started='phase50'in result['phases'],rows=reports)
    dump(OUT/'launcher_result.json',result)
if __name__=='__main__':
    if '--prepare'in sys.argv:prepare()
    if '--run'in sys.argv:
        try:run()
        except Exception as e:
            dump(HERE/'controller_failure.json',dict(status='failed',error_type=type(e).__name__,message=str(e),time=time.time()))
            raise
