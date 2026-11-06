"""冻结原40输入上的数值/契约回归；只执行生产纯函数，禁止网络或仿真调用。"""
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np

ROOT=Path('D:/RiskWeaver/outputs/skills_integration_20261020')
EXPERIMENT=ROOT/'experiments/skills_modelpool_development10_paired_20261023_r1'
SOURCE=EXPERIMENT/'source_snapshot'
INPUTS=Path(__file__).with_name('original40_inputs.json')
OLD=ROOT/'experiments/skills_stage_a_replay_20261020_r1/records.json'
OUT=ROOT/'stage_a_verification_20261023'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pure_function(path,name):
    # 仅提取已冻结、哈希已验证的生产纯函数，不初始化任何LLM客户端或读取.env。
    tree=ast.parse(path.read_text(encoding='utf8'))
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1:raise ValueError('pure_function_not_unique_'+name)
    node=copy.deepcopy(nodes[0]);node.decorator_list=[];scope=dict(np=np,json=json)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope[name]


def transform_state(state,angle,shift):
    moved=copy.deepcopy(state);c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s],[s,c]])
    def move(values):
        v=list(values);v[:2]=(np.asarray(v[:2])@rot.T+shift).tolist();v[2:4]=(np.asarray(v[2:4])@rot.T).tolist();v[4]+=angle;return v
    moved['ego_state']=move(moved['ego_state'])
    moved['route']=(np.asarray(moved['route'])@rot.T+shift).tolist()
    for actor in moved['agents']:
        actor['state']=move(actor['state'])
        for h in actor.get('history',[]):
            if len(h.get('state',[]))==5:h['state']=move(h['state'])
    for obstacle in moved.get('static_obstacles',[]):
        obstacle['center']=(np.asarray(obstacle['center'])@rot.T+shift).tolist();obstacle['yaw']+=angle
    return moved,rot


def main():
    if OUT.exists():raise FileExistsError(OUT)
    names=['policies/llm_adversarial_planner.py','policies/intent_validation.py','policies/skills_planner.py']
    snapshot_hashes=read(EXPERIMENT/'source_sha256.json')
    v4=read(ROOT/'work/skills_recovery_paired_protocol_20261023_v4/source_sha256.json')
    for n in names:
        if sha(SOURCE/n)!=snapshot_hashes[n] or sha(SOURCE/n)!=v4[n]:raise ValueError('source_hash_not_current_frozen_'+n)
    if sha(OLD)!=read(ROOT/'remote_sha256.json')[OLD.relative_to(ROOT).as_posix()]:raise ValueError('historical_replay_hash_changed')
    normalize=pure_function(SOURCE/names[0],'_normalize_env_state');validate=pure_function(SOURCE/names[0],'_validate_attack_plan')
    response=pure_function(SOURCE/names[2],'validate_response')
    spec=importlib.util.spec_from_file_location('_stage_a_intent_validation',SOURCE/names[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    planner=NS(_validate_attack_plan=validate)
    planner._normalize_env_state=lambda state:normalize(planner,state)
    planner._request_attack_plan=lambda *a,**k:(_ for _ in ()).throw(AssertionError('network_prohibited'))
    payload=read(INPUTS);inputs=payload['records'];old={(r['scene'],r['step']):r for r in read(OLD)}
    assert len(inputs)==40 and len(old)==40 and payload['source_unchanged']
    watched=[INPUTS,OLD,*[SOURCE/n for n in names]];before={str(p):sha(p) for p in watched}
    records=[];counts=Counter();failures=[]
    transforms=[('translate',0.,np.array([17.,-31.])),('rotate',.8,np.zeros(2)),
                ('quarter_turn',np.pi/2,np.array([-40.,80.])),('opposite_turn',-np.pi+.13,np.array([1000.,-2000.]))]
    def test(label,passed,detail=None):
        counts[label]+=1
        if not passed:failures.append(dict(check=label,detail=detail))
    for item in inputs:
        key=(item['scene'],item['step']);historical=old[key];state=item['state'];original=item['original_context']['candidates']
        assert not item['frame']['static_obstacles']
        row=dict(scene=key[0],step=key[1],legacy_supplied_count=len(original),modes={},legacy_rejections=[])
        for candidate in original:
            decision=module.validate_world_intent(planner,state,candidate)
            if not decision['accepted']:row['legacy_rejections'].append(dict(candidate_id=candidate['candidate_id'],reason=decision['reason'],metrics=decision['metrics']))
        for mode in ['False','True']:
            candidates=historical['modes'][mode]['candidates'];published=0;checked=0;max_drift=0.
            for candidate in candidates:
                baseline=module.validate_world_intent(planner,state,candidate);published+=baseline['accepted']
                test('published_candidate_final_acceptance',baseline['accepted'],dict(key=key,mode=mode,candidate=candidate['candidate_id'],reason=baseline['reason']))
                for label,angle,shift in transforms:
                    transformed,rot=transform_state(state,angle,shift)
                    changed=dict(candidate,anchors=(np.asarray(candidate['anchors'])@rot.T+shift).tolist())
                    current=module.validate_world_intent(planner,transformed,changed)
                    test('rigid_transform_decision',current['accepted']==baseline['accepted'],dict(key=key,mode=mode,candidate=candidate['candidate_id'],transform=label,reason=current['reason']))
                    for metric,value in baseline['metrics'].items():
                        difference=float(np.max(np.abs(np.asarray(current['metrics'][metric])-np.asarray(value))))
                        max_drift=max(max_drift,difference)
                        test('rigid_transform_metric',difference<=1e-8,dict(key=key,metric=metric,drift=difference,transform=label))
                    checked+=1
            row['modes'][mode]=dict(published=len(candidates),accepted_on_recheck=published,rigid_transforms=checked,max_metric_drift=max_drift)
        catalog={c['candidate_id']:c for c in historical['modes']['True']['candidates']};ids=list(catalog)[:6]
        rid=f'synthetic-contract-{len(records)}';state_hash=module.canonical_hash(state);catalog_hash=module.canonical_hash(catalog)
        reply=dict(request_id=rid,state_hash=state_hash,catalog_hash=catalog_hash,ranking=ids,
            scene_analysis='Synthetic contract fixture, not an LLM response',candidate_reasons={i:'fixture reason' for i in ids},no_attack_reason='' if ids else 'fixed empty catalog')
        response(reply,rid,state_hash,catalog_hash,catalog,False);test('synthetic_contract_positive',True)
        corruptions=[('request_id',dict(reply,request_id='other')),('state_hash',dict(reply,state_hash='other')),('catalog_hash',dict(reply,catalog_hash='other')),
            ('invented_candidate',dict(reply,ranking=['invented'])),('dense_controls',dict(reply,controls=[])),('solver_settings',dict(reply,inner_lr=.1)),
            ('missing_reason_field',{k:v for k,v in reply.items() if k!='candidate_reasons'}),('missing_scene_analysis',dict(reply,scene_analysis=''))]
        if ids:corruptions.append(('duplicate_ranked_ID',dict(reply,ranking=[ids[0],ids[0]])))
        for name,bad in corruptions:
            try:response(bad,rid,state_hash,catalog_hash,catalog,False);rejected=False
            except (ValueError,TypeError):rejected=True
            test('synthetic_contract_negative_'+name,rejected,dict(key=key,corruption=name))
        advanced=copy.deepcopy(state);advanced['current_step']=int(state['current_step'])+1
        try:response(reply,rid,module.canonical_hash(advanced),catalog_hash,catalog,False);rejected=False
        except ValueError:rejected=True
        test('old_state_identity_after_step_change',rejected,dict(key=key))
        if catalog:
            c=next(iter(catalog.values()));bad=copy.deepcopy(c);target=next(a['state'] for a in state['agents'] if a['id']==c['target_id'])
            bad['anchors'][0]=(np.asarray(target[:2])+[4.001,0]).tolist()
            test('t0_anchor_position_limit',not module.validate_world_intent(planner,state,bad)['accepted'],dict(key=key))
            bad=copy.deepcopy(c);bad['anchors'][2][0]=float('nan')
            test('nonfinite_anchor_rejection',not module.validate_world_intent(planner,state,bad)['accepted'],dict(key=key))
            for duration in [0,401,True]:
                bad=dict(c,duration=duration)
                test('invalid_duration_rejection',not module.validate_world_intent(planner,state,bad)['accepted'],dict(key=key,duration=duration))
            ego=np.asarray(state['ego_state']);angle=np.pi/2-ego[4];rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            local=(np.asarray(c['anchors'])-ego[:2])@rot.T
            base_plan=dict(attack=True,attack_target_id=c['target_id'],strategy=c['strategy'],anchors=local.tolist())
            for field in ['trajectory','trajectories','controls','inner_lr','inner_beta','n_guide_steps']:
                try:validate(dict(base_plan,**{field:[]}),normalize(planner,state));rejected=False
                except ValueError:rejected=True
                test('LLM_continuous_boundary_rejection',rejected,dict(key=key,field=field))
        records.append(row)
    assert all(sha(p)==digest for p,digest in before.items())
    summary=dict(planning_inputs=40,scenes=len(payload['source_traces']),legacy_nonempty=sum(r['legacy_supplied_count']>0 for r in records),
        unified_nonempty=sum(r['modes']['False']['published']>0 for r in records),expanded_nonempty=sum(r['modes']['True']['published']>0 for r in records),
        published_candidates=sum(r['modes'][m]['published'] for r in records for m in ['False','True']),
        total_checks=sum(counts.values()),checks_by_kind=dict(counts),failures=len(failures),passed=not failures,
        maximum_rigid_transform_metric_drift=max(r['modes'][m]['max_metric_drift'] for r in records for m in ['False','True']),
        API_calls=0,new_diffusion_samples=0,new_executed_frames=0,source_and_inputs_unchanged=True,
        original_source_traces=payload['source_traces'],input_hashes=before,
        semantics='world-coordinate candidates; four anchors at0/1/2/3s; first point checked against current position, no pure-future reinterpretation',
        scope='40 original exposed development inputs, existing deterministic catalogs and production numeric/schema functions; not new LLM or closed-loop coverage',
        remaining_call_boundary_check='duplicate response replay and stale state after actual mocked transport are separate full planner tests')
    OUT.mkdir(exist_ok=False)
    with (OUT/'stage_a_verification.json').open('x',encoding='utf8') as f:json.dump(dict(summary=summary,records=records,failures=failures),f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['original_source_traces','input_hashes','checks_by_kind']},ensure_ascii=False))
    if failures:sys.exit(2)


if __name__=='__main__':main()
