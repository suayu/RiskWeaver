"""对历史19个接收输出的原文作主代理窄范围审查，确定性复核具体矛盾。"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import os
from build_review import Inputs


def radial_gap_rate(ego,actor):
    """源帧r·相对速度/|r|；正值是间距增大，不能称为当前接近。"""
    if len(ego)<4 or len(actor)<4 or not all(math.isfinite(float(x)) for x in list(ego[:4])+list(actor[:4])):
        raise ValueError('invalid_motion_state')
    dx,dy=actor[0]-ego[0],actor[1]-ego[1];distance=math.hypot(dx,dy)
    if distance==0:raise ValueError('radial_rate_undefined_at_same_center')
    return (dx*(actor[2]-ego[2])+dy*(actor[3]-ego[3]))/distance


def heading_difference(ego,actor):
    return abs(math.degrees(math.atan2(math.sin(actor[4]-ego[4]),math.cos(actor[4]-ego[4]))))


def field_text(record,field,cid=None):
    if field=='scene_analysis':return record['scene_analysis']
    return record['all_ranked_reasons'][cid]


def add_finding(findings,record,field,quote,kind,evidence,explanation,cid=None):
    if quote not in field_text(record,field,cid):raise ValueError('review_quote_not_in_original_text')
    findings.append(dict(review_id=record['review_id'],scenario_index=record['scenario_index'],source_step=record['source_step'],
        candidate_id=cid,field=field,quote=quote,kind=kind,evidence=evidence,explanation_zh=explanation,
        assessment='clear_specific_factual_contradiction',
        scope='quoted_claim_only; not_a_general_tactical_or_scene_correctness_score'))


def assess(pack,inputs):
    """显式审查条目不泛化成全语言自动判分器，未知语义继续保持未知。"""
    selected=[r for r in pack['records'] if r['arm']=='skills_full_shared_recovery' and r['plan_accepted']]
    if len(selected)!=19:raise ValueError('this_review_is_for_exact_historical19_not_new_trial')
    experiment=Path(pack['experiment'])
    if not (experiment/'launcher_result.json').is_file():raise ValueError('historical_terminal_required')
    if experiment.name!='skills_four_arm_incremental_development10_20261026_r2':raise ValueError('review_source_mismatch')
    by_key={(r['scenario_index'],r['source_step']):r for r in selected}
    states={}
    for path in sorted((experiment/'skills_full_shared_recovery').rglob('attempt_result.json')):
        scene=inputs.json(path)['scenario_index'];rows,_=inputs.jsonl(path.with_name('execution_trace.jsonl'))
        states.update({(scene,r['step']):r for r in rows if r.get('kind')=='state'})
    findings=[]
    r=by_key[56,22];s=states[56,22];angles={i:heading_difference(s['ego'],s['agents'][i]) for i in [4,6]}
    assert all(80<a<100 for a in angles.values())
    add_finding(findings,r,'scene_analysis','oncoming actors 4 and 6','crossing_called_oncoming',
        dict(heading_difference_degrees=angles), '两车相对航向差约90°，属于横向来车；原文称其为对向来车。')
    r=by_key[140,0];fact=next(f for f in r['source_pose_facts'] if f['target_id']==0)
    assert fact['forward_m']<-24 and abs(fact['right_m'])<.1
    add_finding(findings,r,'scene_analysis','ahead-left of ego','behind_called_ahead_left',
        {k:fact[k] for k in ['right_m','forward_m','heading_difference_degrees']},
        '目标位于后方约24.86米、横向接近对齐；不是前左方。')
    for key,quote in [((140,0),'closing slowly'),((140,11),'slight closing velocity')]:
        r=by_key[key];s=states[key];rate=radial_gap_rate(s['ego'],s['agents'][0])
        card=r['all_incremental_cards'][r['validated_plan']['candidate_id']]
        assert rate>1. and math.isclose(rate,card['radial_gap_rate_mps'],abs_tol=1e-9)
        add_finding(findings,r,'scene_analysis',quote,'separating_source_motion_called_closing',
            dict(source_ego=s['ego'],source_actor=s['agents'][0],radial_gap_rate_mps=rate,
                formula='(actor_position-ego_position) dot (actor_velocity-ego_velocity) / center_distance',
                positive_means='current_center_distance_increasing',future_acceleration_not_assumed=True),
            '从当前源帧位置和速度独立计算，间距变化率为正，当前正在拉开间距；未来攻击意图可能接近，不等于当前已在接近。')
    r=by_key[140,44];s=states[140,44];angle=heading_difference(s['ego'],s['agents'][1]);assert 80<angle<100
    add_finding(findings,r,'scene_analysis','near-opposite heading','crossing_called_near_opposite',
        dict(target_id=1,heading_difference_degrees=angle), '目标1与自车相对航向差约90.72°，不是近乎反向。')
    candidates={c['candidate_id']:c for c in r['all_context_candidates']};a=candidates['a1_8']['background_safety_margin_m'];b=candidates['a1_5']['background_safety_margin_m']
    assert b<a
    add_finding(findings,r,'candidate_reasons','lowest supplied margin','incorrect_margin_superlative',
        dict(claimed_candidate='a1_8',claimed_margin_m=a,counterexample_candidate='a1_5',counterexample_margin_m=b,
            same_returned_ranking=True), 'a1_8的供应背景裕度16.164米，高于同一排名中的a1_5的14.851米，因此不能称为最低。',cid='a1_8')
    r=by_key[182,0];cards=r['all_incremental_cards'];value=cards['a0_7']['min_gap_difference_m'];others={cid:cards[cid]['min_gap_difference_m'] for cid in ['x2','x5']}
    assert all(value>v for v in others.values())
    add_finding(findings,r,'candidate_reasons','lower than x2/x5 on incremental proxy','reversed_numeric_comparison',
        dict(candidate='a0_7',increment_m=value,other_increment_m=others,quantity='supplied_complete_active_window_min_gap_difference_m'),
        'a0_7的增量代理1.000米，大于x2的0.766米和x5的0.795米；原文把比较大小写反。',cid='a0_7')
    # 以下涉及含糊措辞或未校准的战术判断，不为增加错误数量而强行归类。
    caveats={
        (53,0):'a2_5裕度小于部分、但并非全部排名候选；other options指代含糊，未计明确矛盾。',
        (56,11):'strongest supported incremental proximity可同时涉及绝对接近程度与CV增量；未强行解释为单一数值最大化。',
        (74,55):'shorter active basis 0.5s是配方构造时刻，不是1.0s有效期；其增加不确定性的因果解释未得到校准证据，标不确定而非自动判错。',
        (140,0):'五候选差值相差不足0.0005米，identical small gap reduction可以是近似描述，不将浮点微差计为错误；x0不是active-window recipe也可能指配方族，未当成预检失败断言。',
        (182,0):'forward-opposite with near-crossing heading含混，已单列精确数值比较错误；不另重复计模糊航向用词。',
        (182,11):'Smaller positive contrast未明确比较对象；未据排名邻项强行认定矛盾，也不认证该排序最优。',
        (207,0):'0.5s recipe basis与1.0s有效期的关系有混淆风险；原句未指定later的精确起点，不强行计明确矛盾。',
    }
    reviews=[]
    for r in selected:
        related=[f for f in findings if f['review_id']==r['review_id']]
        reviews.append(dict(review_id=r['review_id'],scenario_index=r['scenario_index'],source_step=r['source_step'],
            scene_analysis_text_sha256=hashlib.sha256((r['scene_analysis'] or '').encode()).hexdigest(),
            ranked_reason_sha256={cid:hashlib.sha256(r['all_ranked_reasons'][cid].encode()).hexdigest() for cid in r['ranking']},
            ranked_reasons_inspected=len(r['ranking']),specific_findings=related,
            status='specific_contradiction_identified' if related else 'no_additional_clear_contradiction_identified_in_narrow_review',
            caveat=caveats.get((r['scenario_index'],r['source_step'])),
            coordinate_time_numeric_review_performed=True,all_possible_claims_certified=False,
            map_topology_and_road_shape_certified=False,tactical_optimality_certified=False,
            actual_threat_certified=False,overall_correctness=None))
    inputs.verify()
    return dict(version='primary_agent_narrow_reason_review_20261028_v1',experiment=str(experiment),
        arm='skills_full_shared_recovery',accepted_outputs_reviewed=len(selected),
        ranked_reasons_inspected=sum(len(r['ranking']) for r in selected),
        quoted_contradictions=len(findings),outputs_with_specific_contradiction=len({f['review_id'] for f in findings}),
        scenes_with_specific_contradiction=sorted({f['scenario_index'] for f in findings}),
        findings=findings,reviews=reviews,reviewer='primary_agent_with_deterministic_source_geometry_and_numeric_checks; not_human_gold_or_independent_panel',
        interpretation='identified_error_lower_bound; other_outputs_not_certified_correct; no_correct_attack_rate_claim',
        hypotheses_for_next_revision=['make_current_radial_separation_vs_candidate_future_closure_explicit',
            'tie_numeric_comparisons_and_superlatives_to_quantity_and_candidate_set',
            'distinguish_recipe_design_time_from_attack_validity_and_forecast_horizon'],
        original_first_rank_rejects_and_unaccepted_outputs_reviewed=False,
        new_API_requests=0,new_execution_frames=0,changes_to_runtime_or_gates=False,goal_complete=False)


def write(result,out,source_pack):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'accepted_reason_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    evidence={r['review_id']:r for r in source_pack['records']}
    doc='<meta charset="utf-8"><style>body{font:16px/1.8 Microsoft YaHei;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}details{margin:12px 0}</style><h1>历史完整skills组：全部排序理由的窄范围事实核查</h1>'
    doc+=f"<p>审查19个已接收输出的场景分析和{result['ranked_reasons_inspected']}条全部排名理由，定位7处具体矛盾，涉及5个输出、3个场景。包括前次已知的3个方位/航向错误及本次额外核实的运动趋势、裕度极值和比较大小错误。全部是同一历史开发轮次的零API重审，不是当前四组新性能。</p>"
    doc+='<p>审查人是主代理，几何和数值由独立计算复核，不是人工金标或独立多评审共识。未发现额外明确矛盾不等于全部正确；含糊措辞、车道拓扑、战术最优及实际威胁仍不认证，正确攻击率保持未知。未审所有未接收或跳过输出。</p><table><tr><th>场景/源帧</th><th>原文</th><th>矛盾依据</th></tr>'
    for f in result['findings']:
        doc+='<tr><td>'+str(f['scenario_index'])+'/'+str(f['source_step'])+'</td><td>'+html.escape(f['quote'])+'</td><td>'+html.escape(f['explanation_zh'])+'</td></tr>'
    doc+='</table><h2>全部19条审查记录和原始排名理由</h2>'
    for item in result['reviews']:
        r=evidence[item['review_id']]
        doc+='<details><summary>'+html.escape(item['review_id'])+'：'+('有具体矛盾' if item['specific_findings'] else '窄范围所审材料未发现额外明确矛盾；非正确证书')+'</summary><pre>'+html.escape(json.dumps(dict(review=item,scene_analysis=r['scene_analysis'],all_ranked_reasons=r['all_ranked_reasons'],source_pose_facts=r['source_pose_facts'],selected_proxy=r['supplied_proxy_comparison']),ensure_ascii=False,indent=2))+'</pre></details>'
    doc+='<h2>后续改进假设</h2><p>当前运动与攻击意图下的未来运动必须分开；比较大小应明确对应量和候选集合；配方构造时刻、攻击有效期及预测终点必须分开。此处尚未修改任何技能或门控，不能声称提出建议已解决问题。继续保留正在运行的267源码四组对照。</p><p><a href="accepted_reason_review.json">完整证据、哈希和审查边界</a> · <a href="../p1_review_historical_four_arm_20261028_v2/index.html">全部350规划输出待审材料</a></p>'
    videos=Path(result['experiment']).parents[1]/'terminal_report_20261028/index.html'
    if videos.exists():doc+='<p><a href="'+Path(os.path.relpath(videos,out)).as_posix()+'">历史四组全部视频与原始LLM对话</a></p>'
    (out/'index.html').write_text(doc,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pack',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();inputs=Inputs();pack=inputs.json(args.pack);result=assess(pack,inputs)
    result['input_sha256']=inputs.hashes;result['inputs_unchanged']=True
    result['analysis_source_sha256']={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.output.resolve()==Path(pack['experiment']).resolve() or Path(pack['experiment']).resolve() in args.output.resolve().parents:raise ValueError('do_not_write_experiment')
    write(result,args.output,pack)
    print(json.dumps({k:v for k,v in result.items() if k not in ['reviews','findings','input_sha256','analysis_source_sha256']},ensure_ascii=True))


if __name__=='__main__':main()
