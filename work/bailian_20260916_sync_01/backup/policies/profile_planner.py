"""画像开启时的封闭候选排序协议，不允许退回自由策略生成。"""
import json
import uuid
import time
import numpy as np


def rank_profile_candidates(planner, env_state, instruction, context, scene_image=None):
    """API 返回完整排序；Codex 的既有严格协议返回第一名的离散意图。"""
    planner.last_request_failed = False
    started = time.monotonic()
    planner.last_trace = {'mode':'profile_candidate_ranking'}
    candidates = context['candidates']
    if not candidates:
        planner.last_trace['plan_validation'] = 'no_feasible_candidates'
        planner.last_trace['elapsed_seconds'] = time.monotonic()-started
        return dict(attack=False,attack_target_id=-1,anchors=[],strategy='none',reason='no_feasible_candidates')
    catalog = {c['candidate_id']:c for c in candidates}
    request_id = uuid.uuid4().hex
    transport_complete = False
    try:
        payload = dict(context, instruction=instruction,request_id=request_id,
                       coordinate_frame='world metres; sparse anchors at 0,1,2,3 seconds',
                       hard_rules='Rank only listed candidates. Copy target, strategy and anchors exactly. No continuous controls. Prefer avoidable small response margin; collision is not success. No obstacles.')
        if getattr(planner,'provider',None)=='codex':
            payload['response_instruction'] = 'Return the first ranked candidate using the existing plan schema, or no attack with a reason. Never invent or modify anchors, target or strategy.'
            response = planner.client.request(payload)
            transport_complete = True
            request_id = response['request_id']
            planner.last_trace.update(planner.client.last_trace or {})
            if response['target_id'] is None:
                if response['anchors'] or not response['no_attack_reason']:
                    raise ValueError('invalid_no_attack')
                return dict(attack=False,attack_target_id=-1,strategy='none',anchors=[],reason=response['no_attack_reason'],request_id=request_id)
            matches = [c for c in candidates if c['target_id']==response['target_id'] and c['strategy']==response['strategy'] and np.asarray(response['anchors']).shape==(4,2) and np.allclose(c['anchors'],response['anchors'],rtol=0,atol=1e-6)]
            if len(matches)!=1 or response['duration']!=matches[0]['duration']:
                raise ValueError('candidate_not_in_catalog')
            selected = matches[0]
            ranking = [selected['candidate_id']]
        else:
            prompt = ('You rank a closed catalog of adversarial interaction intents. All safety and avoidability constraints are mandatory. '
                      'Return JSON only with exactly request_id, ranking, no_attack_reason. ranking is a permutation of every candidate_id, best first; '
                      'or [] with a nonempty no_attack_reason if none should be attempted. Repeat request_id exactly. '
                      'Use the scene-conditioned profile, uncertainty, historical failures and measured returns. Do not invent trajectories, controls or candidates.\n'
                      +json.dumps(payload,ensure_ascii=False,allow_nan=False))
            _,response = planner._request_attack_plan(prompt,scene_image=scene_image)
            transport_complete = True
            if not isinstance(response,dict) or set(response)!={'request_id','ranking','no_attack_reason'} or response['request_id']!=request_id:
                raise ValueError('ranking_schema_or_request_id')
            ranking = response['ranking']
            reason = response['no_attack_reason']
            if not isinstance(ranking,list) or any(not isinstance(x,str) for x in ranking) or not isinstance(reason,str):
                raise ValueError('ranking_type')
            if not ranking:
                if not reason.strip():
                    raise ValueError('missing_no_attack_reason')
                planner.last_trace['plan_validation'] = 'no_attack'
                return dict(attack=False,attack_target_id=-1,strategy='none',anchors=[],reason=reason,request_id=request_id)
            if len(ranking)!=len(catalog) or set(ranking)!=set(catalog) or reason:
                raise ValueError('ranking_not_catalog_permutation')
            selected = catalog[ranking[0]]
        plan = dict(attack=True,attack_target_id=selected['target_id'],strategy=selected['strategy'],
                    attack_family=selected['attack_family'],anchors=selected['anchors'],duration=selected['duration'],
                    reason='profile_candidate_ranking',request_id=request_id,candidate_id=selected['candidate_id'])
        # 复用旧语义验证器；其输入坐标为自车局部系，输出仍保留候选的世界坐标。
        state = planner._normalize_env_state(env_state)
        ego = np.asarray(env_state['ego_state'],float)
        angle = np.pi/2-ego[4]
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        local = (np.asarray(plan['anchors'])-ego[:2])@rotation.T
        planner._validate_attack_plan(dict(plan,anchors=local.tolist()),state)
        planner.last_trace.update(plan_validation='accepted',candidate_ranking=ranking,request_id=request_id)
        return plan
    except Exception as error:
        # 传输/服务失败与返回后的语义拒绝区分；均不得进入自由规划回退路径。
        planner.last_request_failed = not transport_complete
        planner.last_trace.update(plan_validation='planner_service_failure' if planner.last_request_failed else 'plan_validation_failed',error_type=type(error).__name__,request_id=request_id)
        return None
    finally:
        planner.last_trace['elapsed_seconds'] = time.monotonic()-started
