"""逐候选调用原严格校验器；不改写模型回复、不弱化身份和被选候选的门槛。"""
import copy


class CandidateGroundingFailure(ValueError):
    """所有候选都不合格；保留原首项错误及每项审计，不伪装成无攻击。"""
    def __init__(self, reason, audit):
        super().__init__(reason)
        self.candidate_audit = audit


def validate_candidate_local(response, request_id, state_hash, catalog_hash, catalog,
                             obstacles, strict_validator, base_validator):
    """全局契约仍严格；仅隔离各候选grounding错误，编译器随后跳过不合格项。"""
    if obstacles or not isinstance(response, dict) or 'grounding' not in response:
        # 静态和未采用typed grounding的旧接口不引入本策略，完全沿用原校验。
        strict_validator(response, request_id, state_hash, catalog_hash, catalog, obstacles)
        return dict(policy='original_global_interface', eligible_ids=list(response['ranking']), candidates=[])
    # 与PrioritySkillsPlanner相同的全局字段、身份、目录ID、理由及no_attack校验。
    base = {k: v for k, v in response.items() if k != 'grounding'}
    base_validator(base, request_id, state_hash, catalog_hash, catalog, False)
    ranked = response['ranking']; grounding = response['grounding']
    if not isinstance(grounding, dict) or set(grounding) != set(ranked):
        raise ValueError('priority_grounding_keys')
    if not ranked:
        strict_validator(response, request_id, state_hash, catalog_hash, catalog, False)
        return dict(policy='candidate_local_strict', eligible_ids=[], candidates=[])
    audit = []; eligible = []
    for cid in ranked:
        # 这是校验视图，不是被采纳/回写的LLM响应；原始全文、排名和理由原封保留。
        view = copy.deepcopy(response)
        view['ranking'] = [cid]
        view['candidate_reasons'] = {cid: response['candidate_reasons'][cid]}
        view['grounding'] = {cid: grounding[cid]}
        try:
            strict_validator(view, request_id, state_hash, catalog_hash, catalog, False)
        except (ValueError, TypeError) as error:
            audit.append(dict(candidate_id=cid, accepted=False, reason=str(error)))
        else:
            eligible.append(cid)
            audit.append(dict(candidate_id=cid, accepted=True, reason=None))
    if not eligible:
        raise CandidateGroundingFailure(audit[0]['reason'], audit)
    return dict(policy='candidate_local_strict', eligible_ids=eligible, candidates=audit,
                selected_candidate_still_requires_original_semantic_and_execution_checks=True,
                response_not_rewritten=True,prose_or_threat_not_certified=True)
