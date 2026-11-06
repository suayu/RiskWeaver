"""使无画像并行仲裁的可行性质疑成为显式、可审计的规则。"""
from pathlib import Path
p=Path('/home2/zhaoyx/scenario-dreamer/policies/multiagent_planner.py')
s=p.read_text(encoding='utf-8')
old='''            valid.sort(key=lambda e: (-scores.get(e['plan'].get('candidate_id'), 0), e['role']))
            selected = valid[0]['plan'] if valid else None
        # 任一服务失败均使本次协作失败，不能用另一提案掩盖免费额度耗尽。
'''
new='''            arbitration_reason = 'profile_candidate_borda'
            if context is None:
                # 无画像模式没有结构化候选分；可行性角色明确不攻击时保守否决。
                veto = next((e for e in valid if e['role']=='proposer_feasibility'
                             and not e['plan'].get('attack')), None)
                if veto is not None:
                    selected = veto['plan']
                    arbitration_reason = 'feasibility_veto'
                else:
                    attacks = [e for e in valid if e['plan'].get('attack')]
                    selected = sorted(attacks or valid, key=lambda e:e['role'])[0]['plan'] if valid else None
                    arbitration_reason = 'validated_attack_preferred' if attacks else 'no_valid_attack'
            else:
                valid.sort(key=lambda e: (-scores.get(e['plan'].get('candidate_id'), 0), e['role']))
                selected = valid[0]['plan'] if valid else None
        # 任一服务失败均使本次协作失败，不能用另一提案掩盖免费额度耗尽。
'''
assert s.count(old)==1
s=s.replace(old,new)
old='''                               critic_triggers=reasons, selected_plan=selected,
                               memory_hash=fingerprint(self.memory) if self.memory else None,
'''
new='''                               critic_triggers=reasons, selected_plan=selected,
                               arbitration_reason=arbitration_reason if self.mode != 'conditional_critic' else None,
                               memory_hash=fingerprint(self.memory) if self.memory else None,
'''
assert s.count(old)==1
p.write_text(s.replace(old,new),encoding='utf-8')
