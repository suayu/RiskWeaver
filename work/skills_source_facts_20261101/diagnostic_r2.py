"""保留R1服务失败；以未尝试的获批模型优先，在新目录重做全部四状态。"""
import argparse
import json
import diagnostic as base

base.PROTOCOL = base.ROOT/'work/skills_source_facts_protocol_20261101_r3'
base.OUTPUT = base.ROOT/'experiments/skills_source_facts_frozen4_diagnostic_20261101_r2'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        base.prepare()
        path = base.PROTOCOL/'protocol.json'
        protocol = json.loads(path.read_text())
        order = ['deepseek-v4.1-flash-1017', 'glm-5.3', 'qwen3.8-2.4t-a95b']
        protocol['frozen_model_pool'] = [m for m in order if m in protocol['model_source']['active_models']]
        if not protocol['frozen_model_pool']:
            raise ValueError('remaining_approved_pool_empty')
        protocol['preceding_service_failure'] = str(base.ROOT/'experiments/skills_source_facts_frozen4_diagnostic_20261101_r1')
        protocol['recovery_scope'] = 'new_full_four_state_run; preceding_four_planned_denominator_not_overwritten'
        protocol['model_order_reason'] = 'first_model_not_reached_before_R1_four_physical_call_budget_exhaustion'
        base.base.save(path, protocol)
        print(json.dumps(dict(status='R2_prepared', pool=protocol['frozen_model_pool'], API_requests=0)))
    else:
        base.run()
