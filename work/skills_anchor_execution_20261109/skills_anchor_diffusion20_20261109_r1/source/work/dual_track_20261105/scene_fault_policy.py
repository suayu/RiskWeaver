"""单场故障继续运行；有效效果统计与完整故障台账分离。"""
import inspect


def is_fault(record):
    # 无攻击、计划被安全门控拒绝不是系统故障，不能为提高覆盖率删除。
    return bool(record.get('runtime_failure') or record.get('planner_service_failure_count')
                or record.get('prediction_service_failure_count'))


def effect_population(records):
    return [r for r in records if not is_fault(r)]


def install_continue(runner):
    original=runner.main;source=inspect.getsource(original)
    old="""            if record.get('runtime_failure') or record.get('planner_service_failure_count'):
                if record.get('planner_service_failure_count') and args.stop_file:
                    dump_json(args.stop_file,{'reason':'planner_service_failure','group':args.group,'scenario_index':global_index})
                raise RuntimeError('scene_runtime_or_planner_service_failure; see persisted attempt')"""
    if source.count(old)!=1:raise ValueError('scene_loop_source_contract_changed')
    source=source.replace(old,"""            # 单场故障原件已落盘，继续下一固定索引，不补样或拼接重试。
            if record.get('runtime_failure') or record.get('planner_service_failure_count'):
                continue""")
    namespace={};exec(compile(source,inspect.getsourcefile(original),'exec'),runner.__dict__,namespace)
    runner.main=namespace['main']
    return lambda:setattr(runner,'main',original)
