"""单场故障不能阻止后续固定索引，正常无攻击不能被剔除。"""
import types
from pathlib import Path
from scene_fault_policy import install_continue,is_fault,effect_population


def test_continue_after_failed_scene(tmp_path):
    source='''def main():
    for global_index in range(3):
        if True:
            record={'runtime_failure':global_index==1}
            seen.append(global_index)
            if record.get('runtime_failure') or record.get('planner_service_failure_count'):
                if record.get('planner_service_failure_count') and args.stop_file:
                    dump_json(args.stop_file,{'reason':'planner_service_failure','group':args.group,'scenario_index':global_index})
                raise RuntimeError('scene_runtime_or_planner_service_failure; see persisted attempt')
'''
    path=tmp_path/'fixture.py';path.write_text(source)
    module=types.ModuleType('fixture');module.seen=[]
    exec(compile(source,str(path),'exec'),module.__dict__)
    original=module.main;restore=install_continue(module);module.main()
    assert module.seen==[0,1,2]
    restore();assert module.main is original


def test_faults_separate_from_normal_attack_failure():
    rows=[{'scenario_index':1,'runtime_failure':True},
          {'scenario_index':2,'planner_service_failure_count':1},
          {'scenario_index':3,'prediction_service_failure_count':1},
          {'scenario_index':4,'attack_plan_count':0},
          {'scenario_index':5,'generation_failure':True,'generation_failure_reason':'no_safe_candidate'}]
    assert [r['scenario_index'] for r in effect_population(rows)]==[4,5]
    assert is_fault(rows[0])
