"""两个固定闭环场景的状态重放；同CV输入比较计算器，不冒充有效动态D。"""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from policies.risk_metrics import RouteReachability
from policies.commonroad_reachability import CommonRoadReachability


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cases=json.loads((args.source/'results.json').read_text())
    common=CommonRoadReachability();grid=RouteReachability();results=[]
    three_circle=CommonRoadReachability({'commonroad_inflation_mode':3})
    try:
        for index,case in enumerate(cases):
            trace=args.source/'movies'/f'scene_{index}'/'execution_trace.jsonl'
            states={r['step']:r for r in map(json.loads,trace.read_text().splitlines()) if r.get('kind')=='state'}
            scene=pickle.loads(Path(case['scene']).read_bytes())
            result={'scene':case['scene'],'scene_index':case['scene_index'],'measurements':[],
                    'input_basis':'same_recorded_state_and_CV_future_for_both_backends; diagnostic_only',
                    'source_execution_steps':case['actual_steps']}
            for step in [0,10,20]:
                row=states[step]
                env=SimpleNamespace(scenario_dict=scene,ego_state=np.asarray(row['ego']),
                    data_dict={'agent':[np.asarray(row['agents'])]},agent_active=np.asarray(row['active'],bool),
                    current_step=step,dt=.1,pending_joint_trajectory=None)
                observation={'source_step':step,'D':None,'D_status':'CV_calculator_diagnostic_not_valid_joint_prediction'}
                for name,backend in [('route_grid',grid),('commonroad',common),('commonroad_three_circle',three_circle)]:
                    started=time.monotonic()
                    try:
                        observation[name]={**backend.compute(env,False,row['static_obstacles']),
                                           'status':'valid','evaluation_wall_seconds':time.monotonic()-started}
                    except Exception as error:
                        # 每个失败计算都留下null、原因和耗时，不能以零或忽略失败代替。
                        observation[name]={'status':'invalid','area_m2':None,'area_curve':[],
                            'reason':type(error).__name__+': '+str(error),
                            'evaluation_wall_seconds':time.monotonic()-started}
                result['measurements'].append(observation)
                (args.output_dir/'results.partial.json').write_text(json.dumps(results+[result],indent=2))
            results.append(result)
        (args.output_dir/'results.json').write_text(json.dumps(results,indent=2))
        (args.output_dir/'exit_status.json').write_text(json.dumps({'exit_code':0,'scenes':len(results),'snapshots':6}))
    except Exception as error:
        (args.output_dir/'exit_status.json').write_text(json.dumps({'exit_code':1,'reason':type(error).__name__+':'+str(error)}))
        raise
    finally:
        common.close()
        three_circle.close()


if __name__=='__main__':
    main()
