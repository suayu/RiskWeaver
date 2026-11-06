"""只重放失败日志中的轨迹投影，比较BLAS线程数；不采样、不执行、不调用LLM。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
OUT = ROOT / 'experiments/skills_projection_thread_replay_20261020_r1'
CASES = {
    'gpu_catalog_053': ROOT / 'experiments/skills_stage_d_development10_bailian_20261020_r2/catalog_expanded',
    'cpu_skills_056': ROOT / 'experiments/skills_v2_frozen_development10_cpu_pair_20261020_r1/skills_v2',
}
SCENES = {'gpu_catalog_053': 53, 'cpu_skills_056': 56}


def worker(case, threads, all_actors):
    # 数值库只在子进程按指定线程环境加载，避免父进程已初始化线程池。
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from omegaconf import OmegaConf
    from threadpoolctl import threadpool_info
    from policies.trajectory_projection import project_positions
    from policies.ego_profile import JointTrajectorySafetyGate
    from policies.joint_safety import NoSafeJointCandidate
    base = CASES[case]
    trace = base / f'movies/scenario_{SCENES[case]:03d}/execution_trace.jsonl'
    rows = [json.loads(line) for line in trace.open()]
    proposal = next(row for row in rows if row['kind'] == 'proposal_prediction')
    state = next(row for row in rows if row['kind'] == 'state' and row['step'] == proposal['step'])
    config_path = base / 'resolved_config.yaml'
    cfg = OmegaConf.load(config_path)
    settings = cfg.sim.traffic_model.dynamics_projection
    limits = JointTrajectorySafetyGate(cfg.sim).dynamic_limits
    source = Path(ROOT / 'policies/trajectory_projection.py')
    records = []
    for index, agent_id in enumerate(proposal['joint']['agent_ids']):
        if index and not all_actors:
            break
        reference = np.asarray(proposal['joint']['positions_global'][index], float)
        initial = np.asarray(state['agents'][int(agent_id)], float)
        started = time.monotonic()
        record = dict(agent_id=int(agent_id), source_step=proposal['step'], requested_threads=threads,
                      input_sha256=hashlib.sha256(reference.tobytes()+initial.tobytes()).hexdigest(),
                      original_max_seconds=float(settings.max_seconds_per_vehicle),
                      original_max_deviation_m=float(settings.max_deviation_m))
        try:
            p, v, y, audit = project_positions(reference, initial, .1, limits,
                float(settings.max_seconds_per_vehicle), float(settings.max_deviation_m))
            record.update(status='passed_original_projection', audit=audit,
                          output_sha256=hashlib.sha256(p.tobytes()).hexdigest())
        except NoSafeJointCandidate as error:
            record.update(status='rejected', reason=str(error))
        record['elapsed_seconds'] = time.monotonic()-started
        # 原门控拒绝照实保留；单线程另算最优位移用于解释，不视为执行许可。
        if threads == 1 and record.get('reason') == 'projection_excessive_deviation':
            try:
                _, _, _, diagnostic = project_positions(reference, initial, .1, limits,
                    float(settings.max_seconds_per_vehicle), 1e6)
                record['diagnostic_only_unbounded_deviation'] = diagnostic
            except NoSafeJointCandidate as error:
                record['diagnostic_only_error'] = str(error)
        records.append(record)
    return dict(case=case, threads=threads, all_actors=all_actors, limits=limits,
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                projection_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                threadpools=threadpool_info(), records=records,
                LLM_calls=0, actual_executed_frames=0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        print(json.dumps(worker(sys.argv[2], int(sys.argv[3]), sys.argv[4] == 'all')))
        return
    OUT.mkdir(exist_ok=False)
    source = Path(__file__)
    (OUT/'probe_projection_threads.py').write_bytes(source.read_bytes())
    # 两次反向顺序，降低机器负载随时间变化造成的偏差；每项固定同一原始输入。
    jobs = [(case, count, False) for case in CASES for count in [1, 64, 2, 2, 64, 1]]
    jobs += [(case, 1, True) for case in CASES]
    launch = dict(started_at=time.time(), jobs=jobs, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope='recorded-proposal projection replay only; not execution feasibility, challenge, generation coverage or LLM comparison',
                  controls='same source, raw proposal, current state, 2-second deadline, 0.25m deviation and physics limits; only BLAS threads change',
                  LLM_calls=0, actual_executed_frames=0)
    (OUT/'launch.json').write_text(json.dumps(launch, indent=2))
    results=[]
    for case, threads, all_actors in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS=str(threads),
                   OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
        try:
            result = subprocess.run([sys.executable, str(source), '--worker', case, str(threads),
                                     'all' if all_actors else 'first'], env=env, capture_output=True, text=True, timeout=90)
            if result.returncode:
                entry = dict(case=case, threads=threads, all_actors=all_actors, worker_failure=True,
                             exit_code=result.returncode, stderr=result.stderr[-6000:])
            else:
                entry = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            entry = dict(case=case, threads=threads, all_actors=all_actors, worker_failure=True,
                         reason='diagnostic_worker_wall_timeout_90s')
        results.append(entry)
        (OUT/'probe_result.partial.json').write_text(json.dumps(dict(results=results), indent=2))
    final=dict(results=results, LLM_calls=0, actual_executed_frames=0, finished_at=time.time())
    (OUT/'probe_result.json').write_text(json.dumps(final, indent=2))
    (OUT/'exit_status.json').write_text(json.dumps(dict(exit_code=0, worker_failures=sum(bool(r.get('worker_failure')) for r in results))))


if __name__ == '__main__':
    main()
