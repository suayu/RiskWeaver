"""下一轮显式开关入口：默认保留旧策略，只接受新冻结协议，不重启既有实验。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_video_export_fix_20261023'),str(ROOT/'work/skills_v2_compact_dev_20261020')]
import run_with_video_export as base
from route_tail_preflight import RouteTailReadyPlanner
from temporal_joint_recovery import install_temporal_recovery


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--execution-area-policy',choices=('legacy_relative','positive_area'),default='legacy_relative')
    p.add_argument('--temporal-recovery',choices=('off','on'),default='off')
    p.add_argument('--temporal-validation-seconds',type=float,default=30.)
    p.add_argument('--route-tail-preflight',choices=('off','on'),default='off')
    args,remaining=p.parse_known_args()
    if not 1<=args.temporal_validation_seconds<=120:raise ValueError('invalid temporal validation budget')
    output=Path(remaining[remaining.index('--output-dir')+1])
    if output.exists():raise FileExistsError('preserve_existing_experiment')
    def selected(flag):return remaining[remaining.index(flag)+1] if flag in remaining else None
    if args.route_tail_preflight=='on' and any(selected(f)!='on' for f in ['--skills','--compact-view','--reachability-preflight']):
        raise ValueError('route_tail_requires_explicit_skills_compact_and_current_preflight')
    files=[Path(__file__).resolve(),Path(__file__).with_name('route_tail_preflight.py'),Path(__file__).with_name('temporal_joint_recovery.py'),ROOT/'policies/attack_execution_gate.py']
    protocol=dict(execution_area_policy=args.execution_area_policy,temporal_recovery=args.temporal_recovery,
        route_tail_preflight=args.route_tail_preflight,temporal_max_consecutive=1,temporal_acceptance_budget_seconds=args.temporal_validation_seconds,
        original_safety_thresholds_preserved=True,new_LLM_or_same_frame_diffusion_calls_for_recovery=0,
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        scope='prospective common continuous recovery plus skills-only route-tail scheduling; no change to D definition')
    sys.argv=[sys.argv[0]]+remaining
    if '--prepare-only' in remaining:
        text=io.StringIO()
        with redirect_stdout(text):base.main()
        result=json.loads(text.getvalue());result['recovery_v4_protocol']=protocol
        print(json.dumps(result,ensure_ascii=False));return
    manifest=output.parent/'source_sha256.json'
    if not manifest.is_file():raise ValueError('new_v4_source_manifest_required')
    frozen=json.loads(manifest.read_text())
    if any(frozen.get(n)!=d for n,d in protocol['source_sha256'].items()):raise ValueError('v4_sources_not_frozen_or_changed')
    from experiments.comparison import run_e1_group as runner
    import reachability_ready_planner as readiness
    overrides=runner.group_overrides;original_planner=readiness.ReachabilityReadyPlanner
    def configured(*a,**kw):
        values=overrides(*a,**kw)
        return values+[f'++sim.evaluation.reachability.execution_gate_policy={args.execution_area_policy}']
    runner.group_overrides=configured
    if args.route_tail_preflight=='on':readiness.ReachabilityReadyPlanner=RouteTailReadyPlanner
    restore_temporal=lambda:None
    try:
        if args.temporal_recovery=='on':restore_temporal=install_temporal_recovery(output/'temporal_recovery_events.jsonl',max_consecutive=1,max_seconds=args.temporal_validation_seconds)
        base.main()
    finally:
        restore_temporal();readiness.ReachabilityReadyPlanner=original_planner;runner.group_overrides=overrides
        if output.exists():
            with (output/'recovery_v4_protocol.json').open('x') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
