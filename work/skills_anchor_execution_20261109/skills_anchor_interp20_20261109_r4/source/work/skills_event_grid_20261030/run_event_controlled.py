"""未来事件技能的真实运行入口；默认off保持原时序版，不修改运行中的实验。"""
from pathlib import Path
from functools import partial
import argparse,hashlib,json,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--future-event-grid',choices=['off','on'],default='off')
    option,remaining=p.parse_known_args();enabled=option.future_event_grid=='on'
    if enabled:
        for flag in ['--skills','--rolling-intent','--fractional-interaction','--active-window-recipes']:
            if flag not in remaining or remaining[remaining.index(flag)+1]!='on':raise ValueError('event_grid_entry_requires_temporal_skills_path')
    sys.path.insert(0,str(ROOT/'work/skills_temporal_candidates_20261029'))
    import temporal_planner,run_temporal_controlled
    original=temporal_planner.TemporalCandidatePlanner;argv=sys.argv
    output=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if output.exists():raise FileExistsError('preserve_previous_output')
    if enabled:
        if not prepare:
            frozen=json.loads((output.parent/'source_sha256.json').read_text())
            for path in [*HERE.glob('*.py'),HERE/'SKILL.md']:
                if frozen.get(path.relative_to(ROOT).as_posix())!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('event_grid_runtime_source_not_frozen')
        from event_planner import FutureEventPlanner
        temporal_planner.TemporalCandidatePlanner=partial(FutureEventPlanner,event_grid=True)
    try:
        sys.argv=[sys.argv[0]]+remaining;return run_temporal_controlled.main()
    finally:
        temporal_planner.TemporalCandidatePlanner=original;sys.argv=argv
        if enabled and not prepare and output.exists():
            with (output/'future_event_entry_protocol.json').open('x',encoding='utf-8') as file:
                json.dump(dict(version='full_grid_source_CV_sparse_intercept_v1',default_off=True,grid_seconds=[i/10 for i in range(1,31)],
                    source_future_only=True,original_response_and_budget_and_safety_preserved=True,continuous_refinement_changed=False),file,indent=2)


if __name__=='__main__':main()
