import json, math
from pathlib import Path
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
runs={
 '修改前': ROOT/'experiments/riskweaver_profile_projection_common_smoke_20260922/profile_on',
 '修改后': ROOT/'experiments/riskweaver_anchor_drive_smoke_20260922/anchor_drive_on',
}
out={}
for label,run in runs.items():
    scene=run/'movies/scenario_000'
    result=json.loads((scene/'attempt_result.json').read_text())
    errors=[]
    audits=[]
    for line in (scene/'execution_trace.jsonl').open():
        row=json.loads(line)
        if row.get('kind')!='prediction':
            continue
        intent=row.get('attack_intent')
        joint=row.get('joint')
        if not intent or not joint:
            continue
        target=int(intent['target_id'])
        ids=list(joint['agent_ids'])
        if target not in ids:
            continue
        anchors=np.asarray(intent['anchors'],dtype=float)
        traj=np.asarray(joint['positions'][ids.index(target)],dtype=float)
        elapsed=(int(row['step'])-int(intent['source_step']))*0.1
        times=elapsed+(np.arange(len(traj))+1)*0.1
        active=times <= 3.0+1e-6
        target_xy=np.column_stack([np.interp(times[active],np.arange(4),anchors[:,d]) for d in range(2)])
        errors.extend(np.linalg.norm(traj[active]-target_xy,axis=1).tolist())
        audit=joint.get('metadata',{}).get('anchor_satisfaction')
        if audit:
            audits.append(audit)
    feasibility=result.get('feasibility',{})
    out[label]={
      'anchor_error_mean_m': float(np.mean(errors)) if errors else None,
      'anchor_error_p95_m': float(np.percentile(errors,95)) if errors else None,
      'prediction_count': len(audits) if audits else sum(1 for _ in errors),
      'attack_plan_count': result.get('attack_plan_count'),
      'attack_executed_frames': result.get('attack_executed_frames'),
      'danger_D': result.get('scenario_danger_score'),
      'background_collision_frames': feasibility.get('background_collision_frames'),
      'background_static_collision_frames': feasibility.get('background_static_collision_frames'),
      'generation_failure': result.get('generation_failure'),
      'audited_anchor_error_mean_m': float(np.mean([x['mean_error_m'] for x in audits])) if audits else None,
    }
path=ROOT/'experiments/riskweaver_anchor_drive_smoke_20260922/before_after_summary.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))

