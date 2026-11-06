"""创建独立技能版本，旧知识包与旧实验源码保持原样。"""
from pathlib import Path
import hashlib
import json
import shutil


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    previous = root/'work/skills_v2_compact_dev_20261020/skills'
    dest = here/'skills'
    if dest.exists():
        raise FileExistsError('preserve_existing_priority_pack')
    shutil.copytree(previous, dest)
    contract = '''
## Priority contract v5
The ordered objectives are: correct scene understanding and feasible discrete attacks; actual ego threat; enabled natural-language control; enabled difficulty targeting.
The FIRST objective is mandatory. Later objectives never justify fabricated facts, imaginary opportunities or bypassing background safety/dynamics.
Use priority_evidence as the authority for target identity, active duration and time-aligned proxy distances. Background target0 is NOT the separately represented ego.
Response grounding is mandatory per ranked ID: target_id, target_role=background_vehicle, active_horizon_seconds, evidence_ids, interaction_basis.
Cite :actor and :validity plus :active or :after as required by the response schema. A correct reference is evidence hygiene, not a guarantee that the tactic will threaten ego.
Prefer active-window interaction evidence. A3s anchor with1s validity is a later forecast, not an executed3s attack. Keep exploratory probes explicit and uncertain.
Do not invent profile history, claim positive reachable area from background clearance, or report estimated risk/TTC as measured D.
'''
    with (dest/'shared/contract.md').open('a', encoding='utf-8') as f:
        f.write(contract)
    with (dest/'rw-scene-analysis/SKILL.md').open('a', encoding='utf-8') as f:
        f.write('\n## Mandatory identity and time checks\nStart with ego_reference=separate_ego, then identify each selected background target. Compare source-time roles, current relative motion and the active-window proxy before discussing later forecasts. If no published catalog exists, preserve the exact no-opportunity/preflight reason; do not call it a service failure.\n')
    with (dest/'rw-dynamic-attack/SKILL.md').open('a', encoding='utf-8') as f:
        f.write('\n## Opportunity-first ranking\nChoose a supported candidate when a usable catalog is supplied; never refuse merely because the profile is disabled or a proxy risk is null. Prefer relevant active-window closeness/closing evidence, then compatible later opportunities. A high acceleration, small terminal lateral offset or family label alone is insufficient. Cite exact facts and label later_forecast_only/exploratory_probe honestly. Never promise collision or low reaction margin without execution evidence.\n')
    with (dest/'rw-plan-repair/SKILL.md').open('a', encoding='utf-8') as f:
        f.write('\nPriority grounding repair must correct roles, durations and existing fact references. Do not invent evidence or turn missing measurements into0. Use only the existing repair and physical-request budget.\n')
    files = {p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in dest.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    manifest = dict(version='priority-v5-20261024', evidence_status='development_candidate_not_closed_loop_validated',
        parent_version='skills-v2-compact-dev-20261020', continuous_refinement_modified=False, files=files)
    (dest/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(pack=str(dest), documents=len(files), old_pack_unchanged=True, API_calls=0)))


if __name__ == '__main__':
    main()
