"""独立技能知识包：旧v5和既有实验根不动，明确代理增量的适用边界。"""
from pathlib import Path
import hashlib
import json
import shutil


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    target = here/'skills'
    if target.exists():
        raise FileExistsError('preserve_existing_incremental_pack')
    shutil.copytree(root/'work/skills_priority_v5_20261024/skills', target)
    additions = {
        'shared/contract.md': '''
## Incremental evidence prototype (not closed-loop validated)
P1 remains mandatory. Each ranked candidate's grounding additionally includes contrast_direction copied from :contrast.proxy_direction, and cites :contrast.
These facts compare sparse intent vs constant-velocity target with the same constant-velocity ego and timebase. They are neither actual attack effects nor safety certificates.
The difference of two minima can use different times; never describe it as a same-time intervention effect. A separate same-time field is supplied.
Do not discard earlier unfavorable results or relabel prototype recipes as validated. Never replace original model text with a rendered factual summary in the raw record.
''',
        'rw-scene-analysis/SKILL.md': '''
## Distinguish natural motion from attack contribution
An oncoming actor can already approach ego without intervention. A slowing candidate may reduce the active-window threat even though its absolute gap keeps shrinking.
Inspect the target/ego natural CV reference and candidate :contrast facts. Relative heading only describes kinematics; it is not proof of intersection/merge topology.
No active background actors now does not imply no interaction throughout the scene. Keep source-time and whole-scene opportunity statements separate.
''',
        'rw-dynamic-attack/SKILL.md': '''
## Evidence-led choice before risk prose
Prefer a usable candidate that adds supported active-window interaction, subject to unchanged background and physical checks. Explain safety-margin and time-horizon tradeoffs explicitly.
If all current candidate proxies weaken active proximity, do not claim that braking or large acceleration automatically adds danger. Later or uncertain exploratory probes remain possible and must be labeled.
Current recipes may include motion-aligned acceleration and active-window sparse intent alignment. They are generated/screened by code and remain prototypes; LLM only ranks IDs.
Keep numeric facts in referenced cards; use a short qualitative reasoning statement and never call the CV reference a real policy counterfactual.
''',
    }
    for name, text in additions.items():
        with (target/name).open('a', encoding='utf-8') as stream:
            stream.write(text)
    hashes = {p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in target.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    (target/'manifest.json').write_text(json.dumps(dict(version='incremental-prototype-20261025',
        evidence_status='unit_and_frozen_state_audit_only_not_closed_loop_validated',
        parent_version='priority-v5-20261024', continuous_refinement_modified=False, files=hashes),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(documents=len(hashes), previous_pack_changed=False, API_requests=0)))


if __name__ == '__main__':main()
