"""保存已终态目录重放的源码；知识外提以前的单文件版本须匹配原SHA才归档。"""
from pathlib import Path
import hashlib,json

ROOT=Path('/home2/zhaoyx/scenario-dreamer');OUT=ROOT/'work/skills_event_grid_20261030/development10_initial_contexts_20261030_r1'
OLD_KNOWLEDGE='''KNOWLEDGE='''+"'''"+'''
Use future_event_evidence for candidates whose recipe is full_grid_source_CV_sparse_intercept_v1.
Their event_time_seconds is a hypothesized encounter with the EGO'S FUTURE constant-velocity
position, NOT the ego's current position and NOT a measured future or certified collision.
The coordinates remain four sparse world anchors at 0/1/2/3 seconds; the hypothesized
encounter can occur BETWEEN anchors. It is not a fifth anchor or a denser trajectory.
If event_within_current_lease is false, this 10-frame lease does not reach that event.
Describe it as the first phase of a future interaction; do not claim contact or threat
already happens during this lease. Future replanning and tracking may invalidate it.
Compare all supported candidate evidence; do not prefer this recipe just because it is new.
Keep the original response JSON fields and typed grounding, citing original evidence IDs.
'''+"'''"+'\n'


def main():
    hashes=json.loads((OUT/'source_sha256.json').read_text());verified={};recovered=[]
    for name,digest in hashes.items():
        data=(ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:
            if name!='work/skills_event_grid_20261030/event_planner.py':raise ValueError('unexpected_dependency_change:'+name)
            text=data.decode('utf-8');start=text.index('def knowledge():');stop=text.index('\n\nclass FutureEventPlanner',start)
            text=text[:start]+OLD_KNOWLEDGE+text[stop:]
            text=text.replace("+'\\n'+knowledge()+'\\n'","+'\\n'+KNOWLEDGE+'\\n'")
            text=text.replace("                skill_sha256=hashlib.sha256((HERE/'SKILL.md').read_bytes()).hexdigest(),\n",'')
            data=text.encode('utf-8')
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('recovered_source_does_not_match_frozen_SHA')
            recovered.append(name)
        verified[name]=data
    snapshot=OUT/'source_snapshot'
    if snapshot.exists():raise FileExistsError('preserve_source_snapshot')
    for name,data in verified.items():
        path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (OUT/'source_archive_receipt.json').write_text(json.dumps(dict(verified_sources=len(verified),
        recovered_from_known_edit_then_exact_SHA_verified=recovered,all_match_original_SHA=True,
        scope='terminal_source_archive_only; runtime_sources_and_results_unchanged'),indent=2))
    print(json.dumps(dict(verified_sources=len(verified),recovered=recovered)))


if __name__=='__main__':main()
