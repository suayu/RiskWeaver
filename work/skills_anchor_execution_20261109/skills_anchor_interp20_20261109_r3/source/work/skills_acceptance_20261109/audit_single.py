"""独立验收单组终态。固定分母，只统计自车碰撞，按实际视频校验来源。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from prepare_single import ROOT, digest

sys.path.insert(0, str(ROOT/"work/skills_danger_acceptance_20261104"))
from audit_campaign import lines, read, video_check
from danger_acceptance import audit_episode


def campaign_gate(planned, rows, issues):
    """缺场不缩分母。碰撞只从真实自车事件取得。"""
    ids = {r["scenario_index"] for r in rows}
    if len(ids) != len(rows) or not ids <= set(planned) or len(set(planned)) != len(planned):
        raise ValueError("duplicate_or_unknown_scene")
    effective = sum(r["effective"] is True for r in rows)
    ego = sum(r["ego_collision"] is True for r in rows)
    official = sum(r["official"] and r["video_verified"] for r in rows)
    videos = sum(r["video_verified"] for r in rows)
    background = sum(r["background_collision"] for r in rows)
    return dict(planned=len(planned), attempted=len(rows), effective=effective,
        ego_collisions=ego, attack_attributed_ego_collisions=sum(r["ego_collision"] and r["attack_contact"] for r in rows),
        background_collision_scenes=background, official_verified_videos=official, verified_videos=videos,
        missing=[i for i in planned if i not in ids], issues=issues,
        acceptance_passed=bool(len(planned)==20 and len(rows)==20 and effective>=18 and ego>=11
                               and official>=16 and videos==20 and not issues and background==0))


def audit(root):
    protocol = read(root/"protocol.json")
    terminal = read(root/"launcher_result.json")
    if protocol["arms"] != ["skills_joint_dual"]:
        raise ValueError("only_single_skills_protocol_supported")
    issues = []
    if terminal["status"] != "completed" or not read(root/"runtime_source_integrity.json")["verified"]:
        issues.append("launcher_or_runtime_integrity_failed")
    manifest_file = root/"manifest.json"
    if digest(manifest_file) != protocol["manifest_sha256"]:
        issues.append("manifest_hash_changed")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    for name, expected in read(root/"source_sha256.json").items():
        if digest(root/"source_snapshot"/name) != expected:
            issues.append(f"source_snapshot_changed:{name}")
    planned = protocol["planned_scenes"]
    provenance = {e["index"]:e for e in manifest["provenance"]}
    if list(provenance) != planned or sum(e["origin"]=="upstream_200m" for e in provenance.values())!=16:
        issues.append("source_mix_invalid")
    for e in provenance.values():
        if digest(Path(e["path"])) != e["sha256"]:
            issues.append(f"input_scene_changed:{e['index']}")

    arm = "skills_joint_dual"
    calls = lines(root/f"{arm}_api_calls.jsonl")
    counts = Counter(c.get("scenario_index") for c in calls)
    if (set(counts)-set(planned) or any(n>4 for n in counts.values())
        or len({c["request_id"] for c in calls}) != len(calls)
        or any(c.get("model_requested")!="deepseek-flash" or c.get("model_returned") not in (None,"deepseek-flash") for c in calls)):
        issues.append("actual_request_model_or_budget_invalid")
    rows = []
    for scene in planned:
        name = f"scenario_{scene:03d}"
        movie = root/arm/"movies"/name
        if not (movie/"attempt_result.json").is_file():
            continue
        attempt = read(movie/"attempt_result.json")
        trace = lines(movie/"execution_trace.jsonl")
        if attempt["scenario_index"] != scene:
            raise ValueError("attempt_scenario_mismatch")
        # 两份视频都要等于初态加实际执行步，400 是仿真上限。
        frames = attempt["executed_steps"]+1
        video_ok = all(video.is_file() and video_check(video,frames)["verified"] for video in
             (movie/f"{name}.mp4",root/f"{arm}_integer_video_exports"/f"{name}.mp4"))
        evidence = audit_episode(attempt,trace,video_ok)
        terminal_collisions = [r for r in trace if r.get("kind")=="state" and (r.get("info") or {}).get("collision") is True]
        ego_collision = attempt.get("collision") is True and bool(terminal_collisions)
        if (attempt.get("collision") is True) != ego_collision:
            issues.append(f"ego_collision_missing_trace:{scene}")
        background = any(r.get("background_pairs") or r.get("background_static_pairs") for r in trace
                         if r.get("kind")=="state" and not r.get("initial"))
        rows.append(dict(scenario_index=scene,effective=evidence["effective_danger_constructed"],
            ego_collision=ego_collision, attack_contact=evidence["attack_contact_observed"],
            official=provenance[scene]["origin"]=="upstream_200m", video_verified=video_ok,
            background_collision=background, executed_steps=attempt["executed_steps"],
            budget=counts.get(scene,0), evidence=evidence))
    return dict(summary=campaign_gate(planned,rows,issues), rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("output_must_be_outside_experiment")
    result=audit(args.root)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
