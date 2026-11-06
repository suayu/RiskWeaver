"""以新目录保留下一阶段运行协议，不修改231份原冻结文件及当前74单场。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
HERE=Path(__file__).resolve().parent


def main():
    old=ROOT/'work/skills_next_stage_protocol_20261025_r1'
    out=ROOT/'work/skills_next_stage_protocol_20261026_r2'
    if out.exists():raise FileExistsError('preserve_existing_four_arm_protocol')
    hashes=json.loads((old/'source_sha256.json').read_text())
    if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):
        raise ValueError('previous_frozen_sources_changed')
    out.mkdir()
    for n in ['development10.json','validation10.json','smoke074.json','prepared_arms.json','prepared_smoke.json']:
        shutil.copy2(old/n,out/n)
    protocol=json.loads((old/'protocol.json').read_text())
    protocol.update(version='prospective_four_arm_incremental_20261026_r2',
        launcher_entry=str(HERE/'launch.py'),
        launch_metadata_correction='planned attempts = 4 arms x 10 frozen scenes = 40; old two-arm constant20 is not retained',
        original_protocol=str(old),status='prepared_not_started',automatic_start=False)
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    names=set(hashes)
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    names.update(p.relative_to(ROOT).as_posix() for p in out.glob('*.json'))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    for n in hashes:
        dest=out/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/n).read_bytes())
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(protocol=str(out),source_files=len(hashes),planned_attempts=40,experiments_started=0)))


if __name__=='__main__':main()
