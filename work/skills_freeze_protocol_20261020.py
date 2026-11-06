"""冻结开发/验证清单与代码证据；按来源分层抽样，不读取任何性能结果。"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
OUT=ROOT/'work/skills_validation_protocol_20261020_v1'


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    OUT.mkdir(exist_ok=False)
    frozen=json.loads((ROOT/'experiments/scenario_dreamer_ldm_expand250_20260928_r1/riskweaver_frozen_250_scenes.json').read_text())
    files=frozen['scenario_files'];hashes=frozen['scenario_sha256']
    assert len(files)==250 and len(set(hashes))==250
    dev_indices=[53,56,74,80,103,140,181,182,207,220]
    dev=dict(frozen,scenario_count=10,comparison_scene_count=10,
        scenario_files=[files[i] for i in dev_indices],scenario_sha256=[hashes[i] for i in dev_indices],
        original_scenario_indices=dev_indices,selection_basis='fixed_existing_development10_no_outcome_filtering')
    dump(OUT/'development10.json',dev)
    pools={}
    for index,path in enumerate(files):pools.setdefault(str(Path(path).parent),[]).append(index)
    assert sorted(map(len,pools.values()))==[55,75,120]
    rng=np.random.default_rng(20261020);selected=[];quotas={75:3,55:2,120:5}
    for name,pool in sorted(pools.items()):
        eligible=[i for i in pool if i not in dev_indices]
        selected.extend(map(int,rng.choice(eligible,quotas[len(pool)],replace=False)))
    selected.sort()
    assert len(selected)==10 and not set(selected)&set(dev_indices)
    validation=dict(frozen,scenario_count=10,comparison_scene_count=10,
        scenario_files=[files[i] for i in selected],scenario_sha256=[hashes[i] for i in selected],
        original_scenario_indices=selected,selection_seed=20261020,
        selection_basis='source_stratified_3_2_5_exclude_skills_development10_outcome_independent',
        development_exposure_note='Not used in this skills development; prior project-wide exposure is not excluded or claimed absent')
    dump(OUT/'validation10.json',validation)
    for manifest in (dev,validation):
        for path,digest in zip(manifest['scenario_files'],manifest['scenario_sha256']):
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    # 保存轻量源码；不复制历史日志、凭据、权重或现有环境。
    paths=[ROOT/name for name in ['scenario_generator.py','simulator.py','run_simulation.py']]
    for directory,extension in [('policies','*.py'),('experiments/comparison','*.py'),
                                 ('experiments/skills','*.py'),('cfgs','*.yaml'),('skills/riskweaver','*.md')]:
        paths.extend((ROOT/directory).rglob(extension))
    paths.append(ROOT/'skills/riskweaver/manifest.json')
    digests={}
    for source in sorted(set(paths)):
        relative=source.relative_to(ROOT);target=OUT/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);digests[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    dump(OUT/'source_sha256.json',digests)
    dump(OUT/'protocol.json',dict(version='skills-v1-dev-20261020',created_at=time.time(),
        provider='dashscope',model='qwen3.8-flash',policy='idm',seed=42,max_frames=400,
        max_api_calls_per_scene=4,skills_default=False,solver=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2),
        attack_duration_frames=10,development_indices=dev_indices,validation_indices=selected,
        source_composition=[3,2,5],coverage_denominator='all planned scenes, including failures',
        minimum_actual_attack_scenes=6,background_collision_limit=0,
        arms=['legacy','catalog_expanded_without_knowledge','skills_no_expansion','skills_expanded'],
        selection_note='Development outcomes already inspected; prospective validation list fixed without its skills outcomes',
        stages=['A replay completed','B frozen state completed','C factorial completed','D smoke running; ten-scene paired next','E validation pending']))
    print(json.dumps(dict(directory=str(OUT),development=dev_indices,validation=selected,source_files=len(digests))))


if __name__=='__main__':main()
