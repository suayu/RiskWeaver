"""导出七个静态兜底真实冻结状态的候选几何；不调用API或修改实验。"""
from pathlib import Path
import hashlib,itertools,json,pickle,sys
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
from policies.obstacles import ObstacleCatalog


def main():
    here=Path(__file__).resolve().parent;cases=[]
    review=ROOT/'work/hybrid_public_integration_20261101/catalog_review20_r1'
    for path in sorted(review.glob('*_scene*.json')):
        row=json.loads(path.read_text())
        if row['branch']!='static':continue
        development=row['stage']=='development'
        manifest=ROOT/('experiments/skills_temporal_development10_paired_20261029_r1/manifest.json' if development
            else 'work/skills_candidate_local_holdout_protocol_20261030_r1/validation10.json')
        inputs=ROOT/('work/skills_candidate_local_development_protocol_20261030_r4/inputs.json' if development
            else 'work/skills_candidate_local_holdout_protocol_20261030_r1/inputs.json')
        catalog=json.loads(manifest.read_text());index=catalog['original_scenario_indices'].index(row['scene'])
        raw=Path(catalog['scenario_files'][index]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['scenario_sha256']:raise ValueError('source_pickle_changed')
        sample=next(s for s in json.loads(inputs.read_text()) if s['scene']==row['scene']);geometry=pickle.loads(raw)
        chosen=row['static_catalog'][0];counter=itertools.count();primitives=[]
        for placement in chosen['placements']:
            primitives += [o.to_public_dict() for o in ObstacleCatalog().materialize(placement,new_id=lambda name:f'gallery-{next(counter)}')]
        cases.append(dict(scene=row['scene'],stage=row['stage'],ego=sample['frame']['ego'],
            agents=[s for s,active in zip(sample['frame']['agents'],sample['frame']['active']) if active],
            route=np.asarray(geometry['route']).tolist(),lanes=np.asarray(geometry['lanes']).tolist(),
            candidate=chosen,primitives=primitives,source_pickle_sha256=row['scenario_sha256'],
            scope='first_verified_static_catalog_candidate_not_LLM_choice_or_executed_attack'))
    (here/'gallery_data.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(gallery_cases=len(cases),API_requests=0,new_execution_frames=0)))


if __name__=='__main__':main()
