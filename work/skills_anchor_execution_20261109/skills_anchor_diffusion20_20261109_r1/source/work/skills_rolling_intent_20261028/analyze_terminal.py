"""复用事先冻结的只读审计器，保留全部场景、无效覆盖和不利真实性。"""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('D:/RiskWeaver/outputs/skills_shared_resource_20261028')
EXP=OUT/'experiments/skills_four_arm_provider_shared_development10_20261028_r2'
OLD=Path('D:/RiskWeaver/outputs/skills_four_arm_incremental_20261027')
sys.path[:0]=[str(ROOT/'work/skills_four_arm_analysis_20261027'),
    str(OLD/'work/skills_physical_realism_reference_20261027')]
ARMS=['legacy_shared_recovery','catalog_no_knowledge','skills_no_expansion','skills_full_shared_recovery']


def save(name,result):
    with (OUT/name).open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    import scientific_audit,model_transition_audit,realism_batch,physical_batch
    acceptance=OUT/'terminal_independent_acceptance_20261029.json'
    scientific=scientific_audit.audit(EXP,ARMS,acceptance)
    scientific_audit.write_report(scientific,OUT/'terminal_scientific_audit_20261029')
    save('terminal_model_transition_20261029.json',model_transition_audit.audit(EXP,ARMS,False))
    save('terminal_legacy_realism_20261029.json',realism_batch.batch(EXP,ARMS,
        OLD/'work/realism_reference_waymo_perception_validation100_20260924/reference.json',OLD/'policies/realism_audit.py',8))
    save('terminal_physical_realism_20261029.json',physical_batch.batch(EXP,ARMS,
        OLD/'work/skills_physical_realism_reference_20261027/reference100_r1',8))
    print(json.dumps({k:{x:g.get(x) for x in ['planned_scenes','attempted_scenes','executed_scenes',
        'actual_attack_scenes','accepted_plans','attack_frames','strict_qualified_scenes','D','TTC','EA',
        'collision_rate','generation_failures','runtime_failures','planner_service_failures']} for k,g in scientific['groups'].items()}))


if __name__=='__main__':main()
