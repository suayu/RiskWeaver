"""固定小样本来源比例，验证分母和全局编号不会继续写成100场。"""
from experiments.comparison.e1_protocol import select_scenes,select_100,protocol_snapshot,shard_manifest


def manifest250():
    files=[f'/source_{name}/s_{i}.pkl' for name,count in [('old',75),('ldm1',55),('ldm2',120)] for i in range(count)]
    return {'scenario_files':files,'scenario_sha256':[str(i) for i in range(250)]}


def test_ten_scenes_are_source_stratified_reproducible_and_unfiltered():
    a=select_scenes(manifest250(),10);b=select_scenes(manifest250(),10)
    assert a==b and len(set(a['original_scenario_indices']))==10
    assert sum('/source_old/' in s for s in a['scenario_files'])==3
    assert sum('/source_ldm1/' in s for s in a['scenario_files'])==2
    assert sum('/source_ldm2/' in s for s in a['scenario_files'])==5
    shard=shard_manifest(a,[0,1])
    assert shard['comparison_scene_count']==10 and shard['scenario_count']==2


def test_hundred_scene_selection_and_small_protocol_are_correct():
    old=select_100(manifest250());new=select_scenes(manifest250(),100)
    assert old['original_scenario_indices']==new['original_scenario_indices']
    assert protocol_snapshot(scene_count=10)['planned_attempts']==50
