"""知识库新建/继承/旧表升级/并发不丢记录/未知D语义；不调用模型或仿真。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import copy,hashlib,importlib.util,json,sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source_after/policies'))
from rag_store import AttackRAGStore,SCHEMA


def record(ident):
    return dict(case_id=str(ident),record_type='attack_event',status='closed',D=None,measurement_valid=False,
        policy_id='idm-v1',strategy='cut_in',actual_attack_execution_frames=0,is_failure=True)


def append_many(task):
    path,prefix=task;store=AttackRAGStore(path)
    for i in range(5):store.append(record(f'{prefix}-{i}'))


def test_new_files_never_reuse_old_store_and_do_not_inherit_implicitly(tmp_path):
    first=AttackRAGStore.create_new(tmp_path);first.append(record('old'))
    other=AttackRAGStore.create_new(tmp_path)
    assert other.path!=first.path and other.snapshot()['records']==[] and len(first.snapshot()['records'])==1


def test_explicit_inherit_copies_all_rows_and_keeps_source_bytes_unchanged(tmp_path):
    source=AttackRAGStore.create_new(tmp_path/'source');source.append(record('old'))
    before=source.path.read_bytes();copied=AttackRAGStore.create_new(tmp_path/'new',inherit=True,source_path=source.path)
    assert copied.snapshot()['records']==source.snapshot()['records']
    copied.append(record('new'))
    assert source.path.read_bytes()==before and len(copied.snapshot()['records'])==2
    lineage=copied.snapshot()['experiments'][-1]['lineage']
    assert lineage['copied_sha256']==hashlib.sha256(before).hexdigest() and lineage['inherited_record_count']==1


def test_legacy_jsonl_rows_are_all_preserved_but_unverified_reward_not_transferred(tmp_path):
    rows=[dict(schema_version=1,record_id=str(i),policy_id=p,scene_conditions={'road_shape':'straight'},D=None,reward=.9,
        strategy='cut_in',generation_failure_reason='dynamics' if i else None) for i,p in enumerate(['idm','rl'])]
    source=tmp_path/'strategy_returns.jsonl';source.write_text('\n'.join(json.dumps(r) for r in rows))
    store=AttackRAGStore.create_new(tmp_path/'new',inherit=True,source_path=source)
    data=store.snapshot()['records']
    assert [r['legacy_original'] for r in data]==rows and all(not r['benefit_eligible'] for r in data)
    assert all(r['D'] is None for r in data) and {r['policy_id'] for r in data}=={'idm','rl'}


@pytest.mark.parametrize('inherit,source',[(True,None),(False,'ignored.json')])
def test_inherit_flag_is_explicit(inherit,source,tmp_path):
    with pytest.raises(ValueError):AttackRAGStore.create_new(tmp_path,inherit=inherit,source_path=source)


def test_invalid_measurement_cannot_be_zero_or_committed_open_case(tmp_path):
    store=AttackRAGStore.create_new(tmp_path);row=record('bad');row['D']=0.
    with pytest.raises(ValueError,match='null'):store.append(row)
    row=record('open');row['status']='open'
    with pytest.raises(ValueError,match='closed'):store.append(row)
    assert store.snapshot()['records']==[]


def test_idempotence_does_not_allow_overwriting_a_conflicting_event(tmp_path):
    store=AttackRAGStore.create_new(tmp_path);row=record('one');store.append(row);store.append(row)
    other=copy.deepcopy(row);other['strategy']='hard_brake'
    with pytest.raises(ValueError,match='overwrite'):store.append(other)
    assert len(store.snapshot()['records'])==1


def test_multiple_processes_cannot_lose_appended_cases(tmp_path):
    store=AttackRAGStore.create_new(tmp_path)
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(append_many,[(str(store.path),str(i)) for i in range(3)]))
    rows=store.snapshot()['records'];assert len(rows)==15 and {r['_kb_sequence'] for r in rows}==set(range(1,16))


def test_corrupt_or_rule_documents_and_credentials_do_not_silently_enter_store(tmp_path):
    store=AttackRAGStore.create_new(tmp_path);row=record('secret');row['api_key']='fixture-value'
    with pytest.raises(ValueError,match='credential'):store.append(row)
    row=record('rule');row['record_type']='skill'
    with pytest.raises(ValueError,match='rules'):store.append(row)
