"""验证自动注释与真正算法漂移可区分，及未确认保护时绝不进入推理。"""
import json
import sys
import argparse
import ast
from pathlib import Path
import pytest
from model_pool_protocol import inspect_model_source,verify_pool_transition
import run_pool_validation as entry


def source(path,body='        "a",\n        "b",\n',value='1'):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('class LLMAdversarialPlanner:\n    FREE_MODEL_POOL = (\n'+body+
                    '    )\n    def algorithm(self):\n        return '+value+'\n',encoding='utf8')


def test_only_recorded_disabling_is_allowed(tmp_path):
    p=tmp_path/'planner.py';source(p);before=inspect_model_source(p)
    source(p,'        # "a",  # 百炼免费额度已耗尽，自动停用\n        "b",\n')
    report=verify_pool_transition(before,inspect_model_source(p))
    assert report['verified'] and report['algorithm_unchanged'] and report['disabled_during_run']==['a']
    assert not report['raw_source_unchanged']


@pytest.mark.parametrize('body,value',[
    ('        "b",\n','1'),
    ('        "a",\n        "b",\n        "c",\n','1'),
    ('        "b",\n        "a",\n','1'),
    ('        "a",\n        "b",\n','2'),
])
def test_unexplained_removal_addition_reorder_or_algorithm_change_rejected(tmp_path,body,value):
    p=tmp_path/'planner.py';source(p);before=inspect_model_source(p);source(p,body,value)
    assert not verify_pool_transition(before,inspect_model_source(p))['verified']


def test_prepare_only_does_not_create_result_or_require_credentials(tmp_path,monkeypatch,capsys):
    source(tmp_path/'policies/llm_adversarial_planner.py')
    monkeypatch.setattr(entry,'ROOT',tmp_path)
    for key in ['LLM_MODEL_NAME','LLM_MODEL_NAMES','DASHSCOPE_API_KEY']:monkeypatch.delenv(key,raising=False)
    output=tmp_path/'not_created'
    monkeypatch.setattr(sys,'argv',['entry','--skills','off','--prepare-only','--model','a','--models','a','b','--output-dir',str(output)])
    entry.main();report=json.loads(capsys.readouterr().out)
    assert report['prepare_only'] and report['transport_limit']==4
    assert not report['free_protection_operator_confirmed'] and not output.exists()


def test_real_mode_requires_free_protection_confirmation_before_runner_import(tmp_path,monkeypatch):
    source(tmp_path/'policies/llm_adversarial_planner.py');monkeypatch.setattr(entry,'ROOT',tmp_path)
    for key in ['LLM_MODEL_NAME','LLM_MODEL_NAMES']:monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(sys,'argv',['entry','--skills','off','--model','a','--models','a','b','--output-dir',str(tmp_path/'not_created')])
    with pytest.raises(ValueError,match='free_only_protection_confirmation_required'):entry.main()
    assert not (tmp_path/'not_created').exists()


@pytest.mark.parametrize('model_tokens',[['a','b'],['a,b']])
def test_pool_cli_is_accepted_by_real_e1_parser(monkeypatch,model_tokens):
    # 直接复用当前公共运行器的参数声明，不以复制的假解析器掩盖接口不一致。
    root=Path('/home2/zhaoyx/scenario-dreamer')
    module=ast.parse((root/'experiments/comparison/run_e1_group.py').read_text())
    main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    statements=[]
    for node in main.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parser' for t in node.targets):statements.append(node)
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='parser' and node.value.func.attr=='add_argument':statements.append(node)
    namespace=dict(argparse=argparse,Path=Path,ROOT=root,GROUPS=['riskweaver_profile_off'])
    exec(compile(ast.Module(body=statements,type_ignores=[]),'<real_e1_arguments>','exec'),namespace)
    args=['--group','riskweaver_profile_off','--model','a','--models',*model_tokens,'--manifest','/tmp/list.json','--output-dir','/tmp/not_created']
    for key in ['LLM_MODEL_NAME','LLM_MODEL_NAMES']:monkeypatch.delenv(key,raising=False)
    _,pool=entry.explicit_pool(args)
    parsed=namespace['parser'].parse_args(entry.runner_arguments(args,pool))
    assert pool==['a','b'] and parsed.models=='a,b'
