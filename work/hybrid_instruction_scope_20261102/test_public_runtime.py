"""公共runner装配及两条请求分支的替身检查，无实际模型或仿真。"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT), str(ROOT/'tests'),
    str(ROOT/'work/skills_v2_compact_dev_20261020'), str(Path(__file__).resolve().parent)]
from instruction_scope import INSTRUCTION, install


def test_actual_runner_instruction_is_replaced_then_restored():
    import run_closed_loop
    from experiments.comparison import run_e1_group as runner
    before = list(runner.INSTRUCTION)
    old_generator = runner.AdversarialScenarioGenerator
    old_overrides = runner.group_overrides
    options = NS(compact_view='on', event_budget='on', reachability_preflight='on',
        max_screening_queries=40, device='cpu')
    with install(run_closed_loop):
        undo = run_closed_loop.install_variant(options, runner)
        assert runner.INSTRUCTION == INSTRUCTION
        assert Path(runner.__file__).resolve() == ROOT/'experiments/comparison/run_e1_group.py'
        undo()
    assert runner.INSTRUCTION == before
    assert runner.AdversarialScenarioGenerator is old_generator
    assert runner.group_overrides is old_overrides


def fixture():
    spec = importlib.util.spec_from_file_location('hybrid_instruction_public_fixture',
        ROOT/'work/hybrid_public_integration_20261101/test_public_hybrid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_request_uses_same_hybrid_instruction_and_one_call():
    env, planner, engine, calls = fixture().setup(False)
    plan = engine.plan(planner, env, env.get_state_for_planning(), INSTRUCTION)
    assert plan['static_attack'] and calls[0]['instruction'] == INSTRUCTION
    assert engine.calls == len(calls) == 1


def test_dynamic_request_uses_same_hybrid_instruction_and_one_call():
    env, planner, engine, calls = fixture().setup(True)
    plan = engine.plan(planner, env, env.get_state_for_planning(), INSTRUCTION)
    assert plan['attack'] and plan['attack_kind'] == 'dynamic'
    assert calls[0]['instruction'] == INSTRUCTION
    assert engine.calls == len(calls) == 1
