"""真实审计/预算安装链使用SDK替身验证，零HTTP和零仿真。"""
from pathlib import Path
from types import SimpleNamespace
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
import run_flash_worker_v2 as entry
counter=[]
def exercise():
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    import experiments.skills.run_instrumented as audit
    import physical_call_budget as budget
    output=HERE/'runtime_contract_fake'
    undo_a=audit.install_api_audit(output);undo_b=budget.install_physical_budget(output,4)
    try:
        def create(**kw):
            counter.append(kw['model'])
            return SimpleNamespace(model='deepseek-flash',choices=[SimpleNamespace(message=SimpleNamespace(content='{}'),finish_reason='stop')],usage=None)
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        planner=LLMAdversarialPlanner(model_name='deepseek-flash',model_names=['deepseek-flash'],provider='deepseek',client=client)
        for i in range(4):planner.client.chat.completions.create(model='deepseek-flash',messages=[])
        try:planner.client.chat.completions.create(model='deepseek-flash',messages=[])
        except budget.TransportBudgetExhausted:pass
        else:raise AssertionError('budget_not_enforced')
        assert counter==['deepseek-flash']*4
    finally:undo_b();undo_a()
sys.modules['run_dual_worker_v4']=SimpleNamespace(main=exercise)
sys.argv=['worker','--model','deepseek-flash','--models','deepseek-flash','--output-dir',str(HERE/'runtime_contract_fake')]
entry.main()
receipt=dict(real_api_audit_and_budget_installed=True,sdk_substitute_calls=len(counter),fifth_blocked=True,actual_http_requests=0,pro_requests=0)
(HERE/'runtime_contract.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
