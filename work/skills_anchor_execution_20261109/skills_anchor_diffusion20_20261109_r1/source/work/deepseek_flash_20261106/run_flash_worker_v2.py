"""显式付费Flash入口；原Skills/评分/安全/4次物理预算不变。"""
from pathlib import Path
import os,sys,inspect,json
ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'work/dual_track_20261105'),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from flash_guard import install

def main():
    output=Path(sys.argv[sys.argv.index('--output-dir')+1]);prepare='--prepare-only' in sys.argv
    if sys.argv[sys.argv.index('--model')+1]!='deepseek-flash' or sys.argv[sys.argv.index('--models')+1]!='deepseek-flash':raise ValueError('single_flash_model_required')
    os.environ.update(LLM_PROVIDER='deepseek',DEEPSEEK_BASE_URL='https://api.deepseek.com/v1',LLM_MODEL_NAME='deepseek-flash',LLM_MODEL_NAMES='deepseek-flash')
    # 旧入口的免费保护前置条件不能冒充本次显式付费授权；物理预算检查仍原样保留。
    import run_pool_validation as pool
    # 审计与物理预算原本只接受百炼；仅本独立入口改为严格接受DeepSeek。
    import experiments.skills.run_instrumented as audit_module
    import physical_call_budget as budget_module
    restore_installers=[]
    for module,name,old_guard,new_guard in [
        (audit_module,'install_api_audit',"if self.provider!='dashscope':","if self.provider!='deepseek':"),
        (budget_module,'install_physical_budget',"if planner.provider!='dashscope':","if planner.provider!='deepseek':")]:
        function=getattr(module,name);code=inspect.getsource(function)
        if code.count(old_guard)!=1:raise ValueError('paid_budget_or_audit_source_changed')
        scope={};exec(compile(code.replace(old_guard,new_guard),module.__file__,'exec'),module.__dict__,scope)
        setattr(module,name,scope[name]);restore_installers.append((module,name,function))
    original_main=pool.main;source=inspect.getsource(original_main)
    old="    if not options.free_protection_confirmed:raise ValueError('free_only_protection_confirmation_required_before_new_inference')"
    if source.count(old)!=1:raise ValueError('paid_entry_source_contract_changed')
    source=source.replace(old,"    if os.getenv('LLM_PROVIDER')!='deepseek':raise ValueError('explicit_paid_deepseek_provider_required')")
    ns={};exec(compile(source,str(pool.__file__),'exec'),pool.__dict__,ns);pool.main=ns['main']
    from experiments.comparison import run_e1_group as runner
    original_overrides=runner.group_overrides
    runner.group_overrides=lambda *a,**k:[x.replace('sim.llm.provider=dashscope','sim.llm.provider=deepseek') for x in original_overrides(*a,**k)]
    restore=install(output)
    try:
        import run_dual_worker_v4
        run_dual_worker_v4.main()
    finally:
        restore()
        for module,name,function in restore_installers:setattr(module,name,function)
        runner.group_overrides=original_overrides;pool.main=original_main
        if not prepare and output.exists():
            (output/'deepseek_flash_entry.json').write_text(json.dumps(dict(provider='deepseek',requested_model='deepseek-flash',documented_version='DeepSeek-V4.1-Flash-1017',paid_user_authorized=True,free_only_confirmation_claimed=False,pro_allowed=False,physical_calls_per_scene=4),indent=2))
if __name__=='__main__':main()
