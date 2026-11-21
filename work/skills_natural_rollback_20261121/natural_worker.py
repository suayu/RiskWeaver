# 仅在本实验进程内移除两处帧数截断；保留原缓冲区大小及全部自然终止条件。
from pathlib import Path
import ast,copy,hashlib,inspect,json,runpy,sys,textwrap
ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))

def without_frame_cap(fn,expression):
    source=textwrap.dedent(inspect.getsource(fn));tree=ast.parse(source)
    expected=ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False)
    class Remove(ast.NodeTransformer):
        count=0
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==expected:
                self.count+=1
                return ast.copy_location(ast.Constant(value=expression.startswith('count')),node)
            return self.generic_visit(node)
    changer=Remove();tree=changer.visit(tree);assert changer.count==1,expression
    ast.fix_missing_locations(tree);scope={}
    exec(compile(tree,inspect.getsourcefile(fn),'exec'),fn.__globals__,scope)
    return scope[fn.__name__],hashlib.sha256(source.encode()).hexdigest()

def main():
    from experiments.comparison import run_e1_group as runner
    from simulator import Simulator
    original_episode=runner.run_episode;original_step=Simulator.step
    natural_episode,episode_sha=without_frame_cap(original_episode,'count < args.steps')
    natural_step,step_sha=without_frame_cap(original_step,'self.t == self.cfg.sim.steps')
    def episode(env,policy,index,global_index,args,*a,**kw):
        out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
        (out/'natural_termination_protocol.json').write_text(json.dumps(dict(
            version='natural_no_frame_truncation_v2',scenario_index=global_index,
            bootstrap_cli_steps=args.steps,allocation_horizon=env.steps,effective_frame_cap=None,
            runner_frame_cap_removed=True,simulator_frame_cap_removed=True,
            original_episode_sha256=episode_sha,original_step_sha256=step_sha,
            stop_conditions='original ego collision / route end / configured off-route handling',
            watchdog_is_fault_not_success=True),indent=2))
        return natural_episode(env,policy,index,global_index,args,*a,**kw)
    runner.run_episode=episode;Simulator.step=natural_step
    try:
        # 直接进入已回退的运动学计算入口，仅为记录规范化 NumPy 类型。
        import os
        from json_safe import plain
        os.environ.pop('RW_NL_INSTRUCTION',None)
        os.environ.pop('RW_NL_PREFERENCE_ENABLED',None)
        raw_dump=runner.dump_json
        runner.dump_json=lambda path,value:raw_dump(path,plain(value))
        out=Path(sys.argv[sys.argv.index('--output-dir')+1])

        entry=ROOT/'work/skills_anchor_execution_20261109/run_anchor_worker.py'
        sys.path.insert(0,str(entry.parent));sys.argv[0]=str(entry)
        runpy.run_path(str(entry),run_name='__main__')
    finally:
        runner.run_episode=original_episode;Simulator.step=original_step
        if 'out' in locals() and out.is_dir():
            (out/'generation_mode.json').write_text(json.dumps(dict(extra_language_guidance=False)))
if __name__=='__main__':main()
