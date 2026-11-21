"""百炼技能对照的无凭据API审计入口；旧流程与新流程使用同一记录层。"""
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def install_api_audit(output):
    """安装仅在本进程生效的记录层，并返回恢复函数用于测试和复用。"""
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from policies.skills_planner import SkillsAttackPlanner
    from experiments.comparison import run_e1_group as runner
    # 此文件在结果目录旁保存，避免提前创建目录使旧运行器误判为覆盖运行。
    trace_path=output.with_name(output.name+'_api_calls.jsonl')
    if trace_path.exists():raise FileExistsError(trace_path)
    original_init=LLMAdversarialPlanner.__init__
    original_free=LLMAdversarialPlanner.generate_attack_plan
    original_skills=SkillsAttackPlanner.plan
    original_episode=runner.run_episode

    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        if self.provider!='dashscope':raise ValueError('skills experiment must use dashscope')
        if self.client is None:return
        create=self.client.chat.completions.create
        def traced(*positional,**request):
            record=dict(request_id=uuid.uuid4().hex,model_requested=request.get('model'),
                messages=request.get('messages'),started_at=time.time(),
                source_step=getattr(self,'_audit_step',None),scene_id=getattr(self,'_audit_scene',None),
                scenario_index=getattr(self,'_audit_scene_index',None))
            started=time.monotonic()
            try:
                response=create(*positional,**request)
                record.update(model_returned=response.model,content=response.choices[0].message.content,
                    usage=response.usage.model_dump() if response.usage else None,
                    finish_reason=response.choices[0].finish_reason)
                return response
            except Exception as error:
                # 不记录原始异常正文，防止服务实现把请求头或密钥带入日志。
                record.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
            finally:
                record['elapsed_seconds']=time.monotonic()-started
                with trace_path.open('a',encoding='utf8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        self.client.chat.completions.create=traced

    def free(self,state,*args,**kwargs):
        self._audit_step=state.get('current_step')
        return original_free(self,state,*args,**kwargs)

    def skills(self,planner,env,state,*args,**kwargs):
        planner._audit_step=int(env.current_step);planner._audit_scene=str(getattr(env,'current_scene_id','unknown'))
        return original_skills(self,planner,env,state,*args,**kwargs)

    def episode(env,policy,index,global_index,*args,**kwargs):
        if env.llm_planner is not None:
            env.llm_planner._audit_scene_index=int(global_index)
            # reset尚未执行，不能把上一个场景的ID写到新请求；全局索引已明确。
            env.llm_planner._audit_scene=None
        return original_episode(env,policy,index,global_index,*args,**kwargs)

    LLMAdversarialPlanner.__init__=init
    LLMAdversarialPlanner.generate_attack_plan=free
    SkillsAttackPlanner.plan=skills
    runner.run_episode=episode
    def restore():
        LLMAdversarialPlanner.__init__=original_init
        LLMAdversarialPlanner.generate_attack_plan=original_free
        SkillsAttackPlanner.plan=original_skills
        runner.run_episode=original_episode
    return restore


def main():
    from experiments.skills.run_group import main as run_group
    output=Path(sys.argv[sys.argv.index('--output-dir')+1])
    restore=install_api_audit(output)
    try:run_group()
    finally:restore()


if __name__=='__main__':main()
