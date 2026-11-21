"""将模型池运行状态与算法源码分开审计，只允许已记录的永久停用变更。"""
import ast
import copy
import hashlib
from pathlib import Path
import re


def inspect_model_source(path):
    source=Path(path).read_bytes();module=ast.parse(source.decode('utf8'))
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='LLMAdversarialPlanner')
    assignment=next(n for n in cls.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='FREE_MODEL_POOL' for t in n.targets))
    active=ast.literal_eval(assignment.value)
    if not isinstance(active,tuple) or any(not isinstance(x,str) for x in active):raise ValueError('invalid source model pool')
    if len(set(active))!=len(active):raise ValueError('duplicate active model')
    # 池以外的所有可执行AST仍参与哈希，不能把真实算法变化当作正常额度注释。
    assignment.value=ast.Constant(value='MODEL_POOL_RUNTIME_STATE')
    algorithm=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()
    disabled=set(re.findall(r'^\s*#\s*["\']([^"\']+)["\']\s*,.*(?:百炼免费额度已耗尽|百炼模型404)',
                            source.decode('utf8'),re.MULTILINE))
    return dict(raw_sha256=hashlib.sha256(source).hexdigest(),algorithm_sha256=algorithm,
                active_models=list(active),disabled_models=sorted(disabled))


def verify_pool_transition(before,after):
    """删除活动模型需有停用标记；添加、改序或修改算法均不被默许。"""
    initial=before['active_models'];current=after['active_models'];removed=set(initial)-set(current)
    return dict(verified=before['algorithm_sha256']==after['algorithm_sha256']
                and current==[x for x in initial if x in current]
                and removed<=set(after['disabled_models']),
                algorithm_unchanged=before['algorithm_sha256']==after['algorithm_sha256'],
                disabled_during_run=sorted(removed),raw_source_unchanged=before['raw_sha256']==after['raw_sha256'])
