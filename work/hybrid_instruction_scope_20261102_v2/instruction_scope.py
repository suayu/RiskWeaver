"""混合模式专用指令装配与只读请求审计；不覆盖原纯动态入口。"""
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


INSTRUCTION = ['通过分层动静态攻击挑战自车驾驶策略。有可用动态车辆候选时优先选择车辆离散策略；'
    '没有可用动态车辆候选时，从代码发布的静态物体组合目录中选择道路拦截方案。'
    '仅选择目录中的候选ID，不编造车辆、锚点、障碍物或可行性证书。'
    '保留原动力学、背景交通安全与正可行驶面积要求；未来只是预测，计划不等于实际执行。']


def payload_instruction(messages):
    """只解析实际SDK消息中的末行JSON，不凭stdout或配置替代传输证据。"""
    if not messages or not isinstance(messages[-1].get('content'), str):
        raise ValueError('missing_structured_SDK_message')
    payload = json.loads(messages[-1]['content'].rsplit('\n', 1)[-1])
    if 'instruction' not in payload:
        raise ValueError('instruction_missing_from_actual_payload')
    return payload['instruction']


def audit_calls(calls):
    """保留无法解析和不一致记录，不能跳过失败或截断请求来宣称通过。"""
    rows = []
    for index, call in enumerate(calls):
        try:
            actual = payload_instruction(call.get('messages'))
            rows.append(dict(index=index, matches=actual == INSTRUCTION, actual=actual))
        except (KeyError, TypeError, ValueError) as error:
            rows.append(dict(index=index, matches=False, parse_error=type(error).__name__))
    return dict(checked=len(rows), all_match=bool(rows) and all(r['matches'] for r in rows), rows=rows,
        no_calls_does_not_prove_instruction_delivery=True)


@contextmanager
def install(module, output=None):
    """在真实runner已装配后设置专用指令；退出时完整恢复旧函数与旧指令。"""
    original = module.install_variant
    pending = []
    def variant(options, runner):
        previous = copy.deepcopy(runner.INSTRUCTION)
        restore = original(options, runner)
        runner.INSTRUCTION = copy.deepcopy(INSTRUCTION)
        done = False
        receipt = dict(version='hybrid_instruction_scope_v2', instruction=copy.deepcopy(runner.INSTRUCTION),
            original_instruction=previous, runner_file=str(getattr(runner, '__file__', 'unknown')),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            actual_SDK_delivery_still_requires_journal_audit=True,
            output_directory_created_by_public_runner_only=True)
        def undo():
            nonlocal done
            if done:
                return
            done = True
            try:
                restore()
            finally:
                runner.INSTRUCTION = previous
                if output is not None:
                    # 不能在公共runner之前创建输出目录；失败时把装配收据放在父目录。
                    directory = Path(output)
                    if directory.exists() and not directory.is_dir():
                        raise NotADirectoryError(str(directory))
                    target = (directory/'hybrid_instruction_scope.json' if directory.is_dir()
                        else directory.with_name(directory.name+'_hybrid_instruction_scope.json'))
                    with target.open('x', encoding='utf-8') as file:
                        json.dump(receipt, file, ensure_ascii=False, indent=2)
        pending.append(undo)
        return undo
    module.install_variant = variant
    try:
        yield
    finally:
        module.install_variant = original
        first_error = None
        for undo in reversed(pending):
            try:
                undo()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
