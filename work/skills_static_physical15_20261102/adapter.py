"""独立、默认关闭的过程内后备规则；不写公共或当前冻结源码。"""
import ast
import copy
import hashlib
import inspect
import math
import textwrap
import types
from contextlib import contextmanager

from .geometry import route_clearance

EXPECTED_AST = '621696bb58d682cb94ea6f7a994a101baab9fb09a9c726646d48c915830d3351'


@contextmanager
def install(planner, env, enabled=False, initial_margin=.5):
    """仅原15米检查失败才尝试路线检查；原函数的其他语句完全保留。"""
    if not isinstance(enabled,bool):
        raise ValueError('route_fallback_flag_requires_bool')
    if isinstance(initial_margin,bool) or not math.isfinite(initial_margin) or initial_margin < .5:
        raise ValueError('initial_ego_margin_cannot_be_less_than_half_meter')
    events = []
    if not enabled:
        yield dict(enabled=False,events=events)
        return
    original = planner._validate_obstacle_plan
    function = original.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    if hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest() != EXPECTED_AST:
        raise ValueError('original_obstacle_validator_changed_review_required')
    expected = ast.parse('nearest_forward_edge < minimum_forward_clearance',mode='eval').body
    matches = [item for item in ast.walk(node) if isinstance(item,ast.If)
        and ast.dump(item.test,include_attributes=False) == ast.dump(expected,include_attributes=False)]
    if len(matches) != 1:
        raise ValueError('unexpected_original_clearance_predicate')
    # 原模板、数量、现有障碍等检查和原15米成功路径都不动，只增一个显式后备条件。
    matches[0].test = ast.BoolOp(op=ast.And(),values=[copy.deepcopy(matches[0].test),
        ast.UnaryOp(op=ast.Not(),operand=ast.Call(func=ast.Name(id='_rw_route_fallback',ctx=ast.Load()),
            args=[ast.Name(id=name,ctx=ast.Load()) for name in ['self','obstacles','minimum_forward_clearance','nearest_forward_edge']],keywords=[]))])
    ast.fix_missing_locations(tree)

    def fallback(subject, obstacles, threshold, old_forward_edge):
        import numpy as np
        from policies.joint_safety import obb_clearance
        ego = np.asarray(env.ego_state,dtype=float)
        angle = math.pi/2-ego[4]
        c,s = math.cos(angle),math.sin(angle)
        boxes = []
        for obstacle in obstacles:
            x,y = obstacle.center_global[:2]
            boxes.append([ego[0]+x*c+y*s,ego[1]-x*s+y*c,
                obstacle.yaw-angle,obstacle.length,obstacle.width])
        body = np.array([ego[0],ego[1],ego[4],ego[5],ego[6]])
        gaps = [float(obb_clearance(np.asarray(box),body)) for box in boxes]
        result = route_clearance(env.scenario_dict.get('route',[]),ego[:2],boxes,threshold)
        result.update(original_forward_edge_m=float(old_forward_edge),initial_ego_OBB_clearances_m=gaps,
            initial_margin_m=initial_margin,world_boxes=boxes)
        if not gaps or any(not math.isfinite(x) or x < initial_margin for x in gaps):
            result.update(accepted=False,reason='initial_ego_OBB_clearance')
        events.append(result)
        return result['accepted']

    namespace = dict(function.__globals__,_rw_route_fallback=fallback)
    exec(compile(tree,'<independent_route_static_fallback>','exec'),namespace)
    patched = types.MethodType(namespace[node.name],planner)
    had_own = '_validate_obstacle_plan' in planner.__dict__
    own = planner.__dict__.get('_validate_obstacle_plan')
    planner._validate_obstacle_plan = patched
    try:
        yield dict(enabled=True,events=events,original_validator_AST_sha256=EXPECTED_AST,
            only_original_forward_predicate_augmented=True,original_accepts_preserved=True,
            transformed_validator_source=ast.unparse(tree))
    finally:
        if had_own:
            planner._validate_obstacle_plan = own
        else:
            del planner._validate_obstacle_plan
