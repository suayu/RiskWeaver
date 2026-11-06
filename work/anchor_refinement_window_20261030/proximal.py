"""采样末端动作空间近端修正；不替换位置、不训练网络，原硬门槛仍须重新验证。"""
import torch


def fit_actions(dynamics, states, actions, targets, event_indices, dt, iterations=10):
    """通过现有有界动力学求小幅动作残差，只处理同一攻击车的真实候选。"""
    if not event_indices:
        return actions.detach().clone(), {'iterations': 0}
    if not all(torch.isfinite(x).all() for x in (states, actions, targets)):
        raise ValueError('nonfinite_proximal_input')
    events = torch.as_tensor(event_indices, device=actions.device, dtype=torch.long)
    if events.min() < 0 or events.max() >= actions.shape[1]:
        raise ValueError('proximal_event_out_of_horizon')
    # 将被动力学裁剪的原始动作规范化成等价有效动作，避免饱和区零梯度。
    with torch.no_grad():
        state = states
        clipped = []
        for t in range(actions.shape[1]):
            lo, hi = dynamics.ubound(state)
            value = torch.maximum(torch.minimum(actions[:, t], hi), lo)
            clipped.append(value)
            state = dynamics.step(state, value, dt)
        base = torch.stack(clipped, 1).detach()
    scale = actions.new_tensor([1., .1]).view(1, 1, 2)  # 加速度m/s²、偏航角速度rad/s。
    max_step = actions.new_tensor([.5, .3]).view(1, 1, 2)
    z = torch.zeros_like(base)
    regularization = .01
    history = []

    def evaluate(offset):
        trajectory = dynamics.forward_dynamics(states, base + offset * scale, dt)[0]
        residual = (trajectory[:, events, :2] - targets[:, events]).flatten(1)
        return residual, residual.square().sum(1) + regularization * offset.square().sum((1, 2))

    # 固定10轮、固定5档回溯，预算独立记录；不增加扩散网络调用或样本数。
    with torch.enable_grad():
        initial_error = evaluate(z)[0].detach().square().sum(1)
        for _ in range(iterations):
            z = z.detach().requires_grad_(True)
            residual, cost = evaluate(z)
            jacobian = torch.stack([
                torch.autograd.grad(residual[:, j].sum(), z, retain_graph=True)[0].flatten(1)
                for j in range(residual.shape[1])
            ], 1)
            flat = z.flatten(1)
            gram = jacobian @ jacobian.transpose(1, 2)
            eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
            # 以原候选为中心的近端Gauss-Newton步：-J^T(JJ^T+λI)^-1(r-Jz)-z。
            rhs = residual - (jacobian @ flat.unsqueeze(-1)).squeeze(-1)
            solution = torch.linalg.solve(gram + regularization * eye, rhs.unsqueeze(-1))
            delta = (-(jacobian.transpose(1, 2) @ solution).squeeze(-1) - flat).reshape_as(z)
            delta = torch.maximum(torch.minimum(delta, max_step), -max_step).detach()
            if not torch.isfinite(delta).all():
                raise ValueError('nonfinite_proximal_update')
            with torch.no_grad():
                best, best_cost = z.detach(), cost.detach()
                for alpha in (1., .5, .25, .125, .0625):
                    proposed = z.detach() + alpha * delta
                    new_cost = evaluate(proposed)[1]
                    accept = new_cost < best_cost
                    best = torch.where(accept[:, None, None], proposed, best)
                    best_cost = torch.minimum(best_cost, new_cost)
                z = best
                history.append(best_cost.cpu().tolist())
        final_error = evaluate(z)[0].detach().square().sum(1)
    return (base + z.detach() * scale).detach(), {
        'iterations': iterations, 'regularization': regularization,
        'control_scale_acceleration_yaw_rate': [1., .1],
        'max_step_acceleration_yaw_rate': [.5, .03],
        'initial_event_squared_error_m2': initial_error.cpu().tolist(),
        'final_event_squared_error_m2': final_error.cpu().tolist(),
        'regularized_cost_by_iteration': history,
    }


def refine_samples(sampler, result, current_states, guide, data):
    """保留整组样本编号及非攻击车动作；0号逐车合成样本仅在最后重新过滤。"""
    obj = guide.loss_calculator_dict.get('llm_anchor')
    sampler.anchor_proximal_audit = []
    if obj is None or obj.anchor_mask is None or obj.anchor_positions is None:
        return result
    actions = result['sampled_trajectories'].detach().clone()
    if actions.ndim != 4 or actions.shape[-1] != 2 or actions.shape[1] < 2:
        raise ValueError('unexpected_proximal_action_samples')
    b, n, horizon, _ = actions.shape
    audit = []
    for row in range(b):
        events = torch.nonzero(obj.event_mask[row] & obj.anchor_mask[row]).flatten().tolist()
        if not events:
            continue
        states = current_states[row:row+1].expand(n-1, -1)
        targets = obj.anchor_positions[row:row+1].to(actions).expand(n-1, -1, -1)
        fitted, record = fit_actions(sampler.net.dyn, states, actions[row, 1:], targets, events, sampler.net.step_time)
        actions[row, 1:] = fitted
        audit.append(dict(target_row=row, event_indices=events, **record))
    flat = actions.reshape(b*n, horizon, 2)
    states = current_states[:, None].expand(-1, n, -1).reshape(b*n, -1)
    with torch.no_grad():
        trajectory = sampler.net.dyn.forward_dynamics(states, flat, sampler.net.step_time)[0]
        filtered = guide.filter(flat, trajectory, data)
        actions[:, 0:1] = filtered
    result = dict(result, sampled_trajectories=actions, anchor_proximal_audit=audit)
    sampler.anchor_proximal_audit = audit
    return result
