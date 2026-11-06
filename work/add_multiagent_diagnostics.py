"""增量加入默认关闭的候选诊断，不影响样本选择或安全门槛。"""
from pathlib import Path
import time

root = Path('/home2/zhaoyx/scenario-dreamer')
path = root/'policies/diffusion_model_wrapper.py'
raw = path.read_bytes()
(root/'work'/('diffusion_before_diagnostics_'+str(time.time_ns())+'.py')).write_bytes(raw)
text = raw.decode()
anchor = '        joint = self.adapter.decode(frame, safe_batch, positions_local, yaws_local)\n'
addition = '''        # 可选冻结全部联合样本，离线定位具体车辆、时刻和动力学拒绝原因。
        diagnostic_dir = os.environ.get("RISKWEAVER_JOINT_DIAGNOSTIC_DIR")
        diagnostic_samples = info.get("action_samples") if isinstance(info, dict) else None
        if diagnostic_dir and isinstance(diagnostic_samples, dict):
            diagnostic_positions = diagnostic_samples.get("positions")
            diagnostic_yaws = diagnostic_samples.get("yaws")
            if diagnostic_positions is not None and diagnostic_yaws is not None:
                decoded = [self.adapter.decode(frame, safe_batch,
                           diagnostic_positions[:, index].detach().cpu().numpy(),
                           diagnostic_yaws[:, index].detach().cpu().numpy())
                           for index in range(diagnostic_positions.shape[1])]
                Path(diagnostic_dir).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(diagnostic_dir) / f"joint_{frame.step}_{time.time_ns()}.npz",
                    positions=np.stack([candidate.positions_global for candidate in decoded]),
                    agent_ids=joint.agent_ids, states=frame.states_global, dt=frame.dt,
                    selected_index=-1 if joint_sample_index is None else joint_sample_index,
                    source_step=frame.step)
'''
if text.count(anchor) != 1:
    raise RuntimeError('decode anchor mismatch')
path.write_text(text.replace(anchor, anchor+addition), encoding='utf-8')
