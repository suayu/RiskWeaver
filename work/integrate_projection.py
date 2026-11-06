"""默认关闭地接入投影，并继续调用原画像硬门禁。"""
from pathlib import Path
import time

root = Path('/home2/zhaoyx/scenario-dreamer')
backup = root/'work'/('before_projection_'+str(time.time_ns()))
backup.mkdir()


def edit(name, before, after):
    path = root/name
    raw = path.read_bytes()
    text = raw.decode()
    if text.count(before) != 1:
        raise RuntimeError('projection integration anchor mismatch: '+name)
    target = backup/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    path.write_text(text.replace(before, after), encoding='utf-8')


edit('cfgs/sim/base.yaml', '  backend: safe_sim_diffusion #',
     '  # 可选公共连续修正；四组比较使用相同设置，默认保持原路径。\n'
     '  dynamics_projection:\n'
     '    enabled: false\n'
     '    max_deviation_m: 0.25\n'
     '    max_seconds_per_vehicle: 2.0\n'
     '  backend: safe_sim_diffusion #')
edit('run_simulation.py', '        profile_pipeline = self.generator.profile_pipeline\n',
     '        profile_pipeline = self.generator.profile_pipeline\n'
     '        projection_cfg = getattr(self.cfg.traffic_model, "dynamics_projection", None)\n'
     '        projection_enabled = bool(getattr(projection_cfg, "enabled", False))\n'
     '        if projection_enabled and profile_pipeline is None:\n'
     '            raise ValueError("轨迹投影需要画像硬门禁完成全时域安全复核")\n')
edit('run_simulation.py', '                if profile_attack_active:\n                    profile_pipeline.validate_joint(self.env)\n',
     '                if profile_attack_active:\n'
     '                    if projection_enabled:\n'
     '                        from policies.trajectory_projection import project_joint\n'
     '                        corrected = project_joint(self.env.pending_joint_trajectory,\n'
     '                            np.asarray(self.env.data_dict["agent"][-1], float), self.env.dt,\n'
     '                            profile_pipeline.dynamic_limits, projection_cfg)\n'
     '                        self.env.inject_joint_trajectory(corrected)\n'
     '                        trace.append(dict(kind="dynamics_projection", step=int(self.env.current_step),\n'
     '                                          audit=corrected.metadata["dynamics_projection"]))\n'
     '                    profile_pipeline.validate_joint(self.env)\n')
print(backup)
