import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
BASE = ROOT / 'experiments/riskweaver_profile_candidate_dry_run_20260922'
RUN = BASE / 'screen'
SOURCE_MANIFEST = ROOT / 'experiments/riskweaver_profile_scene_screen_20260922/pilot_manifest.json'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

BASE.mkdir(parents=True, exist_ok=False)
RUN.mkdir()
shutil.copy2(__file__, BASE / 'launch_profile_candidate_dry_run.py')
manifest_data = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
manifest = BASE / 'scenario_manifest.json'
write_json(manifest, manifest_data)
instruction = BASE / 'instruction.txt'
instruction.write_text('Candidate screening only; do not call an LLM and do not inject an attack.\n', encoding='utf-8')
overrides = [
    'sim.seed=42',
    '+sim.continue_after_off_route=true',
    'sim.evaluation.max_scenarios=5',
    'sim.steps=45',
    f'+sim.scenario_manifest={manifest}',
    f'+sim.attack_request_file={instruction}',
    'sim.llm.provider=dashscope',
    'sim.llm.model_name=qwen3.7-max',
    'sim.llm.model_names=null',
    'sim.llm.max_planning_calls=2',
    'sim.llm.profile_candidate_dry_run=true',
    'sim.llm.attack_mode=trajectory_only',
    'sim.llm.multiagent.mode=single',
    'sim.llm.obstacles.enabled=false',
    'sim.traffic_model.iterative_adversarial.profile_enabled=true',
    'sim.traffic_model.iterative_adversarial.escalation_enabled=false',
    'sim.traffic_model.iterative_adversarial.full_method_enabled=false',
    'sim.traffic_model.guidance.mode=llm_joint',
    'sim.traffic_model.dynamics_projection.enabled=false',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
    'sim.evaluation.difficulty_control.mode=off',
    'sim.evaluation.difficulty_control.smoke_acceptance=false',
    'sim.visualize=true',
    'sim.lightweight=true',
    f'sim.movie_path={RUN / "movies"}',
    f'sim.scenario_data_output_path={RUN / "carla"}',
    f'hydra.run.dir={RUN / "hydra"}',
]
command = [
    'bash', 'scripts/run_with_llm_provider.sh', 'dashscope',
    '/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python', 'run_simulation.py', *overrides,
]
sources = [
    'scenario_generator.py', 'policies/ego_profile.py', 'policies/profile_planner.py',
    'policies/diffusion_model_wrapper.py', 'policies/joint_safety.py', 'run_simulation.py',
    'cfgs/sim/base.yaml',
]
write_json(RUN / 'launch.json', {
    'command': command,
    'started_at': time.time(),
    'selection_manifest_sha256': hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
    'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    'purpose': 'profile candidate revalidation without LLM calls or attack injection',
})
environment = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES='0', PROJECT_ROOT=str(ROOT), SCRATCH_ROOT=str(ROOT),
    DATASET_ROOT=str(ROOT / 'metadata'),
    PYTHONPATH=f'{ROOT}:{ROOT / "safe-sim"}:{ROOT / "safe-sim/trajdata/src"}',
    MPLBACKEND='Agg', PYTHONUNBUFFERED='1', HYDRA_FULL_ERROR='1',
)
environment.pop('LLM_MODEL_NAMES', None)
with (RUN / 'resolved_config.yaml').open('w', encoding='utf-8') as output:
    subprocess.run(command + ['--cfg', 'job', '--resolve'], cwd=ROOT, env=environment, stdout=output, check=True)
with (RUN / 'run.log').open('w', encoding='utf-8') as output:
    exit_code = subprocess.call(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
write_json(RUN / 'exit_status.json', {'exit_code': exit_code, 'finished_at': time.time()})
raise SystemExit(exit_code)
