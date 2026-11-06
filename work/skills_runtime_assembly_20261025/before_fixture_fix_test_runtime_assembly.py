"""真实多层入口装配回归：仅替代最末仿真入口和网络，无GPU/车辆执行。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from omegaconf import OmegaConf

import run_incremental as entry
from incremental_planner import IncrementalSkillsPlanner
from policies.llm_adversarial_planner import LLMAdversarialPlanner


class RuntimeAssemblyTests(unittest.TestCase):
    def test_real_entry_builds_selected_planner_and_restores_all_wrappers(self):
        from experiments.comparison import run_e1_group as runner
        root = entry.ROOT
        preparation = json.loads((root/'work/skills_incremental_entry_checks_20261025/actual_entry_prepare_final.json').read_text())
        original_override, original_generator = runner.group_overrides, runner.AdversarialScenarioGenerator
        frozen = json.loads((root/'experiments/skills_priority_v5_closed_loop_scene182_20261025_r1/source_sha256.json').read_text())
        frozen.update({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in entry.HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                       and p.suffix in {'.py','.md','.json'}})
        source = root/'experiments/skills_priority_v5_frozen_development10_pair_20261024_r1/inputs.json'
        item = next(i for i in json.loads(source.read_text()) if i['scenario_index'] == 74)
        checked = []
        for record in preparation['checks']:
            with self.subTest(mode=record['mode'], recipes=record['recipes']), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                (parent/'source_sha256.json').write_text(json.dumps(frozen))
                output = parent/'unit_only_no_vehicle_execution'
                command = list(record['command'])
                command.remove('--prepare-only')
                command[command.index('--output-dir')+1] = str(output)
                command[command.index('--device')+1] = 'cpu'
                command.append('--free-protection-confirmed')
                start = command.index('--models')+1
                end = next(i for i in range(start,len(command)) if command[i].startswith('--'))
                pool = command[start:end]
                expected_extra = record['recipes'] == 'on'
                expected_skills = record['mode'] == 'on'
                observed = []

                def substituted_simulation():
                    # 使用真实叠加后的配置函数，不能在测试中直接指定目标规划器。
                    values = runner.group_overrides('riskweaver_profile_off', model=pool[0], models=pool)
                    cfg = OmegaConf.load(root/'experiments/skills_modelpool_recovery_development10_paired_20261023_r3/skills_full_shared_recovery/resolved_config.yaml')
                    for value in values:
                        key, raw = value.lstrip('+').split('=',1)
                        parsed = OmegaConf.from_dotlist(['value='+raw])['value']
                        OmegaConf.update(cfg, key, parsed, merge=False, force_add=True)
                    planner = object.__new__(LLMAdversarialPlanner)
                    planner.available=True;planner.provider='dashscope';planner.use_multimodal=False
                    planner.allow_trajectory=True;planner.use_obstacles=False;planner.last_trace={}
                    generator = runner.AdversarialScenarioGenerator(cfg, ['unit-only dynamic intent'], llm_planner=planner)
                    if not expected_skills:
                        self.assertIsNone(generator.skills_planner)
                    else:
                        self.assertIsInstance(generator.skills_planner, IncrementalSkillsPlanner)
                        self.assertEqual(generator.skills_planner.extra_recipes, expected_extra)
                        self.assertEqual(str(cfg.sim.llm.skills.root), str(entry.HERE/'skills'))
                        self.assertEqual(generator.skills_planner.registry.root, entry.HERE/'skills')
                        state=copy.deepcopy(item['request']['state']);frame=item['frame'];route=np.asarray(state['route'])
                        env=NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
                            agent_active=np.asarray(frame['active']),current_step=0,current_scene_id='scenario_074',
                            scenario_dict={'route':route,'lanes':[route]},get_static_obstacles=lambda:[],cfg=cfg,
                            dt=.1,pending_joint_trajectory=None,get_state_for_planning=lambda:state)
                        prompts=[]
                        def transport(prompt,scene_image=None):
                            payload=json.loads(prompt.rsplit('\n',1)[1]);prompts.append(payload)
                            candidate=payload['context']['candidates'][0];cid=candidate['candidate_id']
                            contrast=payload['context']['priority_evidence'][cid]['facts'][cid+':contrast']
                            return '',dict(request_id=payload['request_id'],state_hash=payload['state_hash'],catalog_hash=payload['catalog_hash'],
                                ranking=[cid],scene_analysis='Transport fixture; no execution.',candidate_reasons={cid:'Uncertain proxy only.'},no_attack_reason='',
                                grounding={cid:dict(target_id=candidate['target_id'],target_role='background_vehicle',active_horizon_seconds=1.,
                                    evidence_ids=[cid+':actor',cid+':validity',cid+':active',cid+':contrast'],interaction_basis='exploratory_probe',
                                    contrast_direction=contrast['proxy_direction'])})
                        planner._request_attack_plan=transport
                        result=generator.skills_planner.plan(planner,env,state,['unit-only'])
                        self.assertTrue(result and result['attack'],planner.last_trace)
                        self.assertEqual(len(prompts),1)
                    output.mkdir()
                    observed.append(dict(mode=record['mode'],recipes=record['recipes'],configuration_root=str(cfg.sim.llm.skills.root),
                                         planner_type=type(generator.skills_planner).__name__,API_requests=0,execution_frames=0))

                with patch.object(runner,'main',side_effect=substituted_simulation), patch.object(sys,'argv',command[1:]), patch.dict(
                        os.environ,dict(CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=pool[0],LLM_MODEL_NAMES=','.join(pool))):
                    entry.main()
                self.assertEqual(len(observed),1)
                self.assertIs(runner.group_overrides,original_override)
                self.assertIs(runner.AdversarialScenarioGenerator,original_generator)
                protocol=json.loads((output/'incremental_entry_protocol.json').read_text())
                self.assertEqual(protocol['incremental_skills'],record['mode'])
                self.assertEqual(protocol['active_window_recipes'],record['recipes'])
                self.assertEqual(protocol['bailian_capabilities'],'observed')
                if expected_skills:
                    recorded=json.loads((output/'incremental_protocol.json').read_text())
                    self.assertEqual(recorded['knowledge_root'],str(entry.HERE/'skills'))
                checked.extend(observed)
        self.assertEqual(len(checked),3)


if __name__=='__main__':unittest.main()
