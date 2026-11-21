"""公共适配只改参数，不偷偷重试或改变物理请求预算。"""
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from bailian_transport import install_provider_compatibility


class TransportTests(unittest.TestCase):
    def test_glm_adaptation_makes_exactly_one_sdk_call_and_restores(self):
        events=[]
        def create(**kwargs):events.append(kwargs);return 17
        engine=NS(provider='dashscope',last_trace={},client=NS(chat=NS(completions=NS(create=create))))
        def once(planner,model_name,prompt,scene_image=None):
            return planner.client.chat.completions.create(model=model_name,extra_body={'enable_thinking':False})
        with patch.object(LLMAdversarialPlanner,'_request_attack_plan_once',once):
            restore=install_provider_compatibility()
            try:self.assertEqual(LLMAdversarialPlanner._request_attack_plan_once(engine,'glm-5.3','prompt'),17)
            finally:restore()
            self.assertIs(LLMAdversarialPlanner._request_attack_plan_once,once)
        self.assertEqual(len(events),1)
        self.assertTrue(events[0]['extra_body']['enable_thinking'])
        self.assertIs(engine.client.chat.completions.create,create)

    def test_error_is_not_automatically_retried(self):
        calls=[]
        def create(**kwargs):calls.append(kwargs);raise ValueError('service fixture')
        engine=NS(provider='dashscope',last_trace={},client=NS(chat=NS(completions=NS(create=create))))
        def once(planner,model_name,prompt,scene_image=None):return planner.client.chat.completions.create(model=model_name)
        with patch.object(LLMAdversarialPlanner,'_request_attack_plan_once',once):
            restore=install_provider_compatibility()
            try:
                with self.assertRaises(ValueError):LLMAdversarialPlanner._request_attack_plan_once(engine,'glm-5.3','prompt')
            finally:restore()
        self.assertEqual(len(calls),1)
        self.assertIs(engine.client.chat.completions.create,create)


if __name__=='__main__':unittest.main()
