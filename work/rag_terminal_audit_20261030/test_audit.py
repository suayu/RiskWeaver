"""检索字段/实际案例/引用/观察窗口及未知D的独立终态审计单元测试。"""
import json,unittest
from audit import wire_cases,recompute_window,same_number,digest


def call(documents=None):
    context={}
    if documents is not None:context['retrieved_attack_evidence']={'documents':documents}
    return {'messages':[{'role':'user','content':'contract\n'+json.dumps({'context':context})}]}


class AuditTests(unittest.TestCase):
    def test_field_present_is_not_actual_case_exposure(self):
        self.assertFalse(wire_cases(call())['field_present'])
        result=wire_cases(call([]));self.assertTrue(result['field_present']);self.assertEqual(result['case_ids'],[])

    def test_case_ID_exposure_requires_nonempty_documents(self):
        self.assertEqual(wire_cases(call([{'case_id':'a'}]))['case_ids'],['a'])

    def test_weighted_D_uses_declared_observed_range_and_valid_frames(self):
        rows=[{'source_step':0,'valid':True,'difficulty':.9,'weight':2},
            {'source_step':1,'valid':False,'difficulty':None,'weight':1},
            {'source_step':2,'valid':True,'difficulty':.3,'weight':1},
            {'source_step':3,'valid':True,'difficulty':1,'weight':1}]
        result=recompute_window({'source_step':0,'observed_frames':3},rows)
        self.assertEqual(result['observed_frames'],3);self.assertEqual(result['valid_frames'],3);self.assertAlmostEqual(result['D'],.7)

    def test_unknown_D_is_not_zero(self):
        result=recompute_window({'source_step':5,'observed_frames':1},[{'source_step':5,'valid':False,'difficulty':None}])
        self.assertIsNone(result['D']);self.assertFalse(same_number(result['D'],0));self.assertTrue(same_number(None,None))

    def test_KB_payload_tampering_changes_content_hash(self):
        a={'case_id':'a','D':.1};b=dict(a,D=.9);self.assertNotEqual(digest(a),digest(b))


if __name__=='__main__':unittest.main()
