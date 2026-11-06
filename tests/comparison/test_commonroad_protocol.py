"""底层原生日志不得破坏带request_id的JSON响应协议。"""
import subprocess
import sys
import json


def test_native_file_descriptor_output_is_not_json_stdout():
    code='''import os,json
from experiments.comparison.commonroad_worker import native_stdout_to_stderr
with native_stdout_to_stderr():
 os.write(1,b'native warning\\n')
 print('python warning')
print(json.dumps({'request_id':1,'ok':True}))
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)=={'request_id':1,'ok':True}
    assert 'native warning' in result.stderr and 'python warning' in result.stderr
