import pytest
from flash_guard import parameters,check_returned

def test_flash_only():
    for name in ['deepseek-v4-pro','deepseek-v4-pro-0813','deepseek-v4.1-flash-1017','qwen-plus']:
        with pytest.raises(ValueError):parameters({'model':name})

def test_contract():
    v=parameters({'model':'deepseek-flash','messages':[]})
    assert v['max_tokens']==8192 and v['extra_body']['thinking']['type']=='disabled'
    assert v['response_format']=={'type':'json_object'}

def test_returned_pro_refused():
    check_returned('deepseek-flash')
    with pytest.raises(ValueError):check_returned('deepseek-v4-pro')
