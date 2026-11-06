"""只收取不可变准备及终态新实验；远端只读，全部文件在本机核对SHA。"""
from pathlib import Path
import importlib.util

source=Path(__file__).resolve().parents[1]/'skills_integration_20261020/collect_verified.py'
spec=importlib.util.spec_from_file_location('base_collect',source);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=Path('D:/RiskWeaver/outputs/skills_event_grid_20261030')
scopes=['work/skills_event_grid_20261030','work/skills_event_grid_analysis_20261030','work/skills_future_event_probe_20261029','work/skills_event_grid_planning_protocol_20261030_r1']
start=base.REMOTE.index('scopes=[');end=base.REMOTE.index('allowed=',start)
extra='''
resource_terminals=[]
name='skills_event_grid_initial_planning10_paired_20261030_r1'
if (root/'experiments'/name/'results.json').exists():
 scopes+=['experiments/'+name,'work/'+name+'_dispatch.json','work/'+name+'_launcher.log']
'''
base.REMOTE=base.REMOTE[:start]+'scopes='+repr(scopes)+'\n'+extra+base.REMOTE[end:]

if __name__=='__main__':base.main()
