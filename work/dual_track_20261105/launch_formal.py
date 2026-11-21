"""仅在新评分短程接线完成后显式派发本轮两组20场，不拼接旧结果。"""
from pathlib import Path
import json,os,subprocess,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from launch_pool_pair import verify_sources


def main():
    protocol_dir=ROOT/'work/skills_dual_track_paired20_protocol_20261105_r3'
    protocol=json.loads((protocol_dir/'protocol.json').read_text())
    gate=json.loads((ROOT/'work/dual_track_20261105/smoke_gate.json').read_text())
    if gate.get('allowed') is not True:raise RuntimeError('smoke_gate_failed')
    if len(protocol['planned_scenes'])!=20 or protocol['planned_attempts']!=40:raise ValueError('paired_denominator')
    if Path(protocol['experiment_root']).exists():raise FileExistsError('preserve_existing_formal')
    hashes=json.loads((protocol_dir/'source_sha256.json').read_text())
    if not verify_sources(protocol,hashes)['verified']:raise RuntimeError('frozen_source_changed')
    if not (ROOT/'work/skills_kimi_acceptance20_20261105_r1_user_stop.json').exists():raise RuntimeError('missing_previous_stop_receipt')
    command=[sys.executable,str(ROOT/'work/dual_track_20261105/run_pair.py'),'--protocol',str(protocol_dir)]
    prefix=ROOT/'work/skills_dual_track_paired20_20261105_r3'
    env=dict(os.environ,PYTHONPATH='.',PYTHONUNBUFFERED='1')
    with Path(str(prefix)+'_launcher.log').open('xb') as output:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=child.pid,command=command,experiment_root=protocol['experiment_root'],
        protocol=str(protocol_dir),planned_scenes=20,planned_attempts=40,smoke_gate_passed=True,started_at=time.time())
    Path(str(prefix)+'_dispatch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    time.sleep(2)
    if child.poll() is not None:raise RuntimeError('controller_exited_at_launch')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
