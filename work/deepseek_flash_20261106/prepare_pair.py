from pathlib import Path
import json,hashlib,shutil,subprocess,os
ROOT=Path('/home2/zhaoyx/scenario-dreamer')
def main():
    old=ROOT/'work/skills_dual_track_paired20_protocol_20261106_r1'
    new=ROOT/'work/skills_deepseek_flash_paired20_protocol_20261106_r1';new.mkdir(exist_ok=False)
    p=json.loads((old/'protocol.json').read_text());files=json.loads((old/'source_sha256.json').read_text())
    added=['work/deepseek_flash_20261106/flash_guard.py','work/deepseek_flash_20261106/run_flash_worker.py','work/deepseek_flash_20261106/run_pair.py','work/deepseek_flash_20261106/test_guard.py']
    for name in [*files,*added]:
        target=new/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        files[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    shutil.copyfile(old/'manifest.json',new/'manifest.json')
    p.update(version='official_deepseek_flash_paid_paired20_v1',source_count=len(files),models=['deepseek-flash'],
        experiment_root=str(ROOT/'experiments/skills_deepseek_flash_paired20_20261106_r1'),source_protocol=str(new),
        transport='official_deepseek_chat_completions',provider='deepseek',paid_user_authorized=True,
        free_protection_user_confirmed=False,user_approved_model='DeepSeek-V4.1-Flash-1017',
        documented_api_model='deepseek-flash',pro_allowed=False,fallback_models=[],
        supersedes_experiment=str(ROOT/'experiments/skills_dual_track_paired20_20261106_r1'),
        change_scope='provider paid official Flash only; same scene/Skills/scoring/safety/budget; no spliced results')
    for arm,c in p['commands'].items():
        c[1]=str(ROOT/'work/deepseek_flash_20261106/run_flash_worker.py')
        for name in ['--model','--models']:c[c.index(name)+1]='deepseek-flash'
        c[c.index('--manifest')+1]=str(new/'manifest.json');c[c.index('--output-dir')+1]=str(Path(p['experiment_root'])/arm)
        c.remove('--free-protection-confirmed')
    (new/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2));(new/'source_sha256.json').write_text(json.dumps(files,indent=2))
    env=dict(os.environ,PYTHONPATH='.')
    for arm,c in p['commands'].items():
        with (new/(arm+'_prepare.log')).open('xb') as log:code=subprocess.run(c+['--prepare-only'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError(arm+'_prepare_failed')
    print(json.dumps(dict(protocol=str(new),sources=len(files),planned_attempts=40,prepares='both_passed')))
if __name__=='__main__':main()
