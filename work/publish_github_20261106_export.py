from pathlib import Path
import os,subprocess,shutil,json,hashlib,re,collections
ROOT=Path('/home2/zhaoyx/scenario-dreamer');DEST=Path('/home2/zhaoyx/RiskWeaver-github-20261106')
DEST.mkdir(exist_ok=False)
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
roots={'utils','safe-sim','datamodules','metadata','docs','tests','cfgs','experiments','nn_modules','datasets','policies','work','models','scripts','skills','data_processing'}
skip={'.git','__pycache__','.pytest_cache','source_snapshot','source','node_modules','site-packages','.venv','venv','build','dist','before','after','publish_github_20261106'}
ext={'.py','.sh','.bash','.ps1','.md','.rst','.txt','.yaml','.yml','.toml','.ini','.cfg','.h','.hpp','.c','.cpp','.cu','.cuh','.cmake'}
files=[];excluded=[]
for base,dirs,names in os.walk(ROOT):
 rel=Path(base).relative_to(ROOT)
 dirs[:]=[d for d in dirs if d not in skip and (rel.parts or d in roots)]
 for name in names:
  p=Path(base)/name;r=p.relative_to(ROOT);key=r.as_posix()
  if p.is_symlink() or name.startswith('.env') or name.endswith(('.lock','.pem','.key')) or any(w in name.lower() for w in ['private_key','credentials','llm.env']):continue
  allowed=key in tracked or p.suffix.lower() in ext or name in ['LICENSE','LICENSE.txt','LICENSE.md','CMakeLists.txt','.gitignore','.gitmodules']
  if p.suffix.lower()=='.json' and name in ['manifest.json','protocol.json','selection.json','package.json']:allowed=True
  if not allowed or p.stat().st_size>5*2**20:continue
  if p.suffix.lower() in {'.pt','.pth','.ckpt','.pkl','.npz','.png','.jpg','.zip','.mp4'} and key not in tracked:continue
  target=DEST/r;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);files.append(key)
# 根目录的当前tracked文件也包括无扩展名的运行配置；大数据不从历史目录递归收取。
findings=[]
patterns=[('provider_key',r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}'),('github_token',r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,})'),('private_key',r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----')]
for key in files:
 p=DEST/key
 try:s=p.read_text(encoding='utf-8')
 except UnicodeError:continue
 for typ,pat in patterns:
  for m in re.finditer(pat,s):findings.append(dict(file=key,line=s.count('\n',0,m.start())+1,type=typ))
receipt=dict(source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=len(files),bytes=sum((DEST/p).stat().st_size for p in files),secret_findings=findings,source_modified=False)
(ROOT/'work/publish_github_20261106/export_audit.json').write_text(json.dumps(receipt,indent=2))
(DEST/'SOURCE_PROVENANCE.json').write_text(json.dumps({k:v for k,v in receipt.items() if k!='secret_findings'},indent=2))
print(json.dumps(receipt))
