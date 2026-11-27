import os,signal,subprocess,time
from pathlib import Path
while Path('/proc/4104796/cmdline').exists():
 if b'launch_no_skills_fixed70.py' not in Path('/proc/4104796/cmdline').read_bytes():break
 for row in subprocess.run(['ps','--ppid','4104796','-o','pid=,etimes='],capture_output=True,text=True).stdout.splitlines():
  pid,age=map(int,row.split())
  try:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'comparison_no_skills_fixed70_20261123_r1/runs/no_skills/scenario_' in cmd and (age>=1500 or b'/scenario_003/' in cmd):
    print(pid,age,'terminated',flush=True)
    os.kill(pid,signal.SIGTERM)
  except (ProcessLookupError,FileNotFoundError):pass
 time.sleep(1)
