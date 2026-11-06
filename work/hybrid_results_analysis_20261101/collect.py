"""只读收取已终态产物与工程证据；二进制走subprocess管道，逐文件核验SHA256。"""
from pathlib import Path
import argparse,hashlib,io,json,shlex,subprocess,tarfile

REMOTE='/home2/zhaoyx/scenario-dreamer'
DEST=Path('D:/RiskWeaver/outputs/skills_dynamic_static_20261101')
BASE=[
 'work/skills_source_facts_20261101','work/skills_source_facts_protocol_20261101_r1',
 'work/skills_source_facts_protocol_20261101_r2','work/skills_source_facts_protocol_20261101_r3',
 'work/skills_source_facts_protocol_20261101_r4','work/skills_service_health_20261101',
 'experiments/skills_source_facts_frozen4_diagnostic_20261101_r1',
 'experiments/skills_source_facts_frozen4_diagnostic_20261101_r2',
 'experiments/skills_source_facts_frozen4_diagnostic_20261101_r3',
 'work/skills_dynamic_static_20261101','work/hybrid_public_integration_20261101',
 'experiments/skills_dynamic_static_highlevel_probe3_20261101_r1',
 'work/hybrid_results_analysis_20261101']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-pair',action='store_true');args=parser.parse_args()
    roots=list(BASE)
    if args.terminal_pair:
        roots+=['work/hybrid_highlevel_pair_20261101','work/skills_hybrid_rag_highlevel_review10_protocol_20261101_r1',
            'experiments/skills_hybrid_rag_highlevel_review10_20261101_r1']
    script='''import pathlib,tarfile,io,json,hashlib,sys
r=pathlib.Path(%r);roots=%r
suffixes={'.py','.json','.jsonl','.yaml','.yml','.md','.xml','.log','.txt','.sh','.csv','.gz'}
members={}
for root in roots:
 p=r/root
 if not p.exists():continue
 if root.startswith('experiments/') and not (p/'results.json').exists():
  raise RuntimeError('refuse_to_collect_nonterminal_experiment:'+root)
 for f in p.rglob('*'):
  if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts or f.suffix not in suffixes:continue
  if f.name=='.env' or f.name.endswith('.env'):continue
  members[f.relative_to(r).as_posix()]=f
for f in (r/'work').glob('skills_*20261101*launcher.log'):members[f.relative_to(r).as_posix()]=f
for f in (r/'work').glob('skills_*20261101*dispatch.json'):members[f.relative_to(r).as_posix()]=f
b=io.BytesIO();hashes={}
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for name,p in sorted(members.items()):
  raw=p.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
  info=tarfile.TarInfo(name);info.size=len(raw);t.addfile(info,io.BytesIO(raw))
 raw=json.dumps(hashes,indent=2).encode();info=tarfile.TarInfo('collection_sha256.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(b.getvalue())
'''%(REMOTE,roots)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','zhaoyx@172.18.36.182',
        '/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python -c '+shlex.quote(script)],capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    DEST.mkdir(parents=True,exist_ok=True);hashes={}
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile() or '..' in Path(member.name).parts or Path(member.name).is_absolute():raise ValueError('archive_path')
            raw=archive.extractfile(member).read()
            if member.name=='collection_sha256.json':hashes=json.loads(raw);continue
            target=DEST/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    failures=[name for name,digest in hashes.items() if hashlib.sha256((DEST/name).read_bytes()).hexdigest()!=digest]
    receipt=dict(files=len(hashes),archive_bytes=len(result.stdout),hash_failures=failures,
        archive_sha256=hashlib.sha256(result.stdout).hexdigest(),roots=roots,terminal_pair_included=args.terminal_pair)
    name='collection_terminal_pair_receipt.json' if args.terminal_pair else 'collection_engineering_receipt.json'
    (DEST/name).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (DEST/('terminal_pair_sha256.json' if args.terminal_pair else 'engineering_sha256.json')).write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    if failures:raise RuntimeError('collection_hash_mismatch')
    print(json.dumps({k:v for k,v in receipt.items() if k!='roots'}))


if __name__=='__main__':main()
