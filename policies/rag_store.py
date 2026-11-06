"""每轮独立、可继承的攻击案例JSON知识库；不执行案例文字或覆盖历史原件。"""
from contextlib import contextmanager
from pathlib import Path
import copy,hashlib,json,os,time,uuid

SCHEMA='riskweaver_attack_rag_v1'
SECRET_KEYS={'api_key','authorization','private_key','access_token','password','credentials','smtp_password'}


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def validate_content(value):
    """拒绝非标准JSON及凭据字段；错误只说明类型，不回显字段值。"""
    canonical(value)
    if isinstance(value,dict):
        if any(str(k).lower() in SECRET_KEYS for k in value):raise ValueError('credential_field_forbidden_in_RAG')
        for item in value.values():validate_content(item)
    elif isinstance(value,list):
        for item in value:validate_content(item)


@contextmanager
def file_lock(path):
    """跨进程串行追加；锁在独立文件上，原子替换JSON不会使锁失效。"""
    lock=Path(str(path)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            stream.seek(0,2)
            if stream.tell()==0:stream.write(b'0');stream.flush()
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
            try:yield
            finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(stream,fcntl.LOCK_UN)


def atomic_write(path,value):
    validate_content(value);path=Path(path)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with temporary.open('x',encoding='utf-8') as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False);file.flush();os.fsync(file.fileno())
    os.replace(temporary,path)


def legacy_record(row,source_sha,index):
    """旧画像收益行完整保留；缺失身份/协议不能自动变成可迁移收益证据。"""
    validate_content(row)
    if not isinstance(row,dict) or row.get('schema_version')!=1 or 'scene_conditions' not in row or 'policy_id' not in row:
        raise ValueError('unsupported_legacy_attack_record')
    for field in ['D','reward']:
        value=row.get(field)
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=1):
            raise ValueError('invalid_legacy_measurement')
    return dict(case_id='legacy-'+digest([source_sha,index,row.get('record_id')]),schema_version=1,
        record_type='attack_event',status='closed',created_at=None,source='legacy_profile_knowledge',
        policy_id=row['policy_id'],policy_key=None,protocol_key=None,
        scene_conditions=copy.deepcopy(row['scene_conditions']),query_facts=None,
        strategy=row.get('strategy'),target_id=row.get('target'),profile_version=row.get('profile_version'),
        u=row.get('u'),actual_attack_execution_frames=row.get('actual_attack_execution_frames',0),
        D=row.get('D') if row.get('valid_observation_frames',0)>0 else None,
        measurement_valid=row.get('D') is not None and row.get('valid_observation_frames',0)>0,
        valid_observation_frames=row.get('valid_observation_frames',0),
        generation_failure_reason=row.get('generation_failure_reason'),
        is_failure=bool(row.get('generation_failure_reason')),benefit_eligible=False,
        policy_conditioned_evidence=dict(scope='legacy_policy_id_only; checkpoint_and_protocol_unverified',
            observed_reward=row.get('reward'),causal_attack_uplift_certified=False),
        legacy_original=copy.deepcopy(row))


def read_inherited(data):
    """接受本版JSON库或旧strategy_returns.jsonl；任何行错误都拒绝，不能静默筛掉。"""
    text=data.decode('utf-8-sig')
    if not text.strip():return dict(schema=SCHEMA,experiments=[],records=[],revision=0)
    try:value=json.loads(text)
    except json.JSONDecodeError:value=[json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value,dict) and value.get('schema')==SCHEMA:
        validate_content(value)
        if not isinstance(value.get('records'),list) or not isinstance(value.get('experiments'),list):raise ValueError('invalid_RAG_store')
        return copy.deepcopy(value)
    rows=value if isinstance(value,list) else [value];source_sha=hashlib.sha256(data).hexdigest()
    return dict(schema=SCHEMA,experiments=[],records=[legacy_record(row,source_sha,i) for i,row in enumerate(rows)],revision=len(rows))


class AttackRAGStore:
    """一个实验文件包含全部可检索正文；检索不会再去历史日志里补读案例内容。"""
    def __init__(self,path):
        self.path=Path(path)
        if not self.path.is_file():raise FileNotFoundError('RAG_store_missing')
        self.snapshot()

    @classmethod
    def create_new(cls,directory,*,inherit=False,source_path=None,experiment_id=None):
        """先独占创建新文件并复制源字节，再保留全部记录和继承收据；源文件只读。"""
        if not isinstance(inherit,bool):raise ValueError('inherit_requires_boolean')
        if inherit and not source_path:raise ValueError('inherit_source_required')
        if not inherit and source_path:raise ValueError('source_requires_explicit_inherit')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        run_id=experiment_id or uuid.uuid4().hex
        if not isinstance(run_id,str) or not run_id or Path(run_id).name!=run_id:raise ValueError('invalid_experiment_id')
        path=directory/f'attack_knowledge_{run_id}_{uuid.uuid4().hex}.json'
        lineage=None
        if inherit:
            source=Path(source_path)
            if source.suffix.lower() not in ('.json','.jsonl'):raise ValueError('RAG_source_must_be_JSON_or_JSONL')
            data=source.read_bytes();source_sha=hashlib.sha256(data).hexdigest()
            with path.open('xb') as file:file.write(data)
            if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:raise RuntimeError('inherit_source_changed_during_copy')
            document=read_inherited(data)
            lineage=dict(source_path=str(source.resolve()),source_sha256=source_sha,
                copied_byte_count=len(data),copied_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                inherited_record_count=len(document['records']),source_file_not_modified=True)
        else:
            with path.open('xb') as file:file.write(b'{}')
            document=dict(schema=SCHEMA,records=[],experiments=[],revision=0)
        document['experiments'].append(dict(experiment_id=run_id,started_at=time.time(),inherit=inherit,lineage=lineage))
        document['current_experiment_id']=run_id
        with file_lock(path):atomic_write(path,document)
        return cls(path)

    def snapshot(self):
        value=json.loads(self.path.read_text(encoding='utf-8'));validate_content(value)
        if value.get('schema')!=SCHEMA or not isinstance(value.get('records'),list):raise ValueError('invalid_RAG_store_schema')
        ids=[r.get('case_id') for r in value['records']]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('duplicate_or_missing_case_id')
        if any(r.get('record_type')!='attack_event' or r.get('status')!='closed' for r in value['records']):
            raise ValueError('RAG_only_contains_closed_attack_events_not_rules')
        for row in value['records']:validate_measurement(row)
        return value

    def append(self,record):
        """仅提交已结束事件，保持失败/无效指标；幂等重试不能覆盖不同内容。"""
        validate_content(record)
        if not isinstance(record.get('case_id'),str) or not record['case_id'] or record.get('status')!='closed':raise ValueError('closed_case_id_required')
        if record.get('record_type')!='attack_event':raise ValueError('rules_and_skills_do_not_belong_in_RAG_store')
        validate_measurement(record)
        with file_lock(self.path):
            value=self.snapshot()
            old=next((r for r in value['records'] if r['case_id']==record['case_id']),None)
            if old is not None:
                core={k:v for k,v in old.items() if k not in ('_kb_sequence','_record_sha256')}
                if canonical(core)!=canonical(record):raise ValueError('conflicting_case_id_cannot_overwrite')
                return old
            entry=copy.deepcopy(record);entry['_kb_sequence']=len(value['records'])+1;entry['_record_sha256']=digest(record)
            value['records'].append(entry);value['revision']=int(value.get('revision',0))+1
            atomic_write(self.path,value)
            return entry


def validate_measurement(record):
    """有效危险度必须有有效观测帧；复制的旧库也不能绕过null语义检查。"""
    value=record.get('D')
    if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=1):raise ValueError('invalid_RAG_danger_measurement')
    if not record.get('measurement_valid') and value is not None:raise ValueError('invalid_measurement_must_keep_D_null')
    if record.get('measurement_valid') and (value is None or record.get('valid_observation_frames',0)<=0):raise ValueError('valid_D_requires_positive_valid_frames')
