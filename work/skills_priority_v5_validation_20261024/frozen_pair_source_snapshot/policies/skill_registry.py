"""加载项目内技能知识；只读文本，不执行技能中的代码。"""
import hashlib
import json
from pathlib import Path


class SkillRegistry:
    def __init__(self,root=None,level='full'):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]/'skills/riskweaver'
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf8'))
        if level not in ('none','protocol','analysis','full'):
            raise ValueError('unknown skill level')
        self.level=level

    def load(self,obstacles=False,repair=False):
        """按本次模式加载必要知识，并冻结正文哈希；离线评估技能不发给规划模型。"""
        paths=[] if self.level=='none' else ['shared/contract.md']
        if self.level in ('analysis','full'):
            paths+=['rw-scene-analysis/SKILL.md']
        if self.level=='full':
            paths+=['rw-dynamic-attack/SKILL.md','shared/recipes.md']
            if obstacles:paths+=['rw-static-layout/SKILL.md']
            if repair:paths+=['rw-plan-repair/SKILL.md']
        documents=[]
        for name in paths:
            # 哈希原始字节，避免Windows CRLF在Linux文本读取时被规范化后误报漂移。
            raw=(self.root/name).read_bytes()
            text=raw.decode('utf8')
            expected=self.manifest['files'].get(name)
            digest=hashlib.sha256(raw).hexdigest()
            if digest!=expected:
                raise ValueError('skill content changed without manifest update: '+name)
            documents.append(dict(path=name,sha256=digest,text=text))
        return dict(version=self.manifest['version'],level=self.level,documents=documents)
