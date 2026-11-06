"""CPU目录审计的无损JSON类型适配；不改变几何、候选或安全判断。"""
import json
from pathlib import Path
import numpy as np


def normalize(value):
    """仅将NumPy标量/数组转为JSON原生类型，未知类型保持显式失败。"""
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def save(path, value):
    """先完整编码再原子落盘，防止序列化错误截断上一份进度。"""
    path = Path(path)
    content = json.dumps(normalize(value), ensure_ascii=False, indent=2)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(content, encoding='utf-8')
    pending.replace(path)
