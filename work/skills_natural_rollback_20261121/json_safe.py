"""仅规范化JSON写出类型；不改变布尔判定、指标数值或失败语义。"""
import numpy as np


def plain(value):
    """递归处理NumPy标量/数组，保留None及非有限值原严格写出约束。"""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value
