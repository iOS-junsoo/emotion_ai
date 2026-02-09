from typing import Dict, List


def average_softmax(windows: List[Dict[str, float]]) -> Dict[str, float]:
    """
    최근 N창 softmax 평균.
    """
    if not windows:
        return {}
    keys = set().union(*[w.keys() for w in windows])
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = sum(w.get(k, 0.0) for w in windows) / len(windows)
    # 정규화
    s = sum(out.values()) or 1.0
    for k in list(out.keys()):
        out[k] = out[k] / s
    return out


