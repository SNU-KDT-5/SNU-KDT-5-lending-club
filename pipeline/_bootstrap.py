"""폴더 이름이 달라도 lc_config 를 찾아 import 경로에 넣는다.

작업 폴더는 한글('파이프라인', 'step 1'…)이고 저장소는 영문('pipeline', 'steps'…)이다.
두 배치 모두에서 스크립트가 그대로 돌아가도록 lc_config.py 를 직접 찾는다.
"""
import os
import sys

_CANDIDATES = ("파이프라인", "pipeline", ".")


def add_pipeline_to_path(start=None):
    """lc_config.py 가 있는 폴더를 sys.path 에 넣고 그 경로를 돌려준다."""
    here = start or os.getcwd()
    for base in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        for name in _CANDIDATES:
            d = os.path.normpath(os.path.join(base, name))
            if os.path.exists(os.path.join(d, "lc_config.py")):
                if d not in sys.path:
                    sys.path.insert(0, d)
                return d
    raise FileNotFoundError(
        "lc_config.py 를 찾지 못했습니다. 아래 중 한 곳에 있어야 합니다:\n  "
        + "\n  ".join(os.path.join(b, n) for b in (here, os.path.dirname(here))
                      for n in _CANDIDATES))
