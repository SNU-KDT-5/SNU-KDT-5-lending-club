#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전처리 상수 fit / transform 관리
================================

문제
----
이전 파이프라인은 윈저라이징 상한 · 대치 중앙값 · Yeo-Johnson λ · hot-deck 통계를
175만 건 **전체**에서 계산한 뒤 분할했다. test 행의 정보가 train 행의 변환값에 섞였다.

해결
----
"학습이 필요한 상수"를 전부 이 모듈을 거치게 만든다.

    P = Params(mode="fit", split_mask=is_train)   # train 행만 보고 계산 → json 저장
    P = Params(mode="transform")                  # 저장된 값을 읽어 그대로 적용

    cap = P.quantile("annual_inc_cap", df["annual_inc"], 0.995)
    med = P.median("annual_inc_med", df["annual_inc"])

fit 모드에서는 split_mask가 True인 행(=train)만으로 값을 계산하고,
transform 모드에서는 데이터를 아예 보지 않고 저장된 값을 반환한다.
따라서 호출부 코드는 한 줄도 분기할 필요가 없다.

사용 순서
--------
    1) step2a: mode="fit"       → 산출물/fit_params.json 생성
    2) step2b: mode="transform" → 위 json으로 전체 행 변환

주의
----
transform 모드에서 저장되지 않은 키를 요청하면 KeyError를 낸다.
fit 단계를 건너뛰고 실행하는 사고를 막기 위한 것이므로 예외를 삼키지 말 것.
"""

import json
import os

import numpy as np
import pandas as pd

import lc_config as C


class Params:
    def __init__(self, mode, split_mask=None, path=None):
        """
        mode       : "fit" | "transform"
        split_mask : fit 모드에서 train 행을 가리키는 boolean Series/array.
                     None이면 전체 행을 쓰므로 누출이 발생한다 (경고 출력).
        """
        assert mode in ("fit", "transform"), mode
        self.mode = mode
        self.path = path or C.FITPARAMS_PATH
        self._mask = None if split_mask is None else np.asarray(split_mask, dtype=bool)
        self.store = {}

        if mode == "fit":
            if self._mask is None:
                print("  [경고] split_mask 없이 fit 모드 — 전체 행에서 상수를 계산합니다. "
                      "누출이 발생합니다.")
            else:
                print(f"  [fit] train {int(self._mask.sum()):,}행에서만 상수를 학습합니다.")
        else:
            if not os.path.exists(self.path):
                raise FileNotFoundError(
                    f"{self.path} 가 없습니다. step2a(fit)를 먼저 실행하세요.")
            with open(self.path, encoding="utf-8") as f:
                self.store = json.load(f)
            print(f"  [transform] 저장된 상수 {len(self.store)}개를 적용합니다.")

    # ---------------- 내부 ----------------
    def _train_part(self, s):
        """fit 모드에서 train 행만 잘라낸다."""
        if self._mask is None:
            return s
        m = self._mask
        if len(m) != len(s):
            raise ValueError(f"split_mask 길이 {len(m)} != 데이터 길이 {len(s)}. "
                             "reset_index(drop=True) 누락 여부를 확인하세요.")
        return s[m]

    def _resolve(self, key, compute):
        if self.mode == "transform":
            if key not in self.store:
                raise KeyError(f"저장되지 않은 상수: '{key}'. fit 단계에서 누락됐습니다.")
            return self.store[key]
        v = compute()
        # json 직렬화를 위해 numpy 스칼라를 파이썬 타입으로 변환
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, pd.Index):
            v = list(v)
        self.store[key] = v
        return v

    # ---------------- 공개 API ----------------
    def median(self, key, s):
        return self._resolve(key, lambda: float(self._train_part(s).median()))

    def mean(self, key, s):
        return self._resolve(key, lambda: float(self._train_part(s).mean()))

    def std(self, key, s):
        return self._resolve(key, lambda: float(self._train_part(s).std()))

    def quantile(self, key, s, q):
        return self._resolve(key, lambda: float(self._train_part(s).quantile(q)))

    def categories(self, key, s, min_count=None, min_share=None, top_n=None):
        """
        원-핫 대상 범주 목록을 train에서 결정한다.
        희소 범주 통합 기준(min_count / min_share / top_n)도 여기서 고정해야
        val/test에 없는 범주가 생기거나 열 구성이 달라지는 사고를 막을 수 있다.
        """
        def _c():
            vc = self._train_part(s).value_counts()
            if min_count is not None:
                vc = vc[vc >= min_count]
            if min_share is not None:
                vc = vc[vc / vc.sum() >= min_share]
            if top_n is not None:
                vc = vc.head(top_n)
            return list(vc.index.astype(str))
        return self._resolve(key, _c)

    def yeojohnson_lambda(self, key, s):
        """Yeo-Johnson λ를 train에서 추정한다. transform에서는 저장된 λ를 쓴다."""
        from scipy.stats import yeojohnson_normmax

        def _c():
            v = self._train_part(s).to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            return float(yeojohnson_normmax(v))
        return self._resolve(key, _c)

    def value(self, key, fn):
        """위 형태에 맞지 않는 임의의 상수. fn은 인자 없는 콜러블."""
        return self._resolve(key, fn)

    # ---------------- 저장 ----------------
    def save(self):
        if self.mode != "fit":
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"  [fit] 상수 {len(self.store)}개 저장 → {os.path.basename(self.path)}")

    def summary(self):
        print(f"\n  학습된 상수 {len(self.store)}개")
        for k in sorted(self.store):
            v = self.store[k]
            if isinstance(v, list):
                print(f"    {k:<38} [{len(v)}개] {v[:4]}{' ...' if len(v) > 4 else ''}")
            else:
                print(f"    {k:<38} {v}")


def load_split_mask(ids, split_name="train"):
    """cohort의 id 순서에 맞춰 train 여부 boolean 배열을 만든다."""
    sp = pd.read_csv(C.SPLIT_PATH)
    m = dict(zip(sp["id"].astype("int64"), sp["split"]))
    return np.array([m.get(int(i)) == split_name for i in ids], dtype=bool)


if __name__ == "__main__":
    # 자체 점검
    s = pd.Series([1, 2, 3, 4, 100.0])
    mask = np.array([True, True, True, False, False])
    p = Params(mode="fit", split_mask=mask, path="/tmp/_p.json")
    print("  train median =", p.median("t_med", s), "(전체 중앙값 3.0 과 달라야 정상)")
    print("  train q0.9   =", p.quantile("t_q", s, 0.9))
    p.save()
    q = Params(mode="transform", path="/tmp/_p.json")
    print("  transform median =", q.median("t_med", s))
    assert q.median("t_med", s) == 2.0
    print("  OK")
