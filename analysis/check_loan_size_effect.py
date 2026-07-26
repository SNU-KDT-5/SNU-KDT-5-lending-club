#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
중간층 우위에 '대출액 크기'가 개입했는가
========================================

[문제 제기]
중간등급(B~D)의 위험조정 수익이 높게 나왔다. 그런데 우리가 쓰는 Sharpe는
    분자 = funded_amnt로 가중한 평균 spread
    분모 = spread의 단순 표준편차(가중 없음)
이다. 분자에만 금액 가중이 들어간다.

그렇다면 "등급이 애매한 사람이 더 큰 금액을 빌렸다"거나 "큰 대출이 우연히
수익률이 좋았다"면, 중간층 우위가 실력이 아니라 가중 방식의 부산물일 수 있다.

[먼저 정리해둘 것]
IRR은 '비율'이라 규모에 중립적이다. 1억을 빌려주든 100만원을 빌려주든
상환 패턴이 같으면 IRR은 같다. 따라서 대출액이 수익률을 '직접' 부풀릴 수는 없다.
개입 경로는 간접적인 것 두 가지뿐이다.
  (경로1) 가중 — 금액이 큰 건이 분자에서 목소리가 커진다.
  (경로2) 교란 — 금액이 term(36/60)이나 부도 확률과 상관돼 있다.
이 스크립트는 두 경로를 각각 분리해 확인한다.

[검증 항목]
  A. 등급별 대출 규모 프로파일 (금액, 60개월 비중)
  B. 금액 효과의 직접 측정 = 가중평균 spread − 단순평균 spread
  C. 등급 내부에서 금액과 spread / 부도의 관계
  D. 금액 5분위 x 등급 교차표
  E. 핵심 — 등급 내 선별 Sharpe를 '비가중'으로 다시 계산해
     B~D 우위가 유지되는지 (유지되면 가중 탓이 아님)
  F. term(36/60) 분해

[분할 오염 방지]
이 분석은 train + val 만 사용한다. test는 건드리지 않는다.
(ANALYSIS_SPLIT 로 전환 가능)

실행: python check_loan_size_effect.py
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
Y_PATH = os.path.join(BASE, "y_table.csv")
RAW_PATH = os.path.join(BASE, "자료_데이터", "lending_club_2020_train.csv")
SPLIT_PATH = os.path.join(BASE, "splited dataset", "split_assignment.csv")

ANALYSIS_SPLIT = ["train", "val"]     # test 제외
GRADES = list("ABCDEFG")
MID = list("BCD")                      # '중간층' 정의


def sharpe(spread, weight=None):
    """weight=None 이면 비가중(단순평균 / 표준편차)"""
    s = np.asarray(spread, dtype=float)
    if len(s) < 2:
        return np.nan
    den = s.std(ddof=1)
    if not den or np.isnan(den):
        return np.nan
    num = s.mean() if weight is None else np.average(s, weights=np.asarray(weight, float))
    return float(num) / float(den)


def wmean(spread, weight):
    return float(np.average(np.asarray(spread, float), weights=np.asarray(weight, float)))


# ---------------------------------------------------------------- #
def load():
    print("[1/3] y_table + split 적재")
    y = pd.read_csv(Y_PATH)
    sp = pd.read_csv(SPLIT_PATH)
    y = y.merge(sp, on="id", how="left", validate="one_to_one")

    before = len(y)
    y = y.loc[y["split"].isin(ANALYSIS_SPLIT)].reset_index(drop=True)
    print(f"      {before:,}건 -> {len(y):,}건 ({'+'.join(ANALYSIS_SPLIT)}만 사용, test 제외)")

    print("[2/3] 원본에서 grade 추출 (청크)")
    want = set(y["id"].astype("int64").tolist())
    parts = []
    for ch in pd.read_csv(RAW_PATH, usecols=["id", "grade"],
                          dtype={"id": str, "grade": str}, chunksize=200_000):
        ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
        ch = ch.loc[ch["id"].notna()].copy()
        ch["id"] = ch["id"].astype("int64")
        ch = ch.loc[ch["id"].isin(want)]
        if len(ch):
            parts.append(ch)
    raw = pd.concat(parts, ignore_index=True)
    del parts

    print("[3/3] 병합 및 스케일 정리")
    df = y.merge(raw, on="id", how="left", validate="one_to_one")
    df = df.loc[df["grade"].notna()].reset_index(drop=True)

    # spread 스케일 자동 판별 (소수 -> %)
    if float(df["spread"].abs().median()) < 1.5:
        df["spread_pct"] = df["spread"] * 100.0
    else:
        df["spread_pct"] = df["spread"]

    df["grade"] = pd.Categorical(df["grade"], categories=GRADES, ordered=True)
    df = df.loc[df["grade"].notna()].reset_index(drop=True)
    print(f"      분석 대상 {len(df):,}건")
    return df


# ---------------------------------------------------------------- #
# A. 등급별 대출 규모 프로파일
#    "등급이 애매한 사람이 더 크게 빌렸나"에 대한 1차 답
# ---------------------------------------------------------------- #
def profile_by_grade(df):
    rows = []
    for g in GRADES:
        s = df.loc[df["grade"] == g]
        rows.append({
            "등급": g,
            "건수": len(s),
            "평균대출액": round(float(s["funded_amnt"].mean()), 0),
            "중앙값대출액": round(float(s["funded_amnt"].median()), 0),
            "60개월비중(%)": round(float((s["term_n"] == 60).mean()) * 100, 1),
            "금액점유율(%)": round(float(s["funded_amnt"].sum() / df["funded_amnt"].sum()) * 100, 2),
            "건수점유율(%)": round(len(s) / len(df) * 100, 2),
        })
    out = pd.DataFrame(rows)
    # 금액점유율 / 건수점유율 > 1 이면 그 등급이 '금액 기준으로 과대표집'
    out["금액/건수 비율"] = (out["금액점유율(%)"] / out["건수점유율(%)"]).round(3)
    return out


# ---------------------------------------------------------------- #
# B. 금액 가중이 spread 평균을 얼마나 밀어올리는가 (경로1의 크기)
# ---------------------------------------------------------------- #
def weight_effect(df):
    rows = []
    for g in GRADES:
        s = df.loc[df["grade"] == g]
        w = wmean(s["spread_pct"], s["funded_amnt"])
        u = float(s["spread_pct"].mean())
        rows.append({
            "등급": g,
            "가중평균spread(%)": round(w, 3),
            "단순평균spread(%)": round(u, 3),
            "가중효과(%p)": round(w - u, 3),      # 양수면 큰 대출이 더 좋은 성과
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
# C. 등급 내부에서 금액과 결과의 관계 (경로2)
# ---------------------------------------------------------------- #
def within_grade_corr(df):
    rows = []
    for g in GRADES:
        s = df.loc[df["grade"] == g]
        ok = s.loc[s["bad"] == 0]
        rows.append({
            "등급": g,
            "corr(금액, spread)": round(float(s["funded_amnt"].corr(s["spread_pct"])), 4),
            "corr(금액, 부도)": round(float(s["funded_amnt"].corr(s["bad"].astype(float))), 4),
            "corr(금액, spread) 정상건만": round(float(ok["funded_amnt"].corr(ok["spread_pct"])), 4)
            if len(ok) > 2 else np.nan,
            "부도건 평균대출액": round(float(s.loc[s["bad"] == 1, "funded_amnt"].mean()), 0),
            "정상건 평균대출액": round(float(ok["funded_amnt"].mean()), 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
# D. 금액 5분위 x 등급 교차 (등급을 통제한 상태에서 금액 효과)
# ---------------------------------------------------------------- #
def size_quintile(df):
    d = df.copy()
    d["금액5분위"] = pd.qcut(d["funded_amnt"], 5, labels=["Q1(소액)", "Q2", "Q3", "Q4", "Q5(고액)"])
    rows = []
    for g in GRADES:
        s = d.loc[d["grade"] == g]
        for q in ["Q1(소액)", "Q2", "Q3", "Q4", "Q5(고액)"]:
            sub = s.loc[s["금액5분위"] == q]
            if len(sub) < 30:
                continue
            rows.append({
                "등급": g, "금액5분위": q, "건수": len(sub),
                "평균대출액": round(float(sub["funded_amnt"].mean()), 0),
                "부도율(%)": round(float(sub["bad"].mean()) * 100, 2),
                "단순평균spread(%)": round(float(sub["spread_pct"].mean()), 2),
                "60개월비중(%)": round(float((sub["term_n"] == 60).mean()) * 100, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
# E. 핵심 — 등급 내 선별 Sharpe를 가중 / 비가중 양쪽으로
#    비가중에서도 중간층이 이기면 '가중 탓'이 아니다.
# ---------------------------------------------------------------- #
def within_grade_selection(df):
    rows = []
    for g in GRADES:
        s = df.loc[df["grade"] == g].sort_values("spread_pct", ascending=False)
        if len(s) == 0:
            continue
        for p in [1.00, 0.60, 0.40, 0.20]:
            n = max(int(len(s) * p), 1)
            sel = s.iloc[:n]
            rows.append({
                "등급": g,
                "등급내선택률(%)": int(p * 100),
                "건수": n,
                "부도율(%)": round(float(sel["bad"].mean()) * 100, 2),
                "가중Sharpe": round(sharpe(sel["spread_pct"], sel["funded_amnt"]), 3),
                "비가중Sharpe": round(sharpe(sel["spread_pct"]), 3),
                "차이": round(sharpe(sel["spread_pct"], sel["funded_amnt"])
                             - sharpe(sel["spread_pct"]), 3),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
# F. term 분해 — 금액과 term이 얽혀 있으므로 따로 본다
# ---------------------------------------------------------------- #
def by_term(df):
    rows = []
    for g in GRADES:
        for t in [36, 60]:
            s = df.loc[(df["grade"] == g) & (df["term_n"] == t)]
            if len(s) < 30:
                continue
            rows.append({
                "등급": g, "term": t, "건수": len(s),
                "평균대출액": round(float(s["funded_amnt"].mean()), 0),
                "부도율(%)": round(float(s["bad"].mean()) * 100, 2),
                "단순평균spread(%)": round(float(s["spread_pct"].mean()), 2),
                "비가중Sharpe": round(sharpe(s["spread_pct"]), 3),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
def main():
    df = load()

    prof = profile_by_grade(df)
    weff = weight_effect(df)
    corr = within_grade_corr(df)
    quint = size_quintile(df)
    within = within_grade_selection(df)
    term = by_term(df)

    for name, t in [("대출액_등급별프로파일", prof), ("대출액_가중효과", weff),
                    ("대출액_등급내상관", corr), ("대출액_5분위교차", quint),
                    ("대출액_등급내선별_가중비교", within), ("대출액_term분해", term)]:
        t.to_csv(os.path.join(BASE, f"{name}.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 200)

    def show(title, t):
        print("\n" + "=" * 90); print(title); print("=" * 90)
        print(t.to_string(index=False))

    show("[A] 등급별 대출 규모 — 애매한 등급이 더 크게 빌렸는가", prof)
    show("[B] 금액 가중이 평균 spread를 밀어올리는 크기 (경로1)", weff)
    show("[C] 등급 내부에서 금액과 결과의 상관 (경로2)", corr)
    show("[D] 금액 5분위 x 등급", quint)
    show("[E] 등급 내 선별 Sharpe — 가중 vs 비가중  ★핵심", within)
    show("[F] term 분해", term)

    # ---------------- 판정 ----------------
    print("\n" + "=" * 90)
    print("[판정]")
    print("=" * 90)

    max_weff = weff["가중효과(%p)"].abs().max()
    print(f"  · 금액 가중이 평균 spread를 바꾸는 폭(최대) : {max_weff:.3f}%p")
    if max_weff < 1.0:
        print("    -> 미미하다. 큰 대출이 특별히 잘하거나 못하지 않는다.")
    else:
        print("    -> 무시할 수 없다. 가중 방식이 결과에 개입하고 있다.")

    sel40 = within.loc[within["등급내선택률(%)"] == 40]
    if len(sel40):
        best_w = sel40.loc[sel40["가중Sharpe"].idxmax(), "등급"]
        best_u = sel40.loc[sel40["비가중Sharpe"].idxmax(), "등급"]
        print(f"\n  · 등급내 40% 선별 시 최고 등급 — 가중기준: {best_w} / 비가중기준: {best_u}")
        if best_w == best_u:
            print("    -> 가중을 빼도 승자가 같다. 중간층 우위는 가중의 부산물이 아니다.")
        else:
            print("    -> 가중 여부에 따라 승자가 바뀐다. Sharpe 정의를 재검토해야 한다.")

    mid_u = sel40.loc[sel40["등급"].isin(MID), "비가중Sharpe"].max()
    a_u = sel40.loc[sel40["등급"] == "A", "비가중Sharpe"].max()
    print(f"\n  · 비가중 기준 중간층(B~D) 최고 Sharpe : {mid_u}")
    print(f"  · 비가중 기준 A등급 Sharpe            : {a_u}")
    print("    -> 중간층 우위 " + ("유지됨" if mid_u > a_u else "사라짐"))

    print("\n주의: train+val 만 사용. test 미사용.")
    print("      선별은 정답을 아는 oracle 기준이라 상한선이며 실제 모델 성능이 아니다.")
    print("=" * 90)


if __name__ == "__main__":
    main()
