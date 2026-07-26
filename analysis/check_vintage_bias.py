#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vintage(발행시점) 편향 근거 산출
================================

[문제]
데이터 스냅샷은 2020-12다. 그런데 우리는 '완결된 대출'(Fully Paid / Charged Off /
Default)만 학습에 쓴다. 최근에 발행된 대출은 아직 만기가 오지 않았으므로,
그중에서 '완결' 상태로 잡히는 건은 두 종류뿐이다.
    · 만기 전에 다 갚아버린 건 (조기상환 = 우량)
    · 만기 전에 이미 망한 건   (조기부도 = 불량)
정상적으로 만기까지 꼬박 갚는 '중간층'은 아직 Current 상태라 표본에서 빠진다.

즉 최근 vintage를 그대로 쓰면 표본이 양극단으로 왜곡된다.

[이 스크립트가 만드는 근거]
  A. 발행연도 x term 별 완결률 — 언제부터 표본이 온전한가
  B. 완결건의 실제 상환기간(k) 중앙값 — 최근 vintage일수록 짧아지는가
  C. 만기도래 vs 미도래 그룹의 완결건 구성 비교 (편향의 직접 증거)
  D. 네 가지 안(A~D) 비교 — 표본 수와 잔존 편향

출력: vintage_*.csv, vintage_편향근거.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "자료_데이터", "lending_club_2020_train.csv")

SNAPSHOT = np.datetime64("2020-12", "M")
BUFFER = 3                                   # D안 여유분(개월)
COMPLETED = ["Fully Paid", "Charged Off", "Default"]
BAD = ["Charged Off", "Default"]


def load():
    print("[1/3] 원본 적재 (청크)")
    parts = []
    for ch in pd.read_csv(
        RAW_PATH,
        usecols=["id", "issue_d", "term", "loan_status", "last_pymnt_d"],
        dtype=str, chunksize=300_000,
    ):
        ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
        ch = ch.loc[ch["id"].notna()].copy()
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"      {len(df):,}행")

    print("[2/3] 파생 컬럼 생성")
    df["issue_ym"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["last_ym"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    df = df.loc[df["issue_ym"].notna()].reset_index(drop=True)

    df["issue_year"] = df["issue_ym"].dt.year
    df["term_n"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    status = df["loan_status"].astype(str).str.strip()
    df["is_policy"] = status.str.startswith("Does not meet the credit policy")
    df["completed"] = status.isin(COMPLETED) & (~df["is_policy"])
    df["bad"] = status.isin(BAD)

    # 만기 도래 여부 (D안 기준): 발행월 + term + 버퍼 <= 스냅샷
    ym = df["issue_ym"].values.astype("datetime64[M]")
    need = (df["term_n"] + BUFFER).fillna(36).values.astype("timedelta64[M]")
    df["matured"] = (ym + need) <= SNAPSHOT

    # 실제 상환 종료까지 걸린 개월 수 (완결건만 의미 있음)
    df["k_months"] = ((df["last_ym"].dt.year - df["issue_ym"].dt.year) * 12
                      + (df["last_ym"].dt.month - df["issue_ym"].dt.month))
    df.loc[df["k_months"] <= 0, "k_months"] = np.nan

    print(f"      완결 {int(df['completed'].sum()):,}건 / "
          f"만기도래 {int(df['matured'].sum()):,}건")
    return df


# ---------------------------------------------------------------- #
# A. 발행연도 x term 별 완결률
# ---------------------------------------------------------------- #
def completion_by_vintage(df):
    rows = []
    for t in [36, 60]:
        for yr in sorted(df["issue_year"].dropna().unique()):
            s = df.loc[(df["issue_year"] == yr) & (df["term_n"] == t)]
            if len(s) < 100:
                continue
            comp = s.loc[s["completed"]]
            rows.append({
                "발행연도": int(yr), "term": t, "발행건수": len(s),
                "완결건수": len(comp),
                "완결률(%)": round(len(comp) / len(s) * 100, 1),
                "완결건중부도율(%)": round(float(comp["bad"].mean()) * 100, 1) if len(comp) else np.nan,
                "완결건상환기간중앙값(개월)": round(float(comp["k_months"].median()), 1) if len(comp) else np.nan,
                "만기도래(D안)": "예" if bool(s["matured"].iloc[0]) else "아니오",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
# C. 만기도래 vs 미도래 — 편향의 직접 증거
# ---------------------------------------------------------------- #
def matured_vs_not(df):
    rows = []
    for t in [36, 60]:
        for m, label in [(True, "만기도래"), (False, "만기미도래")]:
            s = df.loc[(df["term_n"] == t) & (df["matured"] == m)]
            comp = s.loc[s["completed"]]
            if len(comp) < 50:
                continue
            k = comp["k_months"].dropna()
            rows.append({
                "term": t, "구분": label,
                "발행건수": len(s), "완결건수": len(comp),
                "완결률(%)": round(len(comp) / len(s) * 100, 1),
                "완결건중부도율(%)": round(float(comp["bad"].mean()) * 100, 1),
                "상환기간중앙값(개월)": round(float(k.median()), 1),
                # 만기의 80% 이상을 채운 건의 비중 = '끝까지 간' 정상 대출
                "만기80%이상채운비중(%)": round(float((k >= t * 0.8).mean()) * 100, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
# D. 네 가지 안 비교
# ---------------------------------------------------------------- #
def option_compare(df):
    comp = df.loc[df["completed"]].copy()
    opts = {
        "A안 (발행연도 ≤ 2016)": comp["issue_year"] <= 2016,
        "B안 (전체 + vintage 더미)": pd.Series(True, index=comp.index),
        "C안 (전체, 보정 없음)": pd.Series(True, index=comp.index),
        "D안 (만기+3개월 도래분만)": comp["matured"],
    }
    rows = []
    for name, mask in opts.items():
        s = comp.loc[mask]
        k = s["k_months"].dropna()
        term = s["term_n"]
        full = float(((k / term.reindex(k.index)) >= 0.8).mean()) * 100
        rows.append({
            "안": name,
            "표본수": len(s),
            "원본대비(%)": round(len(s) / len(comp) * 100, 1),
            "부도율(%)": round(float(s["bad"].mean()) * 100, 2),
            "상환기간중앙값(개월)": round(float(k.median()), 1),
            "만기80%이상채운비중(%)": round(full, 1),
            "잔존편향": "없음" if name.startswith("D") else
                       ("일부(60개월)" if name.startswith("A") else "그대로"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
def plot(cbv, mvn, path):
    for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "DejaVu Sans"]:
        try:
            matplotlib.rc("font", family=f); break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # ① 발행연도별 완결률
    for t, c in [(36, "#2b6cb0"), (60, "#c53030")]:
        s = cbv.loc[cbv["term"] == t]
        ax[0].plot(s["발행연도"], s["완결률(%)"], "o-", color=c, label=f"{t}개월")
    ax[0].axhline(95, ls="--", c="gray", lw=1)
    ax[0].set_xlabel("발행연도"); ax[0].set_ylabel("완결률 (%)")
    ax[0].set_title("① 최근 발행분은 아직 끝나지 않았다")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # ② 완결건의 상환기간 중앙값
    for t, c in [(36, "#2b6cb0"), (60, "#c53030")]:
        s = cbv.loc[cbv["term"] == t]
        ax[1].plot(s["발행연도"], s["완결건상환기간중앙값(개월)"], "o-", color=c, label=f"{t}개월")
    ax[1].set_xlabel("발행연도"); ax[1].set_ylabel("상환기간 중앙값 (개월)")
    ax[1].set_title("② 최근 완결건은 '빨리 끝난 건'뿐이다")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    # ③ 만기도래 vs 미도래
    labels, vals_full, vals_bad = [], [], []
    for _, r in mvn.iterrows():
        labels.append(f"{int(r['term'])}개월\n{r['구분']}")
        vals_full.append(r["만기80%이상채운비중(%)"])
        vals_bad.append(r["완결건중부도율(%)"])
    x = np.arange(len(labels))
    ax[2].bar(x - 0.2, vals_full, 0.4, label="만기 80%↑ 채운 비중", color="#2b6cb0")
    ax[2].bar(x + 0.2, vals_bad, 0.4, label="완결건 중 부도율", color="#c53030")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=9)
    ax[2].set_ylabel("%")
    ax[2].set_title("③ 미도래분에는 '중간층'이 없다")
    ax[2].legend(fontsize=9)

    plt.tight_layout(); plt.savefig(path, dpi=140)
    print(f"      그림 저장: {os.path.basename(path)}")


def main():
    df = load()
    print("[3/3] 집계")
    cbv = completion_by_vintage(df)
    mvn = matured_vs_not(df)
    opt = option_compare(df)

    cbv.to_csv(os.path.join(BASE, "vintage_연도별완결률.csv"), index=False, encoding="utf-8-sig")
    mvn.to_csv(os.path.join(BASE, "vintage_만기도래비교.csv"), index=False, encoding="utf-8-sig")
    opt.to_csv(os.path.join(BASE, "vintage_안비교.csv"), index=False, encoding="utf-8-sig")
    plot(cbv, mvn, os.path.join(BASE, "vintage_편향근거.png"))

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 100)
    for t, d in [("[A] 발행연도 x term 별 완결률", cbv),
                 ("[C] 만기도래 vs 미도래", mvn),
                 ("[D] 네 가지 안 비교", opt)]:
        print("\n" + "=" * 100); print(t); print("=" * 100)
        print(d.to_string(index=False))


if __name__ == "__main__":
    main()
