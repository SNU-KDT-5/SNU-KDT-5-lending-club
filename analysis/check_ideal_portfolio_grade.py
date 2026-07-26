#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이상적 포트폴리오의 등급 구성 검증
==================================

[이 스크립트가 답하려는 질문]

우리 팀의 결론 서사는 "LC가 평균으로 뭉뚱그려 등급을 매긴 저등급 집단 안에서,
실제로 상환할 개인을 골라낸다"이다.

이 서사가 참이라면, 정답을 다 아는 상태에서 고른 최선의 포트폴리오(= spread 상위 X%)는
고금리(D~G) 대출로 상당 부분 채워져 있어야 한다. 저등급은 상환만 하면 A등급의 3배를
벌어주기 때문이다.

반대로 최선의 포트폴리오가 A·B로만 채워져 있다면, 우리 서사는 데이터의 지지를 못 받는다.
그 경우 "A·B만 승인" baseline을 이기기 어렵고, 결론을 수정해야 한다.

즉 이 스크립트는 우리 서사를 **반증할 기회를 주는** 코드다.

[검증 항목]
  1. spread 상위 X%의 등급 분포 (전체 분포 대비 lift)
  2. 상위 X%의 부도율 / 평균 spread / Sharpe
  3. baseline 전략 비교 (전부 승인 / A·B만 / A만 / D~G만)
  4. 핵심: 등급 '안에서' 골라내기가 통하는가
     — 각 등급 안에서 정답 기준 상위 p%를 샀을 때 Sharpe가 A등급 전량매수를 넘는가

[주의] 여기서 쓰는 '이상적 포트폴리오'는 미래를 다 아는 상태의 상한선(oracle)이다.
      실제 모델이 도달할 성능이 아니라, "이 방향에 먹을 것이 있는가"를 보는 용도.

실행: python check_ideal_portfolio_grade.py
출력: 이상포트폴리오_등급구성.csv/.png, 이상포트폴리오_선택률별성과.csv,
      등급내선별_샤프.csv, baseline_전략비교.csv, 콘솔 요약
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 폰트 (macOS)
for _f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]:
    try:
        matplotlib.rc("font", family=_f)
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
Y_PATH = os.path.join(BASE, "y_table.csv")
RAW_PATH = os.path.join(BASE, "자료_데이터", "lending_club_2020_train.csv")
SPLIT_PATH = os.path.join(BASE, "splited dataset", "split_assignment.csv")

# test 오염 방지 — 탐색은 train+val 로만 한다.
# 전체를 보고 싶으면 ["train", "val", "test"] 로 바꾼다.
ANALYSIS_SPLIT = ["train", "val"]

SELECT_RATES = [0.05, 0.10, 0.20, 0.30, 0.50]   # 상위 몇 %를 살 것인가
MAIN_RATE = 0.20                                 # 본문에서 주로 인용할 선택률
GRADES = list("ABCDEFG")


# ---------------------------------------------------------------- #
# Sharpe 정의 (팀 문서 정의를 그대로 따름)
#   분자 = 대출금액(funded_amnt)으로 가중한 평균 spread
#   분모 = spread의 단순 표준편차 (가중하지 않음)
# ---------------------------------------------------------------- #
def sharpe(spread: pd.Series, weight: pd.Series) -> float:
    if len(spread) < 2:
        return np.nan
    w = weight.to_numpy(dtype=float)
    s = spread.to_numpy(dtype=float)
    if w.sum() <= 0:
        return np.nan
    den = s.std(ddof=1)
    if not den or np.isnan(den):
        return np.nan
    return float(np.average(s, weights=w)) / den


def wmean(spread: pd.Series, weight: pd.Series) -> float:
    if len(spread) == 0:
        return np.nan
    return float(np.average(spread.to_numpy(dtype=float),
                            weights=weight.to_numpy(dtype=float)))


# ---------------------------------------------------------------- #
# 1. 데이터 적재 — y_table(81만행)에 원본의 grade / int_rate만 붙인다.
#    원본은 175만행 x 141열이라 usecols로 5개만 읽어 메모리를 아낀다.
# ---------------------------------------------------------------- #
def load() -> pd.DataFrame:
    print("[1/5] y_table 적재")
    y = pd.read_csv(Y_PATH)
    print(f"      y_table: {y.shape[0]:,}행 x {y.shape[1]}열")

    # 분할 배정을 붙여 test를 잘라낸다 (탐색 단계에서 test를 보지 않기 위함)
    sp = pd.read_csv(SPLIT_PATH)
    y = y.merge(sp, on="id", how="left", validate="one_to_one")
    before = len(y)
    y = y.loc[y["split"].isin(ANALYSIS_SPLIT)].reset_index(drop=True)
    print(f"      {before:,}건 -> {len(y):,}건 ({'+'.join(ANALYSIS_SPLIT)}만 사용)")

    print("[2/5] 원본에서 grade / sub_grade / int_rate 추출 (청크 처리)")
    # 원본은 175만행 x 141열이라 통째로 읽으면 메모리가 터진다.
    # usecols로 4개 열만, chunksize로 나눠 읽고, y_table에 있는 id만 남긴다.
    #
    # 또한 원본 956,843번째 행 부근에 'Loans that do not meet the credit policy'
    # 텍스트 행이 섞여 있다. id를 문자열로 읽은 뒤 숫자 변환에 실패한 행을 버린다.
    want_ids = set(y["id"].astype("int64").tolist())
    parts, bad_rows = [], 0

    for ch in pd.read_csv(
        RAW_PATH,
        usecols=["id", "grade", "sub_grade", "int_rate"],
        dtype={"id": str, "grade": str, "sub_grade": str, "int_rate": str},
        chunksize=200_000,
    ):
        ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
        bad_rows += int(ch["id"].isna().sum())
        ch = ch.loc[ch["id"].notna()].copy()
        ch["id"] = ch["id"].astype("int64")
        ch = ch.loc[ch["id"].isin(want_ids)]          # 필요한 행만 보관
        if len(ch):
            parts.append(ch)

    if bad_rows:
        print(f"      텍스트/불량 행 {bad_rows}건 제거")
    raw = pd.concat(parts, ignore_index=True)          # 인덱스 구멍 방지
    del parts
    raw["grade"] = raw["grade"].astype("category")
    print(f"      매칭된 원본 행: {len(raw):,}건")

    print("[3/5] 병합")
    df = y.merge(raw, on="id", how="left", validate="one_to_one")

    miss = int(df["grade"].isna().sum())
    if miss:
        print(f"      경고: grade 결측 {miss:,}건 -> 제외")
        df = df.loc[df["grade"].notna()].reset_index(drop=True)

    # int_rate가 '13.56%' 형태의 문자열일 수 있으므로 방어적으로 처리
    if not pd.api.types.is_numeric_dtype(df["int_rate"]):
        df["int_rate"] = (df["int_rate"].astype(str)
                          .str.replace("%", "", regex=False).str.strip())
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    # spread가 소수(0.06)인지 퍼센트(6.0)인지 자동 판별해 %로 통일
    med_abs = float(df["spread"].abs().median())
    if med_abs < 1.5:
        df["spread_pct"] = df["spread"] * 100.0
        print(f"      spread를 소수로 판정 -> 100배해 %로 통일 (중앙값 |spread|={med_abs:.4f})")
    else:
        df["spread_pct"] = df["spread"]
        print(f"      spread를 %로 판정 (중앙값 |spread|={med_abs:.4f})")

    df["grade"] = pd.Categorical(df["grade"], categories=GRADES, ordered=True)
    df = df.loc[df["grade"].notna()].reset_index(drop=True)

    print(f"      최종 분석 대상: {df.shape[0]:,}건")
    return df


# ---------------------------------------------------------------- #
# 2. 이상적 포트폴리오(oracle) 상위 X%의 등급 구성
# ---------------------------------------------------------------- #
def ideal_grade_mix(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] 이상적 포트폴리오 등급 구성 계산")

    # 전체(= '전부 승인' baseline)의 등급 분포
    base_share = df["grade"].value_counts(normalize=True).reindex(GRADES).fillna(0.0)

    rows = []
    ranked = df.sort_values("spread_pct", ascending=False).reset_index(drop=True)

    for rate in SELECT_RATES:
        n = int(len(ranked) * rate)
        top = ranked.iloc[:n]
        share = top["grade"].value_counts(normalize=True).reindex(GRADES).fillna(0.0)

        for g in GRADES:
            sub = top.loc[top["grade"] == g]
            rows.append({
                "선택률(%)": round(rate * 100, 1),
                "등급": g,
                "선택건수": int(len(sub)),
                "포트폴리오내비중(%)": round(float(share[g]) * 100, 2),
                "전체내비중(%)": round(float(base_share[g]) * 100, 2),
                # lift > 1 이면 그 등급이 최선의 포트폴리오에서 '과대표집' 됐다는 뜻
                "lift(배)": round(float(share[g] / base_share[g]), 2) if base_share[g] > 0 else np.nan,
                "평균spread(%)": round(wmean(sub["spread_pct"], sub["funded_amnt"]), 2) if len(sub) else np.nan,
            })

    return pd.DataFrame(rows)


def ideal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """선택률별 전체 성과 + 저등급(D~G) 비중"""
    ranked = df.sort_values("spread_pct", ascending=False).reset_index(drop=True)
    out = []
    for rate in SELECT_RATES:
        n = int(len(ranked) * rate)
        top = ranked.iloc[:n]
        out.append({
            "선택률(%)": round(rate * 100, 1),
            "건수": n,
            "부도율(%)": round(float(top["bad"].mean()) * 100, 2),
            "평균spread(%)": round(wmean(top["spread_pct"], top["funded_amnt"]), 2),
            "변동성": round(float(top["spread_pct"].std(ddof=1)), 2),
            "Sharpe": round(sharpe(top["spread_pct"], top["funded_amnt"]), 3),
            "A·B비중(%)": round(float(top["grade"].isin(list("AB")).mean()) * 100, 1),
            "D~G비중(%)": round(float(top["grade"].isin(list("DEFG")).mean()) * 100, 1),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------- #
# 3. 핵심 검증 — 등급 '안에서' 골라내기가 통하는가
#    각 등급 안에서 정답 기준 상위 p%만 샀을 때의 Sharpe를 구해
#    A등급 전량매수 Sharpe를 넘는지 본다.
# ---------------------------------------------------------------- #
def within_grade_selection(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] 등급 내부 선별 시뮬레이션")
    rows = []
    for g in GRADES:
        sub = df.loc[df["grade"] == g].sort_values("spread_pct", ascending=False)
        if len(sub) == 0:
            continue
        for p in [1.00, 0.80, 0.60, 0.40, 0.20]:
            n = max(int(len(sub) * p), 1)
            sel = sub.iloc[:n]
            rows.append({
                "등급": g,
                "등급내선택률(%)": int(p * 100),
                "건수": n,
                "부도율(%)": round(float(sel["bad"].mean()) * 100, 2),
                "평균spread(%)": round(wmean(sel["spread_pct"], sel["funded_amnt"]), 2),
                "변동성": round(float(sel["spread_pct"].std(ddof=1)), 3),
                "Sharpe": round(sharpe(sel["spread_pct"], sel["funded_amnt"]), 3),
            })
    return pd.DataFrame(rows)


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """비교용 baseline 전략"""
    rows = []
    for name, mask in [
        ("전부 승인", pd.Series(True, index=df.index)),
        ("A·B만 승인", df["grade"].isin(list("AB"))),
        ("A만 승인", df["grade"] == "A"),
        ("D~G만 승인", df["grade"].isin(list("DEFG"))),
    ]:
        sub = df.loc[mask]
        rows.append({
            "전략": name,
            "건수": len(sub),
            "부도율(%)": round(float(sub["bad"].mean()) * 100, 2),
            "평균spread(%)": round(wmean(sub["spread_pct"], sub["funded_amnt"]), 2),
            "변동성": round(float(sub["spread_pct"].std(ddof=1)), 3),
            "Sharpe": round(sharpe(sub["spread_pct"], sub["funded_amnt"]), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #
def plot(mix, summ, within, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # ① 상위 20% 등급 구성 vs 전체 등급 구성
    m = mix.loc[mix["선택률(%)"] == MAIN_RATE * 100]
    x = np.arange(len(GRADES))
    axes[0].bar(x - 0.2, m["전체내비중(%)"], 0.4, label="전체(전부 승인)", color="#b0b0b0")
    axes[0].bar(x + 0.2, m["포트폴리오내비중(%)"], 0.4,
                label=f"이상적 상위 {int(MAIN_RATE*100)}%", color="#2b6cb0")
    axes[0].set_xticks(x); axes[0].set_xticklabels(GRADES)
    axes[0].set_ylabel("비중 (%)")
    axes[0].set_title("① 최선의 포트폴리오는 어느 등급으로 채워지는가")
    axes[0].legend()

    # ② 선택률별 등급 쏠림
    axes[1].plot(summ["선택률(%)"], summ["D~G비중(%)"], "o-", color="#c53030", label="D~G 비중")
    axes[1].plot(summ["선택률(%)"], summ["A·B비중(%)"], "o-", color="#2b6cb0", label="A·B 비중")
    axes[1].set_xlabel("선택률 (%)"); axes[1].set_ylabel("포트폴리오 내 비중 (%)")
    axes[1].set_title("② 선택을 좁힐수록 저등급이 늘어나는가")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # ③ 등급 내부 선별 시 Sharpe
    for g in GRADES:
        sub = within.loc[within["등급"] == g].sort_values("등급내선택률(%)")
        axes[2].plot(sub["등급내선택률(%)"], sub["Sharpe"], "o-", label=g)
    a_full = within.loc[(within["등급"] == "A") & (within["등급내선택률(%)"] == 100), "Sharpe"]
    if len(a_full):
        axes[2].axhline(float(a_full.iloc[0]), ls="--", c="k", lw=1, label="A등급 전량매수")
    axes[2].set_xlabel("등급 내 선택률 (%)"); axes[2].set_ylabel("Sharpe")
    axes[2].set_title("③ 등급 안에서 골라내면 A등급을 이기는가")
    axes[2].legend(fontsize=8, ncol=2); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    print(f"      그림 저장: {os.path.basename(path)}")


# ---------------------------------------------------------------- #
def main():
    df = load()

    mix = ideal_grade_mix(df)
    summ = ideal_summary(df)
    within = within_grade_selection(df)
    base = baseline_table(df)

    mix.to_csv(os.path.join(BASE, "이상포트폴리오_등급구성.csv"), index=False, encoding="utf-8-sig")
    summ.to_csv(os.path.join(BASE, "이상포트폴리오_선택률별성과.csv"), index=False, encoding="utf-8-sig")
    within.to_csv(os.path.join(BASE, "등급내선별_샤프.csv"), index=False, encoding="utf-8-sig")
    base.to_csv(os.path.join(BASE, "baseline_전략비교.csv"), index=False, encoding="utf-8-sig")
    plot(mix, summ, within, os.path.join(BASE, "이상포트폴리오_등급구성.png"))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("\n" + "=" * 78)
    print(f"[A] 이상적 포트폴리오 상위 {int(MAIN_RATE*100)}%의 등급 구성")
    print("=" * 78)
    print(mix.loc[mix["선택률(%)"] == MAIN_RATE * 100].to_string(index=False))

    print("\n" + "=" * 78)
    print("[B] 선택률별 성과와 등급 쏠림")
    print("=" * 78)
    print(summ.to_string(index=False))

    print("\n" + "=" * 78)
    print("[C] baseline 전략 비교")
    print("=" * 78)
    print(base.to_string(index=False))

    print("\n" + "=" * 78)
    print("[D] 등급 내부 선별 시 Sharpe (oracle 상한)")
    print("=" * 78)
    print(within.to_string(index=False))

    # ---------------- 판정 ----------------
    main_row = summ.loc[summ["선택률(%)"] == MAIN_RATE * 100].iloc[0]
    low_share = float(main_row["D~G비중(%)"])
    base_low = float(mix.loc[(mix["선택률(%)"] == MAIN_RATE * 100) &
                             (mix["등급"].isin(list("DEFG"))), "전체내비중(%)"].sum())

    print("\n" + "=" * 78)
    print("[판정] 우리 서사가 데이터의 지지를 받는가")
    print("=" * 78)
    print(f"  · 최선의 포트폴리오 중 D~G 비중 : {low_share:.1f}%")
    print(f"  · 전체 대출 중 D~G 비중        : {base_low:.1f}%")
    if low_share > base_low * 1.1:
        print("  -> 지지받음. 최선의 선택은 저등급을 '더' 담는다.")
        print("     즉 '저등급 중 상환할 사람 골라내기'에 실제로 먹을 것이 있다.")
    elif low_share < base_low * 0.9:
        print("  -> 지지받지 못함. 최선의 선택은 오히려 저등급을 피한다.")
        print("     결론 서사를 수정해야 한다.")
    else:
        print("  -> 중립. 등급보다 등급 내 개별 편차가 더 중요하다는 뜻일 수 있다.")

    a_full = float(base.loc[base["전략"] == "A만 승인", "Sharpe"].iloc[0])
    best_low = float(within.loc[within["등급"].isin(list("DEFG")), "Sharpe"].max())
    print(f"\n  · A등급 전량매수 Sharpe        : {a_full}")
    print(f"  · D~G 등급 내 선별 최고 Sharpe : {best_low}  (oracle 상한)")
    if best_low > a_full:
        print("  -> 저등급 선별의 '천장'이 A등급 전량매수보다 높다. 방향은 유효.")
    else:
        print("  -> 저등급은 완벽히 골라내도 A등급을 못 이긴다. Sharpe 분모 때문.")
        print("     이 경우 우위의 원천을 'A·B 내부 선별'로 재설정해야 한다.")

    print("\n주의: 위 수치는 정답을 다 아는 oracle 기준 상한선이다.")
    print("      실제 모델 성능이 아니라 '이 방향에 먹을 것이 있는가'의 판단 재료다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
