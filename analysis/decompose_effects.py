# %% [markdown]
# # 효과 분해 — 모델 때문인가, 변수 때문인가
#
# ## 무엇을 "빼는" 것인가
#
# **데이터에서 변수를 빼는 게 아닙니다. 샤프 숫자끼리 뺍니다.**
#
# 세 번 학습해서 샤프 3개를 얻은 뒤, 그 스칼라 3개로 산수를 합니다.
#
# | 실험 | 모델 | 변수 | 결과 |
# |---|---|---|---|
# | ① | 선형 | 47열 | S₁ |
# | ② | 트리 | **47열** | S₂ |
# | ③ | 트리 | 122열 | S₃ |
#
# ```
# 모델 효과 = S₂ − S₁     ← 변수를 47로 고정. 달라진 건 모델뿐
# 변수 효과 = S₃ − S₂     ← 모델을 트리로 고정. 달라진 건 변수뿐
# 전체 격차 = S₃ − S₁ = 모델 효과 + 변수 효과   (항등식)
# ```
#
# ②번 실험을 위해 **트리 데이터에서 47열만 골라내는 것**이 유일한 "빼기"이고,
# 그건 학습 전 준비 단계입니다. 분해 자체는 뺄셈 두 번이 전부입니다.
#
# ## 사용법
# 각 실험의 **val 예측값**만 있으면 됩니다.
#
# ```python
# from 실험_효과분해 import decompose
# decompose(pred_lin47, pred_tree47, pred_tree122, y_val, amt=None, pct=0.40)
# ```

# %%
import os
import sys

import numpy as np
import pandas as pd


def sharpe(pred, excess, pct):
    """예측 상위 pct 를 승인했을 때의 동일가중 포트폴리오 샤프.

    거부 건은 초과수익 0(전액 국채)으로 포함한다.
    승인률이 다르면 비교가 불가능하므로 pct 를 반드시 고정한다.
    """
    thr = np.quantile(pred, 1 - pct)
    x = np.where(pred >= thr, excess, 0.0)
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 else float(x.mean() / sd)


def decompose(pred_lin47, pred_tree47, pred_tree122, excess,
              pct=0.40, n_boot=500, seed=42):
    """세 실험의 val 예측값 → 효과 분해표.

    pct : 승인률. 세 실험에 **같은 값**을 써야 한다.
          다르면 샤프 차이가 승인률 차이인지 성능 차이인지 알 수 없다.
    """
    e = np.asarray(excess, dtype=float)
    P = {"①선형×47": np.asarray(pred_lin47, float),
         "②트리×47": np.asarray(pred_tree47, float),
         "③트리×122": np.asarray(pred_tree122, float)}
    for k, v in P.items():
        if len(v) != len(e):
            raise ValueError(f"{k}: 길이 {len(v)} != y {len(e)}")

    S = {k: sharpe(v, e, pct) for k, v in P.items()}
    model_eff = S["②트리×47"] - S["①선형×47"]
    var_eff = S["③트리×122"] - S["②트리×47"]
    total = S["③트리×122"] - S["①선형×47"]

    # 부트스트랩 — 차이가 표본 잡음인지 구분한다.
    # 0.005 수준의 격차는 신뢰구간을 봐야 의미가 있는지 알 수 있다.
    rng = np.random.default_rng(seed)
    n = len(e)
    bm, bv = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        s1 = sharpe(P["①선형×47"][i], e[i], pct)
        s2 = sharpe(P["②트리×47"][i], e[i], pct)
        s3 = sharpe(P["③트리×122"][i], e[i], pct)
        bm.append(s2 - s1)
        bv.append(s3 - s2)
    bm, bv = np.array(bm), np.array(bv)

    print("=" * 62)
    print(f"  효과 분해  (승인률 {pct * 100:.0f}%, 부트스트랩 {n_boot}회)")
    print("=" * 62)
    for k, v in S.items():
        print(f"  {k:<12} 샤프 {v:.4f}")
    print("-" * 62)
    rows = []
    for name, val, b in [("모델 효과 (②−①)", model_eff, bm),
                         ("변수 효과 (③−②)", var_eff, bv)]:
        lo, hi = np.percentile(b, [2.5, 97.5])
        sig = "유의" if lo * hi > 0 else "불확실(0 포함)"
        share = val / total * 100 if total != 0 else np.nan
        print(f"  {name:<16} {val:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  "
              f"기여 {share:5.1f}%  {sig}")
        rows.append({"효과": name, "값": round(val, 4),
                     "CI하한": round(lo, 4), "CI상한": round(hi, 4),
                     "기여율%": round(share, 1), "판정": sig})
    print("-" * 62)
    print(f"  전체 격차 (③−①) {total:+.4f}   "
          f"(검산: {model_eff:+.4f} {var_eff:+.4f} = {model_eff + var_eff:+.4f})")
    print("=" * 62)

    # 해석 안내
    print("\n[해석]")
    if abs(var_eff) < 1e-9:
        pass
    elif var_eff < 0:
        print("  변수 효과가 음수 → 트리 전용 25개가 오히려 해롭다(과적합 신호).")
        print("  트리도 47열로 가는 것을 검토할 것.")
    elif abs(model_eff) > abs(var_eff) * 2:
        print("  모델 효과가 지배적 → 트리의 비선형·상호작용이 실제로 기여.")
        print("  '변수를 더 줘서 이긴 것'이라는 반론을 방어할 수 있다.")
    elif abs(var_eff) > abs(model_eff) * 2:
        print("  변수 효과가 지배적 → 이긴 건 모델이 아니라 정보량이다.")
        print("  '트리가 더 낫다'고 쓰면 안 된다. 선형에도 변수를 더 줘야 공정하다.")
    else:
        print("  둘 다 기여 → 두 요인을 순서대로 서술할 것.")

    return pd.DataFrame(rows)


# %%
# ─────────────────────────────────────────────────────────────
# ②번 실험 준비 — 트리 데이터에서 47열만 골라내기
# ─────────────────────────────────────────────────────────────
# 이게 유일한 "빼기"다. 학습 전 준비 단계일 뿐이다.
def make_controlled_X(tree_csv, common_csv):
    """model_input_tree.csv → 선형용 47열만 남긴 통제용 X"""
    common = pd.read_csv(common_csv)["열"].tolist()
    df = pd.read_csv(tree_csv)
    NOT_FEATURE = ["id", "split", "grade", "term_n", "funded_amnt", "rf", "irr",
                   "excess", "excess_contract", "spread_positive", "bad"]
    keep = [c for c in common if c in df.columns]
    missing = set(common) - set(df.columns)
    if missing:
        raise KeyError(f"트리 데이터에 없는 공통열: {sorted(missing)}")
    print(f"통제용 X: {len(keep)}열 (전체 {df.shape[1] - len(NOT_FEATURE)}열에서 축소)")
    return df[keep], df[[c for c in NOT_FEATURE if c in df.columns]]


# %%
if __name__ == "__main__":
    # 자체 점검 — 가짜 예측값으로 항등식이 성립하는지 확인
    rng = np.random.default_rng(0)
    n = 20_000
    e = rng.normal(0.02, 0.27, n)
    base = e + rng.normal(0, 0.5, n)          # 약한 신호
    p1 = base + rng.normal(0, 0.35, n)        # 선형: 잡음 큼
    p2 = base + rng.normal(0, 0.25, n)        # 트리 47: 잡음 중간
    p3 = base + rng.normal(0, 0.20, n)        # 트리 122: 잡음 작음
    out = decompose(p1, p2, p3, e, pct=0.40, n_boot=200)
    print("\n[자체 점검] 항등식 성립 확인 완료")
