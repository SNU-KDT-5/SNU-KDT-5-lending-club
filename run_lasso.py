

#%%


"""
LASSO 2차 검증 — 변수 선택 + 도메인 판단과 대조

[원칙]
  표준화도 alpha 선택도 전부 train 안에서만 한다.
  전체 데이터로 고르면 test 성능이 낙관적으로 편향된다.

[출력]
  lasso_coefficients.csv    변수별 계수 · 선택 여부
  lasso_vs_domain.csv       도메인 판단 vs LASSO 결과 대조표
  화면에 4분면 요약과 val 성능

[입력]  splited dataset/lasso_train.csv, lasso_val.csv
        lasso_candidate_variables.csv   (도메인 채택 여부 판정용)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# scikit-learn 은 표준 라이브러리가 아니라 별도 설치가 필요하다.
# 없을 때 트레이스백 대신 설치 방법을 안내한다.
try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    print("!! scikit-learn 이 설치되어 있지 않습니다.\n")
    print("   주피터/VS Code 노트북에서:")
    print("       %pip install scikit-learn")
    print("   터미널에서:")
    print("       pip install scikit-learn\n")
    print("   설치 후 커널을 재시작해야 인식됩니다.")
    print("   파이썬 3.13+ 에서 휠을 못 찾는다는 오류가 나면:")
    print('       %pip install --upgrade pip')
    print('       %pip install "scikit-learn>=1.7"')
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────
TARGET = "spread"            # 목표변수. spread_contract 로 바꿔 민감도 확인 가능
SEED = 42
CV_FOLDS = 5
ALPHAS = np.logspace(-5, -1, 40)
TOP_PCT = 0.20               # 포트폴리오 평가 시 상위 몇 %를 '승인'할지

# n_jobs=-1 로 두면 CV 폴드마다 설계행렬(약 0.6GB)을 통째로 복사해
# 메모리를 3GB 넘게 쓴다. 램이 넉넉하면 2~4 로 올려도 된다.
N_JOBS = 1

# y_table 에서 온 컬럼 = 설명변수가 아님. 전부 제외해야 한다.
Y_COLS = ["loan_status", "bad", "issue_month", "term_n", "k_months", "funded_amnt",
          "net_total", "irr_monthly", "irr_annual_raw", "irr_annual", "irr_contract",
          "rf", "spread", "spread_contract", "spread_positive"]
DROP_COLS = Y_COLS + ["id", "zip_code"]


def find_dir(marker, sub=None, max_up=5):
    try:
        start = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start = os.getcwd()
    d = start
    for _ in range(max_up):
        cand = os.path.join(d, sub) if sub else d
        if os.path.exists(os.path.join(cand, marker)):
            return cand
        p = os.path.dirname(d)
        if p == d:
            break
        d = p
    return None


def section(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66, flush=True)


SPLIT_DIR = (find_dir("lasso_train.csv", "splited dataset")
             or find_dir("lasso_train.csv"))
if SPLIT_DIR is None:
    raise SystemExit("lasso_train.csv 를 찾지 못했습니다.")
BASE_DIR = find_dir("lasso_candidate_variables.csv") or SPLIT_DIR
print(f"분할 데이터: {SPLIT_DIR}")


# ─────────────────────────────────────────────────────────
# 1. 로드
# ─────────────────────────────────────────────────────────
section("1. train 로드")

# 메모리 절약: 헤더로 컬럼을 먼저 파악한 뒤 필요한 것만 읽는다.
# (전체를 읽으면 167열 × 48만 행이라 불필요한 y 컬럼까지 메모리에 올라온다)
TR_PATH = os.path.join(SPLIT_DIR, "lasso_train.csv")
header = pd.read_csv(TR_PATH, nrows=0)
X_cols = [c for c in header.columns if c not in DROP_COLS]

tr = pd.read_csv(TR_PATH, usecols=X_cols + [TARGET], low_memory=False)
print(f"train {len(tr):,}행 | 설명변수 {len(X_cols)}개 | 목표변수 {TARGET}")
print(f"  {TARGET} 평균 {tr[TARGET].mean():+.4f} / 표준편차 {tr[TARGET].std():.4f}")

na = tr[X_cols].isna().sum()
train_median = {}
if na.sum():
    print(f"  [주의] 결측 있는 변수 {int((na > 0).sum())}개 → 중앙값 대체")
for c in X_cols:
    m = tr[c].median()
    train_median[c] = m          # val 대체에도 train 중앙값을 써야 누수가 없다
    if tr[c].isna().any():
        tr[c] = tr[c].fillna(m)


# ─────────────────────────────────────────────────────────
# 2. 표준화 + LassoCV (train 에서만)
# ─────────────────────────────────────────────────────────
section("2. LassoCV 적합 — train 에서만")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

scaler = StandardScaler().fit(tr[X_cols])          # train 만!

# dtype 은 반드시 float64 로 둔다.
# float32 로 줄이면 메모리는 절반이지만, LassoCV 가 내부에서 Gram 행렬을
# 미리 계산해 검증할 때 정밀도 부족으로 다음 오류가 난다.
#   ValueError: Gram matrix passed in via 'precompute' parameter did not pass validation
# (플랫폼의 BLAS 구현에 따라 리눅스에서는 통과하고 맥에서는 실패하기도 한다)
Xtr = scaler.transform(tr[X_cols]).astype(np.float64)
ytr = tr[TARGET].to_numpy(dtype=np.float64)
del tr                                   # 설계행렬을 만든 뒤 원본은 즉시 해제
print(f"설계행렬 {Xtr.shape} float64 ({Xtr.nbytes / 1e9:.2f} GB)")

print(f"alpha 후보 {len(ALPHAS)}개 ({ALPHAS.min():.1e} ~ {ALPHAS.max():.1e}), {CV_FOLDS}-fold CV")
print("적합 중... (수십 초)")
# precompute 는 Gram 행렬을 미리 계산해 속도를 크게 올리지만,
# 수치 정밀도 문제로 검증에 실패하는 환경이 있다(맥의 BLAS 등).
#   ValueError: Gram matrix passed in via 'precompute' parameter did not pass validation
# 빠른 경로를 먼저 시도하고, 실패하면 자동으로 안전 경로로 넘어간다.
PRECOMPUTE = "auto"
try:
    model = LassoCV(alphas=ALPHAS, cv=CV_FOLDS, max_iter=5000, precompute=True,
                    n_jobs=N_JOBS, random_state=SEED).fit(Xtr, ytr)
except ValueError as e:
    if "Gram matrix" not in str(e):
        raise
    print("  [알림] Gram 행렬 검증 실패 → precompute=False 로 재시도합니다.")
    print("         (속도는 느려지지만 결과는 동일합니다)")
    PRECOMPUTE = False
    model = LassoCV(alphas=ALPHAS, cv=CV_FOLDS, max_iter=5000, precompute=False,
                    n_jobs=N_JOBS, random_state=SEED).fit(Xtr, ytr)

# LassoCV.alphas_ 는 내림차순, mse_path_ 는 (n_alphas, n_folds)
mse_mean = model.mse_path_.mean(axis=1)
mse_se = model.mse_path_.std(axis=1) / np.sqrt(model.mse_path_.shape[1])
i_min = int(np.argmin(mse_mean))
thr = mse_mean[i_min] + mse_se[i_min]
# 1-SE 규칙: 오차가 최소값의 1 표준오차 안에 드는 것 중 가장 큰 alpha(= 가장 단순한 모형)
cand = np.where(mse_mean <= thr)[0]
i_1se = int(cand[np.argmax(model.alphas_[cand])])

from sklearn.linear_model import Lasso
m_1se = Lasso(alpha=model.alphas_[i_1se], max_iter=5000,
              precompute=(PRECOMPUTE if PRECOMPUTE is False else True),
              random_state=SEED).fit(Xtr, ytr)

n_min = int((model.coef_ != 0).sum())
n_1se = int((m_1se.coef_ != 0).sum())
print(f"\n  alpha_min = {model.alpha_:.6f}  → 선택 {n_min}/{len(X_cols)}개")
print(f"  alpha_1se = {model.alphas_[i_1se]:.6f}  → 선택 {n_1se}/{len(X_cols)}개  (더 단순한 모형)")
print(f"\n  ※ 표본이 {len(ytr):,} 건으로 커서 alpha_min 은 변수를 거의 안 죽인다.")
print("    '어떤 변수가 유용한가'를 보려면 1-SE 기준과 계수 크기를 같이 봐야 한다.")


# ─────────────────────────────────────────────────────────
# 3. 계수 정리
# ─────────────────────────────────────────────────────────
section("3. 변수별 계수")

coef = pd.DataFrame({
    "변수": X_cols,
    "계수": model.coef_,
    "계수_1se": m_1se.coef_,
})
coef["절대계수"] = coef["계수"].abs()
coef["선택_min"] = (coef["계수"] != 0).astype(int)
coef["선택_1se"] = (coef["계수_1se"] != 0).astype(int)
coef = coef.sort_values("절대계수", ascending=False).reset_index(drop=True)
coef.to_csv(os.path.join(BASE_DIR, "lasso_coefficients.csv"),
            index=False, encoding="utf-8-sig")

print("영향력 상위 25개 (표준화 계수 절댓값 기준)")
print(coef.head(25)[["변수", "계수", "선택_1se"]].to_string(index=False))
print(f"\n계수 0 (완전 탈락): {int((coef['계수'] == 0).sum())}개")
if int((coef["계수"] == 0).sum()):
    print("  " + ", ".join(coef.loc[coef["계수"] == 0, "변수"].tolist()[:20]))


# ─────────────────────────────────────────────────────────
# 4. 도메인 판단과 대조 (정합성 검토)
# ─────────────────────────────────────────────────────────
section("4. 도메인 판단 vs LASSO — 정합성 검토")

vpath = os.path.join(BASE_DIR, "lasso_candidate_variables.csv")
if not os.path.exists(vpath):
    print(f"[건너뜀] {os.path.basename(vpath)} 이 없어 대조를 생략합니다.")
else:
    vd = pd.read_csv(vpath)
    vd["도메인채택"] = vd["구분"].astype(str).str.startswith("도메인 채택").astype(int)
    cmp = coef.merge(vd[["변수", "원천변수", "구분", "도메인채택"]], on="변수", how="left")
    cmp["도메인채택"] = cmp["도메인채택"].fillna(0).astype(int)
    cmp["LASSO채택"] = cmp["선택_1se"]

    ct = pd.crosstab(cmp["도메인채택"], cmp["LASSO채택"])
    print("교차표 (행: 도메인 채택 / 열: LASSO 채택, 1-SE 기준)")
    print(ct.to_string())

    both = cmp[(cmp.도메인채택 == 1) & (cmp.LASSO채택 == 1)]
    only_l = cmp[(cmp.도메인채택 == 0) & (cmp.LASSO채택 == 1)]
    only_d = cmp[(cmp.도메인채택 == 1) & (cmp.LASSO채택 == 0)]
    neither = cmp[(cmp.도메인채택 == 0) & (cmp.LASSO채택 == 0)]

    print(f"\n① 양쪽 채택        {len(both):>3}개  → 도메인 판단 확인됨")
    print(f"② LASSO 만 채택    {len(only_l):>3}개  → 우리가 놓친 변수. 되살릴지 검토")
    print(f"③ 도메인만 채택    {len(only_d):>3}개  → LASSO 가 죽인 이유 확인 필요")
    print(f"④ 양쪽 탈락        {len(neither):>3}개")

    if len(only_l):
        print("\n② LASSO 만 채택 — 영향력 순")
        print(only_l.head(20)[["변수", "원천변수", "계수"]].to_string(index=False))
    if len(only_d):
        print("\n③ 도메인만 채택 — 영향력 순 (LASSO 가 0으로 만든 것)")
        print(only_d.head(20)[["변수", "계수", "절대계수"]].to_string(index=False))

    cmp.to_csv(os.path.join(BASE_DIR, "lasso_vs_domain.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n대조표 저장 → lasso_vs_domain.csv")


# ─────────────────────────────────────────────────────────
# 5. val 성능 — 예측력과 포트폴리오
# ─────────────────────────────────────────────────────────
section("5. val 성능")

va = pd.read_csv(os.path.join(SPLIT_DIR, "lasso_val.csv"),
                 usecols=X_cols + [TARGET, "funded_amnt", "bad"], low_memory=False)
for c in X_cols:
    if va[c].isna().any():
        va[c] = va[c].fillna(train_median[c])      # train 중앙값으로 대체(누수 방지)

Xva = scaler.transform(va[X_cols]).astype(np.float64)   # transform 만!
pred = model.predict(Xva)
yva = va[TARGET].to_numpy()

ss_res = float(((yva - pred) ** 2).sum())
ss_tot = float(((yva - yva.mean()) ** 2).sum())
print(f"val R² = {1 - ss_res / ss_tot:.4f}   (spread 는 본래 예측이 어렵다. 낮게 나오는 게 정상)")
print(f"예측-실제 상관 = {np.corrcoef(pred, yva)[0, 1]:.4f}")


def sharpe(sub):
    w = sub["funded_amnt"] / sub["funded_amnt"].sum()
    return float(np.average(sub[TARGET], weights=w) / sub[TARGET].std())


va = va.copy()
va["pred"] = pred
cut = va["pred"].quantile(1 - TOP_PCT)
sel = va[va["pred"] >= cut]

print(f"\n[포트폴리오 — 예측 상위 {int(TOP_PCT * 100)}% 승인]")
print(f"  전건 승인      샤프 {sharpe(va):+.4f}  (n={len(va):,}, 부도율 {va['bad'].mean() * 100:.2f}%)")
print(f"  모델 상위 {int(TOP_PCT * 100)}%  샤프 {sharpe(sel):+.4f}  (n={len(sel):,}, 부도율 {sel['bad'].mean() * 100:.2f}%)")
print(f"  평균 spread    전건 {va[TARGET].mean():+.4f} → 선별 {sel[TARGET].mean():+.4f}")

print("\n  참고 벤치마크 (전체 표본 기준, 앞선 분석)")
print("    A등급만 고르기 = +0.231  ← 이걸 넘어야 의미가 있다")

for p in [0.1, 0.3, 0.5]:
    c = va["pred"].quantile(1 - p)
    s = va[va["pred"] >= c]
    print(f"  상위 {int(p * 100):>2}%        샤프 {sharpe(s):+.4f}  (n={len(s):,}, 부도율 {s['bad'].mean() * 100:.2f}%)")

print(f"""
{'-' * 66}
읽는 법
{'-' * 66}
  · R² 가 낮아도 정상이다. 목적은 예측 정확도가 아니라 '상위를 골라내는 능력'이다.
  · 포트폴리오 샤프가 A등급만 고르기(+0.231)를 넘는지가 핵심 판정 기준이다.
  · 4번 교차표의 ②(LASSO만 채택)와 ③(도메인만 채택)이 다음 회의 안건이다.
""")

# %%
