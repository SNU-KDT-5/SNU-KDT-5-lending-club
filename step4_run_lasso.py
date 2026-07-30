# %% [markdown]
# # STEP 4-B — LASSO 2차 검증
#
# ## 원칙
# **표준화도 alpha 선택도 전부 train 안에서만 한다.**
# 전체 데이터로 고르면 val/test 성능이 낙관적으로 편향된다.
#
# ## 무엇을 묻는가
# "도메인 지식으로 고른 변수 집합이 타당한가"를 **독립적인 기준**으로 검증한다.
# 후보군은 도메인 필터를 거치지 않은 넓은 집합이므로,
# **도메인 트랙이 잘못 버린 변수를 되찾아줄 수 있다.**
#
# ## 결과를 읽는 법 — 4분면
# | | LASSO 채택 | LASSO 미채택 |
# |---|---|---|
# | **도메인 채택** | 합의 — 그대로 간다 | 도메인 근거를 다시 확인 (공선성으로 대체됐을 수 있음) |
# | **도메인 미채택** | **되살릴지 판단** ← 여기가 핵심 | 합의 — 뺀 게 맞다 |
#
# ⚠️ LASSO가 골랐다고 무조건 넣는 것이 아니다. LASSO는 **선형·주효과**만 본다.
# 도메인이 뺀 이유(공선성·해석·수집시점)가 여전히 유효하면 유지한다.

# %%
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    print("!! scikit-learn 이 없습니다.  %pip install scikit-learn  후 커널 재시작")
    raise SystemExit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "파이프라인"))
import lc_config as C

OUT = C.STEP_DIR[4]

TARGET = "excess"                 # 주 지표. excess_contract 로 바꿔 민감도 확인 가능
CV_FOLDS = 5
ALPHAS = np.logspace(-5, -1, 40)
# n_jobs=-1 은 CV 폴드마다 설계행렬을 통째로 복사해 메모리를 폭발시킨다.
# 램이 넉넉하면 2~4 로 올려도 된다.
N_JOBS = 1

_log = []


def log(s=""):
    print(s, flush=True)
    _log.append(s)


# %%
# ─────────────────────────────────────────────────────────────
# 1. 적재 + 분할
# ─────────────────────────────────────────────────────────────
X = pd.read_csv(os.path.join(OUT, "lasso_candidate_X.csv"))
y = pd.read_csv(C.Y_TABLE_PATH, usecols=["id", "split", "grade", TARGET, "funded_amnt"])
vt = pd.read_csv(os.path.join(OUT, "lasso_candidate_variables.csv"))

m = X.merge(y, on="id", how="inner", validate="one_to_one")
assert len(m) == len(X), "y_table 과 행이 어긋납니다"

FEAT = [c for c in X.columns if c != "id"]
tr = m[m["split"] == "train"]
va = m[m["split"] == "val"]
log(f"후보 {len(FEAT)}개 변수 / train {len(tr):,} · val {len(va):,}")
log(f"타깃 {TARGET}  (train 평균 {tr[TARGET].mean():.4f}, 표준편차 {tr[TARGET].std():.4f})")

# %%
# ─────────────────────────────────────────────────────────────
# 2. 표준화 + LassoCV — train 에서만
# ─────────────────────────────────────────────────────────────
log("\n[2] LassoCV 적합 (train 에서만)")
scaler = StandardScaler().fit(tr[FEAT])              # ★ train 만
Xtr = scaler.transform(tr[FEAT]).astype(np.float64)  # float32 는 맥에서 Gram 검증 오류
ytr = tr[TARGET].to_numpy(dtype=np.float64)

log(f"    alpha 후보 {len(ALPHAS)}개 ({ALPHAS.min():.1e} ~ {ALPHAS.max():.1e}), {CV_FOLDS}-fold")
log("    (수 분 소요)")
model = LassoCV(alphas=ALPHAS, cv=CV_FOLDS, max_iter=5000,
                precompute=True, n_jobs=N_JOBS, random_state=C.SEED).fit(Xtr, ytr)

# 1-SE 규칙: 오차가 최소값의 1 표준오차 안에 드는 것 중 가장 큰 alpha(= 가장 단순한 모형)
mse = model.mse_path_.mean(axis=1)
se = model.mse_path_.std(axis=1) / np.sqrt(model.mse_path_.shape[1])
i_min = int(np.argmin(mse))
cand = np.flatnonzero(mse <= mse[i_min] + se[i_min])
i_1se = int(cand[np.argmax(model.alphas_[cand])])
a_1se = float(model.alphas_[i_1se])

m_1se = Lasso(alpha=a_1se, max_iter=5000, random_state=C.SEED).fit(Xtr, ytr)
n_min = int((model.coef_ != 0).sum())
n_1se = int((m_1se.coef_ != 0).sum())

log(f"\n    alpha_min = {model.alpha_:.6f}  → 선택 {n_min}/{len(FEAT)}개")
log(f"    alpha_1se = {a_1se:.6f}  → 선택 {n_1se}/{len(FEAT)}개  (더 단순한 모형)")
log(f"\n    ※ 표본이 {len(ytr):,}건으로 커서 alpha_min 은 변수를 거의 안 죽인다.")
log("      '어떤 변수가 유용한가'를 보려면 1-SE 기준과 계수 크기를 같이 봐야 한다.")

# %%
# ─────────────────────────────────────────────────────────────
# 3. 검증 성능 — 반증 기회
# ─────────────────────────────────────────────────────────────
log("\n[3] val 성능")
Xva = scaler.transform(va[FEAT]).astype(np.float64)
yva = va[TARGET].to_numpy(dtype=np.float64)

for name, mdl in [("alpha_min", model), ("alpha_1se", m_1se)]:
    p = mdl.predict(Xva)
    r2 = 1 - ((yva - p) ** 2).sum() / ((yva - yva.mean()) ** 2).sum()
    log(f"    {name}  R² {r2:.4f}   상관 {np.corrcoef(p, yva)[0, 1]:.4f}")

# ★ R² 가 낮아도 실망할 필요는 없다. 우리가 필요한 건 값의 정확도가 아니라
#   '순위'다. 상위 X% 를 골라 샤프가 개선되면 목적은 달성된다.
log("\n    [순위 성능] 예측 상위 X% 를 승인했을 때의 동일가중 샤프")
log("    (거부 건은 초과수익 0 = 전액 국채로 포함)")
pred = m_1se.predict(Xva)
ev = va[TARGET].to_numpy()
rows = []
for pct in [0.1, 0.2, 0.4, 0.6, 1.0]:
    thr = np.quantile(pred, 1 - pct)
    ap = pred >= thr
    x = np.where(ap, ev, 0.0)
    rows.append({"승인률%": round(pct * 100, 1), "샤프": round(x.mean() / x.std(ddof=1), 4),
                 "평균excess%": round(ev[ap].mean() * 100, 2)})
perf = pd.DataFrame(rows)
log(perf.to_string(index=False))
log("\n    비교 기준: 전건승인 0.0861 / A·B만 0.1297 / Oracle 1.6246 (step3)")

# %%
# ─────────────────────────────────────────────────────────────
# 4. 계수표
# ─────────────────────────────────────────────────────────────
coef = pd.DataFrame({"열": FEAT, "계수_min": model.coef_, "계수_1se": m_1se.coef_})
coef = coef.merge(vt, on="열", how="left")
coef["선택_min"] = (coef["계수_min"] != 0).astype(int)
coef["선택_1se"] = (coef["계수_1se"] != 0).astype(int)
coef["절대계수"] = coef["계수_1se"].abs()
coef = coef.sort_values("절대계수", ascending=False)

log("\n[4] 계수 상위 25개 (1-SE 기준)")
log(coef.head(25)[["열", "원천변수", "계수_1se", "도메인채택", "유형"]].to_string(index=False))

# %%
# ─────────────────────────────────────────────────────────────
# 5. 도메인 판단과 대조 — 원천변수 단위 4분면
# ─────────────────────────────────────────────────────────────
# 원-핫으로 쪼개진 열은 원천변수로 되묶는다.
# "addr_state 가 채택됐는가"를 물으려면 51개 더미를 하나로 봐야 한다.
log("\n[5] 도메인 판단 vs LASSO (원천변수 단위, 1-SE 기준)")

src = (coef.groupby("원천변수")
       .agg(열수=("열", "size"), 도메인채택=("도메인채택", "max"),
            LASSO채택=("선택_1se", "max"), 최대절대계수=("절대계수", "max"),
            유형=("유형", "first"))
       .reset_index().sort_values("최대절대계수", ascending=False))

ct = pd.crosstab(src["도메인채택"], src["LASSO채택"])
log("\n교차표 (행: 도메인 채택 / 열: LASSO 채택)")
log(ct.to_string())

log("\n── ① LASSO만 채택 (도메인이 뺐는데 신호가 있다) ← 되살릴지 판단할 것")
q1 = src[(src["도메인채택"] == 0) & (src["LASSO채택"] == 1)]
log(q1[["원천변수", "최대절대계수", "유형"]].to_string(index=False) if len(q1) else "    (없음)")

log("\n── ② 도메인만 채택 (LASSO가 죽였다) ← 공선성으로 대체됐는지 확인")
q2 = src[(src["도메인채택"] == 1) & (src["LASSO채택"] == 0)]
log(q2[["원천변수", "최대절대계수", "유형"]].to_string(index=False) if len(q2) else "    (없음)")

log("\n── ③ 양쪽 채택 (합의)")
q3 = src[(src["도메인채택"] == 1) & (src["LASSO채택"] == 1)]
log(f"    {len(q3)}개: {', '.join(q3['원천변수'].tolist())}")

log("\n── ④ 양쪽 미채택 (합의 — 뺀 게 맞다)")
q4 = src[(src["도메인채택"] == 0) & (src["LASSO채택"] == 0)]
log(f"    {len(q4)}개: {', '.join(q4['원천변수'].tolist())}")

# %%
# ─────────────────────────────────────────────────────────────
# 6. 저장
# ─────────────────────────────────────────────────────────────
coef.to_csv(os.path.join(OUT, "lasso_coefficients.csv"), index=False, encoding="utf-8-sig")
src.to_csv(os.path.join(OUT, "lasso_vs_domain.csv"), index=False, encoding="utf-8-sig")
perf.to_csv(os.path.join(OUT, "lasso_val_순위성능.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "step4_요약.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_log))

print(f"\n저장: lasso_coefficients.csv / lasso_vs_domain.csv / lasso_val_순위성능.csv")
print("\n✅ STEP 4 완료 — ①②번 목록을 보고 최종 X 변수를 확정할 것")

# %%
