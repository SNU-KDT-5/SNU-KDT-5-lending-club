# %% [markdown]
# # 검증 — 샤프 정의를 무엇으로 할 것인가 (미결 #8)
#
# y_table.csv 만 있으면 돌아간다. 원본 재적재 불필요.
#
# 세 후보:
# | 정의 | 분자 | 분모 |
# |---|---|---|
# | 현행 | 대출금액 가중 평균 | 단순 표준편차 |
# | 동일가중 | 단순 평균 | 단순 표준편차 |
# | 완전금액가중 | 대출금액 가중 평균 | 금액가중 표준편차 |

# %%
import os, sys, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
# 저장소(pipeline/)와 작업 폴더(파이프라인/) 둘 다에서 동작하도록 경로를 찾는다
def _find_pipeline(start):
    for base in (start, os.path.dirname(start), os.path.dirname(os.path.dirname(start))):
        for name in ("파이프라인", "pipeline", "."):
            d = os.path.normpath(os.path.join(base, name))
            if os.path.exists(os.path.join(d, "lc_config.py")):
                return d
    raise FileNotFoundError("lc_config.py 를 찾지 못했습니다")


PIPE_DIR = _find_pipeline(HERE)
if PIPE_DIR not in sys.path:
    sys.path.insert(0, PIPE_DIR)
import lc_config as C

C.ensure_dirs()
y = pd.read_csv(C.Y_TABLE_PATH)
e = y["excess"].to_numpy()
amt = y["funded_amnt"].to_numpy()
g = y["grade"].to_numpy()
print(f"표본 {len(y):,}행")


def sharpe(x, w, mode):
    """거부 건을 0으로 포함한 포트폴리오 샤프."""
    if mode == "현행":            # 분자만 금액가중 — 분자·분모의 측도가 다르다
        return np.average(x, weights=w) / x.std(ddof=1)
    if mode == "동일가중":
        return x.mean() / x.std(ddof=1)
    m = np.average(x, weights=w)  # 완전금액가중
    v = np.average((x - m) ** 2, weights=w) * len(x) / (len(x) - 1)
    return m / np.sqrt(v)


MODES = ["현행", "동일가중", "완전금액가중"]

# %%
# ─────────────────────────────────────────────────────────────
# 1. 세 정의가 전략 순위를 바꾸는가
# ─────────────────────────────────────────────────────────────
strat = {"전건승인": np.ones(len(e), bool), "A·B만": np.isin(g, ["A", "B"])}
for q in [0.2, 0.4]:
    strat[f"전역 상위{int(q * 100)}%"] = e >= np.quantile(e, 1 - q)
for gr in ["A", "B", "C", "D"]:
    m = g == gr
    a = np.zeros(len(e), bool)
    a[m & (e >= np.quantile(e[m], 0.6))] = True
    strat[f"{gr}등급내 상위40%"] = a

rows = []
for name, a in strat.items():
    x = np.where(a, e, 0.0)
    r = {"전략": name, "승인률%": round(a.mean() * 100, 1)}
    r.update({m: round(sharpe(x, amt, m), 4) for m in MODES})
    rows.append(r)
t = pd.DataFrame(rows).sort_values("동일가중", ascending=False)
for m in MODES:
    t[f"순위_{m}"] = t[m].rank(ascending=False).astype(int)
print(t.to_string(index=False))

print("\n[순위 일치 여부]")
for a, b in itertools.combinations(MODES, 2):
    same = bool((t[f"순위_{a}"] == t[f"순위_{b}"]).all())
    print(f"  {a:8s} vs {b:10s} 완전일치 {same}  spearman {t[a].corr(t[b], method='spearman'):.4f}")

# %%
# ─────────────────────────────────────────────────────────────
# 2. 우리 논지의 핵심 — B가 C보다 낫다고 말할 수 있는가
# ─────────────────────────────────────────────────────────────
# 방향성이 '중간 등급 살리기'라면 B > C 를 말할 수 있어야 한다.
# 정의별로 그 판별력이 어떻게 다른지 부트스트랩으로 본다.
print("\nB등급내 상위40% vs C등급내 상위40% (부트스트랩 300회)")
rng = np.random.default_rng(42)
n = len(e)
xB = np.where(strat["B등급내 상위40%"], e, 0.0)
xC = np.where(strat["C등급내 상위40%"], e, 0.0)

for mode in MODES:
    pt = sharpe(xB, amt, mode) - sharpe(xC, amt, mode)
    d = np.array([sharpe(xB[i], amt[i], mode) - sharpe(xC[i], amt[i], mode)
                  for i in (rng.integers(0, n, n) for _ in range(300))])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {mode:8s} B−C = {pt:+.4f}   95%CI [{lo:+.4f}, {hi:+.4f}]   "
          f"B 승률 {(d > 0).mean() * 100:5.1f}%")

# %%
# ─────────────────────────────────────────────────────────────
# 3. 금액가중이 정보를 더하는가
# ─────────────────────────────────────────────────────────────
# 대출액과 excess 의 상관이 0에 가까우면, 가중은 정보가 아니라 잡음만 더한다.
print("\ncorr(대출액, excess)")
for gr in ["A", "B", "C", "D", "E", "F", "G"]:
    m = g == gr
    print(f"  {gr}등급 {np.corrcoef(amt[m], e[m])[0, 1]:+.4f}")
print(f"  전체   {np.corrcoef(amt, e)[0, 1]:+.4f}")

t.to_csv(os.path.join(HERE, "샤프정의_전략순위.csv"), index=False, encoding="utf-8-sig")
print(f"\n저장: 샤프정의_전략순위.csv")
