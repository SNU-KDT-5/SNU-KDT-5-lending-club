# %% [markdown]
# # STEP 3 — y_table (IRR · 국채 · 초과수익)
#
# **이 단계는 IRR을 다시 구현하지 않습니다.** `irr/irr_module.py` 를 그대로 호출합니다.
#
# 이유: IRR은 상수 몇 개의 조합이 아니라 **현금흐름 구성 · 이분법 솔버 · 국채 매칭 ·
# 윈저화가 얽힌 하나의 절차**입니다. 상수만 복사하면 절차는 복사되지 않습니다.
# 실제로 `lc_config` 에 `IRR_CASHFLOW="equal"` 이라 적혀 있었지만 모듈은 처음부터
# `A = installment` (installment 고정)로 계산하고 있었습니다. 설정 파일이 코드와
# 다른 말을 하고 있었고, 그래서 상수를 지우고 모듈을 단일 출처로 삼았습니다.
#
# ### 확정된 정의
#
# | 쟁점 | 확정 | 근거 |
# |---|---|---|
# | 현금흐름 | **installment 고정** | 계약 현실. 부도 건에서 `A×(k−1) > 순회수` 일 때만 A 축소 |
# | 순회수 | `total_pymnt − collection_recovery_fee` | `recoveries` 는 이미 `total_pymnt` 에 포함(항등식 100% 성립) |
# | 연율화 | **실제 보유기간 k** → `irr` | `irr_contract` 는 강건성 검증 전용 |
# | 주 지표 | **`excess`** = `irr` − 만기매칭 국채 | 모형 선택·임계값 결정은 전부 이 컬럼으로 |
# | 윈저화 | ±100% | 무처리 std 10.34 → 0.3158, 클립 15건 |
#
# **`excess` 와 `excess_contract` 의 샤프를 비교해 높은 쪽을 고르지 않습니다.**
# 수익률 정의는 모형의 입력이 아니라 **샤프를 재는 자(尺)** 입니다.
# 자를 바꿔가며 유리한 쪽을 고르면 더 좋은 모델이 아니라 유리한 자를 고른 것이 됩니다.
#
# ### 사후변수를 여기서 쓰는 것이 모순이 아닌 이유
# `installment`, `total_pymnt`, `last_pymnt_d` 는 X(모형 입력)에서는 배제했지만
# y(실현 수익률) 계산에는 씁니다. **"사후 정보는 feature 금지, 성과 평가에는 사용"**
# 이라는 원칙 그대로입니다.

# %%
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
PROJ = os.path.dirname(HERE)
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

C.check_paths()
C.describe()

irr = C.load_irr_module()
OUT = C.STEP3_DIR
print(f"\nirr_module : {irr.__file__}")
print(f"산출물 폴더: {OUT}")

_log = []


def log(s=""):
    print(s)
    _log.append(s)


# %%
# ─────────────────────────────────────────────────────────────
# 1. cohort + 현금흐름 원본 컬럼 적재
# ─────────────────────────────────────────────────────────────
# build_cashflow_inputs 가 요구하는 컬럼. 전부 사후·내생 변수지만
# y 계산에는 정당하게 쓰인다 (위 설명 참조).
CF_COLS = ["id", "funded_amnt", "installment", "term", "issue_d",
           "last_pymnt_d", "total_pymnt", "collection_recovery_fee"]

log("\n[1/5] 적재")
cohort = pd.read_csv(C.COHORT_PATH, usecols=["id", "split", "grade", "bad", "int_rate"])
want = set(cohort["id"].astype("int64"))
log(f"      cohort {len(cohort):,}행")

parts = []
for ch in pd.read_csv(C.RAW_PATH, usecols=CF_COLS, chunksize=200_000, low_memory=False):
    ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
    ch = ch.loc[ch["id"].notna()]
    ch = ch.loc[ch["id"].astype("int64").isin(want)]
    if len(ch):
        parts.append(ch)
df = pd.concat(parts, ignore_index=True)
del parts
df["id"] = df["id"].astype("int64")
df = df.set_index("id").loc[cohort["id"].to_numpy()].reset_index()
log(f"      현금흐름 원본 {len(df):,}행 × {df.shape[1]}열")
assert len(df) == len(cohort) and df["id"].is_unique

tr = C.load_treasury()
log(f"      국채 {tr.index.min():%Y-%m} ~ {tr.index.max():%Y-%m} (결측 0)")
log("      36개월 → 3년물(y3) / 60개월 → 5년물(y5) 만기 매칭")

# %%
# ─────────────────────────────────────────────────────────────
# 2. IRR · 초과수익 계산 — 모듈 호출
# ─────────────────────────────────────────────────────────────
log("\n[2/5] compute_returns")
t0 = time.time()

ret = irr.compute_returns(
    df,
    treasury=tr,
    snapshot_period=C.VINTAGE_CUT_PERIOD,          # step1 과 같은 상수
    vintage_buffer_months=C.MATURITY_BUFFER_MONTHS,
)
log(f"      {time.time() - t0:.1f}초, {ret.shape[0]:,}행 × {ret.shape[1]}열")

# ★ 교차 검증 — step1(전처리팀)과 irr_module(IRR팀)이 같은 표본을 보고 있는가
# 같은 함수(vintage_buffer_mask)에 같은 상수를 넘겼으니 전부 True 여야 한다.
# 하나라도 False 면 두 팀의 표본 정의가 다시 어긋난 것이다.
n_out = int((~ret["in_sample"]).sum())
log(f"\n      [교차검증] in_sample False {n_out}건 "
    f"(cohort 는 이미 같은 컷을 통과했으므로 0이어야 정상)")
assert n_out == 0, "step1 cohort 와 irr_module 의 vintage 컷이 어긋났습니다"

# %%
# ─────────────────────────────────────────────────────────────
# 3. 검증 — 결론 내기 전에 반증 기회를 준다
# ─────────────────────────────────────────────────────────────
log("\n[3/5] 검증")
checks = []


def check(ok, msg):
    checks.append(bool(ok))
    log(f"      [{'OK ' if ok else 'FAIL'}] {msg}")


check(ret["excess"].isna().sum() == 0, f"excess 결측 0 (실제 {int(ret['excess'].isna().sum())})")
check(ret["irr"].between(-1.0, 1.0).all(), "irr 이 윈저화 범위 ±100% 안에 있음")
check((ret["k"] >= 1).all(), f"보유기간 k >= 1 (최소 {ret['k'].min():.0f}개월)")
check((ret["net_total"] >= 0).all(), "순회수 >= 0")

# 순회수 보존: P(원금) 대비 A×n_reg + F 가 순회수와 일치해야 한다
cf = irr.build_cashflow_inputs(df)
resid = float(np.abs(cf["A"] * cf["n_reg"] + cf["F"] - cf["net_total"]).max())
check(resid < 1e-6, f"순회수 보존 잔차 max {resid:.2e}")

# 윈저화가 실제로 필요한지 — 무처리 표준편차와 비교
raw_std, w_std = float(ret["irr_raw"].std()), float(ret["irr"].std())
n_clip = int((ret["irr_raw"].abs() > 1.0).sum())
log(f"\n      [윈저화] 무처리 std {raw_std:.4f} → 윈저 후 {w_std:.4f}  (클립 {n_clip}건)")
log(f"               무처리 최댓값 {ret['irr_raw'].max() * 100:,.0f}%")
log("               상한이 임의 상수라는 게 유일한 약점 → 아래 민감도로 방어")
log("")
log("               ※ [기각된 가설] '극단값은 만기 미도래 vintage의 초단기 조기상환이므로")
log("                 vintage 컷이 이미 걷어냈을 것'이라고 예상했으나 데이터가 부정했다.")
log("                 IRR팀 완결건 전체(1,115,888) 무처리 std 10.34 → 우리 cohort(717,969)")
log("                 에서 오히려 12.89 로 **올라간다**. 최댓값 +1,091,937% 도 그대로 남는다.")
log("                 즉 극단값은 성숙한 vintage 안에 있다. 윈저화는 여전히 필수다.")
log("                 실제로 무처리 샤프는 0.0030 으로 무너지고, ±100% 에서 0.0861 이 된다.")
log("                 다만 ±100% 와 ±200% 가 동일하므로 **상한 위치는 결론을 좌우하지 않는다.**")

# 정상 완납 건의 IRR 이 명목금리와 비슷한가 (계산이 맞다는 간접 증거)
ok_ = cohort["bad"].to_numpy() == 0
log(f"\n      [정합성] 완납 건 IRR 중앙값 {ret.loc[ok_, 'irr'].median() * 100:.2f}%  vs  "
    f"명목금리 중앙값 {cohort.loc[ok_, 'int_rate'].median():.2f}%")

if not all(checks):
    raise SystemExit("!! 검증 실패 — 위 로그 확인")

# %%
# ─────────────────────────────────────────────────────────────
# 4. 윈저화 민감도 — cap 이 결론을 바꾸는가
# ─────────────────────────────────────────────────────────────
# 결과가 cap 에 둔감하면 그 자체가 방어 근거다.
log("\n[4/5] 윈저화 민감도 (전건승인 기준)")
# ★ 클립을 IRR 에 먼저 하고 rf 를 그다음 차감한다.
#   excess 를 직접 클립하면 시점별로 다른 rf 때문에 같은 대출이 cap 마다
#   다른 무위험수익률 취급을 받게 된다.
rf_all = ret["irr"] - ret["excess"]
rows = []
for cap in [0.5, 1.0, 2.0, None]:
    irr_c = ret["irr_raw"] if cap is None else ret["irr_raw"].clip(-cap, cap)
    e = irr_c - rf_all
    rows.append({"cap": "무처리" if cap is None else f"±{cap * 100:.0f}%",
                 "평균": e.mean(), "표준편차": e.std(),
                 "샤프": e.mean() / e.std() if e.std() > 0 else np.nan,
                 "클립건수": 0 if cap is None else int((ret["irr_raw"].abs() > cap).sum())})
sens = pd.DataFrame(rows)
log(sens.round(4).to_string(index=False))
log("      → cap 을 두 배로 늘려도 샤프가 거의 안 움직이면 상한 선택이 결론을 좌우하지 않는다는 뜻")

# %%
# ─────────────────────────────────────────────────────────────
# 5. 벤치마크 사다리 — 모형이 넘어야 할 선
# ─────────────────────────────────────────────────────────────
# 거부한 건은 초과수익 0(전액 국채)으로 포함시켜 포트폴리오 전체로 계산한다.
# 승인률이 다른 전략을 같은 자로 재려면 이렇게 해야 한다.
log("\n[5/5] 벤치마크 사다리")

e = ret["excess"].to_numpy()
amt = ret["P"].to_numpy()
grade = cohort["grade"].to_numpy()


def port_sharpe(approve, weight_numerator=True):
    """거부 건을 0으로 포함한 포트폴리오 샤프.

    분자만 금액가중하는 현재 정의는 팀 문서를 따른 것이며 재검토 안건이다.
    weight_numerator=False 면 단순 동일가중.
    """
    x = np.where(approve, e, 0.0)
    if weight_numerator:
        num = np.average(x, weights=amt)
    else:
        num = x.mean()
    sd = x.std(ddof=1)
    if sd == 0:
        # 전액 국채: 초과수익 0, 변동성 0. 관례상 샤프 0 으로 둔다(0/0 아님).
        return 0.0
    return float(num / sd)


ladder = [
    ("전액 국채 (전건거부)", np.zeros(len(e), dtype=bool)),
    ("전건승인 (A~G)", np.ones(len(e), dtype=bool)),
    ("A·B 등급만 승인", np.isin(grade, ["A", "B"])),
    ("Oracle (excess>0 만)", e > 0),
]
rows = []
for name, ap in ladder:
    rows.append({"전략": name, "승인률(%)": round(ap.mean() * 100, 1),
                 "샤프(금액가중 분자)": round(port_sharpe(ap, True), 4),
                 "샤프(동일가중)": round(port_sharpe(ap, False), 4)})
bench = pd.DataFrame(rows)
log(bench.to_string(index=False))
log("\n      ★ 'A·B 등급만 승인' 이 모형이 실제로 넘어야 할 선이다.")
log("      ★ Oracle 은 정답을 다 안다고 가정한 상한선이지 달성 목표가 아니다.")
log("      ※ 두 샤프 정의가 다르게 나오면 정의를 먼저 확정해야 한다 (미결 안건 #8).")

# 등급별 초과수익 — 단조 감소해야 정상
log("\n      [등급별 excess]")
g = pd.DataFrame({"grade": grade, "excess": e, "bad": cohort["bad"].to_numpy()})
gt = g.groupby("grade").agg(건수=("excess", "size"), 평균excess=("excess", "mean"),
                            표준편차=("excess", "std"), 부도율=("bad", "mean"))
gt["평균excess"] = (gt["평균excess"] * 100).round(2)
gt["표준편차"] = gt["표준편차"].round(4)
gt["부도율"] = (gt["부도율"] * 100).round(2)
log(gt.to_string())

# %%
# ─────────────────────────────────────────────────────────────
# 6. y_table 저장
# ─────────────────────────────────────────────────────────────
y = pd.DataFrame({
    "id": df["id"].to_numpy(),
    "split": cohort["split"].to_numpy(),
    "grade": grade,
    "bad": cohort["bad"].to_numpy(),
    "term_n": ret["term_n"].to_numpy(),
    "k": ret["k"].to_numpy(),                       # 실제 보유기간(개월)
    "funded_amnt": ret["P"].to_numpy(),
    "net_total": ret["net_total"].to_numpy(),       # 순회수
    "hpr": ret["hpr"].to_numpy(),                   # 보유기간 총수익률
    "rf": (ret["irr"] - ret["excess"]).to_numpy(),  # 만기매칭 국채
    "irr": ret["irr"].to_numpy(),
    "irr_contract": ret["irr_contract"].to_numpy(),
    "excess": ret["excess"].to_numpy(),             # ★ 주 지표
    "excess_contract": ret["excess_contract"].to_numpy(),   # 강건성 전용
})
# 미결 안건 #2(이진 vs 연속) 를 둘 다 돌려볼 수 있도록 이진 타깃도 함께 만든다
y["spread_positive"] = (y["excess"] > 0).astype("int8")

y.to_csv(C.Y_TABLE_PATH, index=False, encoding="utf-8-sig", float_format="%.10g")
sens.to_csv(os.path.join(OUT, "step3_윈저화민감도.csv"), index=False, encoding="utf-8-sig")
bench.to_csv(os.path.join(OUT, "step3_벤치마크사다리.csv"), index=False, encoding="utf-8-sig")
gt.to_csv(os.path.join(OUT, "step3_등급별수익구조.csv"), encoding="utf-8-sig")

log(f"\n저장 위치: {OUT}")
for f in ["y_table.csv", "step3_윈저화민감도.csv", "step3_벤치마크사다리.csv",
          "step3_등급별수익구조.csv"]:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        log(f"  {f:<28} {os.path.getsize(p) / 1e6:>7.1f} MB")

log(f"\n  excess>0 비율 {y['spread_positive'].mean() * 100:.2f}%  "
    f"(이진 타깃으로 쓸 때의 양성 비율)")

with open(os.path.join(OUT, "step3_요약.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_log))
print(f"\n✅ STEP 3 완료 — y_table {len(y):,}행")
