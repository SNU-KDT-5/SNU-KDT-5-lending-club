# %% [markdown]
# # STEP 1 — 분석 대상(cohort) 확정 및 6:2:2 분할
#
# **이 단계가 하는 일은 하나뿐입니다: "어떤 대출을 분석에 쓸 것인가."**
#
# y(IRR) 값이 무엇이 되든 *어떤 행이 남을지*에는 영향을 주지 않으므로,
# 이 단계는 IRR 정의와 독립입니다. 그래서 IRR 논의와 병렬로 돌릴 수 있습니다.
#
# 이후 모든 단계(step2 X 전처리, step3 y_table, step4 LASSO, step5 모델)는
# 여기서 만든 `cohort.csv` / `split_assignment.csv` 를 **단일 기준**으로 삼습니다.
#
# ---
# ### 필터 순서
#
# | | 필터 | 왜 |
# |---|---|---|
# | ① | 원본에 섞인 텍스트 행 제거 | 956,843행 부근에 "Loans that do not meet..." 구분행이 섞여 있음 |
# | ② | `loan_status` ∈ {Fully Paid, Charged Off} | 현금흐름이 **끝난** 건만. Default는 `out_prncp`가 남아 진행 중 |
# | ③ | "Does not meet the credit policy" 제외 | LC가 사후 재분류한 건. 정상 승인 건과 성격이 다름 |
# | ④ | 발행월 + term + 6개월 ≤ 2020-10 | 만기가 도래한 vintage만. **term별로** 기준이 달라야 함 |
# | ⑤ | 발행연도 ≥ 2010 | 2007~2009는 저등급 셀이 통계적으로 비어 있음 |
#
# ### 실행 방법
# VS Code에서 이 파일을 열고 `Shift+Enter`로 셀을 순서대로 실행하거나,
# 터미널에서 `python "step1_cohort_and_split.py"`.

# %%
# ─────────────────────────────────────────────────────────────
# 0. 설정 로드
# ─────────────────────────────────────────────────────────────
import os
import sys

import numpy as np
import pandas as pd

# 파이프라인/ 에 있는 lc_config 를 import (이 파일은 'step 1/' 에 있다)
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

C.check_paths()
C.describe()

# IRR 쪽과 **같은 함수**로 vintage 마스크를 만든다.
# 두 팀이 각자 구현하면 표본이 어긋난다 — 실제로 어긋났던 것을 이번에 고쳤다.
irr_module = C.load_irr_module()
print(f"\nvintage 마스크: irr_module.vintage_buffer_mask() 사용 (IRR 쪽과 동일 구현)")

USECOLS = ["id", "issue_d", "term", "loan_status", "grade", "sub_grade",
           "funded_amnt", "out_prncp", "int_rate", "last_pymnt_d"]

_log = []


def log(s=""):
    print(s)
    _log.append(s)


# %%
# ─────────────────────────────────────────────────────────────
# 1. 원본 적재
# ─────────────────────────────────────────────────────────────
# 175만행 × 141열을 통째로 읽으면 메모리가 터진다. usecols + chunksize 필수.
log("\n[1/5] 원본 적재 (청크)")

parts = []
for ch in pd.read_csv(C.RAW_PATH, usecols=USECOLS, dtype=str, chunksize=300_000):
    # id가 숫자로 변환되지 않는 행 = 섹션 구분용 텍스트 행. 여기서 버린다.
    ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
    parts.append(ch.loc[ch["id"].notna()])

df = pd.concat(parts, ignore_index=True)   # ← reset_index 필수. 안 하면 인덱스에 구멍
del parts
df["id"] = df["id"].astype("int64")
log(f"      원본(텍스트 행 제거 후)            {len(df):>10,}")

# 파생 컬럼
df["issue_ym"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
assert df["issue_ym"].notna().all(), "issue_d 파싱 실패 행이 있습니다"
df["issue_year"] = df["issue_ym"].dt.year
df["term_n"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
for c in ["funded_amnt", "out_prncp"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["int_rate"] = pd.to_numeric(
    df["int_rate"].astype(str).str.replace("%", "", regex=False), errors="coerce")

status = df["loan_status"].astype(str).str.strip()
is_policy = status.str.startswith("Does not meet the credit policy")

log(f"      발행월 범위 {df['issue_ym'].min():%Y-%m} ~ {df['issue_ym'].max():%Y-%m}")

# %%
# ─────────────────────────────────────────────────────────────
# 2. 컷 기준 시점 확인 — 상수를 쓰기 전에 데이터로 확인한다
# ─────────────────────────────────────────────────────────────
# irr_module 규약: snapshot_period 는 런타임 자동도출 금지, 호출자가 1회 확인 후
# 상수로 넘긴다. 그 "1회 확인"이 이 셀이다.
log("\n[2/5] 컷 기준 시점 확인")

last_pymnt = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
obs_max = last_pymnt.max().to_period("M")

log(f"      실제 관측 지평 (last_pymnt_d.max())   {obs_max}")
log(f"      우리가 쓰는 컷    (VINTAGE_CUT_PERIOD) {C.VINTAGE_CUT_PERIOD}")
log("      → 컷이 관측 지평보다 이르다. 이건 데이터가 없어서가 아니라,")
log("        2020년 만기 이연(코로나 유예) 때문에 보수적으로 자르는 것이다.")

assert pd.Period(C.VINTAGE_CUT_PERIOD, "M") <= obs_max, \
    "컷 기준이 관측 지평보다 미래입니다 — 관측되지 않은 구간을 성숙하다고 판정하게 됩니다"

# %%
# ─────────────────────────────────────────────────────────────
# 3. 필터 ② ~ ⑤
# ─────────────────────────────────────────────────────────────
log("\n[3/5] 필터 적용")

# ② 완결 상태 — Default 제외
#    Default는 121일+ 연체지만 아직 상각 전이라 out_prncp가 남아 있다.
#    '끝난 상태'가 아니라 '진행 중'이므로 IRR을 계산할 근거가 없다.
n_default = int((status == "Default").sum())
m = status.isin(C.COMPLETED_STATUS)
log(f"      ② 완결 상태 {C.COMPLETED_STATUS}")
log(f"         Default 제외분: {n_default:,}건")
df = df[m].reset_index(drop=True)
status, is_policy = status[m].reset_index(drop=True), is_policy[m].reset_index(drop=True)
log(f"         잔존                            {len(df):>10,}")

# ③ policy 건
if C.EXCLUDE_POLICY_LOANS:
    keep = ~is_policy
    log(f"      ③ 'Does not meet credit policy' 제외 {int((~keep).sum()):,}건")
    df, status = df[keep].reset_index(drop=True), status[keep].reset_index(drop=True)
    log(f"         잔존                            {len(df):>10,}")

# ④ vintage — IRR 쪽과 동일 함수. 월 단위 Period 연산이라 일 단위 경계 모호성 없음.
keep = irr_module.vintage_buffer_mask(
    issue_dt=df["issue_ym"],
    term_n=df["term_n"],
    snapshot_period=C.VINTAGE_CUT_PERIOD,
    buffer_months=C.MATURITY_BUFFER_MONTHS,
)
log(f"      ④ vintage 만기+{C.MATURITY_BUFFER_MONTHS}개월 도래분만 "
    f"(제외 {int((~keep).sum()):,}건)")
df, status = df[keep.to_numpy()].reset_index(drop=True), status[keep.to_numpy()].reset_index(drop=True)
log(f"         잔존                            {len(df):>10,}")

# 컷 경계가 의도대로인지 확인 (snapshot − term − buffer)
snap = pd.Period(C.VINTAGE_CUT_PERIOD, "M")
for t in sorted(df["term_n"].dropna().unique()):
    edge = snap - int(t + C.MATURITY_BUFFER_MONTHS)
    actual = df.loc[df["term_n"] == t, "issue_ym"].max().to_period("M")
    log(f"         {int(t)}개월 최대 발행월 {actual}  (경계 {edge})")
    assert actual <= edge, f"{int(t)}개월 컷 경계 위반"

# ⑤ 발행연도
keep = df["issue_year"] >= C.ISSUE_YEAR_MIN
if C.ISSUE_YEAR_MAX is not None:
    keep &= df["issue_year"] <= C.ISSUE_YEAR_MAX
log(f"      ⑤ 발행연도 >= {C.ISSUE_YEAR_MIN} (제외 {int((~keep).sum()):,}건)")
df, status = df[keep].reset_index(drop=True), status[keep].reset_index(drop=True)
log(f"         잔존                            {len(df):>10,}")

# 부도 라벨
df["bad"] = status.isin(C.BAD_STATUS).astype("int8")

# %%
# ─────────────────────────────────────────────────────────────
# 4. 무결성 점검 — 결론을 내기 전에 반증 기회를 준다
# ─────────────────────────────────────────────────────────────
log("\n[4/5] 무결성 점검")

checks = []


def check(ok, msg):
    checks.append(bool(ok))
    log(f"      [{'OK ' if ok else 'FAIL'}] {msg}")


# 완결건은 정의상 미상환 원금이 0이어야 한다. 이게 '완결'의 조작적 정의였다.
n_out = int((df["out_prncp"].fillna(0) > 0).sum())
check(n_out == 0, f"전 행 out_prncp = 0 (위반 {n_out}건)")

check(df["id"].is_unique, f"id 중복 없음 ({df['id'].nunique():,}개)")
check(df["bad"].notna().all(), "부도 라벨 결측 없음")
check(df["term_n"].isin([36, 60]).all(), "term은 36 또는 60만")
check(df["issue_year"].min() >= C.ISSUE_YEAR_MIN,
      f"최소 발행연도 {int(df['issue_year'].min())} >= {C.ISSUE_YEAR_MIN}")
check(df["grade"].notna().all(), "등급 결측 없음")

# 남은 vintage가 실제로 성숙했는지 — 이번 컷 변경의 핵심 근거를 재확인
#
# ★ 주의: 완결률은 반드시 **월 단위**로 재야 한다.
#   컷이 월 단위(issue_ym + term + 6 <= 2020-10)로 걸리기 때문이다.
#   연 단위로 재면 "2017년 36개월 완결률 77.09%"가 나와서 마치 미성숙한 칸을
#   채택한 것처럼 보이는데, 실제로 채택된 건 2017-01~04뿐이고 2017-05~12는
#   컷이 이미 걸러낸다. 연 단위 집계는 버린 달까지 섞어서 재는 셈이라 틀린다.
#   (실제로 이 스크립트 초안이 이 오탐으로 FAIL을 냈다.)
log("\n      [핵심 점검] 채택한 (발행월 × term) 칸의 완결률 — 월 단위")
log("      성숙하지 않은 칸이 섞여 있으면 정상 상환 중인 건이 빠져 부도가 과대표집된다.")

raw_status = []
for ch in pd.read_csv(C.RAW_PATH, usecols=["id", "issue_d", "term", "loan_status"],
                      dtype=str, chunksize=300_000):
    ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
    raw_status.append(ch.loc[ch["id"].notna()])
raw = pd.concat(raw_status, ignore_index=True)
del raw_status

raw = raw[~raw["loan_status"].astype(str).str.strip()
          .str.startswith("Does not meet the credit policy")]
raw["ym"] = pd.to_datetime(raw["issue_d"], format="%b-%Y").dt.to_period("M")
raw["term_n"] = raw["term"].astype(str).str.extract(r"(\d+)").astype(int)
raw["done"] = raw["loan_status"].astype(str).str.strip().isin(["Fully Paid", "Charged Off"])

# 채택된 (발행월, term) 조합만 대상으로 한다
taken = df.assign(ym=df["issue_ym"].dt.to_period("M"))[["ym", "term_n"]].drop_duplicates()
taken["term_n"] = taken["term_n"].astype(int)

cell_m = (raw.groupby(["ym", "term_n"])["done"].agg(칸전체="size", 완결률="mean")
          .reset_index().merge(taken, on=["ym", "term_n"], how="inner"))
cell_m["완결률"] = (cell_m["완결률"] * 100).round(2)

for t in sorted(cell_m["term_n"].unique()):
    sub = cell_m[cell_m["term_n"] == t]
    w = sub.nsmallest(3, "완결률")
    log(f"         {t}개월  채택 {len(sub)}개월  "
        f"가중완결률 {(sub['칸전체'] * sub['완결률']).sum() / sub['칸전체'].sum():.2f}%  "
        f"최저 {w['완결률'].min():.2f}% ({w.iloc[0]['ym']})")

worst = cell_m["완결률"].min()
check(worst >= 95.0, f"채택한 모든 발행월의 완결률 >= 95% (최저 {worst:.2f}%)")

# 보고서용 연 단위 요약표 (점검 기준으로는 쓰지 않는다 — 위 주의 참조)
cells = (df.groupby(["issue_year", "term_n"]).size().reset_index(name="채택건수"))
cells["term_n"] = cells["term_n"].astype(int)
cells = cells.merge(
    cell_m.assign(issue_year=cell_m["ym"].dt.year)
          .groupby(["issue_year", "term_n"])
          .apply(lambda g: pd.Series({
              "채택월수": len(g),
              "완결률": round((g["칸전체"] * g["완결률"]).sum() / g["칸전체"].sum(), 2)}),
              include_groups=False)
          .reset_index(),
    on=["issue_year", "term_n"], how="left")
log("")
log(cells.sort_values(["term_n", "issue_year"]).to_string(index=False))

del raw

if not all(checks):
    raise SystemExit("\n!! 점검 실패 항목이 있습니다. 위 로그를 확인하세요.")

# %%
# ─────────────────────────────────────────────────────────────
# 5. 6:2:2 분할
# ─────────────────────────────────────────────────────────────
# 분할에 필요한 건 행 집합(id 목록)뿐이다. y_table보다 먼저 만들 수 있다.
log("\n[5/5] 6:2:2 분할")

rng = np.random.default_rng(C.SEED)
n = len(df)
idx = rng.permutation(n)
n_tr = int(n * C.SPLIT_RATIOS[0])
n_va = int(n * C.SPLIT_RATIOS[1])

split = np.empty(n, dtype=object)
split[idx[:n_tr]] = "train"
split[idx[n_tr:n_tr + n_va]] = "val"
split[idx[n_tr + n_va:]] = "test"
df["split"] = split

log(f"      seed={C.SEED}")
vc = df["split"].value_counts()
for k in ["train", "val", "test"]:
    log(f"      {k:<6} {vc[k]:>9,}  ({vc[k] / n * 100:.2f}%)")

# 층화 없이 무작위로 나눴으니, 실제로 고르게 섞였는지 확인한다.
log("\n      [점검] 분할별 부도율 / 평균 발행연도 / 60개월 비중")
for k in ["train", "val", "test"]:
    s = df[df["split"] == k]
    log(f"      {k:<6} 부도율 {s['bad'].mean() * 100:5.2f}%   "
        f"평균 발행연도 {s['issue_year'].mean():.2f}   "
        f"60개월 {(s['term_n'] == 60).mean() * 100:5.2f}%")

spread = df.groupby("split")["bad"].mean()
log(f"\n      분할 간 부도율 최대 격차 {(spread.max() - spread.min()) * 100:.2f}%p")
log("      (0.5%p 안쪽이면 45만 건 규모에서 무작위 분할로 충분하다는 뜻)")

# %%
# ─────────────────────────────────────────────────────────────
# 6. 분포 (보고서용)
# ─────────────────────────────────────────────────────────────
log("\n[분포] 발행연도별")
t = df.groupby("issue_year").agg(건수=("id", "size"), 부도율=("bad", "mean"))
t["부도율"] = (t["부도율"] * 100).round(2)
log(t.to_string())

log("\n[분포] 등급별")
t = df.groupby("grade").agg(건수=("id", "size"), 부도율=("bad", "mean"))
t["비중(%)"] = (t["건수"] / len(df) * 100).round(2)
t["부도율"] = (t["부도율"] * 100).round(2)
log(t.to_string())

log("\n[분포] term별")
t = df.groupby("term_n").agg(건수=("id", "size"), 부도율=("bad", "mean"))
t["부도율"] = (t["부도율"] * 100).round(2)
log(t.to_string())

# %%
# ─────────────────────────────────────────────────────────────
# 7. 저장
# ─────────────────────────────────────────────────────────────
OUT = C.STEP1_DIR

cohort = df.drop(columns=["out_prncp", "last_pymnt_d"])
cohort.to_csv(C.COHORT_PATH, index=False, encoding="utf-8-sig")
cohort[["id", "split"]].to_csv(C.SPLIT_PATH, index=False, encoding="utf-8-sig")
cells.to_csv(os.path.join(OUT, "step1_채택칸_완결률.csv"),
             index=False, encoding="utf-8-sig")

log(f"\n저장 위치: {OUT}")
for f in ["cohort.csv", "split_assignment.csv", "step1_채택칸_완결률.csv"]:
    p = os.path.join(OUT, f)
    log(f"  {f:<28} {os.path.getsize(p) / 1e6:>8.1f} MB")

with open(os.path.join(OUT, "step1_요약.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_log))
print(f"\n  step1_요약.txt 저장 완료")
print(f"\n✅ STEP 1 완료 — 최종 표본 {len(cohort):,}건")

# %%
