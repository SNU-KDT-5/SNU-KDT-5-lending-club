


#%%

"""
y_table 생성 — 실현 IRR · 국채수익률 · spread

[정의 출처]
  IRR_계산_구현.ipynb          현금흐름 균등분배 방식
  part5_사후내생변수_검증.ipynb  순회수 정의·k 규칙·윈저화 (검증 결과 반영본)

[두 파일이 충돌하는 지점과 처리]
  1) recoveries 별도 가산
     IRR 노트북은 t=n+1 에 (recoveries - collection_recovery_fee) 를 더한다.
     그러나 원본에서 recoveries>0 인 부도건 전부가 total_pymnt >= recoveries 를 만족한다.
     = 회수금이 이미 total_pymnt 에 포함돼 있다.
     따라서 별도로 더하면 이중 계상이 되어 부도 대출 수익률이 과대평가된다.
     → part5 결정대로 '별도 가산하지 않고', 수수료만 차감한다.
         순회수(net_total) = total_pymnt - collection_recovery_fee

  2) 중간 회차 배분
     part5 는 installment 고정 + 잔여액을 마지막에 두는 방식,
     IRR 노트북은 균등분배 방식을 쓴다.
     → 팀 결정에 따라 균등분배(IRR 노트북 방식)를 채택한다.

[극단 케이스 4종 규칙 — part5 §9·§12]
  (a) last_pymnt_d 결측  → k = 1
  (b) k <= 0             → k = 1
  (c) k > term           → 허용 (추심 회수는 정상 현상)
  (d) 연율 IRR 을 [-100%, +100%] 로 윈저화
      보유기간 1~2개월 건을 연율화하면 (1+r)^12 가 폭발해
      소수 건이 샤프 분모를 혼자 지배한다.

[국채수익률]
  FRED DGS3(3년물) / DGS5(5년물), 대출 실행월(issue_d)의 일별 수익률 월평균.
  36개월 대출 → 3년물, 60개월 대출 → 5년물.

[입력]  자료_데이터/lending_club_2020_train.csv
        treasury_monthly.csv  (없으면 FRED 에서 받아 생성. 이후 오프라인 재실행 가능)
[출력]  y_table.csv
"""

import os
import numpy as np
import pandas as pd

VERSION = "v2 (vintage 필터 포함)"

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────
COMPLETED_STATUS = ["Fully Paid", "Charged Off", "Default"]
EXCLUDE_POLICY_LOANS = True     # "Does not meet the credit policy..." 1,683건 제외
IRR_WINSOR = (-1.0, 1.0)        # 연율 IRR 윈저화 범위
FRED_SERIES = {36: "DGS3", 60: "DGS5"}

# ─── vintage 편향 필터 ───────────────────────────────────
# 데이터 스냅샷이 2020-12 라서, 최근 발행 대출은 아직 만기가 오지 않았다.
# 그런데도 '완결'로 잡히는 건 조기상환(우량)과 조기부도(불량)뿐이라
# 정상적으로 만기까지 간 중간층이 통째로 빠진다.
#
#   "maturity" (D안, 권장) : 발행일 + term + 버퍼 <= 스냅샷 인 건만 사용
#                            term 별로 필요한 기간이 다르므로 각각 적용한다.
#                            36개월 → 2017-09까지 / 60개월 → 2015-09까지
#   "year"     (A안)       : 발행연도 <= YEAR_CUTOFF 인 건만 사용
#                            연도로만 자르면 60개월의 편향이 남는다(2016년 완결률 71.8%)
#   "none"     (C안)       : 필터 없음. 편향이 성능 수치에 그대로 섞인다
VINTAGE_FILTER = "maturity"
MATURITY_BUFFER_MONTHS = 3      # 연체→상각 인식 지연을 감안한 여유분
YEAR_CUTOFF = 2016              # VINTAGE_FILTER="year" 일 때만 사용


def find_base_dir(marker="자료_데이터", max_up=5):
    try:
        start = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start = os.getcwd()
    d = start
    for _ in range(max_up):
        if os.path.isdir(os.path.join(d, marker)):
            return d
        p = os.path.dirname(d)
        if p == d:
            break
        d = p
    raise FileNotFoundError(f"'{marker}' 폴더를 찾지 못했습니다.")


def section(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64, flush=True)


BASE_DIR      = find_base_dir()
SRC_PATH      = os.path.join(BASE_DIR, "자료_데이터", "lending_club_2020_train.csv")
TREASURY_PATH = os.path.join(BASE_DIR, "treasury_monthly.csv")
OUT_PATH      = os.path.join(BASE_DIR, "y_table.csv")


# ─────────────────────────────────────────────────────────
# 1. 국채수익률 — FRED 수집 + 캐시
# ─────────────────────────────────────────────────────────
def fetch_fred_series(series_id):
    """FRED CSV 엔드포인트에서 일별 시계열을 받는다. API 키 불필요."""
    import requests
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = requests.get(url, params={"id": series_id}, timeout=90)
    r.raise_for_status()
    from io import StringIO
    s = pd.read_csv(StringIO(r.text))
    s.columns = ["date", "value"]
    s["date"] = pd.to_datetime(s["date"])
    # FRED 는 휴일을 '.' 으로 표기한다
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    return s.dropna()


def load_treasury():
    """월평균 국채수익률 테이블(소수 단위). 캐시가 있으면 그대로 쓴다."""
    if os.path.exists(TREASURY_PATH):
        t = pd.read_csv(TREASURY_PATH, parse_dates=["month"])
        print(f"[캐시 사용] {os.path.basename(TREASURY_PATH)}  "
              f"{len(t)}개월 ({t['month'].min():%Y-%m} ~ {t['month'].max():%Y-%m})")
        return t

    print("[캐시 없음] FRED 에서 국채수익률 수집")
    frames = {}
    for term, sid in FRED_SERIES.items():
        try:
            s = fetch_fred_series(sid)
        except Exception as e:
            raise SystemExit(
                f"\n!! FRED 수집 실패 ({sid}): {type(e).__name__}: {e}\n"
                f"!! 네트워크가 되는 환경에서 한 번 실행해 캐시를 만들거나,\n"
                f"!! 조원에게 받은 treasury_monthly.csv 를 아래에 두고 다시 실행하세요:\n"
                f"!!   {TREASURY_PATH}\n"
                f"!! (형식: month, rf_36m, rf_60m — rf 는 소수 단위. 예: 0.0254)"
            )
        # 일별 → 월평균. FRED 는 퍼센트 단위이므로 100 으로 나눈다.
        s["month"] = s["date"].values.astype("datetime64[M]")
        m = s.groupby("month")["value"].mean() / 100.0
        frames[term] = m
        print(f"  {sid}: {len(s):,}일 → {len(m)}개월, "
              f"평균 {m.mean() * 100:.2f}%")

    t = pd.DataFrame({"rf_36m": frames[36], "rf_60m": frames[60]}).reset_index()
    t.to_csv(TREASURY_PATH, index=False)
    print(f"[캐시 저장] {os.path.basename(TREASURY_PATH)}")
    return t


# ─────────────────────────────────────────────────────────
# 2. IRR — 벡터화 이분법
# ─────────────────────────────────────────────────────────
def npv(r, P, A, F, k):
    """
    현금흐름  t=0: -P,  t=1..k-1: A,  t=k: F  의 현재가치.
    r 은 월별 할인율. 배열 연산으로 한 번에 계산한다.
    """
    d = 1.0 + r
    n = k - 1                                   # 균등분배 회차 수
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # 연금현가계수. r≈0 이면 극한값 n 을 쓴다.
        small = np.abs(r) < 1e-12
        af = np.where(small, n, (1.0 - np.power(d, -n)) / np.where(small, 1.0, r))
        val = -P + A * af + F * np.power(d, -k)
    return val


def irr_vectorized(P, A, F, k, lo=-0.9999, hi=10.0, iters=80):
    """
    월별 IRR 을 이분법으로 구한다.

    현금 유입(A, F)이 모두 0 이상이면 NPV(r) 은 r 에 대해 단조감소하고,
    r→-1+ 에서 +∞, r→+∞ 에서 -P(<0) 이므로 근이 유일하다.
    numpy_financial.irr 을 행마다 호출하면 100만 행에서 수십 분이 걸리지만
    이 방식은 배열 연산이라 수십 초면 끝난다.
    """
    lo_a = np.full(P.shape, lo, dtype=float)
    hi_a = np.full(P.shape, hi, dtype=float)
    for _ in range(iters):
        mid = (lo_a + hi_a) / 2.0
        v = npv(mid, P, A, F, k)
        pos = v > 0                    # NPV>0 이면 할인율을 더 올려야 한다
        lo_a = np.where(pos, mid, lo_a)
        hi_a = np.where(pos, hi_a, mid)
    return (lo_a + hi_a) / 2.0


# ─────────────────────────────────────────────────────────
# 3. 실행
# ─────────────────────────────────────────────────────────
print("#" * 64)
print(f"#  build_y_table  {VERSION}")
print(f"#  VINTAGE_FILTER = {VINTAGE_FILTER!r}"
      + (f"  /  버퍼 {MATURITY_BUFFER_MONTHS}개월" if VINTAGE_FILTER == "maturity" else "")
      + (f"  /  발행 {YEAR_CUTOFF}년 이하" if VINTAGE_FILTER == "year" else ""))
print("#" * 64)

section("0. 데이터 로드")

COLS = ["id", "issue_d", "loan_status", "term", "funded_amnt", "installment",
        "total_pymnt", "last_pymnt_d", "last_pymnt_amnt",
        "recoveries", "collection_recovery_fee"]
df = pd.read_csv(SRC_PATH, usecols=COLS, low_memory=False)

# 원본에 섞인 섹션 구분용 텍스트 행 제거 후 인덱스 재설정
df = df[pd.to_numeric(df["id"], errors="coerce").notna()].copy()
df["id"] = df["id"].astype("int64")
df = df.reset_index(drop=True)
print(f"전체: {len(df):,}행")


section("1. 완결 대출 필터링")

status = df["loan_status"].astype(str).str.strip()
mask = status.isin(COMPLETED_STATUS)
policy_mask = status.str.startswith("Does not meet the credit policy")
n_policy = int(policy_mask.sum())
print(f"완결 상태({COMPLETED_STATUS}): {int(mask.sum()):,}건")
print(f"'Does not meet the credit policy...': {n_policy:,}건 "
      f"→ {'제외' if EXCLUDE_POLICY_LOANS else '포함'}")
if not EXCLUDE_POLICY_LOANS:
    mask |= policy_mask & status.str.contains("Fully Paid|Charged Off")

d = df[mask].copy().reset_index(drop=True)
d["bad"] = status[mask].isin(["Charged Off", "Default"]).astype("int8").to_numpy()
n_completed = len(d)
print(f"완결 대출: {n_completed:,}건 ({n_completed / len(df) * 100:.1f}%)")

# 스냅샷 시점 = 데이터에 기록된 마지막 상환월
snapshot = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce").max()
del df
print(f"데이터 스냅샷: {snapshot:%Y-%m}")


# ── vintage 필터 ──
_iss = pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce")
_term = pd.to_numeric(d["term"].astype(str).str.strip().str.extract(r"(\d+)")[0],
                      errors="coerce")
snap_m = np.datetime64(snapshot, "M")

if VINTAGE_FILTER == "maturity":
    _ym = _iss.values.astype("datetime64[M]")
    keep = (_ym + (_term + MATURITY_BUFFER_MONTHS).values.astype("timedelta64[M]")) <= snap_m
    print(f"\n[vintage 필터] maturity (버퍼 {MATURITY_BUFFER_MONTHS}개월)")
    for t in sorted(_term.dropna().unique()):
        sub = _iss[(_term == t) & keep]
        if len(sub):
            print(f"  {int(t)}개월 대출 → {sub.max():%Y-%m} 까지 발행분 사용")
elif VINTAGE_FILTER == "year":
    keep = _iss.dt.year <= YEAR_CUTOFF
    print(f"\n[vintage 필터] year (발행 {YEAR_CUTOFF}년 이하)")
elif VINTAGE_FILTER == "none":
    keep = pd.Series(True, index=d.index)
    print("\n[vintage 필터] none — 필터 없음")
    print("  주의: 최근 발행분의 선택 편향이 성능 수치에 그대로 섞입니다.")
else:
    raise ValueError(f"VINTAGE_FILTER 값이 잘못됨: {VINTAGE_FILTER}")

# maturity 분기는 numpy 배열, 나머지는 Series 를 돌려주므로 형식을 통일한다
keep = pd.Series(np.asarray(keep), index=d.index).fillna(False).astype(bool)
d = d[keep].copy().reset_index(drop=True)
print(f"  {n_completed:,}건 → {len(d):,}건 (완결건의 {len(d) / n_completed * 100:.1f}% 유지)")
print(f"  부도(bad=1) 비율: {d['bad'].mean() * 100:.2f}%")


section("2. 보유기간 k 결정")

issue_dt = pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce")
last_dt = pd.to_datetime(d["last_pymnt_d"], format="%b-%Y", errors="coerce")
k_raw = ((last_dt.dt.year - issue_dt.dt.year) * 12
         + (last_dt.dt.month - issue_dt.dt.month))
term_n = pd.to_numeric(d["term"].astype(str).str.strip().str.extract(r"(\d+)")[0],
                       errors="coerce")

n_na = int(k_raw.isna().sum())
n_le0 = int((k_raw <= 0).sum())
n_gt = int((k_raw > term_n).sum())
print(f"(a) last_pymnt_d 결측 : {n_na:,}건 → k=1  (한 푼도 안 갚은 즉시 부도)")
print(f"(b) k <= 0           : {n_le0:,}건 → k=1")
print(f"(c) k > term         : {n_gt:,}건 → 그대로 허용 (추심 회수는 정상)")

k = k_raw.fillna(1).clip(lower=1).astype("int64")
d["k_months"] = k
d["term_n"] = term_n.astype("int64")
print(f"k 분포: 중앙값 {k.median():.0f}, 최대 {k.max():.0f}개월")


section("3. 현금흐름 구성")

P = d["funded_amnt"].to_numpy(dtype=float)

# 순회수 = total_pymnt - collection_recovery_fee
# recoveries 는 이미 total_pymnt 에 포함돼 있어 더하지 않는다(이중 계상 방지).
net_total = (d["total_pymnt"] - d["collection_recovery_fee"]).to_numpy(dtype=float)
net_total = np.maximum(net_total, 0.0)
d["net_total"] = net_total

F = d["last_pymnt_amnt"].to_numpy(dtype=float)
kk = k.to_numpy(dtype=float)

# 균등분배: t=1..k-1 에 (순회수 - 마지막회차)/(k-1)
with np.errstate(divide="ignore", invalid="ignore"):
    A = np.where(kk > 1, (net_total - F) / np.maximum(kk - 1, 1), 0.0)

# 중간 회차가 음수가 되는 경우(순회수 < 마지막회차)는 사실상 전액이 만기에 들어온 건이다.
# 음수 유입을 두면 NPV 가 단조성을 잃어 IRR 이 불안정해지므로,
# A=0 으로 두고 전액을 마지막 시점에 몰아준다. 총액은 그대로 보존된다.
neg = A < 0
n_neg = int(neg.sum())
A = np.where(neg, 0.0, A)
F = np.where(neg | (kk <= 1), net_total, F)
print(f"중간회차 음수 → 만기 일괄로 보정: {n_neg:,}건 ({n_neg / len(d) * 100:.2f}%)")

# 총액 보존 검증
resid = np.abs(A * np.maximum(kk - 1, 0) + F - net_total)
print(f"총액 보존 최대 오차: {resid.max():.6f}")

no_cash = net_total <= 0
print(f"순회수 0 이하(전손): {int(no_cash.sum()):,}건 → IRR = -100%")


section("4. IRR 계산")

r_m = irr_vectorized(P, A, F, kk)
irr_annual_raw = np.power(1.0 + r_m, 12.0) - 1.0
irr_annual_raw = np.where(no_cash, -1.0, irr_annual_raw)

d["irr_monthly"] = r_m
d["irr_annual_raw"] = irr_annual_raw
d["irr_annual"] = np.clip(irr_annual_raw, *IRR_WINSOR)

n_hi = int((irr_annual_raw > IRR_WINSOR[1]).sum())
n_lo = int((irr_annual_raw < IRR_WINSOR[0]).sum())
print(f"윈저화 전 표준편차 {np.nanstd(irr_annual_raw):.3f} "
      f"→ 후 {d['irr_annual'].std():.4f}")
print(f"  상한 {IRR_WINSOR[1] * 100:.0f}% 초과 {n_hi:,}건 / "
      f"하한 미만 {n_lo:,}건 (합 {(n_hi + n_lo) / len(d) * 100:.4f}%)")
print(f"  윈저화 전 최댓값 {np.nanmax(irr_annual_raw) * 100:,.0f}%")

# 대안 지표: 계약 만기(term) 기준 연율화 (논문 각주 5 방식)
# 초단기 상환건의 연율화 폭발을 구조적으로 피한다.
with np.errstate(invalid="ignore", divide="ignore"):
    gross = np.where(P > 0, net_total / P, np.nan)
    irr_contract = np.power(np.maximum(gross, 1e-9), 12.0 / d["term_n"].to_numpy()) - 1.0
d["irr_contract"] = np.clip(irr_contract, *IRR_WINSOR)
print(f"irr_contract 중앙값 {d['irr_contract'].median():.4f}")


section("5. 국채수익률 매칭")

tr = load_treasury()
tr["month"] = pd.to_datetime(tr["month"]).values.astype("datetime64[M]")

d["issue_month"] = issue_dt.values.astype("datetime64[M]")
d = d.merge(tr, left_on="issue_month", right_on="month", how="left")

# 36개월 → 3년물, 60개월 → 5년물
d["rf"] = np.where(d["term_n"] == 60, d["rf_60m"], d["rf_36m"])
n_rf_na = int(d["rf"].isna().sum())
print(f"rf 결측: {n_rf_na:,}건")
if n_rf_na:
    bad_m = d.loc[d["rf"].isna(), "issue_month"].drop_duplicates().sort_values()
    print(f"  매칭 안 된 월: {[str(x)[:7] for x in bad_m[:12]]}")
print(f"rf 범위 {d['rf'].min() * 100:.2f}% ~ {d['rf'].max() * 100:.2f}%, "
      f"평균 {d['rf'].mean() * 100:.2f}%")
print("\nterm별 rf 평균 (기간 프리미엄: 36개월 < 60개월 이어야 정상)")
print((d.groupby("term_n")["rf"].mean() * 100).round(3).to_string())


section("6. spread 계산")

d["spread"] = d["irr_annual"] - d["rf"]
d["spread_contract"] = d["irr_contract"] - d["rf"]
# 0/1 판단값: 초과수익이 양수인가 (승인 라벨 후보)
d["spread_positive"] = (d["spread"] > 0).astype("int8")

print(f"spread 평균 {d['spread'].mean():.4f} / 표준편차 {d['spread'].std():.4f}")
print(f"spread > 0 비율: {d['spread_positive'].mean() * 100:.2f}%")

# 자본가중 샤프 (참고값)
w = d["funded_amnt"] / d["funded_amnt"].sum()
sharpe_all = float(np.average(d["spread"], weights=w) / d["spread"].std())
print(f"\n[참고] 전건 승인 샤프(자본가중): {sharpe_all:.4f}")
print("  part5 에서 -0.0421 로 보고됨. 음수면 '전건 승인은 국채만 못하다'는 뜻으로,")
print("  스크리닝 모델이 성립하는 근거가 된다.")


section("7. 검증")

ok = True


def check(cond, msg_ok, msg_fail):
    global ok
    if cond:
        print(f"[통과] {msg_ok}")
    else:
        print(f"[실패] {msg_fail}")
        ok = False


check(d["id"].is_unique and d["id"].notna().all(), "id 고유·결측 없음", "id 중복/결측")
check(int(d["irr_annual"].isna().sum()) == 0, "irr_annual 결측 0",
      f"irr_annual 결측 {int(d['irr_annual'].isna().sum()):,}")
check(n_rf_na == 0, "rf 결측 0", f"rf 결측 {n_rf_na:,}건")
check(int(d["spread"].isna().sum()) == 0, "spread 결측 0",
      f"spread 결측 {int(d['spread'].isna().sum()):,}")
check(bool(d["irr_annual"].between(*IRR_WINSOR).all()),
      f"irr_annual 윈저화 범위 {IRR_WINSOR}", "윈저화 범위 이탈")

# 방향성: 정상상환이 부도보다 수익률이 높아야 정상
m_good = float(d.loc[d["bad"] == 0, "irr_annual"].median())
m_bad = float(d.loc[d["bad"] == 1, "irr_annual"].median())
check(m_good > m_bad,
      f"정상({m_good:.4f}) > 부도({m_bad:.4f}) 방향 정상",
      f"방향 역전! 정상 {m_good:.4f} vs 부도 {m_bad:.4f}")
check(m_bad < 0, f"부도 대출 중앙 IRR 음수 ({m_bad:.4f})",
      f"부도인데 IRR 양수 ({m_bad:.4f}) — 회수금 이중계상 의심")

# 기간 프리미엄
rf36 = float(d.loc[d["term_n"] == 36, "rf"].mean())
rf60 = float(d.loc[d["term_n"] == 60, "rf"].mean())
check(rf36 < rf60, f"기간 프리미엄 정상 (36개월 {rf36 * 100:.2f}% < 60개월 {rf60 * 100:.2f}%)",
      f"기간 프리미엄 역전 ({rf36 * 100:.2f}% vs {rf60 * 100:.2f}%)")

print("\n[loan_status별 IRR]")
print(d.groupby("loan_status")[["irr_annual", "spread"]]
      .agg(["median", "mean", "count"]).round(4).to_string())

print("\n[대출 실행연도별 rf — 실제 금리 사이클과 맞는지]")
yr = pd.to_datetime(d["issue_month"]).dt.year
print((d.assign(yr=yr).groupby("yr")["rf"].mean() * 100).round(2).to_string())


section("8. 저장")

Y_COLS = ["id", "loan_status", "bad", "issue_month", "term_n", "k_months",
          "funded_amnt", "net_total",
          "irr_monthly", "irr_annual_raw", "irr_annual", "irr_contract",
          "rf", "spread", "spread_contract", "spread_positive"]
y = d[Y_COLS].copy()
y.to_csv(OUT_PATH, index=False, float_format="%.10g")
print(f"저장 → {os.path.basename(OUT_PATH)}  {y.shape[0]:,}행 × {y.shape[1]}열")
print(f"적용된 vintage 필터: {VINTAGE_FILTER}"
      + (f" (버퍼 {MATURITY_BUFFER_MONTHS}개월)" if VINTAGE_FILTER == "maturity" else "")
      + (f" (발행 {YEAR_CUTOFF}년 이하)" if VINTAGE_FILTER == "year" else ""))
print("\n주요 컬럼")
print("  irr_annual      보유기간 기준 연율 IRR (윈저화 완료) ← 기본 지표")
print("  irr_contract    계약 만기 기준 연율화 (대안 지표)")
print("  rf              만기·시점 매칭 국채수익률")
print("  spread          irr_annual - rf  ← LASSO 의 y")
print("  spread_positive spread > 0 (0/1 라벨 후보)")
print("\n다음 단계: X 테이블과 id 로 merge → 6:2:2 분할 → train 에서만 LASSO")

if not ok:
    raise SystemExit("\n!! 검증 실패 항목이 있습니다.")
print("\n전체 검증 통과.")

# %%
