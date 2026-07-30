

#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — 도메인 필터 X 전처리 (함수형 재구성)
==============================================

무엇을 다시 만들었나
-------------------
기존 `domain_filtered_X_code.py`(2,215줄)는 **5개의 독립 스크립트가 이어붙은**
형태였다. 각 스크립트가 `X_table_raw.csv`를 읽고 → 변환하고 → **같은 파일에
덮어쓰는** 방식이라, 전체를 한 프로세스에서 돌릴 수 없었다(중복실행 가드가
`SystemExit`를 낸다). 게다가 파트마다 1.2GB 백업을 떠서 디스크를 4배로 썼다.

이 파일은 **도메인 로직만 그대로 이식**하고, 버린 것은 다음뿐이다.
  · 파일 I/O 와 5번의 재적재      → 메모리에서 한 번에
  · 파트별 1.2GB 백업             → 불필요 (원본은 건드리지 않음)
  · `find_data_dir` / 중복실행 가드 → lc_config 가 대신함
  · 175만 행 전체 처리            → cohort 71.8만 행만

즉 **변환·대치·검증의 판단은 한 줄도 바꾸지 않았다.** 바꾼 것은 껍데기다.

누출 차단
--------
"데이터를 보고 값을 정하는" 지점을 전부 `lc_params.Params`를 거치게 만들었다.
fit 모드는 train 행만 보고 상수를 계산해 JSON에 저장하고, transform 모드는
데이터를 보지 않고 저장된 값만 적용한다. 호출부는 분기가 없다.

  P.quantile("annual_inc_cap", s, 0.995)   P.median("dti_med", s)
  P.value("yj_lambda_revol_bal", fn)       P.categories(...)

hot-deck 공여자 풀도 train 행에서만 뽑는다(fit 시 parquet 저장 → transform 시 재사용).

변수명 원칙 (기존 유지)
---------------------
기존 변수를 정제한 결과는 원래 이름을 유지한다. 원-핫처럼 여러 컬럼이 되는
경우만 원래 이름을 접두어로 쓴다. 원본에 없던 새 정보만 새 이름을 받는다.
"""

import os
import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import yeojohnson

# ══════════════════════════════════════════════════════════════════
# 설정 — 기존 파일의 상수를 그대로 옮겼다
# ══════════════════════════════════════════════════════════════════
ANNUAL_INC_CLIP_Q = 0.995   # 연소득 윈저라이징 분위수
ANNUAL_INC_FLOOR = 0        # 이하이면 입력 오류로 보고 결측 처리
DTI_SENTINEL = 999          # '측정 불가' 코드값
DTI_CAP = 100               # 센티넬 제거 후 상한
MONTH_SENTINEL = 999        # 경과월류 결측 대체값
HOTDECK_K = 200             # donor 탐색 개수

# purpose: 'other'(106,217건)보다 적은 범주는 전부 other 로 통합
PURPOSE_KEEP = ["debt_consolidation", "credit_card", "home_improvement", "other"]
PURPOSE_FALLBACK = "other"
HOME_KEEP = ["MORTGAGE", "RENT", "OWN", "OTHER"]
HOME_FALLBACK = "OTHER"

CENSUS_COLS = ["zip3_median_income", "zip3_median_home_value", "zip3_no_earnings_ratio"]
BANK_COL = "zip3_bank_concentration"

# 극단 왜도 → 발생여부 0/1 (크기보다 '있다/없다'가 정보)
BINARY_VARS = ["pub_rec", "tax_liens", "collections_12_mths_ex_med",
               "chargeoff_within_12_mths", "tot_coll_amt"]
LOG1P_VARS = ["delinq_2yrs", "num_accts_ever_120_pd", "delinq_amnt"]
MONTH_VARS = ["mths_since_last_delinq", "mths_since_last_major_derog",
              "mths_since_last_record", "mo_sin_old_rev_tl_op",
              "mo_sin_rcnt_tl", "mo_sin_rcnt_rev_tl_op", "mths_since_recent_bc"]
MEDIAN_VARS = ["tot_hi_cred_lim", "total_bc_limit", "bc_open_to_buy",
               "percent_bc_gt_75", "all_util", "max_bal_bc", "revol_util", "revol_bal"]

# log1p 는 과교정되어 왜도가 음수로 뒤집힌다 → Yeo-Johnson
YJ_VARS = ["revol_bal", "bc_open_to_buy", "tot_hi_cred_lim", "total_bc_limit", "max_bal_bc"]

ACCOUNT_DETAIL_COLS = [
    "acc_open_past_24mths", "num_tl_op_past_12m",
    "num_rev_accts", "num_op_rev_tl", "num_actv_rev_tl",
    "num_bc_tl", "num_bc_sats", "num_actv_bc_tl", "num_il_tl", "mort_acc",
]
HIERARCHY_PAIRS = [
    ("acc_open_past_24mths", "num_tl_op_past_12m"),
    ("num_rev_accts", "num_op_rev_tl"),
    ("num_op_rev_tl", "num_actv_rev_tl"),
    ("num_rev_accts", "num_bc_tl"),
    ("num_bc_tl", "num_bc_sats"),
    ("num_bc_tl", "num_actv_bc_tl"),
]
# 원본 관측값에서 위반 0건인 조합만 남겼다.
# num_bc_sats > open_acc 인 행이 6,939건 있어 open_acc 상한에서 제외했다
# (num_bc_sats 는 open_acc 에 안 잡히는 계좌도 세므로 애초에 성립하지 않는 제약).
UPPER_BOUND_COLUMNS = {
    "open_acc": ["num_op_rev_tl", "num_actv_rev_tl", "num_actv_bc_tl"],
    "total_acc": ["acc_open_past_24mths", "num_tl_op_past_12m", "num_rev_accts",
                  "num_bc_tl", "num_il_tl", "mort_acc",
                  "num_op_rev_tl", "num_actv_rev_tl", "num_bc_sats", "num_actv_bc_tl"],
}
HELPER_COLS = ["total_acc", "inq_last_12m", "inq_fi"]  # 처리 후 제거

SEC_APP_KEEP_NAN = [
    "sec_app_fico_avg", "sec_app_credit_history_mths", "sec_app_inq_last_6mths",
    "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_revol_util",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
]
DROP_COLS = ["annual_inc_joint", "dti_joint", "revol_bal_joint",
             "sec_app_fico_range_low", "sec_app_fico_range_high",
             "sec_app_earliest_cr_line", "issue_d", "zip_code"]

# 원본에서 읽어야 할 컬럼
SOURCE_COLS = sorted(set(
    ["id", "zip_code", "loan_amnt", "term", "purpose", "annual_inc", "dti",
     "emp_length", "home_ownership", "application_type", "issue_d",
     "fico_range_low", "fico_range_high", "earliest_cr_line",
     "open_acc", "inq_last_6mths", "mths_since_recent_inq"]
    + BINARY_VARS + LOG1P_VARS + MONTH_VARS + MEDIAN_VARS
    + ACCOUNT_DETAIL_COLS + HELPER_COLS
    + ["annual_inc_joint", "dti_joint", "revol_bal_joint",
       "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
       "sec_app_inq_last_6mths", "sec_app_open_act_il", "sec_app_num_rev_accts",
       "sec_app_revol_util", "sec_app_chargeoff_within_12_mths",
       "sec_app_collections_12_mths_ex_med"]
))


def _sec(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64, flush=True)


def _save_table(df, path):
    """donor 풀 저장. parquet 엔진이 없으면 csv.gz 로 자동 대체한다.

    pyarrow / fastparquet 는 파이썬 새 버전에 휠이 늦게 올라온다(3.14 등).
    전처리 전체가 여기서 멈출 이유가 없으므로 조용히 형식을 바꾼다.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError) as e:
        alt = os.path.splitext(path)[0] + ".csv.gz"
        df.to_csv(alt, index=False, compression="gzip")
        print(f"  [알림] parquet 엔진 없음 ({type(e).__name__}) → {os.path.basename(alt)} 로 저장")
        return alt


def _load_table(path):
    """_save_table 이 쓴 파일을 읽는다. parquet 이 없으면 csv.gz 를 찾는다."""
    alt = os.path.splitext(path)[0] + ".csv.gz"
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError):
            pass
    if os.path.exists(alt):
        return pd.read_csv(alt)
    raise FileNotFoundError(
        f"donor 풀을 찾을 수 없습니다: {path} / {alt}\n"
        "step2_run.py 의 fit 셀을 먼저 실행하세요.")


def yj_forward(x, lam):
    """Yeo-Johnson (x >= 0, lam != 0). 대상 변수는 전부 0 이상이다."""
    return ((x + 1.0) ** lam - 1.0) / lam


def yj_inverse(y, lam):
    return (y * lam + 1.0) ** (1.0 / lam) - 1.0


# ══════════════════════════════════════════════════════════════════
# 0. 적재
# ══════════════════════════════════════════════════════════════════
def load_source(raw_path, cohort_ids):
    """원본에서 cohort 행 · 도메인 컬럼만 읽는다.

    메모리 주의사항 두 가지:
      · 175만행×141열을 통째로 읽으면 터진다 → usecols + chunksize 필수
      · 전 컬럼을 dtype=str 로 읽으면 718k행×70열이 전부 문자열 객체가 되어
        수 GB를 먹는다 → 문자열이 꼭 필요한 컬럼만 str 로 읽고,
        나머지는 **청크 단위로 즉시 수치 변환**한 뒤 이어붙인다.

    id 가 숫자로 변환되지 않는 텍스트 행을 여기서 버리고 reset_index 한다
    (인덱스에 구멍이 생기면 이후 concat/대입에서 NaN 과 유령 행이 생긴다).
    """
    _sec("0. 원본 적재 (cohort 행만)")
    want = set(int(i) for i in cohort_ids)

    # 문자열로 남겨야 하는 컬럼 (범주·날짜·'14%' 형태)
    STR_COLS = {"id", "zip_code", "term", "purpose", "emp_length", "home_ownership",
                "application_type", "issue_d", "earliest_cr_line",
                "sec_app_earliest_cr_line", "revol_util"}

    parts = []
    for ch in pd.read_csv(raw_path, usecols=SOURCE_COLS, chunksize=200_000,
                          low_memory=False, dtype={c: str for c in STR_COLS}):
        ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
        ch = ch.loc[ch["id"].notna()]
        ch = ch.loc[ch["id"].astype("int64").isin(want)]
        if not len(ch):
            continue
        # ★ pandas 3.x 에서는 문자열 컬럼 dtype 이 object 가 아니라 StringDtype 이라
        #   `dtype == object` 로 판별하면 조용히 건너뛴다. is_numeric_dtype 으로 판별할 것.
        for c in ch.columns:
            if c in STR_COLS:
                continue
            if not pd.api.types.is_numeric_dtype(ch[c]):
                ch[c] = pd.to_numeric(
                    ch[c].astype(str).str.strip().str.rstrip("%"), errors="coerce")
        parts.append(ch)

    df = pd.concat(parts, ignore_index=True)
    del parts
    df["id"] = df["id"].astype("int64")

    # revol_util 은 '14%' 형태라 str 로 읽었다. 여기서 수치화한다.
    df["revol_util"] = pd.to_numeric(
        df["revol_util"].astype(str).str.strip().str.rstrip("%"), errors="coerce")

    # cohort 순서에 맞춘다 (split_mask 와 위치가 일치해야 한다)
    df = df.set_index("id").loc[[int(i) for i in cohort_ids]].reset_index()
    mem = df.memory_usage(deep=True).sum() / 1e6
    print(f"적재 {df.shape[0]:,}행 × {df.shape[1]}열  ({mem:,.0f} MB)")
    assert df["id"].is_unique and len(df) == len(cohort_ids), "cohort 행 수 불일치"
    return df


# ══════════════════════════════════════════════════════════════════
# Part 1 — 식별자/메타 (zip3 외부변수)
# ══════════════════════════════════════════════════════════════════
def part1_meta(df, P, zip3_cache_path):
    """Census ACS · FDIC 를 zip3 로 병합한다.

    개별 차주 정보만으로는 지역 경제 환경을 알 수 없다. 같은 소득이라도
    물가가 다른 지역이면 상환 여력이 다르다.
    """
    _sec("Part 1 — 식별자/메타 (zip3 외부변수)")
    cache = pd.read_csv(zip3_cache_path, dtype={"zip3": str})

    zc = df["zip_code"].astype(str).str.strip()
    df["zip3"] = zc.str[:3].where(zc.str.match(r"^\d{3}"), other=np.nan)
    df = df.merge(cache, on="zip3", how="left")

    n_na = int(df["zip3_median_income"].isna().sum())
    print(f"Census 매칭 실패 {n_na:,}건 ({n_na / len(df) * 100:.3f}%) "
          "— 군사우편 APO/FPO·사서함 전용 대역 등 지리적 실체가 없는 권역")
    for c in CENSUS_COLS:
        df[c] = df[c].fillna(P.median(f"p1_{c}_med", df[c]))

    n_bk = int(df[BANK_COL].isna().sum())
    print(f"FDIC 매칭 실패   {n_bk:,}건 ({n_bk / len(df) * 100:.3f}%) "
          "— 은행 지점이 실제로 없는 권역이라 Census 보다 실패율이 높음")
    df[BANK_COL] = df[BANK_COL].fillna(P.median(f"p1_{BANK_COL}_med", df[BANK_COL]))

    return df.drop(columns=["zip3"])


# ══════════════════════════════════════════════════════════════════
# Part 2 — 대출 자체 조건
# ══════════════════════════════════════════════════════════════════
def part2_loan_terms(df, P):
    _sec("Part 2 — 대출 자체 조건")

    # loan_amnt: 결측 0%, 왜도 0.78, 500~40,000 로 제도적 상한이 있어 변환 없이 사용
    print(f"[loan_amnt] 왜도 {df['loan_amnt'].skew():.3f} → 변환 없이 사용")

    # term: 원본이 ' 36 months' 처럼 앞 공백이 있어 문자열 비교 대신 숫자만 추출
    tm = pd.to_numeric(df["term"].astype(str).str.strip().str.extract(r"(\d+)")[0],
                       errors="coerce")
    bad = set(tm.dropna().unique()) - {36, 60}
    if bad:
        raise ValueError(f"예상치 못한 term 값: {bad}")
    if tm.isna().any():
        raise ValueError(f"term 변환 실패 {int(tm.isna().sum()):,}건")
    df["term"] = (tm == 60).astype("int8")
    print(f"[term] 이진 인코딩 — 60개월 {df['term'].mean() * 100:.2f}%")

    # purpose: 희소범주 통합 후 원-핫.
    # ★ 범주 목록을 train 에서 고정하는 것이 이번 개편의 핵심 중 하나다.
    #   고정하지 않으면 val/test 에서 원-핫 열 구성이 달라져 모델이 아예 안 돌아간다.
    pc = df["purpose"].astype(str).str.strip().str.lower()
    cats = P.value("p2_purpose_cats", lambda: list(PURPOSE_KEEP))
    merged = sorted(set(pc.unique()) - set(cats) - {"nan"})
    n_m = int(pc.isin(merged).sum())
    print(f"[purpose] 단독 유지 {cats}")
    print(f"  '{PURPOSE_FALLBACK}' 통합 {len(merged)}개 범주 {n_m:,}건 "
          f"({n_m / len(df) * 100:.2f}%)")
    pc = pc.where(pc.isin(cats), PURPOSE_FALLBACK)
    dm = pd.get_dummies(pd.Categorical(pc, categories=cats), prefix="purpose", dtype="int8")
    dm.index = df.index
    df = pd.concat([df, dm], axis=1).drop(columns=["purpose"])
    print(f"  원-핫 {list(dm.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════
# Part 3 — 차주 신용 프로필
# ══════════════════════════════════════════════════════════════════
def part3_borrower(df, P):
    _sec("Part 3 — 차주 신용 프로필")

    # ── annual_inc ──
    # 순서가 중요하다: 하한 결측화 → 윈저라이징 → 중앙값 대치 → (마지막에) log1p
    # 로그를 먼저 취하면 극단값이 이미 압축돼 윈저라이징 효과가 희석된다.
    print("[1] annual_inc")
    n_zero = int((df["annual_inc"] <= ANNUAL_INC_FLOOR).sum())
    df.loc[df["annual_inc"] <= ANNUAL_INC_FLOOR, "annual_inc"] = np.nan
    print(f"  하한 {ANNUAL_INC_FLOOR:,} 이하 {n_zero:,}건 결측 처리 "
          "(대출 승인 건에 소득 0 은 구조적으로 불가능)")

    cap = P.quantile("p3_annual_inc_cap", df["annual_inc"], ANNUAL_INC_CLIP_Q)
    n_clip = int((df["annual_inc"] > cap).sum())
    df["annual_inc"] = df["annual_inc"].clip(upper=cap)
    print(f"  상한 {cap:,.0f} 윈저라이징 ({n_clip:,}건)")

    med = P.median("p3_annual_inc_med", df["annual_inc"])
    n_na = int(df["annual_inc"].isna().sum())
    df["annual_inc"] = df["annual_inc"].fillna(med)
    print(f"  결측 {n_na:,}건 → 중앙값 {med:,.0f}")

    # ── fico_avg ── low/high 가 거의 완전상관이라 평균 하나로 줄인다
    print("[2] fico_avg")
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"])
    print(f"  범위 {df['fico_avg'].min():.0f}~{df['fico_avg'].max():.0f}, 원본 2개 삭제")

    # ── dti ── 999 는 '측정 불가' 코드값이지 실제 부채비율이 아니다
    print("[3] dti")
    n_sent = int((df["dti"] >= DTI_SENTINEL).sum())
    n_neg = int((df["dti"] < 0).sum())
    df.loc[df["dti"] >= DTI_SENTINEL, "dti"] = np.nan
    df.loc[df["dti"] < 0, "dti"] = np.nan
    df["dti"] = df["dti"].clip(upper=DTI_CAP)
    dmed = P.median("p3_dti_med", df["dti"])
    n_na = int(df["dti"].isna().sum())
    df["dti"] = df["dti"].fillna(dmed)
    print(f"  센티넬 {n_sent:,} / 음수 {n_neg:,} / 상한 {DTI_CAP} 클리핑 "
          f"/ 결측 {n_na:,}건 → 중앙값 {dmed:.2f}")

    # ── emp_length ── '10년 이상'과 '1년 미만'에 순서가 있다
    print("[4] emp_length — 순서형")
    EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
               "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
               "10+ years": 10}
    ec = df["emp_length"].astype(str).str.strip()
    eo = ec.map(EMP_MAP)
    unmapped = set(ec[eo.isna() & (ec != "nan")].unique())
    if unmapped:
        print(f"  [주의] 매핑되지 않은 값: {unmapped}")
    # 결측은 '무직'을 뜻하므로 더미로 표시하고 순서형은 0 으로 채운다.
    # 더미가 있으므로 모델은 '무직'과 '1년 미만 재직'을 구분할 수 있다.
    df["emp_length_missing"] = eo.isna().astype("int8")
    df["emp_length"] = eo.fillna(0).astype("int8")
    print(f"  0~10 인코딩 | 무직 {df['emp_length_missing'].mean() * 100:.2f}%")

    # ── home_ownership ──
    print("[5] home_ownership — 원-핫")
    hc = df["home_ownership"].astype(str).str.strip().str.upper()
    hcats = P.value("p3_home_cats", lambda: list(HOME_KEEP))
    hmerged = sorted(set(hc.unique()) - set(hcats) - {"NAN"})
    print(f"  '{HOME_FALLBACK}' 통합: {hmerged}")
    hc = hc.where(hc.isin(hcats), HOME_FALLBACK)
    hd = pd.get_dummies(pd.Categorical(hc, categories=hcats),
                        prefix="home_ownership", dtype="int8")
    hd.index = df.index
    df = pd.concat([df, hd], axis=1).drop(columns=["home_ownership"])
    print(f"  {list(hd.columns)}")

    # ── application_type ──
    df["application_type"] = (df["application_type"].astype(str).str.strip()
                              .str.lower().str.startswith("joint").astype("int8"))
    print(f"[6] application_type — Joint {df['application_type'].mean() * 100:.2f}%")

    # ── income_to_zip3_median ──
    # ★ 반드시 log1p '전'에 계산한다. 로그 후에 계산하면 분자가 로그 스케일이 되어
    #   '지역 중위소득 대비 배수'라는 변수의 의미가 무너진다.
    df["income_to_zip3_median"] = df["annual_inc"] / df["zip3_median_income"].replace(0, np.nan)
    print(f"[7] income_to_zip3_median 중앙값 {df['income_to_zip3_median'].median():.3f}")

    # ── annual_inc log1p (제자리) ──
    df["annual_inc"] = np.log1p(df["annual_inc"])
    print(f"[8] annual_inc log1p → 왜도 {df['annual_inc'].skew():.3f} "
          "(이제 로그 스케일. 원 단위 해석 시 expm1 필요)")
    return df


# ══════════════════════════════════════════════════════════════════
# Part 4 — 연체/부실 이력 (가장 복잡한 부분)
# ══════════════════════════════════════════════════════════════════
def part4_delinquency(df, P, donor_mask=None, donor_pool_path=None):
    """4-1 ~ 4-5.

    donor_mask : hot-deck 공여자로 쓸 행 (= train). None 이면 전체에서 뽑으므로 누출.
    """
    _sec("Part 4 — 연체/부실 이력")
    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    # ── B. 4-1 연체 이력 ──────────────────────────────────────────
    # 이 3개의 결측은 '값을 모른다'가 아니라 '해당 이력 자체가 없다'는 뜻이다.
    # 999 로 덮기 전에 플래그로 먼저 보존한다.
    df["has_delinq_history"] = df["mths_since_last_delinq"].notna().astype("int8")
    df["has_major_derog_history"] = df["mths_since_last_major_derog"].notna().astype("int8")
    df["has_public_record"] = df["mths_since_last_record"].notna().astype("int8")
    print(f"[이력 플래그] delinq {df['has_delinq_history'].mean() * 100:.2f}% / "
          f"major_derog {df['has_major_derog_history'].mean() * 100:.2f}% / "
          f"public_record {df['has_public_record'].mean() * 100:.2f}%")

    # 2012년 이전에는 major_derog 항목 자체가 수집되지 않아 100% 결측이다(스키마 결측).
    # '이력이 없어서 결측'인 것과 구분해야 모델이 오해하지 않는다.
    df["major_derog_data_available"] = (issue_dt.dt.year >= 2012).fillna(False).astype("int8")
    print(f"[major_derog_data_available] {df['major_derog_data_available'].mean() * 100:.2f}%")

    # 극단 왜도 → 0/1. 대부분 0 이고 일부만 큰 값이라 크기보다 '있다/없다'가 정보다.
    print("\n[극단 왜도 → 발생여부 0/1]")
    for c in BINARY_VARS:
        sk = float(df[c].skew())
        df[c] = (df[c] > 0).astype("int8")   # 결측은 자동으로 0(발생 없음)
        print(f"  {c:32s} 왜도 {sk:9.2f} → 1 비율 {df[c].mean() * 100:5.2f}%")

    print("\n[log1p 제자리]")
    for c in LOG1P_VARS:
        n = int(df[c].isna().sum())
        df[c] = df[c].fillna(0)              # 결측 2.4% 이하, 0 대체 시 손실 미미
        sk = float(df[c].skew())
        df[c] = np.log1p(df[c])
        print(f"  {c:32s} 결측 {n:>7,} 0대체 | 왜도 {sk:8.2f} → {df[c].skew():.2f}")

    # ── C. 4-2·4-4 결측 구조 플래그 ───────────────────────────────
    # all_util, max_bal_bc 의 결측은 "값이 없다"가 아니라 "그 시절엔 항목이 없었다".
    df["data_available_2015_12"] = (issue_dt >= pd.Timestamp("2015-12-01")).astype("int8")
    print(f"\n[data_available_2015_12] {df['data_available_2015_12'].mean() * 100:.2f}% "
          f"(all_util 결측률 {df['all_util'].isna().mean() * 100:.2f}%)")

    # tot_hi_cred_lim 등 5개는 결측 패턴이 거의 일치하는 동일 집단에서 발생한다.
    # 5개를 각각 플래그로 만들면 완전 중복되어 다중공선성이 생기므로 대표 1개만.
    systemic = ["tot_hi_cred_lim", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_tl",
                "mo_sin_rcnt_rev_tl_op", "total_bc_limit"]
    base = df["tot_hi_cred_lim"].isna()
    print("[시스템성 결측 패턴 일치율]")
    for c in systemic:
        print(f"  {c:25s} {(df[c].isna() == base).mean():.4f}")
    df["credit_info_linked"] = df["tot_hi_cred_lim"].notna().astype("int8")

    # 결측률이 시스템성 그룹보다 살짝 높아 완전히 같은 집단이 아닐 가능성 → 개별 플래그
    for c in ["mths_since_recent_bc", "bc_open_to_buy", "percent_bc_gt_75"]:
        df[f"{c}_available"] = df[c].notna().astype("int8")

    # ── D. 4-3·4-5 계좌 수 변수 ───────────────────────────────────
    print("\n[D] 계좌 수 변수")
    # 기준 변수를 먼저 채워야 뒤 단계의 조건이 조용히 False 가 되는 사고를 막는다.
    # ★ 상수는 결측이 없더라도 **항상** 학습·저장한다.
    #   fit 때 결측이 0이라 건너뛰면, 나중에 결측이 있는 데이터(예: 7/27 배부 test)를
    #   transform 할 때 "저장되지 않은 상수" KeyError 로 멈춘다.
    #   조건부 저장은 fit/transform 구조에서 반드시 사고를 낸다.
    for c in ["open_acc", "inq_last_6mths", "total_acc"]:
        n = int(df[c].isna().sum())
        m = P.median(f"p4_{c}_med", df[c])
        df[c] = df[c].fillna(m)
        if n:
            print(f"  {c} 결측 {n:,}건 → 중앙값 {m:g} (기준 변수라 선행 처리)")

    # 노트북은 open_acc > total_acc 인 행을 제거하지만 여기서는 행을 보존해야 한다
    # (y 테이블과 id 로 병합해야 하므로). total_acc 는 상한 검증용 보조 컬럼이니
    # 모순 행만 total_acc 를 open_acc 로 올려 상한이 성립하게 만든다.
    viol = (df["open_acc"] > df["total_acc"]).fillna(False)
    if int(viol.sum()):
        df.loc[viol, "total_acc"] = df.loc[viol, "open_acc"]
    print(f"  open_acc > total_acc 모순 {int(viol.sum()):,}건 → total_acc 보정 (행 제거 안 함)")

    detail_missing_before = df[ACCOUNT_DETAIL_COLS].isna().copy()
    print(f"  세부 계좌 변수 결측 행 {int(detail_missing_before.any(axis=1).sum()):,}건 "
          f"({detail_missing_before.any(axis=1).mean() * 100:.2f}%)")

    # D-1. 산식 대치 — 회전계좌 ≈ 전체계좌 - 할부 - 모기지, 단 현재 열린 회전계좌가 하한
    fm = (df["num_rev_accts"].isna()
          & df[["total_acc", "num_il_tl", "mort_acc"]].notna().all(axis=1))
    if fm.any():
        val = df["total_acc"] - df["num_il_tl"] - df["mort_acc"]
        val = pd.concat([val, df["num_op_rev_tl"]], axis=1).max(axis=1).clip(lower=0).round()
        df.loc[fm, "num_rev_accts"] = val.loc[fm]
    print(f"  num_rev_accts 산식 대치 {int(fm.sum()):,}건")

    # D-2. 논리적으로 확정 가능한 0 (기존 관측값은 건드리지 않고 NaN 셀만)
    zero_rules = [
        (df["total_acc"].eq(0), ACCOUNT_DETAIL_COLS, "total_acc=0"),
        (df["open_acc"].eq(0),
         ["num_op_rev_tl", "num_actv_rev_tl", "num_bc_sats", "num_actv_bc_tl"], "open_acc=0"),
        (df["revol_bal"].eq(0), ["num_actv_rev_tl", "num_actv_bc_tl"], "revol_bal=0"),
        (df["acc_open_past_24mths"].eq(0), ["num_tl_op_past_12m"], "acc_open_past_24mths=0"),
        (df["total_bc_limit"].eq(0), ["num_actv_bc_tl"], "total_bc_limit=0"),
    ]
    n_cells = 0
    for cond, cols, _ in zero_rules:
        cond = cond.fillna(False)
        for c in cols:
            m = cond & df[c].isna()
            k = int(m.sum())
            if k:
                df.loc[m, c] = 0.0
                n_cells += k
    print(f"  0 규칙 대치 {n_cells:,}셀")

    # D-3. mths_since_recent_inq — 조회 이력을 교차 확인해 구간 판정
    miss = df["mths_since_recent_inq"].isna()
    m_0_6 = miss & df["inq_last_6mths"].gt(0)
    m_6_12 = miss & df["inq_last_6mths"].eq(0) & df["inq_last_12m"].gt(0)
    m_none = (miss & df["inq_last_6mths"].eq(0)
              & (df["inq_last_12m"].isna() | df["inq_last_12m"].eq(0))
              & (df["inq_fi"].isna() | df["inq_fi"].eq(0)))
    # ★ sentinel 은 관측 최댓값에서 나오므로 fit 대상이다 (원 코드는 전체에서 계산했다)
    sent = P.value("p4_inq_sentinel",
                   lambda: float(np.nanmax(df["mths_since_recent_inq"].to_numpy(dtype=float))) + 1)
    df.loc[m_0_6, "mths_since_recent_inq"] = 3.0
    df.loc[m_6_12, "mths_since_recent_inq"] = 9.0
    df.loc[m_none, "mths_since_recent_inq"] = sent
    n_amb = int(df["mths_since_recent_inq"].isna().sum())
    df["mths_since_recent_inq"] = df["mths_since_recent_inq"].fillna(sent)
    print(f"  mths_since_recent_inq 결측 {int(miss.sum()):,} 분해 → "
          f"0~6개월 {int(m_0_6.sum()):,} / 6~12개월 {int(m_6_12.sum()):,} / "
          f"조회없음 {int(m_none.sum()):,} / 확인불가 {n_amb:,} (sentinel {sent:g})")
    # 부도율과의 관계가 초기에 급하고 이후 완만해지는 패턴을 반영해 로그 변환
    df["mths_since_recent_inq"] = np.log1p(df["mths_since_recent_inq"])

    # D-4. 계층 제약 공동 hot-deck
    df = _hotdeck(df, P, detail_missing_before, donor_mask, donor_pool_path)

    # ── E. 마무리 ─────────────────────────────────────────────────
    print("\n[E] 마무리")
    for c in MONTH_VARS:
        n = int(df[c].isna().sum())
        df[c] = df[c].fillna(MONTH_SENTINEL)
        if n:
            print(f"  {c:32s} {n:>8,}건 → {MONTH_SENTINEL}")

    for c in MEDIAN_VARS:
        n = int(df[c].isna().sum())
        m = P.median(f"p4_{c}_med", df[c])
        df[c] = df[c].fillna(m)
        if n:
            print(f"  {c:32s} {n:>8,}건 → 중앙값 {m:,.2f}")

    # credit_age_months — 날짜 자체가 아니라 '신용거래 시작 후 경과'가 정보다
    ear = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_age_months"] = (issue_dt - ear).dt.days / 30.44
    n_neg = int((df["credit_age_months"] < 0).sum())
    df.loc[df["credit_age_months"] < 0, "credit_age_months"] = np.nan
    cam = P.median("p4_credit_age_med", df["credit_age_months"])
    df["credit_age_months"] = df["credit_age_months"].fillna(cam)
    df = df.drop(columns=["earliest_cr_line"])
    print(f"  credit_age_months 파생 (음수 {n_neg:,}건 결측화 → 중앙값 {cam:.1f}), "
          "원본 earliest_cr_line 삭제")

    # Yeo-Johnson — log1p 는 과교정되어 왜도가 음수로 뒤집힌다
    # ★ 원 코드는 조원 노트북의 하드코딩 λ 를 썼다(전체 데이터 기준 = 누출).
    #   여기서는 train 에서 다시 추정한다.
    print("\n[Yeo-Johnson]")
    for c in YJ_VARS:
        lam = P.yeojohnson_lambda(f"p4_yj_lambda_{c}", df[c])
        sk = float(df[c].skew())
        df[c] = yeojohnson(df[c].to_numpy(dtype=float), lmbda=lam)
        print(f"  {c:20s} λ={lam:<8.4f} 왜도 {sk:8.2f} → {df[c].skew():.3f}")

    return df, {c: float(P.store.get(f"p4_yj_lambda_{c}", np.nan)) for c in YJ_VARS}


def _hotdeck(df, P, detail_missing_before, donor_mask, donor_pool_path):
    """계층 제약 공동 hot-deck.

    한 행의 세부 계좌 변수를 **같은 donor 에서** 가져와 변수 간 결합구조를 보존한다.
    개별 변수를 따로 대치하면 "회전계좌 3개인데 그중 활성이 5개" 같은 모순이 생긴다.

    누출 차단: donor 는 train 행에서만 뽑는다. fit 시 donor 풀을 저장하고
    transform 시 그대로 재사용한다.
    """
    recip = df[ACCOUNT_DETAIL_COLS].isna().any(axis=1)
    n_recip = int(recip.sum())
    print(f"\n  hot-deck 대상 행 {n_recip:,}건", flush=True)
    if not n_recip:
        return df

    match_numeric = ["open_acc", "total_acc", "revol_bal", "revol_util",
                     "inq_last_6mths", "mths_since_recent_inq",
                     "annual_inc", "dti", "fico_avg", "loan_amnt"]
    home_cols = [c for c in df.columns if c.startswith("home_ownership_")]

    mf = df[match_numeric].astype(float).copy()
    # 큰 양의 왜도를 줄여 거리 계산이 금액 변수에 지배되지 않게 한다
    for c in ["revol_bal", "loan_amnt"]:
        mf[c] = np.log1p(mf[c].clip(lower=0))
    mf = pd.concat([mf, df[home_cols].astype(float)], axis=1)

    if P.mode == "fit":
        # donor = 완전 관측 + 비음수 정수 + 모든 hard constraint 만족 + train 행
        dm = df[ACCOUNT_DETAIL_COLS].notna().all(axis=1)
        dm &= df[ACCOUNT_DETAIL_COLS].ge(0).all(axis=1)
        dv = df[ACCOUNT_DETAIL_COLS].to_numpy(dtype=float)
        dm &= pd.Series(np.isclose(dv, np.rint(dv), rtol=0, atol=1e-9).all(axis=1),
                        index=df.index)
        del dv
        for p_, c_ in HIERARCHY_PAIRS:
            dm &= df[p_].ge(df[c_])
        for b_, cols in UPPER_BOUND_COLUMNS.items():
            for c_ in cols:
                dm &= df[c_].le(df[b_])
        dm = dm.fillna(False)
        if donor_mask is not None:
            dm &= pd.Series(np.asarray(donor_mask, dtype=bool), index=df.index)
            print(f"  donor 를 train 행으로 제한")
        else:
            print("  [경고] donor_mask 없음 — 전체에서 donor 를 뽑습니다 (누출)")

        # 거리 표준화 상수도 donor(=train)에서만 계산한다
        center = P.value("p4_hd_center", lambda: mf.loc[dm].median().to_dict())
        mu = P.value("p4_hd_mu", lambda: mf.loc[dm].mean().to_dict())
        sd = P.value("p4_hd_sd", lambda: mf.loc[dm].std().replace(0, 1).to_dict())

        pool = pd.concat([mf.loc[dm], df.loc[dm, ACCOUNT_DETAIL_COLS],
                          df.loc[dm, ["open_acc", "total_acc"]].add_prefix("bnd_")], axis=1)
        saved = _save_table(pool, donor_pool_path)
        print(f"  donor {int(dm.sum()):,}건 → {os.path.basename(saved)} 저장")
    else:
        pool = _load_table(donor_pool_path)
        print(f"  저장된 donor {len(pool):,}건 로드")
        # transform 모드에서는 저장된 값만 읽는다 (없으면 KeyError — fit 누락 방지)
        center = P.store["p4_hd_center"]
        mu = P.store["p4_hd_mu"]
        sd = P.store["p4_hd_sd"]

    center, mu, sd = pd.Series(center), pd.Series(mu), pd.Series(sd)

    feat = list(mf.columns)
    mf = mf.fillna(center[feat])          # 거리 계산용. 최종 대치에는 쓰지 않는다
    ms = (mf - mu[feat]) / sd[feat]
    ms[["open_acc", "total_acc"]] *= 2.0  # 계좌 총계가 비슷한 donor 를 우선
    del mf

    dpool = ((pool[feat].fillna(center[feat]) - mu[feat]) / sd[feat])
    dpool[["open_acc", "total_acc"]] *= 2.0

    recip_pos = np.flatnonzero(recip.to_numpy())
    donor_mat = np.ascontiguousarray(dpool.to_numpy(dtype=np.float32))
    recip_mat = np.ascontiguousarray(ms.to_numpy(dtype=np.float32)[recip_pos])
    del ms, dpool

    print(f"  KD트리 구축 ({donor_mat.shape[0]:,} × {donor_mat.shape[1]}차원)", flush=True)
    tree = cKDTree(donor_mat, compact_nodes=True, balanced_tree=True)
    k = min(HOTDECK_K, len(donor_mat))
    del donor_mat

    donor_detail = pool[ACCOUNT_DETAIL_COLS].to_numpy(dtype=float)
    recip_detail = df.iloc[recip_pos][ACCOUNT_DETAIL_COLS].to_numpy(dtype=float)
    recip_na = np.isnan(recip_detail)
    cpos = {c: i for i, c in enumerate(ACCOUNT_DETAIL_COLS)}
    bnd_vals = {b: df.iloc[recip_pos][b].to_numpy(dtype=float) for b in UPPER_BOUND_COLUMNS}

    # ★ 질의와 제약 검증을 한 배치 루프로 묶는다.
    #   k=200 이면 이웃 배열이 (수여자수 × 200) 이고, 제약을 검사할 때마다
    #   같은 크기의 실수 배열이 새로 생긴다. 수여자가 20만이면 배열 하나가 320MB라
    #   전체를 한 번에 들고 있으면 터진다. 배치마다 만들고 즉시 버린다.
    n_r = len(recip_pos)
    chosen = np.empty(n_r, dtype=np.int64)
    n_has_valid = 0
    BATCH = 20_000
    print(f"  donor {k}개 탐색 — {n_r:,}행을 {BATCH:,}씩 배치 처리", flush=True)

    for st in range(0, n_r, BATCH):
        en = min(st + BATCH, n_r)
        _, nb_b = tree.query(recip_mat[st:en], k=k, workers=-1)
        if k == 1:
            nb_b = nb_b[:, None]
        na_b = recip_na[st:en]
        rd_b = recip_detail[st:en]
        valid = np.ones(nb_b.shape, dtype=bool)

        # 수여자의 '관측된' 값과 충돌하는 donor 후보를 제외한다
        for parent, child in HIERARCHY_PAIRS:
            pp, cp = cpos[parent], cpos[child]
            a = na_b[:, pp] & ~na_b[:, cp]
            b = na_b[:, cp] & ~na_b[:, pp]
            if a.any():
                valid[a] &= donor_detail[nb_b[a], pp] >= rd_b[a, cp, None]
            if b.any():
                valid[b] &= donor_detail[nb_b[b], cp] <= rd_b[b, pp, None]
        for bnd, cols in UPPER_BOUND_COLUMNS.items():
            rb = bnd_vals[bnd][st:en]
            for c in cols:
                bp = cpos[c]
                need = na_b[:, bp]
                if need.any():
                    valid[need] &= donor_detail[nb_b[need], bp] <= rb[need, None]

        hv = valid.any(axis=1)
        rank = np.argmax(valid, axis=1)
        rank[~hv] = 0
        chosen[st:en] = nb_b[np.arange(en - st), rank]
        n_has_valid += int(hv.sum())
        del valid, nb_b, na_b, rd_b
        if (st // BATCH) % 5 == 0:
            print(f"    {en:,}/{n_r:,}", flush=True)

    del tree, recip_mat
    print(f"  유효 donor 확보 {n_has_valid:,} / {n_r:,}", flush=True)

    sel = donor_detail[chosen]
    imputed = recip_detail.copy()
    imputed[recip_na] = sel[recip_na]
    df.loc[df.index[recip_pos], ACCOUNT_DETAIL_COLS] = imputed
    del donor_detail, recip_detail, sel, imputed

    # 제약이 서로 얽혀 있어 한 번만 훑으면 부족하다.
    # 예: num_bc_tl >= num_bc_sats 를 맞추려 num_bc_tl 을 올리면
    #     앞서 통과했던 num_rev_accts >= num_bc_tl 이 다시 깨진다.
    # 대치된 셀만 조정하며 위반이 없을 때까지 반복한다(관측값은 모든 제약을 만족하므로 수렴).
    n_rep, sweep = 0, 0
    for sweep in range(20):
        changed = 0
        for bnd, cols in UPPER_BOUND_COLUMNS.items():
            for c in cols:
                m = df[c].gt(df[bnd]) & detail_missing_before[c]
                if int(m.sum()):
                    df.loc[m, c] = df.loc[m, bnd]
                    changed += int(m.sum())
        for parent, child in HIERARCHY_PAIRS:
            m = df[parent].lt(df[child]) & detail_missing_before[parent]
            if int(m.sum()):
                df.loc[m, parent] = df.loc[m, child]
                changed += int(m.sum())
            m = df[parent].lt(df[child]) & detail_missing_before[child]
            if int(m.sum()):
                df.loc[m, child] = df.loc[m, parent]
                changed += int(m.sum())
        n_rep += changed
        if changed == 0:
            break
    else:
        print("  [주의] 20회 반복 후에도 제약 위반이 남았습니다")
    print(f"  제약 위반 보정 {n_rep:,}셀 ({sweep + 1}회 반복 후 수렴)")

    for c in ACCOUNT_DETAIL_COLS:
        m = detail_missing_before[c]
        df.loc[m, c] = np.rint(df.loc[m, c].clip(lower=0))

    # 위와 같은 이유로 잔여 결측용 중앙값도 항상 학습·저장한다
    left = int(df[ACCOUNT_DETAIL_COLS].isna().sum().sum())
    if left:
        print(f"  잔여 결측 {left}셀 → 중앙값")
    for c in ACCOUNT_DETAIL_COLS:
        df[c] = df[c].fillna(P.median(f"p4_{c}_left_med", df[c]))
    return df


# ══════════════════════════════════════════════════════════════════
# Part 6 — 공동신청자
# ══════════════════════════════════════════════════════════════════
def part6_joint(df, P, yj_lambdas):
    """joint 값 통합 + sec_app_* 파생.

    통합 대상 3개는 이미 변환된 상태(annual_inc=log1p, revol_bal=YJ)라
    **역변환 → 통합 → 재변환** 순서를 지켜야 한다. 스케일이 섞이면 값이 무의미해진다.
    """
    _sec("Part 6 — 공동신청자")
    df = df.rename(columns={"application_type": "is_joint_application"})
    df["is_joint_application"] = df["is_joint_application"].astype("int8")
    print(f"Joint {int(df['is_joint_application'].sum()):,}건 "
          f"({df['is_joint_application'].mean() * 100:.2f}%)")

    # ── annual_inc ── 현재 log1p 상태. 원 단위인 joint 와 합치려면 되돌려야 한다
    print("\n[annual_inc] 로그 역변환 → 통합 → 재정제 → 재변환")
    inc_raw = np.expm1(df["annual_inc"])
    n_j = int(df["annual_inc_joint"].notna().sum())
    inc = df["annual_inc_joint"].fillna(inc_raw)
    inc = inc.mask(inc <= ANNUAL_INC_FLOOR)
    cap = P.quantile("p6_annual_inc_cap", inc, ANNUAL_INC_CLIP_Q)
    inc = inc.clip(upper=cap)
    med = P.median("p6_annual_inc_med", inc)
    inc = inc.fillna(med)
    df["annual_inc"] = np.log1p(inc)
    print(f"  joint 사용 {n_j:,} / 개인 {len(df) - n_j:,} | 상한 {cap:,.0f} 중앙값 {med:,.0f}")
    print(f"  → 왜도 {df['annual_inc'].skew():.3f}")

    # ── dti ── 스케일 변환이 없어 역변환 불필요. 같은 정제만 다시 적용
    print("\n[dti]")
    n_j = int(df["dti_joint"].notna().sum())
    dti = df["dti_joint"].fillna(df["dti"])
    dti = dti.mask(dti >= DTI_SENTINEL).mask(dti < 0).clip(upper=DTI_CAP)
    dmed = P.median("p6_dti_med", dti)
    df["dti"] = dti.fillna(dmed)
    print(f"  joint 사용 {n_j:,} / 개인 {len(df) - n_j:,} | 중앙값 {dmed:.2f}")

    # ── revol_bal ── 현재 YJ 상태. log1p 는 과교정이라 YJ 를 채택했으므로 통합 후에도 YJ 유지
    print("\n[revol_bal] YJ 역변환 → 통합 → 재변환")
    lam = yj_lambdas["revol_bal"]
    bal_raw = pd.Series(yj_inverse(df["revol_bal"].to_numpy(dtype=float), lam),
                        index=df.index).clip(lower=0)
    n_j = int(df["revol_bal_joint"].notna().sum())
    bal = df["revol_bal_joint"].fillna(bal_raw).clip(lower=0)
    bal = bal.fillna(P.median("p6_revol_bal_med", bal))   # 조건부 저장 금지 (위 주석 참조)
    df["revol_bal"] = yj_forward(bal.to_numpy(dtype=float), lam)
    print(f"  joint 사용 {n_j:,} (λ={lam:.4f}) → 왜도 {df['revol_bal'].skew():.3f}")

    # ── sec_app_* : 결측 93%. 대치하면 없는 정보를 지어내는 것이므로 NaN 유지 ──
    # 선형 모델에서는 제외하고, NaN 을 분기 조건으로 다룰 수 있는 트리 모델에서만 쓴다.
    print("\n[sec_app_*] NaN 유지 (결측 93% — 대치하지 않음)")
    df["sec_app_fico_avg"] = (df["sec_app_fico_range_low"] + df["sec_app_fico_range_high"]) / 2

    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    sec_dt = pd.to_datetime(df["sec_app_earliest_cr_line"], format="%b-%Y", errors="coerce")
    mths = (issue_dt.dt.year - sec_dt.dt.year) * 12 + (issue_dt.dt.month - sec_dt.dt.month)
    n_neg = int((mths < 0).sum())
    df["sec_app_credit_history_mths"] = mths.mask(mths < 0)  # 신용개설이 대출일보다 뒤면 오류
    print(f"  sec_app_fico_avg 결측 {df['sec_app_fico_avg'].isna().mean() * 100:.2f}%")
    print(f"  sec_app_credit_history_mths 파생 (음수 {n_neg:,}건 결측화)")

    for c in ["sec_app_open_act_il", "sec_app_num_rev_accts"]:
        sk = float(df[c].skew())
        df[c] = np.log1p(df[c])   # NaN 은 log1p 후에도 NaN
        print(f"  {c:28s} log1p 왜도 {sk:6.2f} → {df[c].skew():.3f}")

    # 나머지 4개는 원본 유지:
    #   sec_app_revol_util — 100% 초과 구간에서도 부도율 차이가 확인되어 캡핑하지 않음
    #   chargeoff/collections — 2건 이상 케이스가 수백~천 건이라 0/1 로 바꾸면 심각도 손실
    print("  원본 유지 4개: sec_app_inq_last_6mths, sec_app_revol_util, "
          "sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med")

    drop = [c for c in DROP_COLS + HELPER_COLS if c in df.columns]
    df = df.drop(columns=drop)
    print(f"\n컬럼 정리 — 삭제 {len(drop)}개: {drop}")
    print("  total_acc: 세부 계좌수의 합에 가까워 VIF 가 극단적으로 높음")
    print("  inq_last_12m·inq_fi: 특정 시점 이후에만 수집되어 전체 기간 후보에서 제외")
    return df


# ══════════════════════════════════════════════════════════════════
# 검증
# ══════════════════════════════════════════════════════════════════
def verify(df):
    """결론을 내기 전에 반증 기회를 주는 점검."""
    _sec("검증")
    ok = []

    def chk(cond, msg):
        ok.append(bool(cond))
        print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")

    chk(df["id"].is_unique, f"id 중복 없음 ({len(df):,}행)")

    # sec_app_* 8개를 뺀 나머지는 결측 0 이 방침이다
    non_sec = [c for c in df.columns if c not in SEC_APP_KEEP_NAN]
    na = df[non_sec].isna().sum()
    bad = na[na > 0]
    chk(len(bad) == 0, f"sec_app_* 외 결측 0 (위반 {len(bad)}개 컬럼)")
    if len(bad):
        print(bad.to_string())

    chk(all(c in df.columns for c in SEC_APP_KEEP_NAN), "sec_app_* 8개 존재")
    print(f"       sec_app_fico_avg 결측률 {df['sec_app_fico_avg'].isna().mean() * 100:.2f}% (의도된 NaN)")

    # 계층 제약이 실제로 성립하는지 — hot-deck 보정이 제대로 됐다는 증거
    for parent, child in HIERARCHY_PAIRS:
        v = int((df[parent] < df[child]).sum())
        chk(v == 0, f"{parent} >= {child} (위반 {v})")

    chk(df["term"].isin([0, 1]).all(), "term 이진")
    chk((df[[c for c in df.columns if c.startswith('purpose_')]].sum(axis=1) == 1).all(),
        "purpose 원-핫 행합 = 1")
    chk((df[[c for c in df.columns if c.startswith('home_ownership_')]].sum(axis=1) == 1).all(),
        "home_ownership 원-핫 행합 = 1")

    num = df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(num.to_numpy(dtype=float)).sum())
    chk(n_inf == 0, f"무한대 값 없음 (발견 {n_inf})")

    print(f"\n  최종 {df.shape[0]:,}행 × {df.shape[1]}열")
    return all(ok)


# ══════════════════════════════════════════════════════════════════
# 전체 파이프라인
# ══════════════════════════════════════════════════════════════════
def build_X(raw_path, cohort_ids, P, zip3_cache_path, donor_pool_path, donor_mask=None):
    """원본 → 전처리 완료된 X 테이블.

    P.mode == "fit"       : train 행(donor_mask)만 보고 상수 학습 → JSON
    P.mode == "transform" : 저장된 상수만 적용, 데이터를 보지 않음
    """
    df = load_source(raw_path, cohort_ids)
    df = part1_meta(df, P, zip3_cache_path)
    df = part2_loan_terms(df, P)
    df = part3_borrower(df, P)
    df, yj = part4_delinquency(df, P, donor_mask, donor_pool_path)
    df = part6_joint(df, P, yj)
    return df

# %%
