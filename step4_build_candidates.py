# %% [markdown]
# # STEP 4-A — LASSO 후보 X 테이블 (원본에서 최소 정제만)
#
# ## 왜 도메인 필터 결과로 돌리면 안 되는가
#
# `step 2` 의 X(76열)로 LASSO를 돌리면 **"우리가 고른 것 중 뭘 더 뺄까"** 만 답할 수 있습니다.
# 잘못 버린 변수를 되찾아주지 못합니다. **검증이 아니라 축소**가 됩니다.
#
# 그래서 원본 141개에서 **기계적으로 반드시 빼야 하는 것만** 제거한 더 넓은 후보군을
# 따로 만들어 LASSO에 넣고, 도메인 판단 결과와 대조합니다.
#
# ## 제외 원칙 — 도메인 취향이 아니라 반드시 빼야 하는 것
#
# | 유형 | 이유 |
# |---|---|
# | **사후 변수** | 대출 실행 후 결정. `total_pymnt`, `recoveries`, `last_fico_*` 등 |
# | **내생 변수** | LC 심사·가격결정의 결과물. `int_rate` 를 넣으면 IRR이 기계적으로 결정되어 예측이 아니라 항등식이 된다 |
# | **y 계산 재료** | `funded_amnt`, `funded_amnt_inv` |
# | **식별자·텍스트** | `url`, `desc`, `title`, `emp_title`, `policy_code` |
#
# **`sec_app_*` 는 다른 이유로 제외합니다.** 결측 93%라 중앙값으로 채우면 93%가 같은 값이
# 되어 사실상 상수가 되고 LASSO가 자동으로 죽입니다. 넣어도 정보를 얻을 수 없으므로
# 트리 모델 + SHAP 경로로 따로 판단합니다.
#
# ## 전처리 수준 — 최소
# 결측 대치 + 인코딩만 합니다. **왜도 변환·윈저라이징은 하지 않습니다.**
# 변수 '선택' 단계이므로 원본 정보를 최대한 보존하는 게 맞고,
# 단조변환은 LASSO의 선택 결과를 크게 바꾸지 않습니다.
#
# ## 표준화는 여기서 하지 않습니다
# LASSO 페널티는 스케일에 민감해 표준화가 필수지만, 평균·표준편차는
# **반드시 train에서만** 학습해야 합니다. → `step4_run_lasso.py` 에서 처리.

# %%
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "파이프라인"))
import lc_config as C

C.check_paths()
OUT = C.STEP_DIR[4]
print(f"산출물 폴더: {OUT}")

# ─────────────────────────────────────────────────────────────
# 1. 변수 분류
# ─────────────────────────────────────────────────────────────
POST_OUTCOME = [
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "pymnt_plan", "debt_settlement_flag", "deferral_term",
    "payment_plan_start_date", "orig_projected_additional_accrued_interest",
    "hardship_flag", "hardship_type", "hardship_status", "hardship_amount",
    "settlement_status", "settlement_date", "settlement_amount",
]
ENDOGENOUS = ["int_rate", "grade", "sub_grade", "installment",
              "funded_amnt", "funded_amnt_inv"]
ID_TEXT = ["member_id", "url", "desc", "title", "emp_title", "policy_code"]
SEC_APP_DEFER = [
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "verification_status_joint",
]

# 팀이 도메인 지식으로 채택했던 수치형
NUM_TEAM = [
    "loan_amnt", "delinq_2yrs", "num_accts_ever_120_pd", "delinq_amnt",
    "mths_since_last_delinq", "mths_since_last_major_derog", "mths_since_last_record",
    "pub_rec", "tax_liens", "collections_12_mths_ex_med", "chargeoff_within_12_mths",
    "bc_open_to_buy", "all_util", "max_bal_bc", "revol_util", "percent_bc_gt_75",
    "tot_hi_cred_lim", "mo_sin_old_rev_tl_op", "total_bc_limit", "mo_sin_rcnt_tl",
    "mo_sin_rcnt_rev_tl_op", "mths_since_recent_bc", "tot_coll_amt",
    "open_acc", "acc_open_past_24mths", "num_tl_op_past_12m", "num_rev_accts",
    "num_op_rev_tl", "num_actv_rev_tl", "num_bc_tl", "num_bc_sats", "num_actv_bc_tl",
    "num_il_tl", "mort_acc", "inq_last_6mths", "mths_since_recent_inq",
]
# 팀이 검토 없이 뺐지만 결측이 적어 되살릴 만한 것 — LASSO 가 판단하게 둔다
NUM_ADDED = [
    "total_acc",                  # VIF 근거로 뺐으나 LASSO 가 판단하게
    "acc_now_delinq",             # 과거 이력만 있고 '현재' 연체 상태가 빠져 있었음
    "pct_tl_nvr_dlq",             # 무연체 계좌 비율 — 연체 이력의 핵심 요약
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_120dpd_2m",
    "pub_rec_bankruptcies",       # pub_rec(공공기록 전체)과 다른 정보
    "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "bc_util",
    "mo_sin_old_il_acct", "num_rev_tl_bal_gt_0", "num_sats",
    "total_bal_ex_mort", "total_il_high_credit_limit",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m",
    "inq_fi", "total_cu_tl", "inq_last_12m",
]
CAT_COLS = ["purpose", "home_ownership", "verification_status",
            "initial_list_status", "addr_state", "disbursement_method"]
SPECIAL_SRC = ["id", "term", "application_type", "issue_d", "zip_code",
               "annual_inc", "annual_inc_joint", "dti", "dti_joint",
               "revol_bal", "revol_bal_joint",
               "fico_range_low", "fico_range_high", "earliest_cr_line", "emp_length"]

# %%
# ─────────────────────────────────────────────────────────────
# 2. 적재 — cohort 행만
# ─────────────────────────────────────────────────────────────
raw_cols = pd.read_csv(C.RAW_PATH, nrows=2).columns.tolist()
print(f"원본 {len(raw_cols)}열")

use = [c for c in (NUM_TEAM + NUM_ADDED + CAT_COLS + SPECIAL_SRC) if c in raw_cols]
missing_req = [c for c in (NUM_TEAM + NUM_ADDED + CAT_COLS + SPECIAL_SRC) if c not in raw_cols]
if missing_req:
    print(f"[주의] 원본에 없는 후보: {missing_req}")

cohort = pd.read_csv(C.COHORT_PATH, usecols=["id", "split"])
want = set(cohort["id"].astype("int64"))

STR_KEEP = {"id", "term", "purpose", "home_ownership", "verification_status",
            "initial_list_status", "addr_state", "disbursement_method",
            "application_type", "issue_d", "zip_code", "earliest_cr_line", "emp_length"}
parts = []
for ch in pd.read_csv(C.RAW_PATH, usecols=use, chunksize=200_000, low_memory=False,
                      dtype={c: str for c in STR_KEEP if c in use}):
    ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
    ch = ch.loc[ch["id"].notna()]
    ch = ch.loc[ch["id"].astype("int64").isin(want)]
    if not len(ch):
        continue
    for c in ch.columns:
        if c in STR_KEEP:
            continue
        if not pd.api.types.is_numeric_dtype(ch[c]):
            ch[c] = pd.to_numeric(ch[c].astype(str).str.strip().str.rstrip("%"),
                                  errors="coerce")
    parts.append(ch)

df = pd.concat(parts, ignore_index=True)
del parts
df["id"] = df["id"].astype("int64")
# ★ cohort 순서로 정렬. reset_index 를 빠뜨리면 이후 concat 에서 유령 행이 생긴다.
df = df.set_index("id").loc[cohort["id"].to_numpy()].reset_index()
print(f"적재 {df.shape[0]:,}행 × {df.shape[1]}열")
assert len(df) == len(cohort) and df["id"].is_unique

# revol_util 등 '%' 문자열
for c in ["revol_util", "il_util", "bc_util", "all_util", "percent_bc_gt_75", "pct_tl_nvr_dlq"]:
    if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip().str.rstrip("%"), errors="coerce")

# %%
# ─────────────────────────────────────────────────────────────
# 3. 특수 처리 — step2 와 같은 파생을 최소한으로만
# ─────────────────────────────────────────────────────────────
out = pd.DataFrame({"id": df["id"].to_numpy()})
issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

# term → 이진
out["term_60m"] = (pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0]) == 60).astype("int8")

# application_type → 이진
out["is_joint_application"] = (df["application_type"].astype(str).str.strip()
                               .str.lower().str.startswith("joint").astype("int8"))

# joint 통합 (공동이면 합산값, 개인이면 개인값) — 변환은 하지 않는다
for base, jc in [("annual_inc", "annual_inc_joint"), ("dti", "dti_joint"),
                 ("revol_bal", "revol_bal_joint")]:
    out[base] = df[jc].fillna(df[base]) if jc in df.columns else df[base]

# fico 평균 (low/high 는 거의 완전상관)
out["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

# credit_age_months
ear = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
out["credit_age_months"] = (issue_dt - ear).dt.days / 30.44

# emp_length 순서형 + 결측 플래그
EMP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
       "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
       "10+ years": 10}
eo = df["emp_length"].astype(str).str.strip().map(EMP)
out["emp_length"] = eo
out["emp_length_missing"] = eo.isna().astype("int8")

# zip3 외부변수 (복구한 캐시에서 병합)
z = pd.read_csv(os.path.join(PROJ, "파이프라인", "zip3_external_cache.csv"), dtype={"zip3": str})
zc = df["zip_code"].astype(str).str.strip()
tmp = pd.DataFrame({"zip3": zc.str[:3].where(zc.str.match(r"^\d{3}"), other=np.nan)})
tmp = tmp.merge(z, on="zip3", how="left")
for c in z.columns.drop("zip3"):
    out[c] = tmp[c].to_numpy()

# 수치형 원본 그대로
for c in NUM_TEAM + NUM_ADDED:
    if c in df.columns:
        out[c] = df[c]

print(f"수치·파생 {out.shape[1] - 1}개")

# %%
# ─────────────────────────────────────────────────────────────
# 4. 범주형 원-핫
# ─────────────────────────────────────────────────────────────
# ★ 범주 목록을 train 에서 고정한다. 고정하지 않으면 val/test 에서 열 구성이 달라진다.
tr_mask = (cohort["split"] == "train").to_numpy()
cat_map = {}
for c in CAT_COLS:
    if c not in df.columns:
        continue
    s = df[c].astype(str).str.strip()
    cats = sorted(s[tr_mask].dropna().unique())
    cats = [x for x in cats if x != "nan"]
    cat_map[c] = cats
    d = pd.get_dummies(pd.Categorical(s, categories=cats), prefix=c, dtype="int8")
    d.index = out.index
    out = pd.concat([out, d], axis=1)
    print(f"  {c:22s} {len(cats):>2}개 범주 → 원-핫")

# %%
# ─────────────────────────────────────────────────────────────
# 5. 결측 대치 — train 중앙값 (누출 방지)
# ─────────────────────────────────────────────────────────────
feat = [c for c in out.columns if c != "id"]
na_before = out[feat].isna().sum()
med = out.loc[tr_mask, feat].median()          # ★ train 에서만
out[feat] = out[feat].fillna(med)

n_filled = int(na_before.sum())
print(f"결측 대치 {n_filled:,}셀 (train 중앙값 기준)")
print("결측률 상위 8개:")
print((na_before[na_before > 0].sort_values(ascending=False).head(8)
       / len(out) * 100).round(2).to_string())

# 상수 컬럼 제거 (LASSO 에서 무의미하고 표준화 시 0으로 나눔)
const = [c for c in feat if out[c].nunique() <= 1]
if const:
    out = out.drop(columns=const)
    feat = [c for c in feat if c not in const]
    print(f"상수 컬럼 제거 {len(const)}개: {const}")

assert out[feat].isna().sum().sum() == 0
print(f"\n최종 후보 X: {out.shape[0]:,}행 × {len(feat)}개 변수")

# %%
# ─────────────────────────────────────────────────────────────
# 6. 변수 분류표 — 도메인 판단과 대조할 수 있게 원천변수 단위로 정리
# ─────────────────────────────────────────────────────────────
# 원-핫으로 쪼개진 열은 원천변수로 되묶는다. "addr_state 가 채택됐는가"를
# 물으려면 addr_state_CA, addr_state_NY ... 를 하나로 봐야 하기 때문이다.
def source_of(col):
    for c in CAT_COLS:
        if col.startswith(c + "_"):
            return c
    return {"term_60m": "term", "is_joint_application": "application_type",
            "emp_length_missing": "emp_length"}.get(col, col)


# step 2 도메인 트랙이 실제로 채택한 열
dom_path = os.path.join(C.STEP2_DIR, "X_table.csv")
if os.path.exists(dom_path):
    dom_cols = pd.read_csv(dom_path, nrows=2).columns.tolist()
else:
    dom_cols = []
    print("[주의] step 2/X_table.csv 가 없어 도메인 대조는 LASSO 실행 시로 미룹니다.")


def dom_source(col):
    for c in ["purpose", "home_ownership"]:
        if col.startswith(c + "_"):
            return c
    return {"is_joint_application": "application_type",
            "emp_length_missing": "emp_length"}.get(col, col)


dom_src = {dom_source(c) for c in dom_cols if c != "id"}
# 도메인 트랙에서 파생으로 흡수된 것들
dom_src |= {"fico_range_low", "fico_range_high", "earliest_cr_line", "zip_code"} & set(raw_cols)

rows = []
for col in feat:
    src = source_of(col)
    rows.append({"열": col, "원천변수": src,
                 "도메인채택": int(src in dom_src) if dom_cols else -1,
                 "유형": ("범주형(원-핫)" if src in CAT_COLS else
                        "팀채택_수치" if src in NUM_TEAM else
                        "되살린_후보" if src in NUM_ADDED else "파생·특수")})
vt = pd.DataFrame(rows)
print("\n[후보 구성]")
print(vt.groupby("유형").agg(열수=("열", "size"),
                          원천변수수=("원천변수", "nunique")).to_string())

excluded = pd.DataFrame(
    [{"변수": c, "제외사유": r} for r, cols in
     [("사후변수", POST_OUTCOME), ("내생변수", ENDOGENOUS),
      ("식별자·텍스트", ID_TEXT), ("sec_app(결측93%, 트리+SHAP로 이관)", SEC_APP_DEFER)]
     for c in cols if c in raw_cols])
print(f"\n기계적 제외 {len(excluded)}개")
print(excluded.groupby("제외사유").size().to_string())

# ─────────────────────────────────────────────────────────────
# 7. 저장
# ─────────────────────────────────────────────────────────────
out.to_csv(os.path.join(OUT, "lasso_candidate_X.csv"), index=False,
           encoding="utf-8-sig", float_format="%.10g")
vt.to_csv(os.path.join(OUT, "lasso_candidate_variables.csv"), index=False,
          encoding="utf-8-sig")
excluded.to_csv(os.path.join(OUT, "lasso_제외변수.csv"), index=False, encoding="utf-8-sig")
pd.Series(cat_map).to_json(os.path.join(OUT, "lasso_cat_levels.json"), force_ascii=False)

print(f"\n저장 위치: {OUT}")
for f in ["lasso_candidate_X.csv", "lasso_candidate_variables.csv", "lasso_제외변수.csv"]:
    pth = os.path.join(OUT, f)
    print(f"  {f:<34} {os.path.getsize(pth) / 1e6:>7.1f} MB")
print(f"\n✅ STEP 4-A 완료 — 후보 {len(feat)}열 / 원천변수 {vt['원천변수'].nunique()}개")
print("   다음: step4_run_lasso.py")

# %%
