"""
LASSO 2차 검증용 후보 변수 테이블 생성

[왜 별도 테이블이 필요한가]
  domain_filtered_X_table_raw.csv 는 이미 도메인 필터가 적용된 결과물이다.
  그걸로 LASSO 를 돌리면 "우리가 고른 것 중 뭘 더 뺄까"만 답할 수 있고,
  잘못 버린 변수를 되찾아주지 못한다. 검증이 아니라 축소가 된다.
  따라서 원본 141개에서 '기계적으로 반드시 빼야 하는 것'만 제거한
  더 넓은 후보군을 만들어 LASSO 에 넣고, 도메인 판단 결과와 대조한다.

[제외 원칙 — 도메인 취향이 아니라 반드시 빼야 하는 것들]
  1) 사후 변수 : 대출 실행 후에 결정되는 값. 넣으면 미래 정보 누수.
                 total_pymnt, recoveries, last_fico_*, hardship_* 등
  2) 내생 변수 : LendingClub 자체 심사·가격결정의 결과물.
                 int_rate, grade, sub_grade, installment
                 → int_rate 를 넣으면 IRR 이 거의 기계적으로 결정되어
                   '예측'이 아니라 항등식이 된다.
  3) y 계산 재료 : funded_amnt, funded_amnt_inv
  4) 식별자·텍스트·상수 : id(키로만 사용), url, desc, title, emp_title, policy_code

[제외하지만 이유가 다른 것 — sec_app 계열]
  sec_app_* 8개는 결측이 93%다. 중앙값으로 채우면 93%가 같은 값이 되어
  사실상 상수가 되고 LASSO 가 자동으로 죽인다. 넣어도 정보를 얻을 수 없다.
  이 8개는 LASSO 가 아니라 트리 모델 + SHAP 으로 따로 판단한다.
  단, 공동신청 정보 자체는 is_joint_application 과
  annual_inc/dti/revol_bal 의 joint 통합값으로 후보에 포함된다.

[전처리 수준 — 최소]
  결측 대치 + 인코딩만 한다. 왜도 변환·윈저라이징은 하지 않는다.
  변수 '선택' 단계이므로 원본 정보를 최대한 보존하는 게 맞고,
  단조변환은 LASSO 의 선택 결과를 크게 바꾸지 않는다.

[표준화는 여기서 하지 않는다]
  LASSO 페널티는 스케일에 민감해 표준화가 필수지만,
  평균·표준편차는 반드시 train 에서만 학습해야 한다.
  분할 이후에 적용하도록 맨 아래 예시 코드를 참고할 것.

[입력]  자료_데이터/lending_club_2020_train.csv
        census_zip3_2016.csv, fdic_zip3_2007.csv  (없으면 지역변수 없이 진행)
[출력]  lasso_candidate_X.csv        후보 X 테이블 (id 포함)
        lasso_candidate_variables.csv 변수 분류표 (리포트 첨부용)
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# 1. 변수 분류
# ─────────────────────────────────────────────────────────

# ── 반드시 제외 ──
POST_OUTCOME = [   # 대출 실행 후 결정 → 미래 정보
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "pymnt_plan", "debt_settlement_flag", "deferral_term",
    "payment_plan_start_date", "orig_projected_additional_accrued_interest",
]
ENDOGENOUS = ["int_rate", "grade", "sub_grade", "installment",
              "funded_amnt", "funded_amnt_inv"]
ID_TEXT = ["member_id", "url", "desc", "title", "emp_title", "policy_code"]

# ── 결측 93%라 LASSO 로는 판단 불가 (트리+SHAP 경로로 이관) ──
SEC_APP_DEFER = [
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "verification_status_joint",
]

# ── 후보: 수치형 (원본 그대로 사용) ──
NUM_TEAM = [   # 팀이 채택했던 것
    "loan_amnt",
    "delinq_2yrs", "num_accts_ever_120_pd", "delinq_amnt",
    "mths_since_last_delinq", "mths_since_last_major_derog", "mths_since_last_record",
    "pub_rec", "tax_liens", "collections_12_mths_ex_med", "chargeoff_within_12_mths",
    "bc_open_to_buy", "all_util", "max_bal_bc", "revol_util", "percent_bc_gt_75",
    "tot_hi_cred_lim", "mo_sin_old_rev_tl_op", "total_bc_limit", "mo_sin_rcnt_tl",
    "mo_sin_rcnt_rev_tl_op", "mths_since_recent_bc", "tot_coll_amt",
    "open_acc", "acc_open_past_24mths", "num_tl_op_past_12m", "num_rev_accts",
    "num_op_rev_tl", "num_actv_rev_tl", "num_bc_tl", "num_bc_sats", "num_actv_bc_tl",
    "num_il_tl", "mort_acc", "inq_last_6mths", "mths_since_recent_inq",
]
NUM_ADDED = [  # 팀이 검토 없이 뺐지만 결측이 적어 후보로 되살릴 만한 것
    "total_acc",                  # VIF 근거로 뺐으나 LASSO 가 판단하게 둔다
    "acc_now_delinq",             # 과거 이력만 있고 '현재' 연체 상태가 빠져 있었음
    "pct_tl_nvr_dlq",             # 무연체 계좌 비율 — 연체 이력의 핵심 요약
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_120dpd_2m",
    "pub_rec_bankruptcies",       # pub_rec(공공기록 전체)과 다른 정보
    "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "bc_util",
    "mo_sin_old_il_acct", "num_rev_tl_bal_gt_0", "num_sats",
    "total_bal_ex_mort", "total_il_high_credit_limit",
]

# ── 후보: 범주형 (원-핫) ──
CAT_COLS = ["purpose", "home_ownership", "verification_status",
            "initial_list_status", "addr_state"]

# ── 후보: 별도 처리 ──
#   term                → 이진
#   application_type    → is_joint_application
#   annual_inc/dti/revol_bal + *_joint → 통합
#   fico_range_low/high → fico_avg
#   earliest_cr_line + issue_d → credit_age_months
#   zip_code            → zip3 외부변수 (캐시 병합)
SPECIAL_SRC = ["id", "term", "application_type", "issue_d", "zip_code",
               "annual_inc", "annual_inc_joint", "dti", "dti_joint",
               "revol_bal", "revol_bal_joint",
               "fico_range_low", "fico_range_high", "earliest_cr_line",
               "emp_length"]

ZIP3_COLS = ["zip3_median_income", "zip3_median_home_value",
             "zip3_no_earnings_ratio", "zip3_bank_concentration"]


def find_data_dir(marker="자료_데이터", max_up=5):
    try:
        start = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start = os.getcwd()
    d = start
    for _ in range(max_up):
        if os.path.isdir(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(f"'{marker}' 폴더를 찾지 못했습니다.")


def section(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64, flush=True)


BASE_DIR   = find_data_dir()
SRC_PATH   = os.path.join(BASE_DIR, "자료_데이터", "lending_club_2020_train.csv")
OUT_PATH   = os.path.join(BASE_DIR, "lasso_candidate_X.csv")
DICT_PATH  = os.path.join(BASE_DIR, "lasso_candidate_variables.csv")
CENSUS     = os.path.join(BASE_DIR, "census_zip3_2016.csv")
FDIC       = os.path.join(BASE_DIR, "fdic_zip3_2007.csv")
# 캐시가 정리돼 없을 때 zip3 변수를 가져올 대체 경로
DERIVED_SRC = os.path.join(BASE_DIR, "domain_filtered_X_derived_variables.csv")


# ─────────────────────────────────────────────────────────
# 2. 로드
# ─────────────────────────────────────────────────────────
section("1. 후보 변수 로드")

all_cols = list(pd.read_csv(SRC_PATH, nrows=0).columns)
use_cols = sorted(set(NUM_TEAM + NUM_ADDED + CAT_COLS + SPECIAL_SRC))
missing = [c for c in use_cols if c not in all_cols]
if missing:
    raise SystemExit(f"원본에 없는 컬럼: {missing}")

excluded = [c for c in all_cols if c not in use_cols]
print(f"원본 {len(all_cols)}개 → 후보 원천 {len(use_cols)}개 (제외 {len(excluded)}개)")

df = pd.read_csv(SRC_PATH, usecols=use_cols, low_memory=False)

# 원본에 섞인 섹션 구분용 텍스트 행 제거
before = len(df)
df = df[pd.to_numeric(df["id"], errors="coerce").notna()].copy()
df["id"] = df["id"].astype("int64")
if before - len(df):
    print(f"비정상 행 {before - len(df)}개 제거")

# 인덱스를 반드시 다시 매긴다.
#
# 위에서 행을 지우면 df 의 인덱스에 구멍이 생긴다(원본 956,843번째 행).
# 아래에서 만드는 out 은 0..n-1 의 새 인덱스를 갖기 때문에, 인덱스를 맞춰두지 않으면
# out["term_60m"] = ... 같은 Series 대입과 pd.concat(axis=1) 에서 pandas 가
# 인덱스 기준으로 정렬해버려 엉뚱한 NaN 과 유령 행이 생긴다.
# (구멍이 데이터 중간에 있어 앞부분만 잘라 테스트하면 재현되지 않는다)
df = df.reset_index(drop=True)

n_before = len(df)
print(f"행 수: {n_before:,}")

# 수치여야 할 컬럼이 문자열로 읽히는 경우 대비 (pandas 3.x StringDtype 포함)
for c in NUM_TEAM + NUM_ADDED + ["annual_inc", "annual_inc_joint", "dti", "dti_joint",
                                 "revol_bal", "revol_bal_joint",
                                 "fico_range_low", "fico_range_high"]:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip().str.rstrip("%"),
                              errors="coerce")
        print(f"  {c} 수치 변환")


# ─────────────────────────────────────────────────────────
# 3. 특수 처리
# ─────────────────────────────────────────────────────────
section("2. 특수 변수 처리")

out = pd.DataFrame({"id": df["id"].to_numpy()})

# ── term ──
term_m = pd.to_numeric(df["term"].astype(str).str.strip().str.extract(r"(\d+)")[0],
                       errors="coerce")
out["term_60m"] = (term_m == 60).astype("int8")
print(f"term_60m: 60개월 {out['term_60m'].mean() * 100:.2f}%")

# ── is_joint_application ──
is_joint = (df["application_type"].astype(str).str.strip().str.lower()
            .str.startswith("joint")).astype("int8")
out["is_joint_application"] = is_joint
print(f"is_joint_application: Joint {is_joint.mean() * 100:.2f}%")

# ── joint 통합 (공동이면 합산값, 개인이면 개인값) ──
for base_c, joint_c in [("annual_inc", "annual_inc_joint"),
                        ("dti", "dti_joint"),
                        ("revol_bal", "revol_bal_joint")]:
    out[base_c] = df[joint_c].fillna(df[base_c])
    print(f"{base_c}: joint 값 사용 {int(df[joint_c].notna().sum()):,}건")

# ── fico_avg ──
out["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

# ── credit_age_months ──
issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
early_dt = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
age = (issue_dt - early_dt).dt.days / 30.44
out["credit_age_months"] = age.mask(age < 0)
print(f"credit_age_months: 중앙값 {out['credit_age_months'].median():.0f}개월")

# ── emp_length 순서형 ──
EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
           "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
           "10+ years": 10}
emp = df["emp_length"].astype(str).str.strip().map(EMP_MAP)
out["emp_length"] = emp                      # 결측은 아래에서 함께 대치
out["emp_length_missing"] = emp.isna().astype("int8")
print(f"emp_length: 결측(무직) {out['emp_length_missing'].mean() * 100:.2f}%")

# ── zip3 외부변수 ──
# 1순위: census/fdic 원본 캐시에서 zip3 로 병합
# 2순위: 캐시가 정리돼 없으면 이미 만들어둔 파생변수 파일에서 id 로 가져온다
#        (같은 값이므로 결과는 동일하다)
if os.path.exists(CENSUS) and os.path.exists(FDIC):
    zc = df["zip_code"].astype(str).str.strip()
    zip3 = zc.str[:3].where(zc.str.match(r"^\d{3}"), other=np.nan)
    census = pd.read_csv(CENSUS, dtype={"zip3": str})
    fdic = pd.read_csv(FDIC, dtype={"zip3": str})
    z = pd.DataFrame({"zip3": zip3.to_numpy()})
    z = z.merge(census[["zip3"] + ZIP3_COLS[:3]], on="zip3", how="left")
    z = z.merge(fdic[["zip3", ZIP3_COLS[3]]], on="zip3", how="left")
    for c in ZIP3_COLS:
        out[c] = pd.to_numeric(z[c], errors="coerce").to_numpy()
    print(f"zip3 외부변수 {len(ZIP3_COLS)}개 병합 (census/fdic 캐시)")
elif os.path.exists(DERIVED_SRC):
    dv = pd.read_csv(DERIVED_SRC, usecols=["id"] + ZIP3_COLS)
    out = out.merge(dv, on="id", how="left", validate="one_to_one")
    print(f"zip3 외부변수 {len(ZIP3_COLS)}개 병합 "
          f"({os.path.basename(DERIVED_SRC)} 에서 id 기준)")
else:
    print("[주의] census/fdic 캐시도 파생변수 파일도 없어 지역 변수를 제외합니다.")
    ZIP3_COLS = []

if ZIP3_COLS:
    out["income_to_zip3_median"] = (
        out["annual_inc"] / out["zip3_median_income"].replace(0, np.nan)
    )
    print("income_to_zip3_median 파생 완료")

# ── 나머지 수치형 그대로 ──
for c in NUM_TEAM + NUM_ADDED:
    out[c] = df[c].to_numpy()


# ─────────────────────────────────────────────────────────
# 4. 범주형 원-핫
# ─────────────────────────────────────────────────────────
section("3. 범주형 원-핫 인코딩")

# 최소 전처리 방침이므로 희소범주를 통합하지 않는다.
# purpose 를 14개 전부 넣어야 "other 로 합친 게 옳았나"를 LASSO 가 판단할 수 있다.
onehot_cols = []
for c in CAT_COLS:
    v = df[c].astype(str).str.strip()
    v = v.replace({"nan": "MISSING", "": "MISSING"})
    d = pd.get_dummies(v, prefix=c, dtype="int8")
    d.index = out.index          # 인덱스 정렬 사고 방지 (위 reset_index 와 이중 안전장치)
    out = pd.concat([out, d], axis=1)
    onehot_cols += list(d.columns)
    print(f"  {c:22s} {d.shape[1]:>3}개 범주")
print(f"원-핫 컬럼 총 {len(onehot_cols)}개")


# ─────────────────────────────────────────────────────────
# 5. 결측 대치 (중앙값)
# ─────────────────────────────────────────────────────────
section("4. 결측 대치")

# LASSO 는 NaN 을 처리하지 못하므로 전부 채운다.
# 왜도 변환은 하지 않는다(변수 선택 단계 — 원본 정보 보존).
num_cols = [c for c in out.columns if c != "id" and c not in onehot_cols]
na = out[num_cols].isna().sum()
na = na[na > 0].sort_values(ascending=False)
print(f"결측 있는 변수 {len(na)}개")
for c, k in na.items():
    print(f"  {c:30s} {k:>9,}건 ({k / len(out) * 100:5.2f}%)")

# 결측률이 높은 변수는 플래그를 함께 만든다.
#
# 이유: mths_since_last_record 는 결측이 85%인데, 이 결측은 "값을 모른다"가 아니라
# "공공기록이 아예 없다"는 뜻이다. 중앙값으로만 채우면 기록이 없는 사람이
# "N개월 전에 기록이 있었던 사람"으로 둔갑하고, 변수의 85%가 같은 조작된 값이 된다.
# 그 상태로 LASSO 를 돌려 변수가 탈락하면 '실제로 무의미해서'인지
# '대치가 변수를 망가뜨려서'인지 구분할 수 없어 검증 자체가 무의미해진다.
# 플래그를 같이 넣으면 LASSO 가 '이력 유무'와 '이력이 있을 때의 경과월'을
# 분리해서 평가할 수 있다.
FLAG_THRESHOLD = 0.05
flag_cols = []
for c in num_cols:
    r = out[c].isna().mean()
    if r > FLAG_THRESHOLD:
        out[f"{c}_isna"] = out[c].isna().astype("int8")
        flag_cols.append(f"{c}_isna")
print(f"\n결측률 {FLAG_THRESHOLD * 100:.0f}% 초과 변수에 결측 플래그 {len(flag_cols)}개 생성")
for c in flag_cols:
    print(f"  {c}")

for c in num_cols:
    if out[c].isna().any():
        out[c] = out[c].fillna(out[c].median())

# 분산 0 변수는 LASSO 에서 무의미하므로 제거
const = [c for c in out.columns if c != "id" and out[c].nunique() <= 1]
if const:
    out = out.drop(columns=const)
    print(f"\n분산 0 변수 제거: {const}")


# ─────────────────────────────────────────────────────────
# 6. 검증
# ─────────────────────────────────────────────────────────
section("5. 검증")

ok = True


def check(cond, msg_ok, msg_fail):
    global ok
    if cond:
        print(f"[통과] {msg_ok}")
    else:
        print(f"[실패] {msg_fail}")
        ok = False


check(len(out) == n_before, f"행 수 유지: {len(out):,}", "행 수 변동")
check(int(out.isna().sum().sum()) == 0, "결측 0셀",
      f"결측 {int(out.isna().sum().sum()):,}셀 잔존")
check(out["id"].is_unique, "id 고유", "id 중복")

leaked = [c for c in POST_OUTCOME + ENDOGENOUS if c in out.columns]
check(not leaked, "사후·내생 변수 미포함", f"누수 변수 발견: {leaked}")

sec_in = [c for c in out.columns if c.startswith("sec_app_")]
check(not sec_in, "sec_app_* 미포함 (트리+SHAP 경로로 이관)", f"포함됨: {sec_in}")

const_left = [c for c in out.columns if c != "id" and out[c].nunique() <= 1]
check(not const_left, "분산 0 변수 없음", f"상수 변수: {const_left}")

feat = [c for c in out.columns if c != "id"]
print(f"\n최종 후보 변수 {len(feat)}개 (id 제외)")


# ─────────────────────────────────────────────────────────
# 7. 변수 분류표 저장 (리포트 첨부용)
# ─────────────────────────────────────────────────────────
section("6. 저장")

DOMAIN_KEPT = set(NUM_TEAM) | {
    "term_60m", "is_joint_application", "annual_inc", "dti", "revol_bal",
    "fico_avg", "credit_age_months", "emp_length", "emp_length_missing",
    "income_to_zip3_median", *ZIP3_COLS,
}
ADDED_CAT = ("verification_status_", "initial_list_status_", "addr_state_")

rows = []
for c in feat:
    src = c
    if c.endswith("_isna"):
        src = c[:-5]
    for pre in CAT_COLS:
        if c.startswith(pre + "_"):
            src = pre
            break

    if src in DOMAIN_KEPT:
        grp = "도메인 채택"
    elif src in NUM_ADDED:
        grp = "★ 재평가 — 저결측 수치형"
    elif src in ("verification_status", "initial_list_status", "addr_state") \
            or c.startswith(ADDED_CAT):
        grp = "★ 재평가 — 범주형"
    elif src in ("purpose", "home_ownership"):
        grp = "도메인 채택 (희소범주 미통합)"
    else:
        grp = "기타"
    rows.append({"변수": c, "원천변수": src, "구분": grp,
                 "결측플래그": "O" if c.endswith("_isna") else ""})
vd = pd.DataFrame(rows)
vd.to_csv(DICT_PATH, index=False)

print("변수 구분별 개수 (컬럼 기준 / 원천변수 기준)")
g = vd.groupby("구분").agg(컬럼수=("변수", "size"), 원천변수수=("원천변수", "nunique"))
print(g.to_string())
print(f"\n※ addr_state 는 51개 주가 원-핫으로 펼쳐져 컬럼 수가 크게 잡힌다.")
print(f"   실제로 재평가되는 '변수' 개수는 원천변수수 기준으로 보는 게 맞다.")
print(f"\n변수 분류표 저장 → {os.path.basename(DICT_PATH)}")

out.to_csv(OUT_PATH, index=False, float_format="%.10g")
print(f"후보 테이블 저장 → {os.path.basename(OUT_PATH)}  "
      f"{out.shape[0]:,}행 × {out.shape[1]}열")

print(f"""
{'-' * 64}
다음 단계
{'-' * 64}
1) y 테이블(spread) 과 id 로 병합
   - IRR 계산이 가능한 완결 대출(Fully Paid / Charged Off)만 남는다.
     전체의 약 63.6% (약 112만 건)

2) 6:2:2 무작위 분할

3) train 에서만 표준화 학습 후 val/test 에 적용
     from sklearn.preprocessing import StandardScaler
     sc = StandardScaler().fit(X_train)          # train 만!
     X_train_s, X_val_s = sc.transform(X_train), sc.transform(X_val)

4) train 에서만 LASSO 적합 (LassoCV 등). alpha 도 train 내 CV 로 결정
   ※ 전체 데이터로 변수를 고르면 test 성능이 낙관적으로 편향된다.

5) 선택된 변수 vs 도메인 채택 변수 대조
   - 양쪽 채택      → 도메인 판단 확인
   - LASSO 만 채택  → 우리가 놓친 변수. 되살릴지 검토
   - 도메인만 채택  → LASSO 가 죽인 이유 확인(공선성인지 무의미인지)
""")

if not ok:
    raise SystemExit("\n!! 검증 실패 항목이 있습니다.")
print("전체 검증 통과.")
