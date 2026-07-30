# %% [markdown]
# # STEP 2 — X 전처리 (train fit → transform)
#
# **이 단계가 고치는 문제: 누출.**
#
# 기존 파이프라인은 윈저라이징 상한 · 대치 중앙값 · Yeo-Johnson λ · hot-deck
# 공여자 풀을 **175만 건 전체**에서 계산한 뒤 분할했습니다.
# 즉 test 행의 정보가 train 행의 변환값에 섞여 있었습니다.
#
# 실제 영향은 작았습니다(중앙값 66,000 vs 67,000 수준). 로버스트 통계량이고
# 표본이 크기 때문입니다. 하지만 **원칙 위반이고 보고서에서 방어하기 껄끄럽습니다.**
#
# ### 새 순서
# ```
# 행 필터(step1) → 분할(step1) → train 에서 fit → 전체에 transform
# ```
#
# ### 이 노트북이 하는 일
# | 단계 | 무엇 |
# |---|---|
# | 1 | cohort · split 로드 |
# | 2 | **fit** — train 430,781행만 보고 상수 학습 → `fit_params.json` |
# | 3 | 그 상수로 전체 717,969행 변환 → `X_table.csv` |
# | 4 | 검증 — 결측·계층제약·원-핫 |
# | 5 | **반증 검증** — transform 재실행이 fit 결과와 일치하는가 |
# | 6 | 기존(누출 포함) 결과물과 대조 |

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
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lc_config as C
import lc_params
import lc_preprocess as PP

C.check_paths()
C.describe()

OUT = C.STEP2_DIR
FITPARAMS = os.path.join(OUT, "fit_params.json")
DONORS = os.path.join(OUT, "hotdeck_donors.parquet")
XPATH = os.path.join(OUT, "X_table.csv")
ZIP3 = os.path.join(PIPE_DIR, "zip3_external_cache.csv")

print(f"\n산출물 폴더: {OUT}")

# %%
# ─────────────────────────────────────────────────────────────
# 1. cohort · split 로드
# ─────────────────────────────────────────────────────────────
cohort = pd.read_csv(C.COHORT_PATH)
split = pd.read_csv(C.SPLIT_PATH)
assert set(cohort["id"]) == set(split["id"])

cohort_ids = cohort["id"].to_numpy()
smap = dict(zip(split["id"].astype("int64"), split["split"]))
is_train = np.array([smap[int(i)] == "train" for i in cohort_ids], dtype=bool)

print(f"cohort {len(cohort):,}행")
print(f"  train {is_train.sum():,} / val+test {(~is_train).sum():,}")
print(f"  train 비율 {is_train.mean() * 100:.2f}%")

# %%
# ─────────────────────────────────────────────────────────────
# 2~3. FIT — train 에서만 상수를 학습하고, 그 상수로 전체를 변환
# ─────────────────────────────────────────────────────────────
# Params(mode="fit", split_mask=is_train) 은 상수를 train 행에서만 계산하지만,
# 변환 자체는 전체 행에 적용된다. 이게 우리가 원하는 동작이다.
t0 = time.time()

P = lc_params.Params(mode="fit", split_mask=is_train, path=FITPARAMS)
X = PP.build_X(
    raw_path=C.RAW_PATH,
    cohort_ids=cohort_ids,
    P=P,
    zip3_cache_path=ZIP3,
    donor_pool_path=DONORS,
    donor_mask=is_train,          # ← hot-deck donor 도 train 에서만
)
P.save()
P.summary()
print(f"\n소요 {time.time() - t0:.1f}초")

# %%
# ─────────────────────────────────────────────────────────────
# 4. 검증
# ─────────────────────────────────────────────────────────────
passed = PP.verify(X)
if not passed:
    raise SystemExit("!! 검증 실패 — 위 로그 확인")

X.to_csv(XPATH, index=False, encoding="utf-8-sig", float_format="%.10g")
print(f"\n저장 {os.path.basename(XPATH)}  {os.path.getsize(XPATH) / 1e6:.1f} MB")

# %%
# ─────────────────────────────────────────────────────────────
# 5. 반증 검증 ① — transform 재실행이 fit 결과와 일치하는가
# ─────────────────────────────────────────────────────────────
# transform 모드는 데이터를 보지 않고 저장된 상수만 쓴다.
# 두 결과가 다르면 어딘가에서 아직 데이터를 보고 값을 정하고 있다는 뜻이다.
print("\n" + "=" * 64)
print("반증 검증 ① — transform 재실행 일치 여부")
print("=" * 64)

P2 = lc_params.Params(mode="transform", path=FITPARAMS)
X2 = PP.build_X(C.RAW_PATH, cohort_ids, P2, ZIP3, DONORS, donor_mask=None)

assert list(X.columns) == list(X2.columns), "컬럼 구성 불일치"
num = X.select_dtypes(include=[np.number]).columns
d = (X[num].to_numpy(dtype=float) - X2[num].to_numpy(dtype=float))
d = np.nan_to_num(d, nan=0.0)
print(f"  최대 절대차 {np.abs(d).max():.3e}")
print(f"  차이 있는 셀 {int((np.abs(d) > 1e-9).sum()):,} / {d.size:,}")
if np.abs(d).max() < 1e-9:
    print("  → 완전 일치. transform 이 데이터를 보지 않는다는 증거.")
else:
    worst = pd.Series(np.abs(d).max(axis=0), index=num).nlargest(5)
    print("  → [주의] 불일치 컬럼:")
    print(worst.to_string())

del X2, P2

# %%
# ─────────────────────────────────────────────────────────────
# 6. 반증 검증 ② — 기존(누출 포함) 결과물과 대조
# ─────────────────────────────────────────────────────────────
# 목적은 "같은가"가 아니라 **"로직이 훼손되지 않았는가"** 확인이다.
# 상수를 train 에서 다시 추정했으므로 값은 조금 달라야 정상이다.
# 컬럼 구성이 같고 분포가 거의 같으면 이식이 성공한 것이다.
OLD = os.path.join(PROJ, "domain_filtered_X_table_raw.csv")
print("\n" + "=" * 64)
print("반증 검증 ② — 기존 결과물과 대조")
print("=" * 64)

if not os.path.exists(OLD):
    print("기존 파일이 없어 건너뜁니다.")
else:
    old = pd.read_csv(OLD, low_memory=False)
    old = old[old["id"].isin(set(cohort_ids))].set_index("id")
    new = X.set_index("id")
    common_id = old.index.intersection(new.index)
    old, new = old.loc[common_id], new.loc[common_id]
    print(f"공통 {len(common_id):,}행")

    only_old = sorted(set(old.columns) - set(new.columns))
    only_new = sorted(set(new.columns) - set(old.columns))
    print(f"기존에만 있는 컬럼 {len(only_old)}개: {only_old}")
    print(f"신규에만 있는 컬럼 {len(only_new)}개: {only_new}")

    both = [c for c in new.columns if c in old.columns]
    rows = []
    for c in both:
        if not (pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c])):
            continue
        a, b = old[c].astype(float), new[c].astype(float)
        m = a.notna() & b.notna()
        rows.append({
            "컬럼": c,
            "기존평균": a[m].mean(), "신규평균": b[m].mean(),
            "상관": a[m].corr(b[m]) if m.sum() > 1 else np.nan,
            "완전일치율": float(np.isclose(a[m], b[m], rtol=1e-6, atol=1e-9).mean()),
        })
    cmp = pd.DataFrame(rows)
    cmp["평균차이%"] = ((cmp["신규평균"] - cmp["기존평균"])
                     / cmp["기존평균"].replace(0, np.nan) * 100).round(3)
    cmp = cmp.sort_values("상관")
    cmp.to_csv(os.path.join(OUT, "step2_기존대조.csv"), index=False, encoding="utf-8-sig")

    print(f"\n공통 수치 컬럼 {len(cmp)}개")
    print(f"  상관 >= 0.999 인 컬럼 {int((cmp['상관'] >= 0.999).sum())}개")
    print(f"  완전일치율 100% 인 컬럼 {int((cmp['완전일치율'] >= 0.9999).sum())}개")
    print("\n상관이 낮은 하위 10개 (여기가 실제로 바뀐 부분):")
    print(cmp.head(10).to_string(index=False))
    del old, new

print("\n✅ STEP 2 완료")
