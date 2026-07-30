# %% [markdown]
# # STEP 4-C — 최종 X 확정 및 y 병합 (선형용 / 트리용 두 벌)
#
# **`변수결정표.csv` 의 `결정` 열만 고치면 결과가 바뀝니다.** 코드는 안 건드려도 됩니다.
#
# | 결정 | 의미 |
# |---|---|
# | `both` | 선형·트리 양쪽 |
# | `tree` | 트리용에만 (공선성·VIF·NaN 유지 등의 이유로 선형 부적합) |
# | `drop` | 양쪽 제외 |
#
# ## 왜 두 벌인가
# LASSO가 죽인 변수의 대부분은 **정보가 없어서가 아니라 대체재가 있어서**입니다
# (`bc_open_to_buy` ↔ `total_bc_limit` r=0.84 등). 선형 모델에서는 공선성이
# 계수를 불안정하게 만들지만, **트리 모델은 공선성에 영향받지 않습니다.**
# `sec_app_*` 8개도 결측 93%라 선형에는 못 넣지만 트리는 NaN을 분기로 다룹니다.
#
# ## 되살린 변수는 어떻게 처리하나
# step 2 X_table에 없던 변수는 여기서 새로 만듭니다.
# **train에서만 학습한 중앙값으로 대치**하고(`step4_fit_params.json`),
# 금액류는 step 2와 같은 기준으로 Yeo-Johnson을 적용합니다.
# 누출 방지 원칙은 step 2와 동일합니다.

# %%
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "파이프라인"))
import lc_config as C
import lc_params

C.ensure_dirs()
OUT = C.STEP_DIR[4]
FITP = os.path.join(OUT, "step4_fit_params.json")

dec = pd.read_csv(os.path.join(OUT, "변수결정표.csv"))
print(f"결정표 {len(dec)}행")
print(dec.groupby("결정").size().to_string())

LIN_SRC = set(dec.loc[dec["결정"] == "both", "원천변수"])
TREE_SRC = set(dec.loc[dec["결정"].isin(["both", "tree"]), "원천변수"])
print(f"\n선형용 원천변수 {len(LIN_SRC)}개 / 트리용 {len(TREE_SRC)}개")

# %%
# ─────────────────────────────────────────────────────────────
# 1. step 2 X_table 적재
# ─────────────────────────────────────────────────────────────
X = pd.read_csv(os.path.join(C.STEP2_DIR, "X_table.csv"))
print(f"step2 X_table {X.shape[0]:,}행 × {X.shape[1]}열")

split = pd.read_csv(C.SPLIT_PATH)
smap = dict(zip(split["id"].astype("int64"), split["split"]))
is_train = np.array([smap[int(i)] == "train" for i in X["id"]], dtype=bool)
print(f"  train {is_train.sum():,}")


def src_of(col):
    """열 이름 → 원천변수. 원-핫으로 쪼개진 것을 되묶는다."""
    for p in ["purpose", "home_ownership", "addr_state", "verification_status"]:
        if col.startswith(p + "_"):
            return p
    return {"is_joint_application": "application_type",
            "emp_length_missing": "emp_length",
            "term": "term"}.get(col, col)


# %%
# ─────────────────────────────────────────────────────────────
# 2. 되살릴 변수 추가 — step2 에 없는 것만
# ─────────────────────────────────────────────────────────────
have = {src_of(c) for c in X.columns if c != "id"}
# step2 에서 파생으로 흡수돼 이름이 사라진 것들은 이미 있는 셈으로 친다
have |= {"fico_range_low", "fico_range_high", "earliest_cr_line", "zip_code"}
need = sorted((TREE_SRC - have))
print(f"\n추가로 만들어야 할 원천변수 {len(need)}개: {need}")

NEW_NUM = [c for c in need if c not in ("addr_state", "verification_status")]
NEW_CAT = [c for c in need if c in ("addr_state", "verification_status")]
# 금액·한도류는 왜도가 커서 step2 와 같은 기준으로 YJ 를 적용한다
YJ_NEW = [c for c in NEW_NUM if c in ("total_il_high_credit_limit",)]

if need:
    raw_cols = pd.read_csv(C.RAW_PATH, nrows=2).columns.tolist()
    use = ["id"] + [c for c in need if c in raw_cols]
    want = set(X["id"].astype("int64"))
    parts = []
    for ch in pd.read_csv(C.RAW_PATH, usecols=use, chunksize=200_000, low_memory=False,
                          dtype={c: str for c in NEW_CAT if c in use}):
        ch["id"] = pd.to_numeric(ch["id"], errors="coerce")
        ch = ch.loc[ch["id"].notna()]
        ch = ch.loc[ch["id"].astype("int64").isin(want)]
        if len(ch):
            parts.append(ch)
    add = pd.concat(parts, ignore_index=True)
    del parts
    add["id"] = add["id"].astype("int64")
    add = add.set_index("id").loc[X["id"].to_numpy()].reset_index()
    for c in NEW_NUM:
        if c in add.columns and not pd.api.types.is_numeric_dtype(add[c]):
            add[c] = pd.to_numeric(add[c].astype(str).str.strip().str.rstrip("%"),
                                   errors="coerce")

    P = lc_params.Params(mode="fit", split_mask=is_train, path=FITP)
    for c in NEW_NUM:
        if c not in add.columns:
            print(f"  [주의] 원본에 없음: {c}")
            continue
        n = int(add[c].isna().sum())
        add[c] = add[c].fillna(P.median(f"s4_{c}_med", add[c]))
        if c in YJ_NEW:
            from scipy.stats import yeojohnson
            lam = P.yeojohnson_lambda(f"s4_yj_{c}", add[c])
            sk = float(add[c].skew())
            add[c] = yeojohnson(add[c].to_numpy(dtype=float), lmbda=lam)
            print(f"  {c:30s} 결측 {n:>7,} → 중앙값 | YJ λ={lam:.4f} 왜도 {sk:.2f}→{add[c].skew():.2f}")
        else:
            print(f"  {c:30s} 결측 {n:>7,} → 중앙값")
        X[c] = add[c].to_numpy()

    for c in NEW_CAT:
        if c not in add.columns:
            continue
        s = add[c].astype(str).str.strip()
        cats = P.categories(f"s4_{c}_cats", s)      # ★ train 에서 고정
        d = pd.get_dummies(pd.Categorical(s, categories=cats), prefix=c, dtype="int8")
        d.index = X.index
        X = pd.concat([X, d], axis=1)
        print(f"  {c:30s} {len(cats)}개 범주 → 원-핫")
    P.save()
    del add

print(f"\n확장 후 X {X.shape[0]:,}행 × {X.shape[1]}열")

# %%
# ─────────────────────────────────────────────────────────────
# 3. 두 벌로 분리
# ─────────────────────────────────────────────────────────────
cols = [c for c in X.columns if c != "id"]
lin_cols = ["id"] + [c for c in cols if src_of(c) in LIN_SRC]
tree_cols = ["id"] + [c for c in cols if src_of(c) in TREE_SRC]

dropped = sorted({src_of(c) for c in cols} - TREE_SRC)
print(f"제외된 원천변수 {len(dropped)}개: {dropped}")

X_lin = X[lin_cols].copy()
X_tree = X[tree_cols].copy()
print(f"\n선형용  {X_lin.shape[0]:,}행 × {X_lin.shape[1] - 1}열")
print(f"트리용  {X_tree.shape[0]:,}행 × {X_tree.shape[1] - 1}열")

# %%
# ─────────────────────────────────────────────────────────────
# 4. 검증
# ─────────────────────────────────────────────────────────────
print("\n[검증]")
ok = []


def chk(c, m):
    ok.append(bool(c))
    print(f"  [{'OK ' if c else 'FAIL'}] {m}")


SEC = [c for c in X.columns if c.startswith("sec_app_")]
chk(X_lin[[c for c in X_lin.columns]].isna().sum().sum() == 0,
    "선형용 결측 0 (sec_app 없어야 정상)")
chk(not any(c.startswith("sec_app_") for c in X_lin.columns),
    "선형용에 sec_app_* 없음")
chk(all(c in X_tree.columns for c in SEC), f"트리용에 sec_app_* {len(SEC)}개 존재")
non_sec = [c for c in X_tree.columns if not c.startswith("sec_app_")]
chk(X_tree[non_sec].isna().sum().sum() == 0, "트리용 결측 0 (sec_app 제외)")
chk(set(X_lin.columns) <= set(X_tree.columns), "선형용 ⊆ 트리용")
chk(X["id"].is_unique, "id 중복 없음")

if not all(ok):
    raise SystemExit("!! 검증 실패")

# %%
# ─────────────────────────────────────────────────────────────
# 5. y 병합 → 모델팀 인계본
# ─────────────────────────────────────────────────────────────
y = pd.read_csv(C.Y_TABLE_PATH)
print(f"\ny_table {y.shape[0]:,}행 × {y.shape[1]}열")

# 모델팀이 쓸 것만: 식별·분할 + 타깃 3종 + 평가에 필요한 보조
Y_KEEP = ["id", "split", "grade", "term_n", "funded_amnt",
          "excess", "excess_contract", "spread_positive", "bad", "rf", "irr"]
y = y[Y_KEEP]

for name, Xv in [("linear", X_lin), ("tree", X_tree)]:
    m = Xv.merge(y, on="id", how="inner", validate="one_to_one")
    assert len(m) == len(Xv), f"{name}: 병합에서 행이 사라졌습니다"
    p = os.path.join(OUT, f"model_input_{name}.csv")
    m.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.10g")
    nx = Xv.shape[1] - 1
    print(f"  model_input_{name}.csv  {len(m):,}행 × {m.shape[1]}열 "
          f"(X {nx} + y {len(Y_KEEP) - 1} + id)  {os.path.getsize(p) / 1e6:.0f} MB")

# 모델팀용 컬럼 안내서
guide = pd.DataFrame({
    "열": list(X_tree.columns[1:]),
    "원천변수": [src_of(c) for c in X_tree.columns[1:]],
})
guide["선형용포함"] = guide["열"].isin(X_lin.columns).astype(int)
guide = guide.merge(dec[["원천변수", "결정", "논의필요", "근거"]], on="원천변수", how="left")
guide.to_csv(os.path.join(OUT, "model_input_컬럼안내.csv"), index=False, encoding="utf-8-sig")

# 통제 비교용 — 트리 모델을 선형용 41개로도 돌릴 수 있게 열 목록을 남긴다.
# 이게 없으면 "트리가 이긴 게 모델 때문인지 변수 24개 때문인지" 구분할 수 없다.
pd.Series(sorted(set(X_lin.columns) - {"id"}), name="열").to_csv(
    os.path.join(OUT, "공통변수_통제비교용.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("모델팀 인계 안내  (상세: 모델팀_실험설계.md)")
print("=" * 60)
print("  ★ 반드시 3개 조합을 돌릴 것 — 안 그러면 비교가 교란된다")
print("      ① 선형 × 41   ② 트리 × 41 ← 통제칸   ③ 트리 × 65")
print("      모델 효과 = ②−①   |   변수 효과 = ③−②")
print("      (선형용 41 ⊂ 트리용 65 이므로 열만 골라내면 된다:")
print("       공통변수_통제비교용.csv)")
print("  ★ 조합 간 비교 시 승인률을 반드시 맞출 것")
print("-" * 60)
print("  X 열      : id 제외 전부")
print("  타깃      : excess(연속, 주 지표) / spread_positive(이진) / bad(부도, 진단용)")
print("  분할      : split 열 (train/val/test). test 는 최종 1회만.")
print("  평가      : 동일가중 샤프. 거부 건은 초과수익 0(전액 국채)으로 포함.")
print("  넘어야 할 선: A·B 등급만 승인 = 0.1297 (전건승인 0.0874 / Oracle 상한 1.6246)")
print("  ⚠️ id·split·grade·term_n·funded_amnt·rf·irr 은 **feature 아님**. 반드시 제외할 것.")
print("\n✅ STEP 4 완료")
