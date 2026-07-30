# Lending Club 신용평가 모델 — 전처리 파이프라인

통계·데이터사이언스 팀 과제 

무위험 국채보다 나은 대출 투자처를 골라내는 신용평가 모델을 만든다.
이 저장소는 그중 **표본 정의 · X 전처리 · y(수익률) 계산 · 변수 선택**을 담당한다.

---

## 파이프라인

```
steps/step1_cohort_and_split.py   표본 확정 + 6:2:2 분할        →  717,969건
steps/step2_run.py                X 전처리 (train fit → transform)
steps/step3_run.py                y_table (IRR · 국채 · excess)
steps/step4_build_candidates.py   LASSO 후보 X (원본에서 최소 정제)
steps/step4_run_lasso.py          LASSO 2차 검증
steps/step4_finalize.py           최종 X 확정 + y 병합 (선형용 / 트리용)
```

경로·상수는 `pipeline/lc_config.py` 하나에 모여 있다. 다른 파일은 필터 조건을 갖지 않는다.

## 구성

| 폴더 | 내용 |
|---|---|
| `pipeline/` | 공용 모듈. 설정·상수관리·전처리 함수·IRR |
| `steps/` | 단계별 실행 스크립트 (VS Code `# %%` 셀) |
| `analysis/` | 검증·진단 스크립트 |
| `legacy/` | 구 파이프라인. 팀의 도메인 판단 원문 보존용 |

### pipeline/

| 파일 | 역할 |
|---|---|
| `lc_config.py` | **모든 필터 조건·상수의 단일 출처.** 각 결정의 근거를 주석에 기록 |
| `lc_params.py` | fit/transform 상수 관리. **누출 방지의 핵심** |
| `lc_preprocess.py` | 도메인 전처리 Part 1~6 을 함수로 재구성 |
| `irr_module.py` | IRR·초과수익 계산. 수익률 정의의 단일 출처 |
| `fetch_treasury.py` | FRED 국채 수집 (3년물·5년물) |

---

## 핵심 설계

### 1. 누출 차단 — 순서를 바로잡았다

```
기존:  원본 → 전처리(전체에서 통계량 계산) → 분할
개선:  행 필터 → 분할 → train 에서 fit → val/test 에 transform
```

윈저라이징 상한 · 대치 중앙값 · Yeo-Johnson λ · hot-deck 공여자 풀을
**175만 건 전체에서 계산한 뒤 분할**하고 있었다. test 행의 정보가 train 행의 변환값에 섞였다.

`lc_params.Params` 가 학습이 필요한 상수 **44개**를 전부 train 에서만 계산해 JSON 에 저장한다.
호출부는 분기가 필요 없다.

```python
P = Params(mode="fit", split_mask=is_train)   # train 행만 보고 계산 → json
P = Params(mode="transform")                  # 저장된 값만 적용, 데이터를 안 봄

cap = P.quantile("annual_inc_cap", df["annual_inc"], 0.995)
lam = P.yeojohnson_lambda("yj_revol_bal", df["revol_bal"])
```

**검증**: fit 결과와 transform 재실행 결과가 **불일치 0 / 14,001,328셀**.

### 2. vintage 편향 통제

만기 전에 끝나는 대출은 조기상환(우량)과 조기부도(불량)뿐이다.
만기를 채우는 평범한 다수는 아직 진행 중이라 표본에 못 들어온다.
**즉 최근 vintage 는 양극단만 남고 중간층이 사라진 표본이다.**

`발행월 + term + 6개월 ≤ 2020-10` 인 건만 학습 표본에 넣는다.
term 별로 만기를 각각 맞추므로 연도 컷으로는 표현할 수 없다.

채택한 발행월의 완결률: 36개월 **99.89%** / 60개월 **99.83%**

### 3. 변수 선택 — 두 트랙 교차검증

도메인 필터 결과로 LASSO 를 돌리면 *"고른 것 중 뭘 더 뺄까"* 만 답할 수 있다.
**검증이 아니라 축소가 된다.**

그래서 원본 141개에서 **기계적으로 반드시 빼야 할 것만**(사후·내생·y재료·식별자)
제거한 넓은 후보군을 따로 만들어(`step4_build_candidates.py`) 도메인 판단과 4분면으로 대조한다.

최종: 선형 모델용 47열 / 트리 모델용 122열. **선형용 ⊂ 트리용** (assert 로 강제).

---

## 실행

```bash
pip install -r requirements.txt
python steps/step1_cohort_and_split.py
python steps/step2_run.py          # 10~20분
python steps/step3_run.py          # 3~5분
python steps/step4_build_candidates.py
python steps/step4_run_lasso.py    # 수 분
python steps/step4_finalize.py
```

VS Code 에서 `# %%` 셀 단위로 실행해도 된다.
저장소 구조(`pipeline/`)와 작업 폴더 구조(`파이프라인/`) 모두에서 동작하도록
경로를 자동 탐색한다.

### 필요한 데이터 (저장소에 없음)

| 경로 | 내용 |
|---|---|
| `자료_데이터/lending_club_2020_train.csv` | 원본 175만행 × 141열 (1.27GB) |
| `pipeline/treasury_monthly.csv` | FRED 국채. `fetch_treasury.py` 로 생성 |
| `pipeline/zip3_external_cache.csv` | Census ACS · FDIC zip3 집계 946개 |

---

## 주의

| 항목 | 내용 |
|---|---|
| pandas 3.x | `dtype == object` 로 문자열 판별 금지. `pd.api.types.is_numeric_dtype()` 사용 |
| 원본 텍스트 행 | 956,843행 부근에 구분용 텍스트 행이 섞여 있다. `to_numeric(errors="coerce")` 후 **`reset_index(drop=True)` 필수** |
| 메모리 | 175만행×141열을 통째로 읽으면 터진다. `usecols` + `chunksize` 필수 |
| sklearn LassoCV | float32 로 넘기면 맥에서 Gram matrix 오류. **float64** 사용. `n_jobs=-1` 은 메모리 폭발 |
| 학습된 상수 | `*fit_params.json`, `hotdeck_donors.*` 는 `.gitignore` 로 제외했다. 새 데이터에 적용하려면 **`mode="transform"`** 으로 별도 전달받을 것 |

---

## 파일명에 한글이 있습니다

`analysis/` 의 일부와 주석·문자열이 한글이다. UTF-8 환경에서는 문제없으나,
Windows 에서 clone 할 때 `git config --global core.quotepath false` 를 권한다.
