#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lending Club 프로젝트 공통 설정
===============================
표본을 정의하는 모든 조건을 이 파일 하나에 모은다.
어떤 스크립트도 필터 조건을 자체적으로 갖지 않는다. 반드시 여기서 import 한다.

2026-07 팀 회의 결정사항 반영:
  · vintage      : D안 유지, 버퍼 3개월 → 6개월
  · 표본 시작연도 : 2007~2009 제외 → 2010년 이후 발행분만
  · loan_status  : Default(268건) 제외
  · IRR 정의     : 재정의 진행 중 (아래 IRR 섹션 참조)

파이프라인 순서 (의존 관계)
  step1  cohort 확정 + 6:2:2 분할      ← IRR 과 무관. 지금 실행 가능
  step2  X 전처리 (train fit → transform) ← IRR 과 무관. 지금 실행 가능
  step3  y_table (IRR · 국채 · spread)  ← IRR 정의 확정 후
  step4  LASSO 2차 검증
  step5  모델 학습 → 챔피언 → test 1회
"""

import os
import numpy as np

# ─────────────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────────────
# BASE_DIR    : 이 파일이 있는 곳 = 파이프라인 코드 디렉터리
# PROJECT_DIR : 그 상위 = 프로젝트 루트. 원본 데이터(1.2GB)는 옮기지 않고 루트에 둔다.
#
# 2026-07-26 : 작업 폴더를 날짜별('7:25 오후', '7:26 오전')로 나누면서 상대경로가 전부
#              깨졌다. 코드는 '파이프라인/' 한 곳에만 두고 데이터는 루트에서 찾도록
#              분리한다. 앞으로 모든 스크립트는 파이프라인/ 안에서 실행할 것.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
RAW_PATH = os.path.join(PROJECT_DIR, "자료_데이터", "lending_club_2020_train.csv")
TREASURY_PATH = os.path.join(BASE_DIR, "treasury_monthly.csv")

# 단계별 산출물 폴더 (프로젝트 루트). 사람이 열어보는 결과물은 여기에 둔다.
STEP_DIR = {n: os.path.join(PROJECT_DIR, f"step {n}") for n in range(1, 6)}


def ensure_dirs(*ns):
    """산출물 폴더를 만든다. import 시점이 아니라 **쓰기 직전에** 호출한다.

    import 만으로 폴더가 생기면 저장소를 clone 했을 때 빈 폴더 5개가 딸려 나온다.
    ns 를 비우면 전부 만든다.
    """
    for n in (ns or STEP_DIR):
        os.makedirs(STEP_DIR[n], exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

STEP1_DIR, STEP2_DIR, STEP3_DIR = STEP_DIR[1], STEP_DIR[2], STEP_DIR[3]

# IRR 모듈 위치. 전처리 쪽에서 import 할 수 있도록 sys.path 에 넣는다.
IRR_DIR = os.path.join(PROJECT_DIR, "irr")


def load_irr_module():
    """irr/irr_module.py 를 import 한다. IRR 정의의 단일 출처."""
    import sys
    if IRR_DIR not in sys.path:
        sys.path.insert(0, IRR_DIR)
    import irr_module
    return irr_module


OUT_DIR = os.path.join(BASE_DIR, "산출물")

COHORT_PATH = os.path.join(STEP1_DIR, "cohort.csv")            # 분석 대상 행 집합
SPLIT_PATH = os.path.join(STEP1_DIR, "split_assignment.csv")   # id -> train/val/test
FITPARAMS_PATH = os.path.join(OUT_DIR, "fit_params.json")    # train에서 학습한 전처리 상수
X_TABLE_PATH = os.path.join(STEP2_DIR, "X_table.csv")          # step2b 산출 (누출 없는 X)
Y_TABLE_PATH = os.path.join(STEP3_DIR, "y_table.csv")


def check_paths():
    """원본 경로가 실제로 존재하는지 확인한다. 모든 step 스크립트 시작 시 호출."""
    ensure_dirs()
    missing = [p for p in (RAW_PATH, TREASURY_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "다음 경로를 찾을 수 없습니다:\n  " + "\n  ".join(missing) +
            f"\n(PROJECT_DIR={PROJECT_DIR})")

# ─────────────────────────────────────────────────────────────
# 1. 표본 정의 — 어떤 행을 분석에 쓸 것인가
# ─────────────────────────────────────────────────────────────

# (1) vintage 컷 기준 시점
#
#     ★ 이름에 주의. 이것은 "데이터 추출 시점"이 아니라 **컷 파라미터**다.
#       실제 추출 시점은 2020-12 가 맞다 (last_pymnt_d.max() == 2020-12,
#       즉 2020년 11·12월 상환이 데이터에 기록돼 있다).
#       이전 코드는 이 상수를 SNAPSHOT 이라 부르며 "추출 시점"으로 오해했고,
#       IRR 쪽 CHANGES.md 는 last_credit_pull_d.max()==2020-10 을 근거로
#       "2020-12는 관측되지 않은 미래"라고 적었다. 둘 다 부정확하다.
#       신용조회는 전 대출에 매월 걸리지 않으므로 그 최댓값은 추출 시점의 하한일 뿐이다.
#
#     그럼에도 2020-10 을 쓰는 이유는 데이터가 거기서 끝나서가 아니라,
#     **2020년 코로나 구간의 만기 이연 때문에 버퍼 6개월로는 부족하기 때문**이다.
#     2020-12 로 하면 두 칸이 추가로 들어오는데, 그 두 칸만 아직 성숙하지 않았다.
#
#       추가되는 칸        건수      그 칸의 완결률
#       2017년 36개월    26,830          77.09%
#       2015년 60개월    10,540          86.67%
#       (이미 포함) 2016년 36개월        99.96%
#       (이미 포함) 2014년 60개월        99.98%
#
#     아직 안 끝난 건은 대부분 정상 상환 중이라, 이들이 빠지면 부도가 과대표집된다.
#     실제로 추가분 부도율은 21.20% (기존 표본 16.20%) 이고, term·등급을 고정해도
#     14칸 전부에서 추가분이 더 나쁘다. 즉 D안 필터가 제거하려던 vintage 편향이
#     그대로 되돌아온다. 표본 3.7만 건을 얻는 대신 편향된 슬라이스를 얻는 셈이다.
#
#     상세 근거: 파이프라인/검증_스냅샷시점.md
#
#     ★ 이 컷은 **학습 표본에만** 적용한다. 7/27 배부 테스트 데이터에는
#       어떤 형태로도 적용하지 않는다 (테스트는 주는 대로 채점).
VINTAGE_CUT_PERIOD = "2020-10"   # 월 단위 Period 문자열

# (2) 완결 상태 정의
#     Fully Paid  : 완납. 미상환 원금 0, 현금흐름 종료 — 확정
#     Charged Off : 상각. 더 이상 회수 안 봄, 현금흐름 종료 — 확정
#     Default     : 121일+ 연체이나 아직 상각 전. out_prncp가 남은 '진행 중' — 미확정
#
#     데이터 확인 결과 (2026-07):
#       Fully Paid  898,522건 → out_prncp = 0 인 건 898,522 (100%)
#       Charged Off 217,366건 → out_prncp = 0 인 건 217,366 (100%)
#       Default        268건 → out_prncp = 0 인 건 1 (중앙값 2,038달러 잔존)
#     즉 Default만 현금흐름이 끝나지 않았다. IRR을 계산할 수 없으므로 제외한다.
COMPLETED_STATUS = ["Fully Paid", "Charged Off"]
BAD_STATUS = ["Charged Off"]          # 부도 라벨 (Default 제외했으므로 Charged Off만)

# (3) "Does not meet the credit policy..." 접두 상태 제외
#     LC가 사후에 심사 기준 미달로 재분류한 건. 정상 승인 건과 성격이 다르다.
EXCLUDE_POLICY_LOANS = True

# (4) vintage 필터 — 만기가 도래한 발행분만 사용
#     최근 발행분에서 '완결'로 잡히는 건은 조기상환(우량)과 조기부도(불량)뿐이라
#     만기를 채운 중간층이 표본에서 빠진다.
#     검증: 만기 도래분은 완결건의 42.2%가 만기 80% 이상을 채웠으나,
#           미도래분은 1.1%에 불과했다.
VINTAGE_FILTER = "maturity"       # "maturity" | "none"
MATURITY_BUFFER_MONTHS = 6        # 회의 결정: 3 → 6개월
#     버퍼 근거: 미국 소비자 대출은 통상 120일(약 4개월) 연체 시 상각 처리한다.
#     3개월로는 만기 직전 연체 건의 상각 인식이 아직 안 됐을 수 있다.
#     버퍼별 민감도 (만기 80%↑ 비중 / 표본 수):
#       0개월 37.5% / 862,236    3개월 39.4% / 813,064
#       6개월 41.2% / 759,372    9개월 42.2% / 702,276    12개월 42.7% / 642,502
#     6~9개월에서 개선 곡선이 꺾인다.

# (5) 발행 시작연도 — LC 초기 구간 제외
#     2007~2009는 저등급 셀이 통계적으로 비어 있다.
#     (G등급 2007년 0건 · 2008년 2건 · 2009년 17건, F등급 2008년 12건)
#     등급 내 개별 편차를 학습하는 전략에서는 사용할 수 없는 구간.
#     또한 평균 대출액 8,600 → 14,000달러, 금리 체계와 심사 기준도 달랐다.
ISSUE_YEAR_MIN = 2010
ISSUE_YEAR_MAX = None             # None이면 상한 없음 (vintage 필터가 알아서 자름)

# 예상 표본 (위 조건 전부 적용): 717,969건
#   원본 1,755,294 → 상태 필터 1,115,888 → vintage 721,880 → 연도 717,969

# ─────────────────────────────────────────────────────────────
# 2. 분할
# ─────────────────────────────────────────────────────────────
SEED = 42
SPLIT_RATIOS = (0.6, 0.2, 0.2)    # train / val / test
# 분할은 cohort(행 집합)만 정해지면 만들 수 있다. IRR 정의와 무관하다.
# 예상: train 430,781 / val 143,593 / test 143,595

# ─────────────────────────────────────────────────────────────
# 3. 전처리 — 누출 방지 원칙
# ─────────────────────────────────────────────────────────────
# 이전 파이프라인은 175만 건 전체에서 윈저라이징 상한 · 대치 중앙값 ·
# Yeo-Johnson λ · hot-deck 공여자 풀을 계산한 뒤 분할했다.
# 즉 test 행의 정보가 train 행의 변환값에 섞였다.
#
# 새 원칙: 학습이 필요한 모든 상수는 train에서만 계산해 fit_params.json에 저장하고,
#          val/test에는 그 값을 그대로 적용(transform)한다.
FIT_ON = "train"                  # 절대 바꾸지 말 것

ANNUAL_INC_CLIP_Q = 0.995         # 연소득 상위 윈저라이징 분위수
DTI_CAP = 60.0                    # dti 상한 클리핑
DTI_SENTINEL = 999                # 결측 센티넬 값

# ─────────────────────────────────────────────────────────────
# 4. IRR — 이 파일은 IRR을 정의하지 않는다
# ─────────────────────────────────────────────────────────────
# 2026-07-26 결정: IRR 정의의 단일 출처는 `irr_module.py` 다.
#
# 이전에 이 자리에 IRR_CASHFLOW / IRR_ANNUALIZE / IRR_WINSOR 상수가 있었으나
# 삭제했다. 이유는 두 가지다.
#
#   (1) 실제로 어긋나 있었다. 여기엔 IRR_CASHFLOW="equal"(균등분배)이라고
#       적혀 있었지만, irr_module.build_cashflow_inputs 는 처음부터
#       A = df["installment"] — 즉 **매월 installment 고정**으로 계산하고 있었다.
#       (A*(k-1) 이 순회수를 넘는 부도 건에서만 A를 축소한다.)
#       설정 파일이 코드와 다른 말을 하고 있었다.
#
#   (2) 상수를 두 곳에 두면 같은 사고가 반복된다. IRR은 단순한 상수 조합이 아니라
#       현금흐름 구성 · 이분법 솔버 · 국채 매칭 · 윈저화가 얽힌 하나의 절차다.
#       상수만 복사하면 절차는 복사되지 않는다.
#
# 따라서 step3 는 IRR을 다시 구현하지 않고 모듈을 그대로 호출한다.
#
#     import irr_module
#     ret = irr_module.compute_returns(df, treasury=tr,
#                                      snapshot_period=SNAPSHOT_PERIOD,
#                                      vintage_buffer_months=MATURITY_BUFFER_MONTHS)
#     y = ret.loc[ret["in_sample"], "excess"]      # ← 주 지표
#
# 주 지표는 `excess`(실제 보유기간 연율화) 하나다.
# `excess_contract`(계약만기 연율화)는 확정 모형을 재적용해 결론이 유지되는지 보는
# **강건성 검증 전용**이며, 두 컬럼의 샤프를 비교해 높은 쪽을 고르는 데 쓰지 않는다.
# 수익률 정의는 모형의 입력이 아니라 **샤프를 재는 자(尺)** 이기 때문이다.
# 자를 바꿔가며 유리한 쪽을 고르면 더 좋은 모델이 아니라 유리한 자를 고른 것이 된다.
#
# 국채 시계열만 이 파일이 계속 관리한다 (표본 정의와 함께 움직이므로).
TREASURY_SERIES = {36: "DGS3", 60: "DGS5"}


def load_treasury():
    """국채 수익률을 irr_module 이 기대하는 형태로 변환해 반환한다.

    ★ 어댑터가 필요한 이유
      fetch_treasury.py 는 (month, rf_36m, rf_60m) 컬럼으로 CSV 를 쓰는데,
      irr_module.excess_return 은 index=월초 Timestamp · columns=['y3','y5'] 를
      기대한다. 게다가 irr_module 은 `from fetch_treasury import load_treasury` 를
      시도하지만 fetch_treasury.py 에는 그런 함수가 없어서 자동 로드가 실패한다.
      따라서 호출자가 treasury= 로 명시적으로 넘겨야 한다. 그 변환을 여기 모아둔다.

    36개월 대출 → 3년물(y3), 60개월 대출 → 5년물(y5) 로 만기를 맞춘다.
    """
    import pandas as pd
    tr = pd.read_csv(TREASURY_PATH, parse_dates=["month"]).set_index("month")
    tr = tr.rename(columns={"rf_36m": "y3", "rf_60m": "y5"})[["y3", "y5"]]
    if tr.isna().any().any():
        raise ValueError(f"국채 결측 {int(tr.isna().sum().sum())}건 — fetch_treasury.py 재실행 필요")
    return tr.sort_index()


# ─────────────────────────────────────────────────────────────
# 5. 평가
# ─────────────────────────────────────────────────────────────
# ─── 샤프 정의 (미결 #8) ────────────────────────────────────
# 세 후보를 전체 표본 717,969행에서 대조했다 (step 3/검증_샤프정의.md).
#
#   · 전략 순위는 세 정의가 완전히 일치한다 (8개 전략, Spearman 1.0000).
#     → 어느 것을 골라도 "어떤 전략이 이기는가"는 안 바뀐다.
#   · 다만 B등급 vs C등급 판별력이 갈린다.
#       현행(혼합)   B−C = +0.0023, 95%CI [−0.0002, +0.0052]  ← 0을 포함
#       동일가중     B−C = +0.0286, 95%CI [+0.0266, +0.0309]
#     우리 논지가 '중간 등급 살리기'라면 B>C 를 말할 수 있어야 하는데
#     현행 정의로는 근거가 약하다.
#   · corr(대출액, excess) 는 전체 −0.010, 등급 내에서도 |r| < 0.04.
#     금액가중은 정보를 더하지 않고 추정 분산만 늘린다.
#
# 결정 근거는 결과가 아니라 원칙이다:
#   ① 내부 일관성 — 분자(금액가중)와 분모(단순)의 측도가 다르면
#      그 비율이 샤프인 포트폴리오가 존재하지 않는다.
#   ② 투자자 현실 — LC 투자자는 $25 단위 note 를 산다. 균등 배분이 실제로 가능하다.
#      금액가중은 "LC가 정한 규모대로 투자한다"를 가정하는데, 우리 전제는
#      "LC의 가격결정을 신뢰하지 않는다"이다. 채점표에 LC의 사이징을 섞을 이유가 없다.
#   ③ 정보량 — 위 상관.
#
# ★ 보고서에는 세 정의 표를 전부 싣는다. 자를 바꿔가며 고르지 않았다는 증거다.
SHARPE_WEIGHT_NUMERATOR = False       # 동일가중
SHARPE_WEIGHT_DENOMINATOR = False
SHARPE_REPORT_ALL_DEFS = True         # 부록에 3정의 대조표 필수

# ─── 학습 타깃 (미결 #2) — 둘 다 유지 ──────────────────────
# 계획서는 이진(spread_positive)이었으나 채점은 연속 spread 다.
# 이진은 "+0.1%p"와 "+20%p"를 같은 1로 취급한다.
# 반대로 샤프 분모(변동성)를 생각하면 이진이 목적함수에 더 가깝다는 반론도 성립한다.
# → 결론을 미리 내지 않고 y_table 에 둘 다 넣어 validation 샤프로 결정한다.
TARGETS = {
    "continuous": "excess",           # 연속 회귀
    "binary": "spread_positive",      # 이진 분류 (excess > 0, 양성 85.44%)
    "default": "bad",                 # 부도 판정 — 비교·진단용
}

# ─── 포트폴리오 구성 규칙 (미결 #3) — 등급대별 배분 유지 ────
# "예측 excess 상위 X% 매수"는 샤프 최적이 아님이 확인됐다.
# 전역 상위컷은 저등급으로 쏠려 변동성이 커진다. 등급 안에서 고르면
# 수익 상한은 낮아지지만 변동성이 훨씬 더 줄어 샤프가 올라간다.
#   전역 상위20% 0.4893  <  B등급내 상위40% 0.3800 (승인률 12.8%)
#   ※ 승인률이 다르므로 직접 비교는 부적절. step5 에서 승인률을 맞춰 재비교할 것.
PORTFOLIO_RULE = "within_grade"       # "within_grade" | "global_topk"
PORTFOLIO_GRADES = ["B", "C"]         # 주력 등급대. A·D 는 민감도 분석으로



def describe():
    """현재 설정 요약 출력 — 모든 step 스크립트 시작 시 호출한다."""
    print("=" * 66)
    print("  Lending Club 파이프라인 설정")
    print("=" * 66)
    print(f"  완결 상태      : {COMPLETED_STATUS}  (Default 제외)")
    print(f"  policy 건      : {'제외' if EXCLUDE_POLICY_LOANS else '포함'}")
    print(f"  vintage 컷     : {VINTAGE_CUT_PERIOD} 기준, 버퍼 {MATURITY_BUFFER_MONTHS}개월")
    print(f"  발행연도       : {ISSUE_YEAR_MIN} 이후")
    print(f"  분할           : {SPLIT_RATIOS}, seed={SEED}")
    print(f"  전처리 fit     : {FIT_ON} 에서만")
    print(f"  IRR 정의       : irr_module.compute_returns() — excess 컬럼")
    print("=" * 66)


if __name__ == "__main__":
    describe()
