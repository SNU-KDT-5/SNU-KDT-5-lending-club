"""
렌딩클럽 건별 IRR / 초과수익률 계산 모듈
=========================================

## 수익률 정의 — 주 지표와 강건성 지표   (작업 1-①)

이 모듈은 두 가지 연율 수익률을 산출한다.

    irr           (주 지표)   : 실제 보유기간(issue_d~last_pymnt_d) 기준 건별 IRR.
                                과제 스펙이 "건별 IRR"이므로 이것이 성과의 정의다.
    irr_contract  (강건성)    : 논문 각주 5 방식. 실현 HPR 을 계약 만기(36/60개월)
                                기준으로 연율화한 대안 자(尺).

모형 선택 · threshold 결정은 **전부 irr(→ excess 컬럼)로 한다.**
irr_contract(→ excess_contract 컬럼)는 확정된 모형을 그대로 재적용해 결론이
유지되는지 보는 **강건성 검증 전용**이다.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 두 지표의 샤프를 비교해 더 높은 쪽을 고르지 않는다.                    │
    └─────────────────────────────────────────────────────────────────────┘

  근거: "전처리 방식도 하나의 모델이니 아웃 오브 샘플 샤프로 고른다"는 원칙은
  목적함수(=샤프를 재는 자)가 고정된 상태에서 X(모형 입력)쪽 처리를 고를 때만
  성립한다. 그런데 수익률 정의는 X가 아니라 **샤프 그 자체를 정의하는 자(尺)**다.
  irr 기준 샤프와 irr_contract 기준 샤프는 애초에 비교 가능한 숫자가 아니다
  (같은 대출이 +14.00%가 되기도 +6.55%가 되기도 한다). 여기서 높은 쪽을 고르면
  더 좋은 모델을 고른 게 아니라 **자기에게 유리한 자를 고른 것**이다. 그래서
  주 지표를 irr 로 못박고, irr_contract 는 하방 꼬리를 다르게 다루는 대안 자로서
  "결론이 자에 둔감한가"만 확인한다. (노트북 §9 참조)


## 이전 구현과의 차이   (작업 1-⑦)

  팀의 이전 IRR 구현(정상 건은 실제 보유기간 IRR, 부도 건은 계약만기 HPR 연율화,
  추심수수료 미차감, last_pymnt_d 결측 건 제외)을 이 모듈이 대체한다.

  | 항목               | 이전                    | 현재                                    | 바꾼 이유
  | ------------------ | ----------------------- | --------------------------------------- | ---------
  | 순회수             | total_pymnt (gross)     | total_pymnt − collection_recovery_fee   | 추심수수료 18%(중앙값) 미차감 시 부도 손실을 체계적으로 과소평가
  | 부도 건 연율화 기준| 계약만기(36/60개월)     | 정상 건과 동일하게 실제 보유기간        | 하방은 계약만기로 누르고 상방은 실제 기간으로 열어두면 초과수익 분포의 왼쪽 꼬리가 인위적으로 잘려 샤프의 분자·분모가 함께 위로 편향된다
  | last_pymnt_d 결측  | 계산 제외(NaN)          | k=1 배치 → −100%                        | 해당 건 100%가 Charged Off. 제외 = 최악 손실만 골라 빼는 생존 편향
  | 극단치             | 부도 건만 억제, 정상 무방비 | 전 표본 동일 규칙 윈저화             | 조기완납 건도 연율화하면 폭발한다. 규칙은 전 표본에 대칭이어야 한다

  요약: 이전 구현은 틀린 게 아니라 **정상 건에는 irr, 부도 건에는 irr_contract 를
  쓰는 하이브리드**였다. 그래서 하방만 눌리는 비대칭이 생긴다(노트북 §4 대조 참조).


## 설계 근거 (5A 파트 변수검증 결과 반영)

1. 순회수 = total_pymnt - collection_recovery_fee
   - total_pymnt 에는 recoveries 가 **이미 포함**되어 있음이 실데이터로 확인됨
     (total_pymnt = total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries,
      완결 건 100.00% 성립)
   - collection_recovery_fee 는 추심업체가 떼어가는 비용 → 투자자 수령액에서 차감

2. 현금흐름 형태 (월 단위)
       t=0        : -funded_amnt                      (자금 유출)
       t=1..k-1   : +installment                      (약정 월납)
       t=k        : +잔여액 (총 순회수 - 위 합계)       (조기상환 잔액정산 / 부도 후 회수 포함)
   - k = issue_d ~ last_pymnt_d 경과 개월수
   - 잔여액을 마지막 시점에 몰아주므로 **순회수 총액은 항상 정확히 보존**됨
   - 정상상환 건의 last_pymnt_amnt/installment 중앙값이 13.07배 → 만기 전 잔액
     일시정산이 지배적이라는 관측과 정합적

3. IRR 해법: 이분법(bisection) 벡터화
   - t=0 이후 현금흐름이 모두 >=0 이므로 NPV(r)은 r에 대해 단조감소 → 해가 유일하고
     이분법이 항상 수렴 (Newton 발산 위험 없음)
   - 100만 건 규모에서 수초 내 처리

4. 극단 케이스 (5A 검증에서 확인된 3종)
   (a) last_pymnt_d 결측 2,034건 → 전량 Charged Off. k=1, 회수액을 1개월차에 배치
   (b) 보유기간 <= 0 (0.69%)     → k=1 로 보정
   (c) 보유기간 > 계약 term (12.68%) → 그대로 허용 (부도 후 추심 회수가 만기 뒤에
       들어오는 정상적 현상). cap 을 원하면 clip_k_to_term=True
   (d) 순회수 = 0 (전손, 부도 건의 0.39%) → IRR = -100%
   (e) 준전손: k=1 이면서 회수액이 원금의 몇 % 뿐인 건은 이론적 월 IRR 이
       탐색 하한(-99.99%/월)보다 낮다. 연율 기준으로는 어차피 -100% 이므로
       하한에 고정하여 -100% 로 배정

5. 대안 지표 (논문 각주 5 방식)
   irr_contract : 실현 HPR 을 **계약 만기 기준**으로 연율화
       r = (1 + HPR)^(12 / T_contract) - 1,  T_contract = 36 or 60
   - 단기 부도 건의 손실이 과대 인코딩되는 문제를 완화하는 **강건성 지표**이며,
     주 지표 irr 을 대체하지 않는다(위 "## 수익률 정의" 참조).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
_DAYS_PER_MONTH = 30.436875  # 평균 그레고리력 월 길이


# --------------------------------------------------------------------------
# 1. 현금흐름 구성
# --------------------------------------------------------------------------
def build_cashflow_inputs(
    df: pd.DataFrame,
    clip_k_to_term: bool = False,
) -> pd.DataFrame:
    """원본 컬럼 -> IRR 계산에 필요한 (P, A, F, k, term_n) 로 변환.

    필요 컬럼: funded_amnt, installment, term, issue_d, last_pymnt_d,
              total_pymnt, collection_recovery_fee
    """
    need = [
        "funded_amnt", "installment", "term", "issue_d", "last_pymnt_d",
        "total_pymnt", "collection_recovery_fee",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}")

    out = pd.DataFrame(index=df.index)

    out["term_n"] = (
        df["term"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    )

    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    last_dt = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    out["issue_dt"] = issue_dt
    out["last_dt"] = last_dt

    # 경과 개월수 k
    k = (last_dt - issue_dt).dt.days / _DAYS_PER_MONTH
    k = np.round(k)
    # (a) last_pymnt_d 결측  (b) k <= 0  ->  k = 1
    k = k.fillna(1.0)
    k = k.where(k >= 1.0, 1.0)
    if clip_k_to_term:
        k = np.minimum(k, out["term_n"])
    out["k"] = k

    # 원금(유출)
    # [각주 ⑥ funded_amnt vs funded_amnt_inv] 투자자 관점을 표방한다면
    # funded_amnt_inv(투자자 실제 출자액)가 더 정확하다. 다만 total_pymnt 와
    # total_pymnt_inv 가 92.8% 동일한 것으로 확인돼(노트북 §2·§7) 실질 영향은
    # 작다. 오늘 밤 표본 정의를 흔들 이유가 없어 funded_amnt 로 확정하고 각주만
    # 남긴다. (바꾸려면 여기 한 줄만 funded_amnt_inv 로 교체)
    out["P"] = df["funded_amnt"].astype(float)

    # 순회수 총액: recoveries 는 total_pymnt 에 이미 포함 -> 더하지 않는다
    net_total = df["total_pymnt"].astype(float) - df["collection_recovery_fee"].astype(float)
    net_total = net_total.clip(lower=0.0)
    out["net_total"] = net_total

    # 월납액 A: k-1 개월간 지급. 단 A*(k-1) 이 순회수를 초과하지 않도록 축소
    #
    # [각주 ⑤ installment 를 여기 쓰는 것과 feature 배제가 모순 아닌 이유]
    #   내생성·사후성 판정은 **X(모형 입력)에만** 적용된다. installment 는
    #   (loan_amnt, term, int_rate)의 결정론적 함수(역산 일치율 99.64%, 노트북
    #   §4 각주)라 int_rate 의 우회 통로가 되므로 feature 에서는 배제한다.
    #   그러나 Y(실현 수익률) 계산에는 실제로 발생한 현금흐름을 그대로 쓰는 것이
    #   옳으며, 이는 "사후 정보는 feature 로 금지하되 성과 평가에는 사용한다"는
    #   원칙 그대로다. installment, total_pymnt, last_pymnt_d 모두 같은 이유로
    #   여기서는 정당하게 쓰인다.
    #
    # [각주 ⑧ 부도 건 균등분배 근사 — 의도적으로 현행 유지]
    #   A*(k-1) > 순회수 인 부도 건은 A 를 축소해 순회수를 전 기간에 고르게
    #   편다. 실제 형태("12회 정상납입 후 중단, 40개월차에 추심 유입")와는 다르나,
    #   검증 결과 두 경로의 건별 IRR 격차 중앙값은 1.43%p 로 무시 가능한 수준이라
    #   현행 유지한다(노트북 §4에서 확인). 솔버·현금흐름 구조는 건드리지 않는다.
    A = df["installment"].astype(float)
    n_reg = (out["k"] - 1.0).clip(lower=0.0)
    over = (A * n_reg) > net_total
    A = A.where(~over, np.where(n_reg > 0, net_total / n_reg.replace(0, np.nan), 0.0))
    A = A.fillna(0.0)
    out["A"] = A
    out["n_reg"] = n_reg

    # 마지막 시점 잔여액 F (>=0 보장)
    out["F"] = (net_total - A * n_reg).clip(lower=0.0)

    out["hpr"] = net_total / out["P"] - 1.0
    return out


# --------------------------------------------------------------------------
# 2. NPV / IRR
# --------------------------------------------------------------------------
def _annuity_factor(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(1 - (1+r)^-m) / r,  r->0 극한은 m."""
    with np.errstate(all="ignore"):
        small = np.abs(r) < 1e-12
        r_safe = np.where(small, 1.0, r)
        af = (1.0 - np.power(1.0 + r_safe, -m)) / r_safe
        return np.where(small, m, af)


def _npv(r: np.ndarray, P, A, F, k, n_reg) -> np.ndarray:
    with np.errstate(all="ignore"):
        return -P + A * _annuity_factor(r, n_reg) + F * np.power(1.0 + r, -k)


def solve_irr(
    cf: pd.DataFrame,
    lo: float = -0.9999,
    hi: float = 1.0,
    n_iter: int = 80,
    clip: tuple[float, float] | None = (-1.0, 1.0),
) -> pd.Series:
    """월 IRR 을 이분법으로 풀고 연율(annualized)로 반환.

    반환값은 연율 IRR. 전손(net_total=0) 건은 -1.0 (=-100%).

    clip : 연율 IRR 의 (하한, 상한). 기본 (-100%, +100%).
        보유기간 k 가 1~2개월인 건을 연율화하면 (1+r)^12 이 폭발한다.
        실측(완결 건 1,115,888 · 캐시로 재현 가능, 노트북 §6):
          - 무처리 시 연율 IRR 최댓값 = +1,091,937%
          - 무처리 시 표준편차       = 10.34   (윈저화 후 0.32)
          - 상한 +100% 초과 건       = 15건 (0.0013%), 하한 -100% 미만 = 0건
        상한 +100% 는 "1년에 원금을 두 배로 회수" 수준이며, 이를 넘는 15건은
        전부 초단기 대량회수 건이므로 윈저화하는 것이 방어적이다.
        상한이 임의 상수라는 점은 유일한 약점이므로 sensitivity_winsor() 로
        cap ∈ {50%,100%,200%,무처리} 민감도를 제시한다(결과가 cap 에 둔감하면
        그 자체가 방어 근거).
        None 을 주면 원시값 그대로 반환(민감도 분석용).
    """
    P = cf["P"].to_numpy(float)
    A = cf["A"].to_numpy(float)
    F = cf["F"].to_numpy(float)
    k = cf["k"].to_numpy(float)
    n_reg = cf["n_reg"].to_numpy(float)

    lo_v = np.full(P.shape, lo)
    hi_v = np.full(P.shape, hi)

    # 상한에서도 NPV>0 이면(초고수익) 상한을 넓힌다
    for _ in range(6):
        need = _npv(hi_v, P, A, F, k, n_reg) > 0
        if not need.any():
            break
        hi_v = np.where(need, hi_v * 2.0, hi_v)

    for _ in range(n_iter):
        mid = 0.5 * (lo_v + hi_v)
        v = _npv(mid, P, A, F, k, n_reg)
        # NPV 는 r 에 대해 단조감소: v>0 이면 해는 오른쪽
        lo_v = np.where(v > 0, mid, lo_v)
        hi_v = np.where(v > 0, hi_v, mid)

    r_m = 0.5 * (lo_v + hi_v)
    irr_annual = np.power(1.0 + r_m, MONTHS_PER_YEAR) - 1.0

    # 전손 / 유효하지 않은 건 처리
    dead = (cf["net_total"].to_numpy(float) <= 0) | (P <= 0)
    irr_annual = np.where(dead, -1.0, irr_annual)

    if clip is not None:
        irr_annual = np.clip(irr_annual, clip[0], clip[1])

    return pd.Series(irr_annual, index=cf.index, name="irr")


def irr_contract_annualized(cf: pd.DataFrame) -> pd.Series:
    """논문 각주 5 방식: 실현 HPR 을 계약 만기(36/60개월) 기준으로 연율화.

        r = (1 + HPR)^(12 / T_contract) - 1
    """
    hpr = cf["hpr"].to_numpy(float)
    T = cf["term_n"].to_numpy(float)
    base = np.clip(1.0 + hpr, 0.0, None)
    r = np.power(base, MONTHS_PER_YEAR / T) - 1.0
    return pd.Series(np.where(base <= 0, -1.0, r), index=cf.index, name="irr_contract")


# --------------------------------------------------------------------------
# 3. 초과수익률
# --------------------------------------------------------------------------
def excess_return(
    irr: pd.Series,
    cf: pd.DataFrame,
    treasury: pd.DataFrame | None = None,
    rf_flat: float | None = None,
) -> pd.Series:
    """초과수익률 = IRR - (대출 실행 시점 · 만기 매칭 국채 수익률).

    treasury : index=월초 Timestamp, columns=['y3','y5'] 형태의 국채 수익률(소수).
               제공되지 않으면 rf_flat(스칼라)을 사용.
    """
    if treasury is not None:
        key = cf["issue_dt"].dt.to_period("M").dt.to_timestamp()
        col = np.where(cf["term_n"].to_numpy() <= 36, "y3", "y5")
        t = treasury.reindex(key)
        rf = pd.Series(
            np.where(col == "y3", t["y3"].to_numpy(), t["y5"].to_numpy()),
            index=cf.index,
        )
        # 해당 월 데이터 결측 시 전후 값으로 보간
        if rf.isna().any():
            filled = treasury.sort_index().ffill().bfill().reindex(key)
            alt = pd.Series(
                np.where(col == "y3", filled["y3"].to_numpy(), filled["y5"].to_numpy()),
                index=cf.index,
            )
            rf = rf.fillna(alt)
    elif rf_flat is not None:
        rf = pd.Series(rf_flat, index=cf.index)
    else:
        raise ValueError("treasury 또는 rf_flat 중 하나는 있어야 합니다")

    return (irr - rf).rename("excess")


# --------------------------------------------------------------------------
# 4. 샤프 비율
# --------------------------------------------------------------------------
def sharpe(excess: pd.Series) -> float:
    """건별 동일가중 샤프 = 초과수익률 평균 / 표준편차."""
    e = pd.Series(excess).dropna()
    if len(e) < 2:
        return np.nan
    sd = e.std(ddof=1)
    return float(e.mean() / sd) if sd > 0 else np.nan


# --------------------------------------------------------------------------
# 4b. 윈저화 민감도   (작업 1-②)
# --------------------------------------------------------------------------
def sensitivity_winsor(
    df: pd.DataFrame,
    treasury: pd.DataFrame | None = None,
    caps: list[float | None] = [0.5, 1.0, 2.0, None],
    rf_flat: float | None = None,
) -> pd.DataFrame:
    """연율 IRR 윈저화 상한(cap)을 바꿔가며 전건승인 샤프가 얼마나 흔들리는지 표로 반환.

    기본 파이프라인의 윈저 cap 은 ±100%(=±1.0)로 고정돼 있고, 그 +100% 상한이
    임의 상수라는 것이 유일한 약점이다(외부 검수자가 가장 먼저 파고들 지점).
    이 함수는 cap ∈ {50%, 100%, 200%, 무처리(None)} 각각에 대해 전건승인 샤프 /
    초과수익 평균 / 표준편차 / 윈저화된 건수를 한 표로 뽑는다.

        ▶ 결과(특히 샤프)가 cap 선택에 둔감하면, 그 둔감함 자체가 방어 근거다.
          cap=None(무처리) 행을 함께 넣어 "왜 윈저화가 필요한가"(표준편차 폭발)도
          같은 표에서 보이게 한다.

    [클립 적용 순서 — 중요]
        각 cap 마다 **IRR 을 먼저 클립하고 그다음 rf 를 뺀다.** 초과수익(excess)을
        직접 클립하면 안 된다. 국채 수익률(rf)은 대출 실행 시점마다 다르므로, 순서가
        바뀌면 같은 대출이 cap 마다 다른 rf 취급을 받는다.
            excess_c = np.clip(irr_raw, -cap, cap) - rf      (rf 는 cap 과 무관, 1회 계산)

    반환: index=cap 라벨, columns=[sharpe_allapprove, excess_mean, excess_std,
          n_winsorized, pct_winsorized]
    """
    if treasury is None and rf_flat is None:
        from fetch_treasury import load_treasury

        treasury = load_treasury()

    cf = build_cashflow_inputs(df)
    irr_raw = solve_irr(cf, clip=None)               # IRR 은 딱 한 번만 푼다
    raw = irr_raw.to_numpy(float)

    # rf 는 cap 과 무관하므로 한 번만 계산해 재사용
    # excess_return(irr) = irr - rf  =>  rf = irr - excess_return(irr)
    rf = irr_raw - excess_return(irr_raw, cf, treasury=treasury, rf_flat=rf_flat)

    rows = []
    for cap in caps:
        if cap is None:
            irr_c = raw
            n_wins = 0
            label = "무처리(None)"
        else:
            irr_c = np.clip(raw, -cap, cap)          # 먼저 클립
            n_wins = int(((raw > cap) | (raw < -cap)).sum())
            label = f"±{cap:g} ({cap*100:.0f}%)"
        excess_c = pd.Series(irr_c, index=cf.index) - rf   # 그다음 rf 차감
        rows.append({
            "cap": label,
            "sharpe_allapprove": sharpe(excess_c),
            "excess_mean": float(excess_c.mean()),
            "excess_std": float(excess_c.std(ddof=1)),
            "n_winsorized": n_wins,
            "pct_winsorized": n_wins / len(raw),
        })
    return pd.DataFrame(rows).set_index("cap")


# --------------------------------------------------------------------------
# 5. vintage 학습 표본 컷 — 만기매칭 버퍼 마스크
# --------------------------------------------------------------------------
def vintage_buffer_mask(
    issue_dt: pd.Series,
    term_n: pd.Series,
    snapshot_period: str | pd.Period,
    buffer_months: int = 6,
) -> pd.Series:
    """만기매칭 버퍼 컷 마스크 (월 단위 Period 벡터 연산).

        in_sample = issue_period + term_n + buffer_months <= snapshot_period

    term(36/60)별 만기 도달 시점을 각각 맞춰 생존편향(만기 미도달 건이 조기상환·
    조기부도 양극단만 완결로 관측되는 검열)을 통제한다. flat 연도 컷은 2016년
    60개월(완결 71.8%, 미성숙)을 넣고 2017년 36개월(완결 77.1%, 만기 도달)을 버리는
    이중 오류가 있는데, term별 만기 매칭이 이를 정공법으로 교정한다.

    ── 구현 규약 ──────────────────────────────────────────────────────────
    · **DateOffset+apply 가 아니라 PeriodIndex 정수 덧셈으로 벡터화한다.**
      issue_period(월) + (term_n + buffer) 개월 <= snapshot_period(월). 전부 월 단위
      Period 비교라 일(day) 단위 경계 모호성이 없다.
    · snapshot_period 는 **학습 표본 정의 전용 고정 상수**다. last_credit_pull_d.max()
      같은 런타임 자동도출을 이 함수 안에서 하지 않는다 — 호출자가 노트북에서 1회
      계산·assert(== 관측 스냅샷) 후 상수로 넘긴다. 이 필터는 7/27 배부 테스트
      데이터에는 **어떤 형태로도 적용되지 않는다**(테스트는 주는 대로 채점).
    · issue_dt 전 건이 월초(day==1)로 파싱됐는지, 결측이 없는지, term 결측이 없는지
      assert 한다. 위반 시 즉시 중단(조용한 오분류 방지).

    참고: SNAP=2020-10 · buffer=6 이면 컷 경계는 36개월 issue_d ≤ 2017-04,
    60개월 issue_d ≤ 2015-04 이다(= snapshot − term − buffer). 호출자는 term별
    최대 발행월이 이 경계와 일치하는지 assert 로 확인할 수 있다.
    """
    if issue_dt.isna().any():
        raise AssertionError(
            f"issue_dt 결측 {int(issue_dt.isna().sum())}건 — 버퍼 컷 전에 처리 필요"
        )
    if not bool((issue_dt.dt.day == 1).all()):
        raise AssertionError("issue_dt 가 전부 월초(day==1)로 파싱되지 않았습니다")
    if term_n.isna().any():
        raise AssertionError(f"term_n 결측 {int(term_n.isna().sum())}건 — 버퍼 컷 불가")

    snap = pd.Period(snapshot_period, freq="M")
    per = pd.PeriodIndex(issue_dt, freq="M")
    offset = term_n.to_numpy().astype("int64") + int(buffer_months)
    matured = np.asarray((per + offset) <= snap)
    return pd.Series(matured, index=issue_dt.index, name="in_sample")


# --------------------------------------------------------------------------
# 6. 원스톱 파이프라인
# --------------------------------------------------------------------------
def compute_returns(
    df: pd.DataFrame,
    treasury: pd.DataFrame | None = None,
    rf_flat: float | None = None,
    clip_k_to_term: bool = False,
    irr_clip: tuple[float, float] | None = (-1.0, 1.0),
    snapshot_period: str | pd.Period | None = None,
    vintage_buffer_months: int = 6,
    issue_year_max: int | None = None,
) -> pd.DataFrame:
    """원본 df -> irr / irr_contract / excess 컬럼이 붙은 DataFrame 반환.

    treasury 를 생략하면 `data/treasury_monthly.csv` 를 자동으로 읽는다.
    (없으면 `python fetch_treasury.py` 로 먼저 생성할 것)
    rf_flat 을 주면 국채 대신 평면 무위험수익률을 쓴다 — 디버깅용.

    [주 지표 — 작업 1-①]
        excess 가 주 지표다. 모형 선택 · threshold 결정은 **전부 excess 컬럼**으로
        한다. excess_contract 는 확정 모형을 그대로 재적용해 결론이 유지되는지 보는
        **강건성 검증 전용**이며, 두 컬럼의 샤프를 비교해 더 높은 쪽을 고르는 용도로
        쓰지 않는다(모듈 최상단 "## 수익률 정의" 참조).

    [vintage 학습 표본 컷 — 만기매칭 버퍼(권장) / 연도 컷(하위호환)]
        두 방식 중 하나로 in_sample 불리언 마스크를 만든다. **동시 지정은 에러.**

        (1) 만기매칭 버퍼 컷 (기본·권장) — `snapshot_period` 를 주면 활성화.
              in_sample = issue_period + term_n + vintage_buffer_months <= snapshot_period
            term별 만기 도달을 각각 맞춰 검열편향을 통제한다. 계산은 월 단위 Period
            벡터 연산(vintage_buffer_mask, DateOffset+apply 아님). snapshot_period 는
            학습 표본 전용 상수이며 호출자가 노트북에서 1회 계산·assert 후 넘긴다
            (런타임 자동도출 금지). 확정안 = snapshot_period='2020-10',
            vintage_buffer_months=6 → 학습 표본 721,880건(검열잔량 0.06%).

        (2) flat 연도 컷 (하위호환/deprecated) — `issue_year_max` 를 주면 활성화.
              in_sample = issue_dt.dt.year <= issue_year_max
            term 이질성을 무시하는 근사라 (1)로 대체됐다. 기존 노트북 호환용으로만 유지.

        (3) 둘 다 None(기본) → 전 건 in_sample=True (원 기본 동작 보존).

        ★ **행을 드롭하지 않고 불리언 컬럼 in_sample 로만 표시한다.** 호출자가
          ret[ret.in_sample] 로 거른다. (a) vintage 편향 증명은 전 vintage 계산이
          있어야 그려지고, (b) 행 수 불변으로 index 정렬 붕괴가 원천 차단되며,
          (c) 컷 변경 시 IRR 재계산 없이 마스크만 다시 계산하면 된다.
        ★ **이 필터는 학습 표본 정의에만 적용된다.** 7/27 배부될 테스트 데이터는
          주는 대로 채점해야 하므로 어떤 형태로도 적용하지 않는다. 테스트가 최근
          vintage 위주로 오면 학습에 없던 검열(만기 미도달) 구조를 만나 샤프가
          떨어질 수 있으며, 그건 재설계 사유가 아니라 보고서에 쓸 원인 분석이다.

    반환 DataFrame 은 입력 df 의 index 를 그대로 보존한다(정렬은 index 기준으로 할 것).
    """
    if treasury is None and rf_flat is None:
        try:
            from fetch_treasury import load_treasury

            treasury = load_treasury()
        except Exception as e:  # 파일이 없으면 안내 후 중단
            raise RuntimeError(
                "국채 수익률을 찾을 수 없습니다. `python fetch_treasury.py` 를 먼저 실행하거나 "
                "treasury= / rf_flat= 을 직접 지정하세요."
            ) from e
    cf = build_cashflow_inputs(df, clip_k_to_term=clip_k_to_term)
    res = cf[["term_n", "issue_dt", "last_dt", "k", "P", "net_total", "hpr"]].copy()
    res["irr_raw"] = solve_irr(cf, clip=None)
    res["irr"] = solve_irr(cf, clip=irr_clip)
    res["irr_contract"] = irr_contract_annualized(cf)
    res["excess"] = excess_return(res["irr"], cf, treasury=treasury, rf_flat=rf_flat)
    res["excess_contract"] = excess_return(
        res["irr_contract"], cf, treasury=treasury, rf_flat=rf_flat
    )
    # vintage 학습 표본 마스크 (행 드롭 X, 마스크 O)
    if snapshot_period is not None and issue_year_max is not None:
        raise ValueError(
            "snapshot_period(만기매칭 버퍼 컷)와 issue_year_max(flat 연도 컷)는 "
            "동시 지정할 수 없습니다. 하나만 쓰세요."
        )
    if snapshot_period is not None:
        res["in_sample"] = vintage_buffer_mask(
            res["issue_dt"], res["term_n"], snapshot_period, vintage_buffer_months
        )
    elif issue_year_max is not None:
        res["in_sample"] = res["issue_dt"].dt.year <= issue_year_max
    else:
        res["in_sample"] = True
    return res
