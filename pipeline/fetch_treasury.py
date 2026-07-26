
#%%
"""
FRED 국채수익률 수집 → treasury_monthly.csv

36개월 대출 → 3년물(DGS3), 60개월 대출 → 5년물(DGS5).
일별 수익률을 월평균으로 집계하고 퍼센트를 소수로 바꿔 저장한다.

[실행]
    python fetch_treasury.py

[출력]
    treasury_monthly.csv   (month, rf_36m, rf_60m)  ← build_y_table.py 가 읽는 파일
    맨 아래에 검증 요약이 출력된다. 그 부분을 복사해서 공유하면 값 확인이 가능하다.

[네트워크가 막혀 있다면]
    브라우저로 아래 두 주소를 열어 CSV 를 받은 뒤 이 스크립트와 같은 폴더에 두고 다시 실행.
        https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3   → DGS3.csv 로 저장
        https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS5   → DGS5.csv 로 저장
"""

import os
import sys
from io import StringIO

import numpy as np
import pandas as pd

SERIES = {"rf_36m": "DGS3", "rf_60m": "DGS5"}   # 36개월→3년물, 60개월→5년물

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
OUT_PATH = os.path.join(BASE_DIR, "treasury_monthly.csv")


def load_series(series_id):
    """
    일별 시계열을 DataFrame(date, value) 로 반환.
    1순위 로컬 CSV, 2순위 FRED 웹 요청.
    """
    local = os.path.join(BASE_DIR, f"{series_id}.csv")
    if os.path.exists(local):
        raw = pd.read_csv(local)
        print(f"  {series_id}: 로컬 파일 사용 ({os.path.basename(local)})")
    else:
        import requests
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        try:
            r = requests.get(url, params={"id": series_id}, timeout=90)
            r.raise_for_status()
        except Exception as e:
            print(f"\n!! {series_id} 수집 실패: {type(e).__name__}: {e}")
            print(f"!! 브라우저로 아래 주소를 열어 CSV 를 받고,")
            print(f"!!   {url}?id={series_id}")
            print(f"!! '{series_id}.csv' 이름으로 이 폴더에 저장한 뒤 다시 실행하세요:")
            print(f"!!   {BASE_DIR}")
            sys.exit(1)
        raw = pd.read_csv(StringIO(r.text))
        print(f"  {series_id}: FRED 에서 수집 완료")

    # FRED 는 컬럼명이 DATE/observation_date 등으로 바뀌어 온 적이 있어 위치로 잡는다
    raw = raw.iloc[:, :2]
    raw.columns = ["date", "value"]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    # 휴장일은 '.' 으로 들어온다 → 숫자 변환 실패분을 버린다
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    return raw.dropna()


print("=" * 62)
print("FRED 국채수익률 수집")
print("=" * 62)

monthly = {}
for col, sid in SERIES.items():
    s = load_series(sid)
    print(f"    일별 {len(s):,}건  ({s['date'].min():%Y-%m-%d} ~ {s['date'].max():%Y-%m-%d})")
    s["month"] = s["date"].values.astype("datetime64[M]")
    # 퍼센트(2.54) → 소수(0.0254)
    monthly[col] = s.groupby("month")["value"].mean() / 100.0

tr = pd.DataFrame(monthly).reset_index()
tr = tr.sort_values("month").reset_index(drop=True)

# 대출 데이터 기간(2007-01 ~ 2020-12)만 남겨도 되지만, 여유 있게 2000년 이후 보관
tr = tr[tr["month"] >= "2000-01-01"].reset_index(drop=True)

tr.to_csv(OUT_PATH, index=False)

print()
print("=" * 62)
print("검증 요약  ↓↓↓ 아래부터 복사해서 공유하세요 ↓↓↓")
print("=" * 62)
print(f"파일: treasury_monthly.csv")
print(f"기간: {tr['month'].min():%Y-%m} ~ {tr['month'].max():%Y-%m}  ({len(tr)}개월)")
print(f"결측: rf_36m {int(tr['rf_36m'].isna().sum())}건 / rf_60m {int(tr['rf_60m'].isna().sum())}건")
print()
print(f"전체 평균: 3년물 {tr['rf_36m'].mean() * 100:.3f}%  /  5년물 {tr['rf_60m'].mean() * 100:.3f}%")
print(f"범위     : 3년물 {tr['rf_36m'].min() * 100:.3f}% ~ {tr['rf_36m'].max() * 100:.3f}%")
print(f"           5년물 {tr['rf_60m'].min() * 100:.3f}% ~ {tr['rf_60m'].max() * 100:.3f}%")

inv = int((tr["rf_36m"] > tr["rf_60m"]).sum())
print(f"장단기 역전(3년물>5년물) 개월수: {inv}개월 / {len(tr)}개월")
print("  (보통 5년물이 높다. 역전은 경기침체 신호로 소수 발생하는 게 정상)")

print()
print("대출 데이터 기간(2007-01 ~ 2020-12) 연도별 평균 (%)")
sub = tr[(tr["month"] >= "2007-01-01") & (tr["month"] <= "2020-12-01")].copy()
sub["yr"] = pd.to_datetime(sub["month"]).dt.year
tbl = (sub.groupby("yr")[["rf_36m", "rf_60m"]].mean() * 100).round(3)
print(tbl.to_string())

print()
print("샘플 (처음 3개월 / 마지막 3개월)")
print(pd.concat([tr.head(3), tr.tail(3)]).to_string(index=False))
print("=" * 62)
print("↑↑↑ 여기까지 복사 ↑↑↑")

print()
print("확인 포인트")
print("  1) 5년물 평균이 3년물보다 높아야 정상 (기간 프리미엄)")
print("  2) 2012~2013년이 최저, 2007년과 2018~2019년이 높게 나와야 실제 금리 사이클과 일치")
print("  3) 결측 0 이어야 함")
print()
print(f"저장 완료 → {OUT_PATH}")
print("이 파일을 build_y_table.py 와 같은 폴더에 두면 자동으로 읽습니다.")

# %%
