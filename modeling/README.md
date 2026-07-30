# Lending Club 신용평가 모델 

Lending Club 데이터로 **IRR(수익률)이 국채 수익률을 상회하는 대출을 선별**하는 신용평가 모델을 개발한다. 평가 지표는 Sharpe Ratio(위험 대비 수익률)이다.
<br>

## 노트북 구성

`01_linear_models.ipynb` 선형 계열 두 모델(LR, K-means+Ridge)의 Sharpe 비교와 Optuna 재튜닝, 부트스트랩 유의성 검증

`02_tree_models.ipynb` 트리 계열 3종 × 47/122열 6개 조합 학습·튜닝

`03_effect_decomposition.ipynb`  ①②③ 예측값을 불러와 모형효과/변수효과를 부트스트랩(500회)으로 분해, 컷오프 선정, 등급 내 AUC·부도율·재분류 분석까지 진행

`04_final_test.ipynb` test 데이터로 챔피언 모델 평가

<br>

## 핵심 결과

| 실험 | 모델 | 변수 | val Sharpe | 승인율 |
|---|---|---|---|---|
| ① 선형×47 | K-means+Ridge (k=7) | 47열 | 0.1717 | 40% |
| ② 트리×47 | CatBoost | 47열 | 0.1921 | 45% |
| ③ 트리×122 | CatBoost | 122열 | **0.1940** | **45%** |

<br>
1. 트리 계열이 선형 계열보다 나은 이유의 92.2%는 모형 구조 차이(모형효과), 변수 개수 차이(변수효과)의 기여는 통계적으로 유의하지 않음 → CatBoost×122열을 챔피언 모델로 확정  <br>
2. 승인 컷오프는 val Sharpe를 최대화하는 45%로 확정  <br>
3. 최종 test(744,443건) 결과: 모델 Sharpe +0.1164 vs LC sub_grade 기준선 +0.0425 (전 구간에서 모델 우위) 

<br>


