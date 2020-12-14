# 서울시 배달건수 회귀분석
- 배달건수 EDA를 통해 얻은 insight를 활용하고, 기상데이터를 추가하여 배달건수를 분석하는 회귀분석 프로젝트입니다.
- 본 프로젝트에서는 2016년 6월 ~ 2019년 9월 서울시 강남구 치킨 통화건수 주문량을 분석하였습니다. 
- 본 프로젝트의 목적은 변수를 다방면으로 전처리 및 스케일링함으로써 전처리의 필요성을 알고, 여러 회귀 모델의 특징과 이에 따른 성능을 비교함을써 최적의 분석 모델을 찾아가기 위함에 있습니다.

## Getting Started
### Requirements
- Python 3.6+
### Installation
The quick way:
```
pip install pandas
pip install matplotlib
pip install seaborn
pip install datetime
pip install sklearn
pip install scipy
```
### Dataset
- 서울시 강남구 치킨 통화주문건수 데이터
  - [SKT data hub](https://www.bigdatahub.co.kr/index.do)
- 서울시 기상 데이터 
  - [기상청](http://www.weather.go.kr/weather/climate/past_cal.jsp)
- 서울시 미세먼지 데이터 
  - [서울시열린데이터광장](https://data.seoul.go.kr/dataList/OA-2218/F/1/datasetView.do)
- 한국 공휴일 데이터
  - [공공데이터포털](https://data.go.kr/index.do)


## 분석 진행순서
1. 데이터 수집 및 결측치 해결
2. 데이터가공 및 전처리
3. 예측변수 스케일링에 따른 모델별 학습, 예측, 평가
4. 모델 검증

### 1. 데이터 수집 및 결측치 입력
- 각각의 배달데이터, 날씨데이터, 공기오염데이터, 공휴일데이터를 수집 및 병합 
- 결측치 해결 (강수량, 적설량 -> 0, 기온 -> 검색하여 수동 입력, 미세먼지 -> 결측발생 월 평균값 입력)

### 2. 데이터 가공 및 전처리
- 범주형 예측변수 원핫인코딩
```python
df = pd.get_dummies(raw_data, columns=['연', '월', '일', '시간대', '요일', '공휴일'])
```

- 훈련용데이터 / 테스트데이터 분리 및 linear regression(initial trial)
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target,
                                                    test_size=0.3, random_state=13)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

# 모델 예측
pred = lr_reg.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)

mae_val = mean_absolute_error(y_test, pred)

r2 = r2_score(y_test, pred)
print(rmse, mae_val, r2)

```

- 목표변수 로그화 
```python
# log 변환
y_target_log = np.log1p(y_target)

# data set 분할
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size=0.3, random_state=13)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)
y_test_exp = np.expm1(y_test)  #로그 처리를 원래 스케일로 변환 
pred_exp = np.expm1(pred)

mse = mean_squared_error(y_test_exp, pred_exp)
rmse_val = np.sqrt(mse)
mae_val = mean_absolute_error(y_test_exp, pred_exp)
r2 = r2_score(y_test_exp, pred_exp)

print('RMSE : {} | MAE : {} | r2 : {} '.format(round(rmse_val,2),round(mae_val,2),round(r2,3)))
```

- 예측변수 이상치 확인(필요시 제거)
```python
# 회귀계수 확인
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
%matplotlib inline
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'

coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)[:20]

sns.barplot(x=coef_sort.values, y=coef_sort.index);
sns.regplot(x=X_features['시간대_18'], y=y_target_log, data =df );
# 회귀계수 상위 5개 이상치 확인되지 않음 
```

### 3. 모델 학습, 예측, 평가
- Linear regression, Ridge, Lasso, Decision Tree Regression, Randomforest Regression 5개 모델의 RMSE, MAE, R2 값을 비교
```
```
  - Ridge, Lasso, Randomforest 모델의 하이퍼파라미터 튜닝 시도 후 최적의 하이퍼파라미터 확인


- 예측 변수 조건을 변화시키며 시도
  - 예측변수 로그화
  - 예측변수 정규화(Standard Scaler)
  - z-score 기준 이상치 제거
  - 예측변수 정규화(Minmax Scaler)
  

### 4. 모델별 교차 검증
- 모델별 k-fold 교차검증
```
from sklearn.model_selection import cross_val_score


def display_socres(model):
    scores = cross_val_score(model, X_test, y_test,
                             scoring="neg_mean_squared_error", cv=5)
    model_rmse_scores = np.sqrt(-socres)
    print('###', model.__class__.__name__, '###')
    print("점수:", model_rmse_scores)
    print("평균:", model_rmse_scores.mean())
    print("표준편차:", model_rmse_scores.std())


for model in [lr_reg, ridge_reg, lasso_reg, tree_reg, forest_reg]:
    display_socres(model)
```


