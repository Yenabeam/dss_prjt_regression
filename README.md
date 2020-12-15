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
- 데이터: "1. data"
- 처리과정: "2. workspace/1211_JR_preprocessing.ipynb"
- 각각의 배달데이터, 날씨데이터, 공기오염데이터, 공휴일데이터를 수집 및 병합 
- 결측치 해결 (강수량, 적설량 -> 0, 기온 -> 검색하여 수동 입력, 미세먼지 -> 결측발생 월 평균값 입력)

  <img src="https://user-images.githubusercontent.com/72846894/102119152-037c1200-3e84-11eb-94ec-3f255d79009e.png"></img>


### 2. 데이터 가공 및 전처리
#### 시간형 예측변수 원핫인코딩
```python
df = pd.get_dummies(raw_data, columns=['연', '월', '일', '시간대', '요일', '공휴일'])
```
  <img src="https://user-images.githubusercontent.com/72846894/102119210-1a226900-3e84-11eb-9f03-e3fab05eb19e.png"></img>
#### 훈련용데이터 / 테스트데이터 분리 및 linear regression(initial trial)
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
<img src="https://user-images.githubusercontent.com/72846894/102119776-fad80b80-3e84-11eb-8851-950722616b1f.png"></img>
#### 목표변수 로그화 - 편중되어 있는 타겟(통화건수)의 왜곡도 낮춤
```python
# log 변환
y_target_log = np.log1p(y_target)
```
<img src="https://user-images.githubusercontent.com/72846894/102119880-21964200-3e85-11eb-8ab4-ba3378b3888a.png"></img>
```python
print('RMSE : {} | MAE : {} | r2 : {} '.format(round(rmse_val,2),round(mae_val,2),round(r2,3)))
```
<img src="https://user-images.githubusercontent.com/72846894/102120195-85206f80-3e85-11eb-82d1-28983df61263.png"></img>
##### -> 목표변수 로그화로 이전 분석보다 r2 score는 증가하고, rmse는 감소하였음.
#### 예측변수 이상치 확인(필요시 제거)
```python
# 회귀계수 확인
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl

coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)[:20]

sns.barplot(x=coef_sort.values, y=coef_sort.index);
sns.regplot(x=X_features['시간대_18'], y=y_target_log, data =df );
```
<img src="https://user-images.githubusercontent.com/72846894/102120332-be58df80-3e85-11eb-8084-bf73753b88b8.png"></img>
### 3. 모델 학습, 예측, 평가
#### Linear regression, Ridge, Lasso, Decision Tree Regression, Randomforest Regression 모델에서도  RMSE, MAE, R2 값을 비교
```python
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test =train_test_split(X_features, y_target_log, test_size=0.3, random_state=13)

# mse / rmse / r2 계산 함수
def evaluate_regr(y, pred):
    mse = mean_squared_error(y, pred)
    rmse_val = np.sqrt(mse)
    mae_val = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print('RMSE : {} | MAE : {} | r2 : {} '.format(round(rmse_val,5),round(mae_val,5),round(r2,5)))

# 여러 모델의 성능 확인 함수 
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1:
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)

    print('###', model.__class__.__name__,'###')
    evaluate_regr(y_test, pred)
#--------------------------------------------------------------------
#모델별로 평가 확인 

lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=0.1)
lasso_reg = Lasso(alpha=0)
tree_reg = DecisionTreeRegressor(random_state=13)
forest_reg = RandomForestRegressor(n_estimators=100,random_state=13)

for model in [lr_reg, ridge_reg, lasso_reg,tree_reg,forest_reg]:
    get_model_predict(model,X_train, X_test, y_train, y_test, is_expm1=True)

```
<img src="https://user-images.githubusercontent.com/72846894/102120513-fe1fc700-3e85-11eb-9a8f-d0a05dab4d06.png"></img>
#### Ridge, Lasso, Randomforest 모델의 하이퍼파라미터 튜닝 시도 후 최적의 하이퍼파라미터 확인
```python
from sklearn.model_selection import GridSearchCV

# 1. Ridge - alpha값이 클수록(penalty 증가) 계수의 크기가 줄어듬
# 영향력이 큰 계수의 영향력을 줄임 / 변수를 축소, 다중공선성을 방지
param_grid = [
    {'alpha': [0, 0.05, 0.1, 0.5, 1, 5]},
]

grid_search = GridSearchCV(ridge, param_grid, cv=5,
                           scoring='r2',
                           return_train_score=True)
grid_search.fit(X_train, y_train)

print('best_params_: ', grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_test_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_test_score, params)
    
# best_params_:  {'alpha': 0.1}, 그러나 alpha 값에 따른 score 변동 거의 없음.
```
<img src="https://user-images.githubusercontent.com/72846894/102120714-47701680-3e86-11eb-8bb4-403e91795e76.png"></img>

```python
# 2. Lasso - alpha 조금만 키워도 계수가 완전히 0이 되는 변수 증가 
# feaure selection, 중요한 변수만 택함
param_grid = [
    {'alpha': [0, 0.05, 0.1, 0.5, 1]},
    ]

grid_search = GridSearchCV(lasso, param_grid, cv=5,
                          scoring='r2',
                          return_train_score=True)
grid_search.fit(X_train, y_train)
print ('best_params_: ', grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_test_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_test_score, params)  
# best_params_:  {'alpha': 0} : 이 데이터로는 Lasso는 하지 않는 것이 바람직.
# alpha 값 감소에 따라 mean_test_score 급격히 감소
```
<img src="https://user-images.githubusercontent.com/72846894/102120849-6ff81080-3e86-11eb-9999-ca991c44bcdc.png"></img>
```python
# 3. Randomforest - 가장 복잡한 모델로, 예측 성능은 좋으나 모델이 복잡하고, 가역성이 좋지않음.
  param_grid = [
    {'n_estimators': [30, 50, 70, 100], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators': [3, 10], 'max_features':[2,3,4] }
    ]

forest_reg = RandomForestRegressor(random_state=13)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(X_train, y_train)
# best_params_: {'max_features': 8, 'n_estimators': 100}
  ```
<img src="https://user-images.githubusercontent.com/72846894/102120913-87cf9480-3e86-11eb-9d74-f321d296739d.png"></img>

#### 예측 변수 조건을 변화시키며 시도
##### 1) raw_data의 분포 - 강수량과 적설량에서 편중 심함
<img src="https://user-images.githubusercontent.com/72846894/102120983-9ddd5500-3e86-11eb-85dd-259f8ef55c73.png"></img>
##### 2) 예측변수(강수량, 적설량) 로그화 -> X축의 범위가 줄어듬
  ```python
  X_features["강수량"] = np.log1p(X_features["강수량"])
  X_features["적설량"] = np.log1p(X_features["적설량"])
  ```
<img src="https://user-images.githubusercontent.com/72846894/102121275-0f1d0800-3e87-11eb-953f-43dce0314364.png"></img>
<img src="https://user-images.githubusercontent.com/72846894/102121302-1ba16080-3e87-11eb-9377-33708d67f8fd.png"></img>
##### 3) 예측변수 정규화(Standard Scaler)
  ```python
  from sklearn.preprocessing import StandardScaler
  # 연속형 예측변수 추출
  scaled_cols = ["기온", "강수량", "풍속", "습도", "적설량", "미세먼지", "초미세먼지"]
  # 정규화
  scaler = StandardScaler()
  scaler.fit(X_features[scaled_cols])
  X_scaled = scaler.transform(X_features[scaled_cols])
  X_features[scaled_cols] = X_scaled
  ```
 <img src="https://user-images.githubusercontent.com/72846894/102121412-4c819580-3e87-11eb-96bd-b6b7cf1dee56.png"></img>
 <img src="https://user-images.githubusercontent.com/72846894/102121428-51dee000-3e87-11eb-9a49-737f25601698.png"></img>
##### 4) z-score 기준 이상치 제거
  ```python
  import scipy as sp
  import scipy.stats

  # check Z score
  df_Zscore = pd.DataFrame()
  outlier_dict = {}
  outlier_idx_list = []

  for one_col in df2[scaled_cols]:
      print("Check", one_col)
      df_Zscore[f'{one_col}_Zscore'] = sp.stats.zscore(df2[one_col])
      outlier_dict[one_col] = df_Zscore[f'{one_col}_Zscore'][(
          df_Zscore[f'{one_col}_Zscore'] > 2) | (df_Zscore[f'{one_col}_Zscore'] < -2)]
      outlier_idx_list.append(list(outlier_dict[one_col].index))
      if len(outlier_dict[one_col]):
          print(one_col, 'Has outliers\n', outlier_dict[one_col])
      else:
          print(one_col, "Has Not outlier")

  # |Z-score| > 2 값 제거        
  all_outlier_idx = sum(outlier_idx_list, [])
  df2 = df2.drop(all_outlier_idx)
  ```
  <img src="https://user-images.githubusercontent.com/72846894/102121462-5a371b00-3e87-11eb-9c1a-66bbb380f4cf.png"></img>
  <img src="https://user-images.githubusercontent.com/72846894/102121485-628f5600-3e87-11eb-9b2c-4ab6ca238e63.png"></img>
##### 5) 예측변수 정규화(Minmax Scaler)
  ```python
  from sklearn.preprocessing import MinMaxScaler

  scaled_cols = ["기온", "강수량", "풍속", "습도", "적설량", "미세먼지", "초미세먼지"]

  scaler = MinMaxScaler()
  scaler.fit(X_features[scaled_cols])
  X_scaled = scaler.transform(X_features[scaled_cols])
  X_features[scaled_cols] = X_scaled
  ```
  <img src="https://user-images.githubusercontent.com/72846894/102121496-67eca080-3e87-11eb-9be2-01d1e2cd8188.png"></img>
  <img src="https://user-images.githubusercontent.com/72846894/102121526-720e9f00-3e87-11eb-8444-0e94aa0fbdc6.png"></img>
  ##### -> 다양한 스케일링을 시도해 보았으나 성능에 두드러진 변화를 주지 않음
  

### 4. 모델별 교차 검증
#### 모델별 k-fold 교차검증
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
  <img src="https://user-images.githubusercontent.com/72846894/102121624-99656c00-3e87-11eb-8558-7fe78866a33e.png"></img>
  
  - scoring = "r2"
  
  <img src=https://user-images.githubusercontent.com/72846894/102121645-9f5b4d00-3e87-11eb-8936-706c1530a9c6.png></img>
##### -> 모든 모델의 교차검증에서 표준편차가 0.01대로 나타남-> 전체적인 데이터에서 일반화된 예측성능을 보인다고 할 수있음. 

### 5. 프로젝트를 마치며 
- 모든 모델에서 R2 score 0.8 이상의 좋은 성능을 나타냈고, RMSE 역시 15 내외로 준수하였음.
- 목표변수만 로그화하고, 예측변수 스케일링 하지 않았을때 가장 좋은 예측 성능을 보인다.
- 복잡한 모델(Randomforest)의 성능은 우수하나, 속도가 느리고 가역성이 떨어지는 단점이 있다.
- 우리의 데이터는 linear regression 분석시, 모델의 단순성에 비해 뛰어난 예측 성능을 가지고 있다고 할수 있다.(lin r2/rmse: 0.912/15.01, rf r2/rmse: 0.941/12.31)
- 선형회귀분석 이외의 다양한 모델을 특징을 학습하였음.
- 교차검증 및 하이퍼파라미터에 대한 이해도와 활용 기술을 익힘.
- 데이터의 스케일링, 모델의 복잡도, 예측성능 등을 다각도로 고려할 것.

## Built with
- 이정려
  - raw_data 통합, 결측치 해결, 원데이터 분석, Gridsearch CV 
  - Readme 작성
  - GitHub: https://github.com/jungryo
  
- 전예나
  - 변수별 특성 파악(EDA), 목표변수 스케일링을 통한 분석
  - 발표자료 작성
  - GitHub: https://github.com/Yenabeam

- 최재철
  - raw data 전처리, 예측변수 스케일링을 통한 분석 
  - 발표
  - GitHub: https://github.com/kkobooc
  
