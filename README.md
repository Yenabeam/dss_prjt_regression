# 서울시 배달건수 회귀분석
- 배달건수 EDA를 통해 얻은 insight를 활용하고, 기상데이터를 추가하여 배달건수를 분석하는 회귀분석 프로젝트입니다.
- 본 프로젝트를 통해 효율적인 식자재관리와 배달관련 마케팅에 활용할 수 있습니다.
- 본 프로젝트에서는 2016년 6월 ~ 2019년 9월 서울시 강남구 치킨 통화건수 주문량을 분석하였습니다. 

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
1. 데이터가공 및 전처리
2. 예측변수 및 목표변수 스케일링
3. 모델 학습, 예측, 평가 
4. 모델 검증

### 1. 데이터가공 및 전처리
- 
