# Hospital Bed Prediction Project

## 데이터 수집 및 전처리 완료

### 데이터 수집 스크립트 작성

- **SSH 터널링 설정**:  
  SSH 터널링을 사용하여 원격 데이터베이스에 안전하게 접속하고 데이터를 수집하는 스크립트를 개발했습니다.
- **데이터 수집 로깅**:  
  데이터 수집 과정에서 발생하는 이벤트와 오류를 로깅하여 문제 발생 시 빠르게 대응할 수 있도록 준비했습니다.

### data-collection-preprocessing 작업 완료

- **데이터 로드**:  
  원본 CSV 파일을 로드하는 데이터 로딩 스크립트를 작성했습니다.
- **결측치 처리**:  
  수치형 변수의 결측치를 `0`으로, 플래그 변수의 결측치를 `'N'`으로 대체했습니다.
- **플래그 변환**:  
  `hvctayn`, `hvmriayn`, `hvventisoayn` 컬럼을 `'Y'`와 `'N'`으로 변환했습니다.
- **이상치 제거**:  
  수치형 변수에서 IQR(Interquartile Range)을 기준으로 이상치를 제거했습니다.
- **데이터 타입 변환**:  
  `hpid`, `duty_name`, `duty_addr` 컬럼을 문자열 타입으로 변환했습니다.
- **전처리 완료 및 저장**:  
  전처리된 데이터를 `processed_hospital_data.csv` 경로에 저장했습니다.

---

## 특성 엔지니어링 (Feature Engineering) 완료

### feature-engineering 작업 완료

- **시간대 분류**:  
  아침(06:00 ~ 12:00), 낮(12:00 ~ 18:00), 저녁(18:00 ~ 24:00), 심야(00:00 ~ 06:00)로 시간대를 분류했습니다.
- **공휴일 및 주말 처리**:  
  공휴일 및 주말 여부를 구분하여 분석에 활용했습니다.
- **병상 가용률 계산**:  
  응급실(`er_bed_utilization`)과 중환자실(`icu_bed_utilization`)의 가용률을 계산했습니다.
- **One-Hot Encoding 적용**:  
  요일(`day_of_week`)과 시간대(`time_of_day`)를 One-Hot Encoding으로 변환했습니다.
- **Feature Engineering 완료 및 저장**:  
  처리된 데이터는 `feature_engineered_hospital_data.csv`에 저장되었습니다.

---

## 모델링 (Modeling) 완료

### modeling 작업 완료

- **SARIMA 모델**:  
  병상 가용성을 예측하기 위해 SARIMA 모델을 사용했습니다. 병원별, 시간대별로 그룹화하여 모델을 학습시키고, 다음 7일 동안의 병상 가용성을 예측합니다.
- **Prophet 모델**:  
  병상 데이터를 바탕으로 Prophet 모델을 활용하여 병상 예측을 수행했습니다. Prophet 모델은 데이터의 시간적 추세와 계절성을 반영하여 중장기 예측에 적합하도록 설정되었습니다.
- **LSTM 모델**:  
  시계열 예측에 적합한 LSTM 신경망 모델을 구축하여 병상 가용성을 예측했습니다. LSTM 모델은 병원의 과거 데이터를 시퀀스로 구성하여 예측을 수행합니다.
- **모델 예측 결과 저장**:  
  각 모델의 예측 결과는 CSV 파일로 저장됩니다. SARIMA, Prophet, LSTM 모델의 예측 결과는 각각 `sarima_predictions.csv`, `prophet_predictions.csv`, `lstm_predictions.csv` 파일에 저장됩니다.
- **앙상블 모델**:  
  SARIMA, Prophet, LSTM 모델의 예측값을 평균하여 최종 예측값으로 활용하는 앙상블 모델을 추가로 구현했습니다. 이 결과는 `ensemble_predictions.csv` 파일에 저장됩니다.

---

## 환경 설정 안내

- 프로젝트를 실행하기 위한 **환경 변수 설정과 설치 방법**을 문서화했습니다.
  - `.env` 파일을 사용하여 **SSH와 데이터베이스 연결 정보**를 관리합니다.
  - 필요한 라이브러리는 `requirements.txt`로 설치 가능합니다.

---

## 향후 계획

- **모델 평가 및 최적화**:  
  모델 성능을 **MAE, RMSE** 등의 평가 지표로 분석하고, **Optuna**를 사용해 하이퍼파라미터를 최적화할 계획입니다.
- **결과 시각화**:  
  예측 결과를 **그래프로 시각화**하여 이해하기 쉽게 표현하고, 주요 인사이트를 도출할 예정입니다.
- **배포 및 모니터링**:  
  완성된 모델을 **Flask** 또는 **FastAPI**로 배포하고, **실시간 성능 모니터링**을 통해 지속적으로 개선할 예정입니다.

---

## 현재까지의 브랜치 작업

- **data-collection-preprocessing**:  
  데이터 로드, 결측치 처리, 이상치 제거, 데이터 타입 변환 작업 완료.
- **feature-engineering**:  
  시간대 분류, 공휴일/주말 처리, 병상 가용률 계산, One-Hot Encoding 완료.
- **modeling**:  
  SARIMA, Prophet, LSTM 모델 예측 수행 및 앙상블 모델 추가.
