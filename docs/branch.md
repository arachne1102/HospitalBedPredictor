```markdown
### data-collection-preprocessing(데이터 수집 전처리)
작업 내용
    src/data_preprocessing.py에서 데이터 전처리 로직 구현
    결측치 처리, 이상치 제거, 데이터 타입 변환 등

### feature-engineering(특성 엔지니어링)
작업 내용
    src/feature_engineering.py에서 시간 관련 특성 추가
    공휴일 여부, 거리 계산 등의 특성 생성
    범주형 변수 인코딩

### modeling(모델링)
작업 내용
    src/modeling.py에서 SARIMA 및 Prophet 모델 구현
    src/lstm_model.py에서 LSTM 모델 구축 및 학습

### model-evaluation-tuning(모델 평가 및 튜닝)
작업 내용
    src/evaluation.py에서 모델 평가 지표 계산(MAE, RMSE 등)
    Optuna를 활용한 하이퍼파라미터 최적화

### results-visualization(결과 시각화)
작업 내용
    src/evaluation.py 및 scripts/visualize.py에서 예측 결과 시각화
    notebooks/dashboard.ipynb에서 대시보드 구축


