import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from src.lstm_model import build_lstm_model  # LSTM 모델 구조를 정의한 함수 가져오기
from config import config
import tensorflow as tf

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 전처리 함수
    """
    df = df.drop(columns=['duty_addr', 'wgs84_lon', 'wgs84_lat', 'hvctayn', 'hvmriayn', 'hvventisoayn'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna({
        'er_bed_utilization': 0,
        'icu_bed_utilization': 0,
        'hpbdn': 0,
        'hvicc': 0,
        'hvec': 0
    }, inplace=True)
    return df

def train_sarima(df: pd.DataFrame):
    """
    SARIMA 모델 학습 및 예측
    """
    logging.info("SARIMA 모델 예측 시작")
    results = []
    grouped = df.groupby(['hpid', 'duty_name'])

    for (hpid, duty_name), group in grouped:
        group.set_index('last_updated', inplace=True)
        
        # 수치형 열만 선택하여 리샘플링
        numeric_group = group.select_dtypes(include=[np.number])
        numeric_group = numeric_group.resample('6H').mean().ffill()

        # SARIMA 모델을 사용해 예측
        model = SARIMAX(numeric_group['hvec'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 28))
        sarima_fit = model.fit(disp=False)
        forecast = sarima_fit.get_forecast(steps=28).predicted_mean
        forecast.index = pd.date_range(start=numeric_group.index[-1], periods=28, freq='6H')

        for time, value in forecast.items():
            results.append({
                'hpid': hpid,
                'duty_name': duty_name,
                'date': time.date(),
                'time_of_day': classify_time_of_day(time.hour),
                'predicted_bed_availability': value
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(config.SARIMA_PREDICTIONS_PATH, index=False)
    logging.info("SARIMA 예측 완료 및 저장 완료")
    return result_df

def train_prophet(df: pd.DataFrame):
    """
    Prophet 모델 학습 및 예측
    """
    logging.info("Prophet 모델 예측 시작")
    results = []
    grouped = df.groupby(['hpid', 'duty_name'])

    for (hpid, duty_name), group in grouped:
        prophet_df = group.rename(columns={'last_updated': 'ds', 'hvec': 'y'})[['ds', 'y']].copy()
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=7 * 4, freq='6H')
        forecast = model.predict(future)
        
        forecast['time_of_day'] = forecast['ds'].dt.hour.apply(classify_time_of_day)
        forecast['date'] = forecast['ds'].dt.date
        for _, row in forecast.iterrows():
            results.append({
                'hpid': hpid,
                'duty_name': duty_name,
                'date': row['date'],
                'time_of_day': row['time_of_day'],
                'predicted_bed_availability': row['yhat']
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(config.PROPHET_PREDICTIONS_PATH, index=False)
    logging.info("Prophet 예측 완료 및 저장 완료")
    return result_df

def train_lstm(df: pd.DataFrame):
    """
    LSTM 모델 학습 및 예측
    """
    logging.info("LSTM 모델 예측 시작")
    results = []
    grouped = df.groupby(['hpid', 'duty_name'])

    for (hpid, duty_name), group in grouped:
        scaler = MinMaxScaler()
        group.set_index('last_updated', inplace=True)
        numeric_group = group.select_dtypes(include=[np.number])
        group = numeric_group.resample('6H').mean().ffill()
        
        data = group[['hvec']].values
        scaled_data = scaler.fit_transform(data)

        # LSTM 모델 생성 및 시퀀스 생성
        X, y = create_sequences(scaled_data)
        input_shape = (X.shape[1], X.shape[2])
        model = build_lstm_model(input_shape)
        
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        predictions = model.predict(X[-28:])  # 7일 동안 예측
        predictions = scaler.inverse_transform(predictions).flatten()

        times = pd.date_range(start=group.index[-1], periods=28, freq='6H')
        for i, time in enumerate(times):
            results.append({
                'hpid': hpid,
                'duty_name': duty_name,
                'date': time.date(),
                'time_of_day': classify_time_of_day(time.hour),
                'predicted_bed_availability': predictions[i]
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(config.LSTM_PREDICTIONS_PATH, index=False)
    logging.info("LSTM 예측 완료 및 저장 완료")
    return result_df

def create_sequences(data, seq_length=28):
    """
    시계열 데이터를 입력 시퀀스와 타겟 시퀀스로 변환하는 함수
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # 타겟 변수는 첫 번째 열로 설정
    return np.array(X), np.array(y)

def classify_time_of_day(hour: int) -> str:
    """
    시간대를 아침, 낮, 저녁, 심야로 분류하는 함수
    """
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 데이터 로드 및 전처리
    input_filepath = config.OUTPUT_FILEPATH_FEATURE_ENGINEERING
    df = pd.read_csv(input_filepath, parse_dates=['last_updated'])
    df = preprocess_data(df)

    # SARIMA 예측 수행
    train_sarima(df)

    # Prophet 예측 수행
    train_prophet(df)

    # LSTM 예측 수행
    train_lstm(df)
