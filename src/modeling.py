import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from src.lstm_model import build_lstm_model, create_sequences
from config import config
import tensorflow as tf

def train_sarima(df: pd.DataFrame):
    """
    SARIMA 모델을 사용한 병상 예측 함수.

    Parameters:
    df (pd.DataFrame): 병상 관련 데이터 프레임

    Returns:
    pd.DataFrame: SARIMA 모델 예측 결과 데이터 프레임
    """
    logging.info("SARIMA 모델 예측 시작")
    results = []

    # 병원 ID(hpid)와 duty_name 기준으로 데이터 그룹화
    grouped = df.groupby(['hpid', 'duty_name'])

    # 각 그룹에 대해 SARIMA 모델을 적용하여 예측
    for (hpid, duty_name), group in grouped:
        group = group.copy()
        group.set_index('last_updated', inplace=True)

        # 수치 데이터만 추출하고 6시간 간격으로 리샘플링하여 결측값을 앞의 값으로 채움
        numeric_group = group.select_dtypes(include=[np.number])
        numeric_group = numeric_group.resample('6h').mean().ffill()

        try:
            # SARIMA 모델 초기화 및 학습
            model = SARIMAX(numeric_group['hvec'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 28))
            sarima_fit = model.fit(disp=False)

            # 28 스텝(7일) 예측
            forecast = sarima_fit.get_forecast(steps=28).predicted_mean
            forecast.index = pd.date_range(start=numeric_group.index[-1] + pd.Timedelta(hours=6), periods=28, freq='6h')

            # 예측 결과를 리스트에 저장
            for time, value in forecast.items():
                time_of_day = classify_time_of_day(time.hour)
                results.append({
                    'hpid': hpid,
                    'duty_name': duty_name,
                    'date': time.date(),
                    'time_of_day': time_of_day,
                    'predicted_bed_availability': value
                })

        except Exception as e:
            logging.error(f"SARIMA 모델 예측 중 오류 발생 (hpid: {hpid}, duty_name: {duty_name}): {e}")

    # 예측 결과를 데이터프레임으로 변환하고 CSV로 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(config.SARIMA_PREDICTIONS_PATH, index=False)
    logging.info("SARIMA 예측 완료 및 저장 완료")
    return result_df

def train_prophet(df: pd.DataFrame):
    """
    Prophet 모델을 사용하여 미래의 병상 예측 결과만을 추출하여 저장하는 함수.

    Parameters:
    df (pd.DataFrame): 병상 관련 데이터 프레임

    Returns:
    pd.DataFrame: 미래 예측 데이터만 포함된 Prophet 모델 예측 결과 데이터 프레임
    """
    logging.info("Prophet 모델 예측 시작")
    results = []

    # 병원 ID(hpid)와 duty_name 기준으로 데이터 그룹화
    grouped = df.groupby(['hpid', 'duty_name'])

    # 각 그룹에 대해 Prophet 모델을 적용하여 예측
    for (hpid, duty_name), group in grouped:
        group = group.copy()
        prophet_df = group.rename(columns={'last_updated': 'ds', 'hvec': 'y'})[['ds', 'y']].copy()
        model = Prophet()

        try:
            # Prophet 모델 학습
            model.fit(prophet_df)

            # 미래 7일간의 데이터 프레임 생성 및 예측
            future = model.make_future_dataframe(periods=7 * 4, freq='6h')
            forecast = model.predict(future)

            # 현재 시간을 기준으로 미래 데이터만 필터링
            forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]

            # 필요한 시간대와 날짜를 추출하여 예측 결과를 리스트에 저장
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

        except Exception as e:
            logging.error(f"Prophet 모델 예측 중 오류 발생 (hpid: {hpid}, duty_name: {duty_name}): {e}")

    # 예측 결과를 데이터프레임으로 변환
    result_df = pd.DataFrame(results)

    # 중복된 예측 시간에 대한 평균 계산 후 중복 제거
    result_df = result_df.groupby(['hpid', 'duty_name', 'date', 'time_of_day'], as_index=False).mean()

    # CSV로 저장
    result_df.to_csv(config.PROPHET_PREDICTIONS_PATH, index=False)
    logging.info("Prophet 미래 예측 완료 및 저장 완료")
    return result_df

def train_lstm(df: pd.DataFrame):
    """
    LSTM 모델을 사용한 병상 예측 함수.

    Parameters:
    df (pd.DataFrame): 병상 관련 데이터 프레임

    Returns:
    pd.DataFrame: LSTM 모델 예측 결과 데이터 프레임
    """
    logging.info("LSTM 모델 예측 시작")
    results = []

    # 병원 ID(hpid)와 duty_name 기준으로 데이터 그룹화
    grouped = df.groupby(['hpid', 'duty_name'])

    # 각 그룹에 대해 LSTM 모델을 적용하여 예측
    for (hpid, duty_name), group in grouped:
        group = group.copy()
        scaler = MinMaxScaler()
        group.set_index('last_updated', inplace=True)

        # 수치 데이터만 추출하고 6시간 간격으로 리샘플링하여 결측값을 앞의 값으로 채움
        numeric_group = group.select_dtypes(include=[np.number])
        numeric_group = numeric_group.resample('6h').mean().ffill()

        # 데이터 스케일링 및 시퀀스 생성
        data = numeric_group[['hvec']].values
        scaled_data = scaler.fit_transform(data)
        X, y = create_sequences(scaled_data)

        # 모델의 입력 형태를 정의하고 LSTM 모델을 생성
        input_shape = (X.shape[1], X.shape[2])
        model = build_lstm_model(input_shape)

        try:
            # 모델 학습 및 예측 수행
            model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
            predictions = model.predict(X[-28:])
            predictions = scaler.inverse_transform(predictions).flatten()

            # 예측 결과를 리스트에 저장
            times = pd.date_range(start=numeric_group.index[-1] + pd.Timedelta(hours=6), periods=28, freq='6h')
            for i, time in enumerate(times):
                time_of_day = classify_time_of_day(time.hour)
                results.append({
                    'hpid': hpid,
                    'duty_name': duty_name,
                    'date': time.date(),
                    'time_of_day': time_of_day,
                    'predicted_bed_availability': predictions[i]
                })

        except Exception as e:
            logging.error(f"LSTM 모델 예측 중 오류 발생 (hpid: {hpid}, duty_name: {duty_name}): {e}")

    # 예측 결과를 데이터프레임으로 변환하고 CSV로 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(config.LSTM_PREDICTIONS_PATH, index=False)
    logging.info("LSTM 예측 완료 및 저장 완료")
    return result_df

def classify_time_of_day(hour: int) -> str:
    """
    시간대를 아침, 낮, 저녁, 심야로 분류하는 함수

    Parameters:
    hour (int): 시각 (0-23)

    Returns:
    str: 시간대 ('morning', 'afternoon', 'evening', 'night')
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
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # 데이터 로드
    input_filepath = config.OUTPUT_FILEPATH_FEATURE_ENGINEERING
    df = pd.read_csv(input_filepath, parse_dates=['last_updated'])

    # SARIMA 예측 수행
    train_sarima(df)

    # Prophet 예측 수행
    train_prophet(df)

    # LSTM 예측 수행
    train_lstm(df)
