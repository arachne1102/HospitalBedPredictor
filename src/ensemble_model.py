import pandas as pd
import numpy as np
import logging
from config import config

def load_predictions(sarima_path, prophet_path, lstm_path):
    """
    각 모델의 예측 결과 파일을 로드하여 DataFrame으로 반환합니다.
    
    Parameters:
    - sarima_path (str): SARIMA 예측 결과 파일의 경로
    - prophet_path (str): Prophet 예측 결과 파일의 경로
    - lstm_path (str): LSTM 예측 결과 파일의 경로
    
    Returns:
    - tuple: SARIMA, Prophet, LSTM 예측 결과를 담은 세 개의 DataFrame
    """
    sarima_df = pd.read_csv(sarima_path)
    prophet_df = pd.read_csv(prophet_path)
    lstm_df = pd.read_csv(lstm_path)

    logging.info(f"SARIMA 예측 결과 로드 완료: {sarima_path}에서 {len(sarima_df)}개의 예측 데이터 로드.")
    logging.info(f"Prophet 예측 결과 로드 완료: {prophet_path}에서 {len(prophet_df)}개의 예측 데이터 로드.")
    logging.info(f"LSTM 예측 결과 로드 완료: {lstm_path}에서 {len(lstm_df)}개의 예측 데이터 로드.")

    return sarima_df, prophet_df, lstm_df

def merge_predictions(sarima_df, prophet_df, lstm_df):
    """
    세 모델의 예측 결과를 병합하고 공통 시간대의 예측값을 필터링하여 앙상블 예측을 생성합니다.
    
    Parameters:
    - sarima_df (DataFrame): SARIMA 모델 예측 결과
    - prophet_df (DataFrame): Prophet 모델 예측 결과
    - lstm_df (DataFrame): LSTM 모델 예측 결과
    
    Returns:
    - DataFrame: 앙상블 예측 결과를 포함한 DataFrame
    """
    # 각 모델의 예측 결과를 'hpid', 'duty_name', 'date', 'time_of_day'를 기준으로 병합
    merged_df = pd.merge(sarima_df, prophet_df, on=['hpid', 'duty_name', 'date', 'time_of_day'], suffixes=('_sarima', '_prophet'))
    merged_df = pd.merge(merged_df, lstm_df, on=['hpid', 'duty_name', 'date', 'time_of_day'])
    
    # 각 모델의 예측값을 추출하여 열 이름 변경
    merged_df = merged_df.rename(columns={
        'predicted_bed_availability_sarima': 'sarima_pred',
        'predicted_bed_availability_prophet': 'prophet_pred',
        'predicted_bed_availability': 'lstm_pred'
    })
    
    # 단순 평균 앙상블: 공통 시간대에 해당하는 예측값의 평균을 ensemble_pred 열에 계산
    merged_df['ensemble_pred'] = merged_df[['sarima_pred', 'prophet_pred', 'lstm_pred']].mean(axis=1)
    
    # 소수점 6자리에서 반올림
    merged_df['ensemble_pred'] = merged_df['ensemble_pred'].round(6)
    
    logging.info(f"공통 시간대 필터링 완료: {len(merged_df)}개의 예측 데이터가 병합되었습니다.")
    
    return merged_df

def save_ensemble_predictions(ensemble_df, output_path):
    """
    앙상블 예측 결과를 CSV 파일로 저장합니다.
    
    Parameters:
    - ensemble_df (DataFrame): 앙상블 예측 결과를 포함한 DataFrame
    - output_path (str): 앙상블 예측 결과를 저장할 경로
    """
    ensemble_df.to_csv(output_path, index=False)
    logging.info(f"앙상블 예측 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 각 모델의 예측 결과 파일 경로를 config에서 가져옴
    sarima_path = config.SARIMA_PREDICTIONS_PATH
    prophet_path = config.PROPHET_PREDICTIONS_PATH
    lstm_path = config.LSTM_PREDICTIONS_PATH
    output_path = config.ENSEMBLE_PREDICTIONS_PATH

    # 각 모델의 예측 결과 로드
    sarima_df, prophet_df, lstm_df = load_predictions(sarima_path, prophet_path, lstm_path)

    # 예측 결과 병합 및 공통 시간대 필터링 후 앙상블 수행
    ensemble_df = merge_predictions(sarima_df, prophet_df, lstm_df)

    # 앙상블 결과 저장
    save_ensemble_predictions(ensemble_df, output_path)
