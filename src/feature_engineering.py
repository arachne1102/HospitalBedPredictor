import logging
import os
import pandas as pd
import numpy as np
import holidays
from datetime import datetime
from config import config

# 한국의 공휴일 정보를 제공하는 holidays 라이브러리 설정
kr_holidays = holidays.KR()

def is_holiday(date: datetime) -> bool:
    """
    주어진 날짜가 공휴일인지 확인하는 함수

    Parameters:
    date (datetime): 검사할 날짜

    Returns:
    bool: 공휴일이면 True, 아니면 False
    """
    return date in kr_holidays

def classify_time_of_day(hour: int) -> str:
    """
    시간대를 아침, 낮, 저녁, 심야로 분류하는 함수

    Parameters:
    hour (int): 시(hour) 정보

    Returns:
    str: 해당 시간대 (아침, 낮, 저녁, 심야)
    """
    if 6 <= hour < 12:
        return 'morning'  # 아침 (06:00 ~ 12:00)
    elif 12 <= hour < 18:
        return 'afternoon'  # 낮 (12:00 ~ 18:00)
    elif 18 <= hour < 24:
        return 'evening'  # 저녁 (18:00 ~ 24:00)
    else:
        return 'night'  # 심야 (00:00 ~ 06:00)

def generate_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    시간 관련 특성을 생성하는 함수
    
    - 요일, 시간대, 주말 여부, 공휴일 여부, 시간대 분류 특성을 추가합니다.

    Parameters:
    data (pd.DataFrame): 원본 데이터프레임

    Returns:
    pd.DataFrame: 시간 관련 특성이 추가된 데이터프레임
    """
    # 요일(day_of_week) 추가 (0: 월요일 ~ 6: 일요일)
    data['day_of_week'] = data['last_updated'].dt.dayofweek
    logging.info("'day_of_week' 특성 생성 완료")

    # 시간(hour) 정보 추가
    data['hour'] = data['last_updated'].dt.hour
    logging.info("'hour' 특성 생성 완료")

    # 시간대를 morning, afternoon, evening, night으로 분류
    data['time_of_day'] = data['hour'].apply(classify_time_of_day)
    logging.info("'time_of_day' 특성 생성 완료")

    # 주말 여부 (1: 주말, 0: 평일)
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    logging.info("'is_weekend' 특성 생성 완료")

    # 공휴일 여부 (1: 공휴일, 0: 비공휴일)
    data['is_holiday'] = data['last_updated'].apply(lambda x: 1 if is_holiday(x) else 0)
    logging.info("'is_holiday' 특성 생성 완료")

    return data

def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 변수를 인코딩하는 함수
    
    - 요일과 시간대를 One-Hot Encoding 방식으로 인코딩합니다.

    Parameters:
    data (pd.DataFrame): 범주형 변수가 포함된 데이터프레임

    Returns:
    pd.DataFrame: 인코딩된 데이터프레임
    """
    # 요일(day_of_week)을 One-Hot Encoding으로 변환
    data = pd.get_dummies(data, columns=['day_of_week'], prefix='dow')
    logging.info("'day_of_week' 컬럼 One-Hot Encoding 완료")

    # 시간대(time_of_day)를 One-Hot Encoding으로 변환
    data = pd.get_dummies(data, columns=['time_of_day'], prefix='tod')
    logging.info("'time_of_day' 컬럼 One-Hot Encoding 완료")

    return data

def create_utilization_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    병상 가용률 특성을 생성하는 함수
    
    - 응급실과 중환자실 병상의 가용률을 계산합니다.

    Parameters:
    data (pd.DataFrame): 병상 관련 데이터가 포함된 데이터프레임

    Returns:
    pd.DataFrame: 병상 가용률 특성이 추가된 데이터프레임
    """
    # 응급실 병상 가용률(er_bed_utilization) 계산
    data['er_bed_utilization'] = data['hvec'] / data['hpbdn']
    logging.info("'er_bed_utilization' 특성 생성 완료")

    # 중환자실 병상 가용률(icu_bed_utilization) 계산
    data['icu_bed_utilization'] = data['hvicc'] / data['hpbdn']
    logging.info("'icu_bed_utilization' 특성 생성 완료")

    return data

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    전체 Feature Engineering 파이프라인 함수
    
    - 시간 관련 특성 생성
    - 범주형 변수 인코딩
    - 병상 가용률 계산

    Parameters:
    data (pd.DataFrame): 원본 데이터프레임

    Returns:
    pd.DataFrame: 특성 엔지니어링이 완료된 데이터프레임
    """
    logging.info("시간 관련 특성 생성 시작")
    data = generate_time_features(data)

    logging.info("범주형 변수 인코딩 시작")
    data = encode_categorical_variables(data)

    logging.info("병상 가용률 계산 시작")
    data = create_utilization_features(data)

    logging.info("Feature Engineering 완료")
    return data

if __name__ == "__main__":
    # 로깅 설정 (파일로 저장하려면 filename 매개변수 추가)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # config.py에서 파일 경로 가져오기
    input_filepath = config.OUTPUT_FILEPATH_PREPROCESS
    output_filepath = config.OUTPUT_FILEPATH_FEATURE_ENGINEERING
    
    logging.info(f"입력 파일 경로: {os.path.abspath(input_filepath)}")
    logging.info(f"출력 파일 경로: {os.path.abspath(output_filepath)}")

    # CSV 파일 존재 여부 확인
    if not os.path.exists(input_filepath):
        logging.error(f"입력 파일이 존재하지 않습니다: {input_filepath}")
        raise FileNotFoundError(f"{input_filepath} 파일을 찾을 수 없습니다.")

    # 데이터 로드 및 Feature Engineering 실행
    try:
        logging.info("데이터 로드 시작")
        data = pd.read_csv(input_filepath, parse_dates=['last_updated'])
        logging.info("데이터 로드 완료")

        logging.info("Feature Engineering 시작")
        processed_data = feature_engineering(data)

        # 처리된 데이터 저장
        processed_data.to_csv(output_filepath, index=False)
        logging.info(f"Feature Engineering 완료: {output_filepath}")
        print(f"Feature Engineering 결과가 {output_filepath}에 저장되었습니다.")

    except Exception as e:
        logging.error(f"Feature Engineering 실패: {e}")
        print(f"Feature Engineering 실패: {e}")



