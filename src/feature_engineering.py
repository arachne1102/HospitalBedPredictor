import logging
import os
import pandas as pd
import numpy as np
from config import config

def create_utilization_features(data: pd.DataFrame) -> pd.DataFrame:
    # 분모가 0인 경우 대비하여 병상 가용률 계산
    data['er_bed_utilization'] = np.where(data['hpbdn'] != 0, data['hvec'] / data['hpbdn'], 0)
    data['icu_bed_utilization'] = np.where(data['hpbdn'] != 0, data['hvicc'] / data['hpbdn'], 0)
    logging.info("병상 가용률 계산 완료")
    return data

def classify_time_of_day(hour: int) -> str:
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

def generate_time_of_day_feature(data: pd.DataFrame) -> pd.DataFrame:
    data['hour'] = data['last_updated'].dt.hour
    data['time_of_day'] = data['hour'].apply(classify_time_of_day)
    data = data.drop(columns=['hour'])
    logging.info("시간대 분류 특성 생성 완료")
    return data

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Feature Engineering 시작")
    data = create_utilization_features(data)
    data = generate_time_of_day_feature(data)
    logging.info("Feature Engineering 완료")
    return data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_filepath = config.OUTPUT_FILEPATH_PREPROCESS
    output_filepath = config.OUTPUT_FILEPATH_FEATURE_ENGINEERING

    if not os.path.exists(input_filepath):
        logging.error(f"입력 파일이 존재하지 않습니다: {input_filepath}")
        raise FileNotFoundError(f"{input_filepath} 파일을 찾을 수 없습니다.")

    try:
        data = pd.read_csv(input_filepath, parse_dates=['last_updated'])
        processed_data = feature_engineering(data)
        processed_data.to_csv(output_filepath, index=False)
        logging.info(f"Feature Engineering 완료: {output_filepath}")
        print(f"Feature Engineering 결과가 {output_filepath}에 저장되었습니다.")
    except Exception as e:
        logging.error(f"Feature Engineering 실패: {e}")
        print(f"Feature Engineering 실패: {e}")