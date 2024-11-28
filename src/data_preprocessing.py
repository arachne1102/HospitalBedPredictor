import logging
import os
import pandas as pd
import numpy as np
from config import config

def load_data(filepath: str) -> pd.DataFrame:
    logging.info(f"데이터 로드 시작: {filepath}")
    try:
        data = pd.read_csv(filepath)
        logging.info(f"데이터 로드 완료: {filepath}")
    except Exception as e:
        logging.error(f"데이터 로드 실패: {e}")
        raise
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["hvec", "hvicc", "hvoc", "hpbdn"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    logging.info("수치형 변수 결측치 처리 완료")

    data['last_updated'] = pd.to_datetime(data.get('last_updated'), errors='coerce')
    data['last_updated'] = data['last_updated'].fillna(pd.to_datetime("1970-01-01"))
    logging.info("'last_updated' 컬럼 결측치 처리 및 변환 완료")

    return data

def remove_unnecessary_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['hvctayn', 'hvmriayn', 'hvventisoayn', 'duty_addr', 'wgs84_lon', 'wgs84_lat']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=existing_columns)
    logging.info(f"불필요한 컬럼 제거 완료: {existing_columns}")
    return data

def replace_infinite_values(data: pd.DataFrame) -> pd.DataFrame:
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    logging.info("무한대 값 처리 완료")
    return data

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["hvec", "hvicc", "hvoc", "hpbdn"]
    for col in numeric_columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before_count = data.shape[0]
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            removed = before_count - data.shape[0]
            logging.info(f"{col}: {removed}개의 이상치 제거")
        else:
            logging.warning(f"{col} 컬럼이 존재하지 않습니다.")
    logging.info("이상치 제거 완료")
    return data

def convert_data_types(data: pd.DataFrame) -> pd.DataFrame:
    string_columns = ["hpid", "duty_name"]
    existing_string_columns = [col for col in string_columns if col in data.columns]
    data[existing_string_columns] = data[existing_string_columns].astype(str)
    logging.info("데이터 타입 변환 완료")
    return data

def preprocess_data(filepath: str) -> pd.DataFrame:
    logging.info("데이터 전처리 시작")
    data = load_data(filepath)
    data = replace_infinite_values(data)
    data = handle_missing_values(data)
    data = remove_unnecessary_columns(data)
    data = remove_outliers(data)
    data = convert_data_types(data)
    logging.info("데이터 전처리 완료")
    return data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_filepath = config.INPUT_FILEPATH
    output_filepath = config.OUTPUT_FILEPATH_PREPROCESS

    if not os.path.exists(input_filepath):
        logging.error(f"입력 파일이 존재하지 않습니다: {input_filepath}")
        raise FileNotFoundError(f"{input_filepath} 파일을 찾을 수 없습니다.")

    try:
        processed_data = preprocess_data(input_filepath)
        processed_data.to_csv(output_filepath, index=False)
        logging.info(f"전처리된 데이터가 저장되었습니다: {output_filepath}")
    except Exception as e:
        logging.error(f"데이터 전처리 실패: {e}")
        print(f"데이터 전처리 실패: {e}")