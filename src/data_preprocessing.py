import logging
import os
import pandas as pd

# config.py에서 config 객체를 import 합니다.
from config import config


def load_data(filepath: str) -> pd.DataFrame:
    """
    데이터 로드 함수
    지정된 파일 경로에서 CSV 파일을 읽어들여 Pandas DataFrame으로 반환

    Parameters:
    filepath (str): 데이터 파일의 경로

    Returns:
    pd.DataFrame: 로드된 데이터프레임
    """
    logging.info(f"데이터 로드 시작: {filepath}")
    try:
        data = pd.read_csv(filepath)
        logging.info(f"데이터 로드 완료: {filepath}")
    except Exception as e:
        logging.error(f"데이터 로드 실패: {e}")
        raise
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    결측치 처리 함수
    데이터프레임 내 결측치를 적절하게 처리

    - 수치형 병상 관련 변수의 결측치는 0으로 대체
    - 플래그 병상 관련 변수의 결측치는 'N'으로 대체
    - 문자열 변수의 결측치는 'Unknown'으로 대체
    - 날짜 형식의 결측치는 '1970-01-01'로 대체

    Parameters:
    data (pd.DataFrame): 결측치가 포함된 원본 데이터프레임

    Returns:
    pd.DataFrame: 결측치가 처리된 데이터프레임
    """
    # 수치형 병상 관련 변수 목록을 정의
    count_bed_columns = [
        "hvec",
        "hvicc",
        "hvoc",
        "hpbdn",
    ]

    # 플래그 병상 관련 변수 목록을 정의
    flag_bed_columns = [
        "hvctayn",
        "hvmriayn",
        "hvventisoayn",
    ]

    # 수치형 병상 관련 변수의 결측치를 0으로 대체
    data[count_bed_columns] = data[count_bed_columns].fillna(0)
    logging.info("수치형 병상 관련 변수의 결측치를 0으로 대체 완료")

    # 수치형 병상 관련 변수들을 숫자형으로 변환 (변환 불가능한 값은 0으로 설정)
    data[count_bed_columns] = (
        data[count_bed_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    logging.info(
        "수치형 병상 관련 변수의 데이터를 숫자형으로 변환 및 NaN을 0으로 대체 완료"
    )

    # 플래그 병상 관련 변수의 결측치를 'N'으로 대체
    data[flag_bed_columns] = data[flag_bed_columns].fillna("N")
    logging.info("플래그 병상 관련 변수의 결측치를 'N'으로 대체 완료")

    # 'duty_addr' 컬럼의 결측치를 'Unknown'으로 대체 (주소 정보 누락 처리)
    if "duty_addr" in data.columns:
        data["duty_addr"] = data["duty_addr"].fillna("Unknown")
        logging.info("'duty_addr' 컬럼의 결측치를 'Unknown'으로 대체 완료")
    else:
        logging.warning("'duty_addr' 컬럼이 데이터프레임에 존재하지 않습니다.")

    # 'last_updated' 컬럼을 datetime 타입으로 변환
    # 변환 과정에서 오류 발생 시 NaT(Not a Time)으로 처리
    if "last_updated" in data.columns:
        data["last_updated"] = pd.to_datetime(data["last_updated"], errors="coerce")
        logging.info("'last_updated' 컬럼을 datetime 타입으로 변환 완료")

        # 'last_updated' 컬럼의 결측치를 기준 날짜인 '1970-01-01'로 대체 (데이터의 일관성 유지)
        data["last_updated"] = data["last_updated"].fillna(pd.to_datetime("1970-01-01"))
        logging.info("'last_updated' 컬럼의 결측치를 '1970-01-01'로 대체 완료")
    else:
        logging.warning("'last_updated' 컬럼이 데이터프레임에 존재하지 않습니다.")

    return data


def handle_existence_flags(data: pd.DataFrame) -> pd.DataFrame:
    """
    존재 여부를 'Y'/'N'으로 변환하는 함수

    Parameters:
    data (pd.DataFrame): 전처리된 데이터프레임

    Returns:
    pd.DataFrame: 변환된 데이터프레임
    """
    existence_columns = ["hvctayn", "hvmriayn", "hvventisoayn"]

    for col in existence_columns:
        if col in data.columns:
            # 'Y'는 그대로 유지하고, 'N1'은 'N'으로 변환
            data[col] = data[col].replace({"N1": "N"})
            # 값이 'Y'이면 'Y', 그 외는 'N'으로 설정
            data[col] = data[col].apply(lambda x: "Y" if x == "Y" else "N")
            logging.info(f"{col} 컬럼을 'Y'/'N'으로 변환 완료")
        else:
            logging.warning(f"{col} 컬럼이 데이터프레임에 존재하지 않습니다.")

    return data


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    이상치 제거 함수
    데이터프레임 내 이상치를 탐지하여 제거

    - 수치형 병상 관련 변수에서 IQR(Interquartile Range)를 이용하여 이상치를 정의
    - 각 변수에 대해 1.5 * IQR을 벗어나는 값을 이상치로 간주하여 제거

    Parameters:
    data (pd.DataFrame): 이상치가 포함된 데이터프레임

    Returns:
    pd.DataFrame: 이상치가 제거된 데이터프레임
    """
    # 이상치 제거 대상 수치형 병상 관련 변수 목록을 정의 (플래그 제외)
    count_bed_columns = [
        "hvec",
        "hvicc",
        "hvoc",
        "hpbdn",
    ]

    for col in count_bed_columns:
        if col in data.columns:
            # 1사분위수(Q1)와 3사분위수(Q3)를 계산
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)

            # 사분위 범위(IQR)를 계산
            IQR = Q3 - Q1

            # 이상치의 하한과 상한을 정의
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 정의된 범위를 벗어나는 데이터를 필터링하여 제거
            before_count = data.shape[0]
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            after_count = data.shape[0]
            removed = before_count - after_count
            logging.info(f"{col}: {removed}개의 이상치 제거")
        else:
            logging.warning(f"{col} 컬럼이 데이터프레임에 존재하지 않습니다.")

    logging.info("이상치 제거 완료")
    return data


def convert_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 타입 변환 함수
    데이터프레임 내 특정 컬럼의 데이터 타입을 변환

    - 'hpid', 'duty_name', 'duty_addr' 컬럼을 문자열 타입으로 변환

    Parameters:
    data (pd.DataFrame): 데이터 타입 변환이 필요한 데이터프레임

    Returns:
    pd.DataFrame: 데이터 타입이 변환된 데이터프레임
    """
    if "hpid" in data.columns:
        data["hpid"] = data["hpid"].astype(str)
        logging.info("'hpid' 컬럼을 문자열 타입으로 변환 완료")
    else:
        logging.warning("'hpid' 컬럼이 데이터프레임에 존재하지 않습니다.")

    if "duty_name" in data.columns:
        data["duty_name"] = data["duty_name"].astype(str)
        logging.info("'duty_name' 컬럼을 문자열 타입으로 변환 완료")
    else:
        logging.warning("'duty_name' 컬럼이 데이터프레임에 존재하지 않습니다.")

    if "duty_addr" in data.columns:
        data["duty_addr"] = data["duty_addr"].astype(str)
        logging.info("'duty_addr' 컬럼을 문자열 타입으로 변환 완료")
    else:
        logging.warning("'duty_addr' 컬럼이 데이터프레임에 존재하지 않습니다.")

    return data


def preprocess_data(filepath: str) -> pd.DataFrame:
    """
    전체 전처리 파이프라인 함수
    데이터 로드부터 결측치 처리, 존재 여부 변환, 이상치 제거, 데이터 타입 변환까지의 전처리 과정을 수행

    Parameters:
    filepath (str): 원본 데이터 파일의 경로

    Returns:
    pd.DataFrame: 전처리가 완료된 데이터프레임
    """
    logging.info("데이터 로드 시작")
    data = load_data(filepath)
    logging.info("데이터 로드 완료")

    logging.info("결측치 처리 시작")
    data = handle_missing_values(data)
    logging.info("결측치 처리 완료")

    logging.info("존재 여부 변환 시작")
    data = handle_existence_flags(data)
    logging.info("존재 여부 변환 완료")

    logging.info("이상치 제거 시작")
    data = remove_outliers(data)
    logging.info("이상치 제거 완료")

    logging.info("데이터 타입 변환 시작")
    data = convert_data_types(data)
    logging.info("데이터 타입 변환 완료")

    logging.info("전체 전처리 과정 완료")
    return data


if __name__ == "__main__":
    # 로깅 설정 (파일로 저장하려면 filename 추가)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("데이터 전처리 스크립트 시작")
    print("현재 작업 디렉토리:", os.getcwd())

    # config.py에서 파일 경로 가져오기
    input_filepath = config.INPUT_FILEPATH
    output_filepath = config.OUTPUT_FILEPATH_PREPROCESS

    # 파일 경로와 존재 여부 출력
    print(f"입력 파일 경로: {os.path.abspath(input_filepath)}")
    print(f"출력 파일 경로: {os.path.abspath(output_filepath)}")
    print(f"입력 파일 존재 여부: {os.path.exists(input_filepath)}")

    logging.info(f"입력 파일 경로: {input_filepath}")
    logging.info(f"출력 파일 경로: {output_filepath}")

    # 전체 전처리 파이프라인을 실행하여 데이터를 전처리
    try:
        processed_data = preprocess_data(input_filepath)
        print("\n데이터 타입 확인:")
        print(processed_data.dtypes)

        # 전처리된 데이터를 지정된 경로에 CSV 파일로 저장 (index=False는 데이터프레임의 인덱스를 저장하지 않음)
        processed_data.to_csv(output_filepath, index=False)
        logging.info(f"데이터 전처리 완료: {output_filepath}")
        print(f"데이터 전처리 완료: {output_filepath}")
    except Exception as e:
        logging.error(f"데이터 전처리 실패: {e}")
        print(f"데이터 전처리 실패: {e}")
