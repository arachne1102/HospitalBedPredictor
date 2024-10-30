import os
from dotenv import load_dotenv

load_dotenv()


def get_env_variable(var_name, cast_type=str):
    value = os.environ.get(var_name)
    if value is None:
        raise EnvironmentError(f"필수 환경 변수 '{var_name}'가 설정되어 있지 않습니다.")
    try:
        return cast_type(value)
    except ValueError:
        raise ValueError(
            f"환경 변수 '{var_name}'의 값을 {cast_type.__name__} 타입으로 변환할 수 없습니다."
        )


class Config:
    # SSH 연결 정보
    SSH_HOST = get_env_variable("SSH_HOST")
    SSH_PORT = get_env_variable("SSH_PORT", int)
    SSH_USER = get_env_variable("SSH_USER")
    SSH_PRIVATE_KEY = get_env_variable("SSH_PRIVATE_KEY")

    # 데이터베이스 연결 정보
    DB_USER = get_env_variable("DB_USER")
    DB_PASS = get_env_variable("DB_PASS")
    DB_NAME = get_env_variable("DB_NAME")

    # 추가 설정
    REMOTE_DB_HOST = get_env_variable("REMOTE_DB_HOST")
    REMOTE_DB_PORT = get_env_variable("REMOTE_DB_PORT", int)
    LOCAL_BIND_PORT = get_env_variable("LOCAL_BIND_PORT", int)
    
    # 파일 경로 설정
    INPUT_FILEPATH = get_env_variable("INPUT_FILEPATH")
    OUTPUT_FILEPATH_PREPROCESS = get_env_variable("OUTPUT_FILEPATH_PREPROCESS")
    OUTPUT_FILEPATH_FEATURE_ENGINEERING = get_env_variable("OUTPUT_FILEPATH_FEATURE_ENGINEERING")


config = Config()
