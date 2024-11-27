import mysql.connector
import pandas as pd
import sys
import os
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로를 추가하여 config 모듈을 찾도록 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import config

def upload_predictions():
    # SSH 터널 설정
    with SSHTunnelForwarder(
        (config.SSH_HOST, config.SSH_PORT),
        ssh_username=config.SSH_USER,
        ssh_pkey=config.SSH_PRIVATE_KEY if hasattr(config, 'SSH_PRIVATE_KEY') else None,
        ssh_password=config.SSH_PASSWORD if hasattr(config, 'SSH_PASSWORD') else None,
        remote_bind_address=(config.REMOTE_DB_HOST, config.REMOTE_DB_PORT),
        local_bind_address=("127.0.0.1", config.LOCAL_BIND_PORT),
    ) as tunnel:
        print("SSH 터널 연결이 설정되었습니다.")

        # MySQL 데이터베이스 연결 설정
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user=config.DB_USER,
            password=config.DB_PASS,
            database=config.DB_NAME,
            port=config.LOCAL_BIND_PORT,
        )

        cursor = connection.cursor()

        # CSV 파일 로드
        csv_file = config.ENSEMBLE_PREDICTIONS_PATH
        try:
            df = pd.read_csv(csv_file)
            print(f"CSV 파일 로드 완료: {csv_file}")
        except FileNotFoundError:
            print(f"CSV 파일을 찾을 수 없습니다: {csv_file}")
            return

        # 테이블 생성 쿼리 (필요한 경우)
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ensemble_predictions (
            hpid VARCHAR(20),
            duty_name VARCHAR(100),
            date DATE,
            time_of_day VARCHAR(10),
            sarima_pred FLOAT(10,6),
            prophet_pred FLOAT(10,6),
            lstm_pred FLOAT(10,6),
            ensemble_pred FLOAT(10,6),
            PRIMARY KEY (hpid, date, time_of_day)
        );
        """

        cursor.execute(create_table_query)
        print("테이블이 준비되었습니다.")

        # 데이터 업로드
        insert_query = """
        REPLACE INTO ensemble_predictions (hpid, duty_name, date, time_of_day, sarima_pred, prophet_pred, lstm_pred, ensemble_pred)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """

        # DataFrame을 튜플의 리스트로 변환
        data = [
            (
                row['hpid'],
                row['duty_name'],
                row['date'],
                row['time_of_day'],
                row['sarima_pred'],
                row['prophet_pred'],
                row['lstm_pred'],
                row['ensemble_pred']
            )
            for index, row in df.iterrows()
        ]

        # 데이터베이스에 데이터 삽입
        try:
            cursor.executemany(insert_query, data)
            connection.commit()
            print("데이터베이스에 데이터 업로드 완료")
        except Exception as e:
            print(f"데이터 업로드 중 오류 발생: {e}")
            connection.rollback()
        finally:
            # 연결 종료
            cursor.close()
            connection.close()
            print("데이터베이스 연결이 종료되었습니다.")

if __name__ == "__main__":
    upload_predictions()
