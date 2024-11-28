import mysql.connector
import csv
import time
import sys
import os
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder

load_dotenv()

# 프로젝트 루트 경로를 추가하여 config 모듈을 찾도록 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import config

# SSH 터널 설정
with SSHTunnelForwarder(
    (config.SSH_HOST, config.SSH_PORT),
    ssh_username=config.SSH_USER,
    ssh_pkey=config.SSH_PRIVATE_KEY,
    remote_bind_address=(config.REMOTE_DB_HOST, config.REMOTE_DB_PORT),  # 수정된 부분
    local_bind_address=(
        "127.0.0.1",
        config.LOCAL_BIND_PORT,
    ),  # 로컬 포트를 원격 DB 포트와 동일하게 설정
) as tunnel:
    print("SSH 터널 연결이 설정되었습니다.")

    # 1. MySQL 데이터베이스 연결 설정
    connection = mysql.connector.connect(
        host="127.0.0.1",
        user=config.DB_USER,
        password=config.DB_PASS,
        database=config.DB_NAME,
        port=config.LOCAL_BIND_PORT,
    )

    cursor = connection.cursor()

    # 2. 페이징 처리를 위한 쿼리 템플릿
    query_template = """
    SELECT h.hpid, h.last_updated, h.hvec, h.hvicc, h.hvoc, h.hvctayn, h.hvmriayn, h.hvventisoayn,
           hi.duty_name, hi.duty_addr, hi.wgs84_lon, hi.wgs84_lat, hi.hpbdn
    FROM emergency_hospital_data_raw h
    INNER JOIN hospital_information hi ON h.hpid = hi.hpid
    LIMIT 1000 OFFSET %s
  """

    # 3. CSV 파일 생성 및 헤더 쓰기
    csv_file = "data/hospital_data.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 헤더 작성
        writer.writerow(
            [
                "hpid",
                "last_updated",
                "hvec",
                "hvicc",
                "hvoc",
                "hvctayn",
                "hvmriayn",
                "hvventisoayn",
                "duty_name",
                "duty_addr",
                "wgs84_lon",
                "wgs84_lat",
                "hpbdn",
            ]
        )

    # 4. 페이징 처리 및 데이터 수집
    offset = 0
    batch_size = 1000

    while True:
        # 쿼리 실행
        cursor.execute(query_template, (offset,))
        rows = cursor.fetchall()

        # 데이터가 더 이상 없으면 종료
        if not rows:
            break

        # 결과를 CSV 파일에 추가
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        print(f"{offset}부터 {offset + batch_size}까지 데이터를 저장했습니다.")

        # 다음 페이지로 이동
        offset += batch_size

        # 서버 부하를 줄이기 위해 잠시 대기
        time.sleep(0.5)  # 0.5초 대기 (필요에 따라 조절 가능)

    # 5. 연결 종료
    cursor.close()
    connection.close()

    print("CSV 파일 저장이 완료되었습니다.")
