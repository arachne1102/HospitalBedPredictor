# 설치 방법

프로젝트를 실행하기 위한 설치 방법을 안내합니다.

## 전제 조건

- **Python 3.7 이상**이 설치되어 있어야 합니다.
- **가상 환경**(venv, conda 등)의 사용을 권장합니다.
- **MySQL 데이터베이스**에 접근할 수 있는 권한이 있어야 합니다.
- **SSH 키**를 생성하고 원격 서버에 접근할 수 있어야 합니다.

## 설치 절차

### 1. 소스 코드 다운로드

GitHub 저장소를 클론합니다.

```bash
git clone https://github.com/yourusername/HospitalBedPredictor.git
cd HospitalBedPredictor
```

### 2. 가상 환경 설치

Python 가상 환경을 생성하고 활성화합니다.

```
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (Windows)
venv\Scripts\activate

# 가상 환경 활성화 (macOS/Linux)
source venv/bin/activate
```

### 3. 의존성 설치

필요한 라이브러리를 requirements.txt 파일을 통해 설치합니다.

```
pip install -r requirements.txt
```

### 4. 환경 변수 설정

프로젝트 루트 디렉토리에 .env 파일을 생성하고 다음과 같이 설정합니다.

```
# SSH 연결 정보
SSH_HOST=your_ssh_host
SSH_PORT=your_ssh_port
SSH_USER=your_ssh_user
SSH_PRIVATE_KEY=path/to/your/private_key.pem

# 데이터베이스 연결 정보
DB_USER=your_db_user
DB_PASS=your_db_password
DB_NAME=your_db_name

# 추가 설정
REMOTE_DB_HOST=your_remote_db_host
REMOTE_DB_PORT=3306
LOCAL_BIND_PORT=3307
```

### 5. 데이터베이스 설정

MySQL 데이터베이스에 필요한 데이터베이스와 사용자 계정을 생성하고 권한을 부여합니다.

```
# MySQL에 접속 후 실행
CREATE DATABASE your_db_name;
CREATE USER 'your_db_user'@'%' IDENTIFIED BY 'your_db_password';
GRANT ALL PRIVILEGES ON your_db_name.* TO 'your_db_user'@'%';
FLUSH PRIVILEGES;
```
