import tensorflow as tf
import numpy as np

def build_lstm_model(input_shape):
    """
    LSTM 모델을 생성하는 함수
    
    Parameters:
    input_shape (tuple): LSTM 모델의 입력 형태 (sequence length, feature dimension)
    
    Returns:
    tf.keras.Model: 생성된 LSTM 모델
    """
    # Sequential 모델을 생성하고, LSTM 레이어 추가 후 출력 레이어 연결
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    
    # 모델 컴파일: 최적화 및 손실 함수 지정
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, seq_length=28):
    """
    시계열 데이터를 입력 시퀀스와 타겟 시퀀스로 변환하는 함수
    
    Parameters:
    data (np.array): LSTM 모델 학습을 위한 스케일링된 데이터 배열
    seq_length (int): 입력 시퀀스의 길이
    
    Returns:
    tuple: (X, y) 형태로 입력 시퀀스와 타겟 배열 반환
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        # 시퀀스 길이만큼 데이터 추출하여 X에 추가, 그 다음 값을 타겟으로 설정하여 y에 추가
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # 타겟 변수는 첫 번째 열로 설정
    return np.array(X), np.array(y)
