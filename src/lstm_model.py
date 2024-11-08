import tensorflow as tf

def build_lstm_model(input_shape):
    """
    LSTM 모델을 구성하는 함수

    Parameters:
    input_shape (tuple): 입력 데이터의 형태 (seq_length, num_features)

    Returns:
    model (tf.keras.Model): LSTM 모델 객체
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
