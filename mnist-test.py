import streamlit as st
import tensorflow as tf  # 다른 ML 라이브러리를 사용한다면 변경
import numpy as np
from PIL import Image

# 미리 훈련된 모델 불러오기
model = tf.keras.models.load_model('my_model.h5')

st.title("손글씨 문자 분류기")

# Streamlit을 통해 이미지 업로드
uploaded_image = st.file_uploader("손글씨 문자를 업로드하세요", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="업로드한 문자", use_column_width=True)

    # 이미지 전처리 및 예측
    image = np.array(image.resize((28, 28)))  # 사이즈와 전처리를 모델에 맞게 조정
    image = image / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가

    predictions = model.predict(image)
    class_idx = np.argmax(predictions[0])

    st.write(f"예측: 클래스 {class_idx}")
