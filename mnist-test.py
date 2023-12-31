import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageOps

# 미리 훈련된 모델 불러오기
model = tf.keras.models.load_model('my_model.h5')

st.title("손글씨 문자 분류기")

# Streamlit을 통해 이미지 업로드
uploaded_image = st.file_uploader("손글씨 문자를 업로드하세요", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = ImageOps.grayscale(image)  # 채널을 1개로 변경
    st.image(image, caption="Uploaded Character", use_column_width=True)

    # 이미지 전처리 및 예측
    image = np.array(image.resize((28, 28)))  # 사이즈 조절
    image = image / 255.0  # 정규화
    image = image.reshape(1, 784)  # 차원을 (None, 784)로 변경

    predictions = model.predict(image)
    class_idx = np.argmax(predictions[0])
    st.write(f"Prediction: Class {class_idx}")
