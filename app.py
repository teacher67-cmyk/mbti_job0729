import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 🎈 귀여운 페이지 기본 설정
st.set_page_config(page_title="내 기분 알아맞히기 ✨", page_icon="🧸", layout="centered")

# 🎨 제목과 설명
st.title("🧸 나의 기분은 어떨까요? ✨")
st.write("지금 당신의 기분을 인공지능이 맞춰볼게요! 얼굴이 나온 사진을 올려주세요. 📸")

# 🛠️ 모델 로드 함수 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_model():
    # Teachable Machine에서 다운받은 모델 로드
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    # 라벨 파일 로드
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = f.readlines()
    return model, labels

model, labels = load_model()

# 🖼️ 이미지 업로드 버튼
uploaded_file = st.file_uploader("사진을 여기에 쏙 넣어주세요! (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 업로드된 이미지 화면에 예쁘게 보여주기
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="업로드된 사진 찰칵! 📷", use_column_width=True)
    
    st.write("인공지능이 열심히 기분을 분석하고 있어요... 🔍✨")
    
    # ⚙️ Teachable Machine 형식에 맞게 이미지 전처리
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    
    # 이미지를 정규화 (-1 ~ 1 사이의 값으로 변환)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # 모델 예측을 위해 배열 형태 변경 (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 🚀 모델 예측
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index].strip()
    confidence_score = prediction[0][index]
    
    # 클래스 이름에서 숫자 제거 (예: "0 웃음" -> "웃음")
    # 라벨 형식이 어떻게 되어있든 텍스트만 깔끔하게 빼옵니다.
    if " " in class_name:
        clean_label = class_name.split(" ", 1)[1]
    else:
        clean_label = class_name

    # 🎀 결과 화면 출력
    st.divider() # 귀여운 구분선
    
    if "웃음" in clean_label:
        st.success(f"### 분석 결과: 활짝 웃는 얼굴이네요! 😊")
        st.write(f"정확도: {confidence_score:.0%}")
        st.write("당신의 예쁜 미소를 보니 저도 기분이 좋아져요! 🌻")
        st.balloons() # 풍선 애니메이션 효과
        
    elif "슬픔" in clean_label:
        st.info(f"### 분석 결과: 조금 슬퍼 보여요... 😢")
        st.write(f"정확도: {confidence_score:.0%}")
        st.write("무슨 일인지는 몰라도, 다 괜찮아질 거예요. 토닥토닥 ☁️💙")
        st.snow() # 눈 내리는 애니메이션 효과 (감성적인 느낌)
        
    else:
        # 혹시 다른 라벨이 있을 경우
        st.write(f"### 분석 결과: {clean_label} 🧐")
        st.write(f"정확도: {confidence_score:.0%}")