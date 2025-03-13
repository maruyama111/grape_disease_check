import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import io

# ===============================================
# 1) 学習済みモデルをロード
# ===============================================
model_path = "model/grape_disease_classification_RNN-9_finetuned.h5"
model = tf.keras.models.load_model(model_path)

# ===============================================
# 2) 学習時のクラス順をロード
# ===============================================
class_labels_path = "model/class_indices_finetuned.pkl"
with open(class_labels_path, 'rb') as f:
    class_labels = pickle.load(f)

# ===============================================
# 3) Streamlit のレイアウト
# ===============================================
st.title("🍇 ブドウの葉の病害診断 AI 🍂")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込み
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="アップロードされた画像", use_column_width=True)
    
    # 画像を推論用に整形
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0  # 正規化
    img_array = np.expand_dims(img_array, axis=0)  # バッチ次元追加
    
    # 予測
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    st.write(f"**予測クラス**: {class_labels[predicted_class]}")
    st.write(f"**信頼度**: {confidence:.4f}")
