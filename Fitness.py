import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Fitness & Nutrition Recommender", layout="centered", page_icon="🏋️‍♂️")

# --- CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
        }
        .result-box {
          
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Fungsi Rekomendasi ---
def generate_nutrition_recommendation(activity, bmi, height_cm=170):
    height_m = height_cm / 100
    weight = bmi * (height_m ** 2)

    if activity == "Running":
        diet = "Konsumsi protein tinggi dan kalori cukup untuk menunjang aktivitas"
        calories = round(15 * weight + 300)
    elif activity == "Walking":
        diet = "Pola makan seimbang dengan protein dan karbohidrat kompleks"
        calories = round(13 * weight + 150)
    elif activity == "Resting":
        diet = "Fokus pada makanan ringan, hindari makanan berat"
        calories = round(10 * weight)
    else:
        diet = "Tidak diketahui"
        calories = round(12 * weight)
    return diet, calories

# --- Load Model ---
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
interpreter = tf.lite.Interpreter(model_path="Fitness_tracker.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Judul ---
st.markdown('<div class="main-title">🏃‍♀️ Fitness Activity & Nutrition Recommender</div>', unsafe_allow_html=True)
st.write("Masukkan data harian Anda:")

# --- Input Form ---
steps = st.number_input("👣 Jumlah Langkah (Steps)", min_value=0, value=8000)
heart_rate = st.number_input("❤️ Detak Jantung (bpm)", min_value=40, max_value=200, value=85)
calories = st.number_input("🔥 Kalori Terbakar", min_value=0, value=400)
bmi = st.number_input("📏 BMI", min_value=10.0, max_value=40.0, value=22.5)
active_minutes = st.number_input("⏱️ Menit Aktif", min_value=0, max_value=300, value=60)
height_cm = st.number_input("📐 Tinggi Badan (cm)", min_value=100, max_value=250, value=170)

# --- Tombol Prediksi ---
if st.button("🔍 Prediksi Aktivitas & Rekomendasi"):
    # Validasi input
    if steps == 0 and heart_rate == 0 and calories == 0 and active_minutes == 0:
        st.error("Mohon masukkan data yang valid untuk melakukan prediksi.")
    else:
        input_data = np.array([[steps, heart_rate, calories, bmi, active_minutes]])
        scaled_input = scaler.transform(input_data).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], scaled_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction)
        activity = label_encoder.inverse_transform([predicted_class])[0]

        # Logging untuk debugging
        print("Output prediksi mentah:", prediction)
        print("Kelas yang diprediksi:", predicted_class)
        print("Aktivitas yang diprediksi:", activity)

        # Ambil rekomendasi berdasarkan hasil prediksi
        diet, recommended_cal = generate_nutrition_recommendation(activity, bmi, height_cm)

        # --- Output Section ---
        st.subheader("📊 Hasil Prediksi")
        st.success(f"**Aktivitas Anda:** {activity}")
        st.info(f"**Pola Makan yang Disarankan:** {diet}")
        st.warning(f"**Kebutuhan Kalori Harian:** {recommended_cal} kcal")
