import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =========================================
# Load Model dengan caching
# =========================================
@st.cache_resource
def load_model():
    MODEL_PATH = Path(__file__).parent / "model_orange.pickle"
    if not MODEL_PATH.exists():
        st.error("❌ File model_orange.pickle tidak ditemukan di repository. Pastikan sudah di-upload ke GitHub.")
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"🚨 Gagal memuat model: {e}")
        return None


# =========================================
# Konfigurasi fitur (harus disesuaikan dengan model Orange)
# =========================================
FEATURE_CONFIG = {
    "umur": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 100,
        "default": 30
    },
    "pendapatan": {
        "type": "numeric",
        "input": "number",
        "min": 0,
        "max": 100000000,
        "default": 5000000
    },
    "lama_bekerja": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 40,
        "default": 5
    },
    "jenis_kelamin": {
        "type": "categorical",
        "options": ["Laki-laki", "Perempuan"]
    },
    "status_pernikahan": {
        "type": "categorical",
        "options": ["Belum Menikah", "Menikah", "Cerai"]
    }
}


# =========================================
# Membuat form input user
# =========================================
def create_input_form():
    input_data = {}
    with st.form("prediction_form"):
        st.subheader("Masukkan Data untuk Prediksi:")
        for feature, config in FEATURE_CONFIG.items():
            if config["type"] == "numeric":
                if config.get("input") == "slider":
                    value = st.slider(
                        label=feature.capitalize(),
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"]
                    )
                else:
                    value = st.number_input(
                        label=feature.capitalize(),
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"]
                    )
            elif config["type"] == "categorical":
                value = st.selectbox(
                    label=feature.capitalize(),
                    options=config["options"]
                )
            else:
                value = None
            input_data[feature] = value

        submitted = st.form_submit_button("Prediksi")
    return input_data, submitted


# =========================================
# Fungsi prediksi utama
# =========================================
def predict_with_model(model, input_df):
    try:
        prediction = model.predict(input_df)
        # Jika model memiliki pred_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            return prediction, proba
        else:
            return prediction, None
    except Exception as e:
        st.warning(f"Prediksi menggunakan model sklearn-like gagal: {e}")
        return predict_with_orange_fallback(model, input_df)


# =========================================
# Fallback prediksi dengan Orange Model
# =========================================
def predict_with_orange_fallback(model, input_df):
    try:
        import Orange

        # Buat domain dari FEATURE_CONFIG
        variables = []
        for feature, config in FEATURE_CONFIG.items():
            if config["type"] == "numeric":
                variables.append(Orange.data.ContinuousVariable(feature))
            else:
                variables.append(Orange.data.DiscreteVariable(feature, values=config["options"]))
        domain = Orange.data.Domain(variables)

        # Konversi data input ke format numpy sesuai domain
        data_row = []
        for feature, config in FEATURE_CONFIG.items():
            val = input_df.iloc[0][feature]
            if config["type"] == "categorical":
                data_row.append(config["options"].index(val))
            else:
                data_row.append(val)

        table = Orange.data.Table(domain, np.array([data_row]))

        # Lakukan prediksi
        prediction = model(table)
        pred_value = prediction[0].value if hasattr(prediction[0], "value") else prediction
        return [pred_value], None

    except ImportError:
        st.error("Library Orange3 belum terinstall. Tambahkan 'orange3' dalam requirements.txt.")
        return None, None
    except Exception as e:
        st.error(f"Prediksi dengan Orange gagal: {e}")
        return None, None


# =========================================
# User Interface (main function)
# =========================================
def main():
    st.title("Aplikasi Prediksi Berbasis Model Orange")
    st.write("Aplikasi ini menggunakan model machine learning hasil training dari **Orange Data Mining** dan dijalankan melalui **Streamlit Cloud**.")

    st.sidebar.title("📘 Petunjuk Penggunaan")
    st.sidebar.markdown("""
    1. Masukkan data pada form input.
    2. Klik tombol **Prediksi**.
    3. Hasil prediksi akan muncul di bawah form.
    
    🔹 Model dimuat dari file `model_orange.pickle` di repository GitHub.
    """)

    model = load_model()
    if model is None:
        return

    input_data, submitted = create_input_form()

    if submitted:
        try:
            input_df = pd.DataFrame([input_data], columns=FEATURE_CONFIG.keys())
            st.subheader("Data Input Pengguna:")
            st.table(input_df)

            prediction, probability = predict_with_model(model, input_df)

            if prediction is not None:
                st.success(f"✅ Hasil Prediksi: **{prediction[0]}**")
                if probability is not None:
                    st.write("Probabilitas (jika tersedia):")
                    st.write(probability)
            else:
                st.error("Prediksi gagal dilakukan.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")


if __name__ == "__main__":
    main()
