import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ---------------------
# Konfigurasi Streamlit
# ---------------------
st.set_page_config(page_title="Dashboard Kepuasan Pelanggan", layout="wide")

# ---------------------
# Load Data
# ---------------------
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Diva-auliya/Kepuasan-Pelanggan/main/data_intro.csv'
    return pd.read_csv(url)

df = load_data()

# ---------------------
# Sidebar Navigasi
# ---------------------
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Dataset & Visualisasi", "🤖 Pelatihan Model", "🔍 Form Prediksi"])

# ---------------------
# Halaman 1: Visualisasi
# ---------------------
if page == "📊 Dataset & Visualisasi":
    st.title("📊 Eksplorasi Data Kepuasan Pelanggan")

    with st.expander("🔎 Tampilkan Dataframe"):
        st.dataframe(df, use_container_width=True)

    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("📊 Korelasi Antar Fitur")
    corr = df.corr(numeric_only=True)
    if corr.empty or corr.isnull().values.all():
        st.warning("⚠️ Tidak bisa menampilkan korelasi karena tidak ada data numerik.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

# ---------------------
# Halaman 2: Pelatihan Model
# ---------------------
elif page == "🤖 Pelatihan Model":
    st.title("🤖 Pelatihan Model Prediksi Kepuasan")

    st.info("Pilih target variabel yang ingin diprediksi berdasarkan fitur lain.")

    target_column = st.selectbox("🎯 Pilih kolom target:", df.columns)
    features = [col for col in df.columns if col != target_column]

    X = df[features]
    y = df[target_column]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("📋 Evaluasi Model")
    st.code(classification_report(y_test, y_pred))

# ---------------------
# Halaman 3: Prediksi
# ---------------------
elif page == "🔍 Form Prediksi":
    st.title("🔍 Form Interaktif Prediksi Kepuasan")

    st.write("Masukkan data pelanggan untuk memprediksi tingkat kepuasan.")

    inputs = {}
    for col in df.columns[:-1]:  # Asumsi kolom terakhir adalah target
        dtype = df[col].dtype
        if dtype == 'object':
            val = st.selectbox(f"{col}:", df[col].unique())
        else:
            val = st.number_input(f"{col}:", float(df[col].min()), float(df[col].max()))
        inputs[col] = val

    if st.button("🚀 Prediksi Sekarang"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)
        st.success(f"📌 Hasil Prediksi: **{prediction[0]}**")

