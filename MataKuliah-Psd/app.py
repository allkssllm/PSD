# App.py
import streamlit as st
import librosa
import numpy as np
import sounddevice as sd
import joblib
import time

# --- Konfigurasi Awal ---
PATH_MODEL = "./dataset/data-hasil/model_audio.pkl"
PATH_ENCODER = "./dataset/data-hasil/label_encoder.pkl"

DURASI_REKAMAN = 1.5  # Sesuaikan dengan durasi saat latihan
SAMPLE_RATE = 22050     # HARUS SAMA dengan saat latihan
N_MFCC = 20           # HARUS SAMA dengan saat latihan

# --- Fungsi Helper ---

@st.cache_resource
def muat_model():
    """Memuat model dan encoder dari file."""
    try:
        model = joblib.load(PATH_MODEL)
        encoder = joblib.load(PATH_ENCODER)
        return model, encoder
    except FileNotFoundError:
        st.error(f"Error: File model/encoder tidak ditemukan.")
        st.error(f"Pastikan file '{PATH_MODEL}' dan '{PATH_ENCODER}' ada.")
        return None, None

def ekstrak_fitur(audio_data, sr, n_mfcc):
    """Mengekstrak fitur MFCC dari data audio mentah."""
    # Pastikan audio mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
        
    # Ekstrak MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    
    # Ambil rata-rata
    mfccs_rata_rata = np.mean(mfccs.T, axis=0)
    
    return mfccs_rata_rata

def rekam_audio(durasi, sr):
    """Merekam audio dari mikrofon."""
    audio = sd.rec(int(durasi * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Menunggu rekaman selesai
    return audio.flatten() # Kembalikan sebagai array 1D

# --- Muat Model ---
model, encoder = muat_model()

# --- Tampilan Aplikasi Streamlit ---
st.title("🎙️ Klasifikasi Audio 'BUKA' dan 'TUTUP'")
st.write("Proyek ini mendemonstrasikan klasifikasi audio secara real-time.")
st.write("Model dilatih untuk membedakan antara kata 'buka' dan 'tutup'.")

st.markdown("---")

if model and encoder:
    st.header("Uji Coba Langsung")
    
    # Tombol untuk memicu rekaman
    if st.button("Tekan untuk Bicara (1.5 detik)"):
        with st.spinner("Merekam..."):
            # 1. Rekam audio
            audio_rekaman = rekam_audio(DURASI_REKAMAN, SAMPLE_RATE)
            st.success("Rekaman selesai, memproses...")
            time.sleep(0.5)

        with st.spinner("Mengekstrak fitur..."):
            # 2. Ekstrak fitur (HARUS SAMA PERSIS DENGAN PELATIHAN)
            fitur = ekstrak_fitur(audio_rekaman, SAMPLE_RATE, N_MFCC)
            
            # 3. Ubah bentuk data untuk prediksi (sklearn butuh 2D array)
            fitur_2d = fitur.reshape(1, -1)
            
        with st.spinner("Memprediksi..."):
            # 4. Lakukan prediksi
            prediksi_kode = model.predict(fitur_2d)
            
            # 5. Ubah kode (angka) kembali ke label (teks)
            hasil_prediksi = encoder.inverse_transform(prediksi_kode)
            
            # Tampilkan hasil
            hasil_final = hasil_prediksi[0].upper()
            
            if hasil_final == "BUKA":
                st.image("https://i.imgur.com/gA0jXoP.png", width=150) # Gambar pintu terbuka
            else:
                st.image("https://i.imgur.com/v8FVyCT.png", width=150) # Gambar pintu tertutup

            st.metric(label="Hasil Prediksi", value=hasil_final)
            
            # (Opsional) Tampilkan audio yang direkam
            st.audio(audio_rekaman, format='audio/wav', sample_rate=SAMPLE_RATE)
else:
    st.warning("Model tidak dapat dimuat. Silakan periksa path file di kode.")