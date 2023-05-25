import streamlit as st
import librosa as rosa
import pandas as pd
import numpy as np

st.title("異音チェッカー シミュレーション")

uploaded_files = st.file_uploader(
    "OKデータのオーディオファイルをアップロードしてください", accept_multiple_files=True
)

for uploaded_file in uploaded_files:
    st.write("filename:", uploaded_file.name)
    wave, sr = rosa.load(uploaded_file, sr=None)
    times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
    ts = pd.Series(wave, index=times, name=str(uploaded_file))
    st.write(ts)
