import streamlit as st
import librosa as rosa
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path

st.title("異音チェッカー シミュレーション")

uploaded_files = st.file_uploader(
    "OKデータのオーディオファイルをアップロードしてください", accept_multiple_files=True
)

for uploaded_file in uploaded_files:
    st.write("filename:", uploaded_file.name)
    with TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir, uploaded_file.name)
        temp_file_path.write_bytes(uploaded_file.read())
        wave, sr = rosa.load(temp_file_path, sr=None)
    times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
    ts = pd.Series(wave, index=times, name=uploaded_file.name)
    st.write(ts)
