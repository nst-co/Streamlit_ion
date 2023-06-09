# Copyright 2023 NST Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import librosa as rosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from tempfile import TemporaryDirectory
from pathlib import Path

st.set_page_config(
    page_title="異音チェッカー シミュレーション",
    page_icon="random",
    menu_items={"About": "Copyright 2023 NST Co., Ltd. All rights reserved."},
)

sns.set("talk", "whitegrid")
japanize_matplotlib.japanize()


def plotMod(plt):
    plt.minorticks_on()
    plt.grid(which="major", color="black", alpha=0.5)
    plt.grid(which="minor", color="gray", linestyle=":")
    st.pyplot(plt)


def showWaveSpec(
    y: np.ndarray, sr: int, frame_length=4800, hop_length=1200, title=None
) -> np.ndarray:
    y = y[np.isfinite(y)]  # 有効値のみにする
    S = np.abs(rosa.stft(y, n_fft=frame_length, hop_length=hop_length))  # STFT of y
    S_db = rosa.amplitude_to_db(S)

    st.audio(y, sample_rate=sr)

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 4))
    rosa.display.waveshow(y, sr=sr, axis="time", ax=axes[0])
    rosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        n_fft=frame_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
    )
    fig.suptitle(title)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xlim(0, len(y) / sr)
    st.pyplot(plt)
    freq = rosa.fft_frequencies(sr=sr, n_fft=frame_length)
    return pd.Series(np.max(S, axis=1), index=freq, name=title)


st.title("異音チェッカー シミュレーション")

tabOK, tabNG = st.tabs(["🆗 OK", "🆖 NG"])

with tabOK:
    uploaded_ok = st.file_uploader(
        "OKデータのオーディオファイルをアップロードしてください（複数可）",
        type=["wav", "mp3", "m4a", "aac", "mp4"],
        accept_multiple_files=True,
    )

    @st.cache_data
    def ok_master(uploaded_ok):
        totalContainer = st.container()
        ss = []
        for uploaded_file in uploaded_ok:
            st.write("filename:", uploaded_file.name)
            with TemporaryDirectory() as temp_dir:
                temp_file_path = Path(temp_dir, uploaded_file.name)
                temp_file_path.write_bytes(uploaded_file.read())
                wave, sr = rosa.load(temp_file_path, sr=None)
            times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
            ts = pd.Series(wave, index=times, name=uploaded_file.name)
            s = showWaveSpec(ts.values, sr=sr, title=ts.name)
            ss.append(s)

        with totalContainer:
            smax = pd.concat(ss, axis=1).max(axis=1)
            smax.name = "OKマスター"
            sns.relplot(data=smax, aspect=3, kind="line").set(
                title="OKマスター（最大値）周波数分布", yscale="log", xlim=(0, smax.index[-1])
            ).set_xlabels("Hz").set_ylabels("amplitude")
            plotMod(plt)
        return smax

    if len(uploaded_ok) > 0:
        ok = ok_master(uploaded_ok)

with tabNG:
    if len(uploaded_ok) == 0:
        st.write("↑OKタブを選択してOKデータをアップロードしてください")
    else:
        uploaded_ng = st.file_uploader(
            "NGデータのオーディオファイルをアップロードしてください（一つのみ）",
            type=["wav", "mp3", "m4a", "aac", "mp4"],
        )
        if uploaded_ng is None:
            st.stop()

        @st.cache_data
        def ng_wave(uploaded_ng):
            st.write("filename:", uploaded_ng.name)
            with TemporaryDirectory() as temp_dir:
                temp_file_path = Path(temp_dir, uploaded_ng.name)
                temp_file_path.write_bytes(uploaded_ng.read())
                wave, sr = rosa.load(temp_file_path, sr=None)
            times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
            ts = pd.Series(wave, index=times, name=uploaded_ng.name)
            return showWaveSpec(ts.values, sr=sr, title=ts.name)

        ng = ng_wave(uploaded_ng)
        ngmax = float(ng.index[-1])
        ngxlim = st.slider("表示周波数範囲", 0.0, ngmax, (0.0, ngmax))

        def wave_dB(ok, ng, xlim):
            df = pd.DataFrame({"マスター": ok, "NG": ng})
            sns.relplot(data=df, aspect=3, kind="line", dashes=False, alpha=0.75).set(
                title=str(ng.name) + "周波数分布比較", yscale="log", xlim=xlim
            ).set_xlabels("Hz").set_ylabels("amplitude")
            plotMod(plt)
            ngdb = pd.Series(20 * np.log10(ng / ok), name=ng.name)
            okdb = ok.apply(lambda x: 0)
            dfdb = pd.DataFrame({"マスター": okdb, "NG": ngdb})
            sns.relplot(data=dfdb, aspect=3, kind="line", dashes=False).set(
                title=str(ng.name) + " マスター比周波数分布比較", ylim=(-15, 15), xlim=xlim
            ).set_xlabels("Hz").set_ylabels("dB")
            plotMod(plt)

        wave_dB(ok, ng, ngxlim)
