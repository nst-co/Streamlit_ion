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
import dspfir
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Tuple

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
    y: np.ndarray, sr: int, frame_length=8192, hop_length=1024, title=None
) -> np.ndarray:
    y = y[np.isfinite(y)]  # 有効値のみにする
    S = np.abs(rosa.stft(y, n_fft=frame_length, hop_length=hop_length))  # STFT of y
    S_db = rosa.amplitude_to_db(S)

    st.write(title)
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


def rosa_temp_load_series(file_data) -> Tuple[pd.Series, float]:
    with TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir, file_data.name)
        temp_file_path.write_bytes(file_data.read())
        wave, sr = rosa.load(temp_file_path, sr=None)
    times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
    return pd.Series(wave, index=times, name=file_data.name), sr


decimation_rates = [1, 2, 4, 8]
sr = 48000

st.title("異音チェッカー シミュレーション")

tabOK, tabNG = st.tabs(["🆗 OK", "🆖 NG"])

with tabOK:
    totalContainer = st.container()
    uploaded_ok = st.file_uploader(
        "OKデータのオーディオファイルをアップロードしてください（複数可）",
        type=["wav", "mp3", "m4a", "aac", "mp4"],
        accept_multiple_files=True,
    )

    @st.cache_data
    def ok_master(uploaded_ok, d_rate):
        ss = []
        for uploaded_file in uploaded_ok:
            ts, sr = rosa_temp_load_series(uploaded_file)
            y = dspfir.downsample(ts.values, d_rate)
            s = showWaveSpec(y, sr=sr / d_rate, title=ts.name)
            ss.append(s)

        smax = pd.concat(ss, axis=1).max(axis=1)
        smax.name = "OKマスター"
        return smax

    if len(uploaded_ok) > 0:
        with totalContainer:
            st.write("OKマスターの作成後、NGタブを選択してNGデータをアップロードしてください")
            decimation_rate = st.radio(
                "サンプリング周波数(周波数分布の最大値はこの半分)",
                decimation_rates,
                horizontal=True,
                format_func=lambda x: f"{round(sr/x/1000):d}kHz",
            )
        ok = ok_master(uploaded_ok, decimation_rate)
        with totalContainer:
            sns.relplot(data=ok, aspect=3, kind="line").set(
                title="OKマスター（最大値）周波数分布", yscale="log", xlim=(0, ok.index[-1])
            ).set_xlabels("Hz").set_ylabels("amplitude")
            plotMod(plt)

with tabNG:
    if len(uploaded_ok) == 0:
        st.warning("↑OKタブを選択してOKデータをアップロードしてください", icon="⚠️")
    else:
        uploaded_ng = st.file_uploader(
            "NGデータのオーディオファイルをアップロードしてください（一つのみ）",
            type=["wav", "mp3", "m4a", "aac", "mp4"],
        )
        if uploaded_ng is None:
            st.stop()

        @st.cache_data
        def ng_fft(file, d_rate):
            ng_ts, ng_sr = rosa_temp_load_series(file)
            y = dspfir.downsample(ng_ts.values, d_rate)
            return showWaveSpec(y, sr=(ng_sr / d_rate), title=ng_ts.name)

        ng = ng_fft(uploaded_ng, decimation_rate)

        ngmax = float(ng.index[-1])
        ngxlim = st.slider("表示周波数範囲", 0.0, ngmax, (0.0, ngmax), 1000.0)

        df = pd.DataFrame({"マスター": ok, "NG": ng})
        sns.relplot(data=df, aspect=3, kind="line", dashes=False, alpha=0.75).set(
            title=str(ng.name) + "周波数分布比較", yscale="log", xlim=ngxlim
        ).set_xlabels("Hz").set_ylabels("amplitude")
        plotMod(plt)
        ngdb = pd.Series(20 * np.log10(ng / ok), name=ng.name)
        okdb = ok.apply(lambda x: 0)
        dfdb = pd.DataFrame({"マスター": okdb, "NG": ngdb})
        sns.relplot(data=dfdb, aspect=3, kind="line", dashes=False).set(
            title=str(ng.name) + " マスター比周波数分布比較", ylim=(-15, 15), xlim=ngxlim
        ).set_xlabels("Hz").set_ylabels("dB")
        plotMod(plt)
