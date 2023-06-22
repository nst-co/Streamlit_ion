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
import dspfir
import plotly.express as px
from plotly.subplots import make_subplots
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Tuple

st.set_page_config(
    page_title="異音チェッカー シミュレーション",
    page_icon="random",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/nst-co/Streamlit_ion/blob/master/README.md",
        "About": "Copyright 2023 NST Co., Ltd. All rights reserved.",
    },
)


def plotMod(plt):
    plt.update_xaxes(showline=True, showgrid=True)
    plt.update_yaxes(showline=True, showgrid=True)
    st.plotly_chart(
        plt,
        use_container_width=True,
        config={
            "toImageButtonOptions": {"filename": plt.layout.title.text, "scale": 2}
        },
    )


def waveSpec(
    y: np.ndarray, sr: int, frame_length=8192, hop_length=1024, title=None
) -> np.ndarray:
    y = y[np.isfinite(y)]  # 有効値のみにする
    S = np.abs(rosa.stft(y, n_fft=frame_length, hop_length=hop_length))  # STFT of y
    freq = rosa.fft_frequencies(sr=sr, n_fft=frame_length)
    return pd.Series(np.max(S, axis=1), index=freq, name=title)


def rosa_temp_load_series(file_data) -> Tuple[pd.Series, float]:
    with TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir, file_data.name)
        temp_file_path.write_bytes(file_data.read())
        wave, sr = rosa.load(temp_file_path, sr=None)
    times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
    return pd.Series(wave, index=times, name=file_data.name), sr


@st.cache_data
def ok_master(uploaded_ok, d_rate):
    ss = []
    master_bar = st.progress(0)
    for i, file in enumerate(uploaded_ok):
        ts, sr = rosa_temp_load_series(file)
        y = dspfir.downsample(ts.values, d_rate)
        s = waveSpec(y, sr=(sr / d_rate), title=ts.name)
        ss.append(s)
        master_bar.progress((i + 1) * 100 // len(uploaded_ok))
    smax = pd.concat(ss, axis=1).max(axis=1)
    smax.name = "OKマスター"
    master_bar.empty()
    return smax, sr


@st.cache_data
def ng_fft(file, d_rate):
    ts, sr = rosa_temp_load_series(file)
    y = dspfir.downsample(ts.values, d_rate)
    return waveSpec(y, sr=(sr / d_rate), title=ts.name)


decimation_rates = [1, 2, 4, 8]
if "decimation_rate" not in st.session_state:
    st.session_state.decimation_rate = 1

st.title("異音チェッカー シミュレーション")
placeholder = st.empty()

with st.sidebar:
    uploaded_ok = st.file_uploader(
        "OKデータのオーディオファイルをアップロードしてください（複数可）",
        type=["wav", "mp3", "m4a", "aac", "mp4"],
        accept_multiple_files=True,
    )
if len(uploaded_ok) == 0:
    placeholder.warning("←サイドバーからOKデータをアップロードしてください")
    st.stop()

placeholder.empty()

with st.spinner("OKマスター作成中"):
    ok, sr = ok_master(uploaded_ok, st.session_state.decimation_rate)
    fig = px.line(
        ok, log_y=True, labels={"value": "amplitude", "index": "Hz"}
    ).update_layout(title="OKマスター（最大値）周波数分布", showlegend=False, height=400)
    plotMod(fig)
with st.sidebar:
    st.divider()
    st.radio(
        "サンプリング周波数(周波数分布の最大値はこの半分)",
        decimation_rates,
        key="decimation_rate",
        horizontal=True,
        format_func=lambda x: f"{round(sr/x/1000):d}kHz",
    )
    st.divider()
    uploaded_ng = st.file_uploader(
        "NGデータのオーディオファイルをアップロードしてください（一つのみ）",
        type=["wav", "mp3", "m4a", "aac", "mp4"],
    )
if uploaded_ng is None:
    placeholder.warning("←サイドバーからNGデータをアップロードしてください")
    st.stop()

placeholder.empty()

ng = ng_fft(uploaded_ng, st.session_state.decimation_rate)


df = pd.DataFrame({"OKマスター": ok, "NG": ng})
fig1 = px.line(df, color_discrete_sequence=px.colors.qualitative.Plotly, log_y=True)
fig1.update_traces(opacity=0.7)
ngdb = pd.Series(20 * np.log10(ng / ok), name=ng.name)
okdb = ok.apply(lambda x: 0)
dfdb = pd.DataFrame({"OKマスター": okdb, "NG": ngdb})
fig2 = px.line(dfdb, color_discrete_sequence=px.colors.qualitative.Plotly)
fig2.update_traces(showlegend=False, opacity=0.7)
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("周波数分布比較", "OKマスター比 周波数分布比較"),
    shared_xaxes=True,
    vertical_spacing=0.1,
).update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis2=dict(title="Hz", rangeslider_visible=True, rangeslider_thickness=0.0625),
    yaxis=dict(title="amplitude", type="log"),
    yaxis2=dict(title="dB", range=[-14, 14], dtick=2),
    title_text=str(ng.name),
    height=750,
)
for trace in fig1["data"]:
    fig.add_trace(trace, row=1, col=1)
for trace in fig2["data"]:
    fig.add_trace(trace, row=2, col=1)
plotMod(fig)

st.write(f"NG-OK比の最大値 **{ngdb.max():.2f}**dB（**{round(ngdb.idxmax())}**Hzにおいて）")
