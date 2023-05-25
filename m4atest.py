import librosa as rosa
import pandas as pd
import numpy as np
import pathlib

print("audio check")

cwd = pathlib.Path.cwd()
uploaded_files = sorted(cwd.glob("*.m4a"))

for uploaded_file in uploaded_files:
    print("filename:", uploaded_file.name)
    wave, sr = rosa.load(uploaded_file, sr=None)
    times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
    ts = pd.Series(wave, index=times, name=str(uploaded_file))
    print(ts)
