import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def look_bear(s):
    tol = 0
    s = s.reset_index(drop=True)
    mid_pos = int(len(s) / 2)
    if mid_pos - tol <= s.idxmax() <= mid_pos + tol:
        return 2
    else:
        return 0

def look_bull(s):
    tol = 0
    s = s.reset_index(drop=True)
    mid_pos = int(len(s) / 2)
    if mid_pos - tol <= s.idxmin() <= mid_pos + tol:
        return 1
    else:
        return 0

def look_range(s):
    thresold = 4
    s = s.reset_index(drop=True)
    if (s.max() - s.min()) / s.min() * 100 < thresold:
        return 3
    else:
        return 0

def generate_labels(df, sequence_length):
    var = int((sequence_length - 1) / 2)
    df['bull'] = df.close.rolling(sequence_length).apply(look_bull)
    df['bear'] = df.close.rolling(sequence_length).apply(look_bear)
    df["bull"].shift(periods=-var)
    df["bear"].shift(periods=-var)

    df['signal'] = np.where((df['bull'] == 1), 1, df['bear'])
    s1 = df[df["signal"] == 1]
    s2 = df[df["signal"] == 2]
    s1 = s1["signal"]
    s2 = s2["signal"]

    result = pd.concat([s1, s2], axis=0)
    result.sort_index(ascending=True, inplace=True)
    result = result.to_frame()
    result["trash"] = np.where((result["signal"] == result["signal"].shift(-1)), "drop", 0)
    result.drop(result[result.trash == "drop"].index, inplace=True)
    index = result.index.tolist()
    signal = result.signal.tolist()

    for i in range(len(index) - 1):
        df.loc[index[i]:index[i + 1], "signal"] = signal[i]

    seq = 2500
    df['signal_hold'] = df.close.rolling(seq).apply(look_range)
    df['signal'] = np.where((df['signal_hold'] == 3), 0, df['signal'])
    s = df[df['signal'] == 0]
    index = s.index.tolist()

    for i in range(seq, len(index)):
        df.loc[index[i] - seq:index[i], "signal"] = 0
    plt.plot(np.where(df["signal"] == 2, df["close"], None), color="red", label="bear")
    plt.plot(np.where(df["signal"] == 1, df["close"], None), color="blue", label="bull")
    plt.plot(np.where(df["signal"] == 0, df["close"], None), color="black", label="range")

    plt.show()
    df_signal = df['signal']

    return df_signal

