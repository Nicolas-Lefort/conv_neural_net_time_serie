import pandas_ta as ta

def clean(df, df_signal):
    df = df.join(df_signal)
    df.replace("", "NaN", inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    df_signal = df['signal']
    df.drop(columns=['volume', 'high', 'low', 'open', 'signal', "close_timestamp", "close_date"], inplace=True)

    return df, df_signal

def augment(df):
    df.ta.rsi(close=df["close"], length=14, append=True)
    df.ta.willr(close=df["close"], low=df["low"], high=df["high"], length=14, append=True)
    df.ta.macd(close=df["close"], append=True)

    return df

def split(df, train_size=0.9, val_size=0.05):
    train_df = df[0:int(len(df) * train_size)]
    val_df = df[int(len(df) * train_size):int(len(df) * (train_size + val_size))]
    test_df = df[int(len(df) * (train_size + val_size)):]

    return train_df, val_df, test_df

