import pandas as pd
import numpy as np

file_speed = "LUBW/table26102021103855270.csv"
file_direction = "LUBW/table26102021103927431.csv"
if __name__ == '__main__':
    START = "2021-07-01 23:00:00"
    END = "2021-10-20 00:00:00"
    # Lade csv für speed und direction
    df_speed = pd.read_csv(file_speed, error_bad_lines=False, skiprows=[i for i in range(0, 6)]
                           , encoding='ISO-8859-1',
                           usecols=["Datum / Uhrzeit", "Wert"], decimal=','
                           )
    df_direction = pd.read_csv(file_direction, error_bad_lines=False, skiprows=[i for i in range(0, 6)],
                               encoding='ISO-8859-1',
                               usecols=["Datum / Uhrzeit", "Wert"]
                               )
    # füge beide zusammen
    df = df_speed.merge(df_direction, how='inner', left_index=True, right_index=True)
    df.drop('Datum / Uhrzeit_y', axis=1, inplace=True)
    df.rename(columns={'Datum / Uhrzeit_x': "dateObserved", 'Wert_x': 'windSpeed', 'Wert_y': 'windDirection'},
              inplace=True)
    # replace 24:00 mit 00:00
    time = df["dateObserved"]
    time_new = np.array([t.replace("24:00", "00:00") for t in time])
    df["dateObserved"] = time_new
    # mean über Stunden
    df["dateObserved"] = df["dateObserved"].astype('datetime64[ns]')
    df_mean = df.resample('H', on='dateObserved').mean()
    df_mean = df_mean.reset_index()
    df_mean.index = pd.to_datetime(df_mean.pop('dateObserved'))
    df_mean.index = df_mean.index.tz_localize('Europe/Berlin')
    df_mean.index = df_mean.index.tz_convert('UTC')
    df_mean = df_mean.reset_index()
    df_mean = df_mean[(df_mean["dateObserved"] > START) & (df_mean["dateObserved"] < END)]
    df_mean["u"] = (df_mean["windSpeed"] * np.sin((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["v"] = (df_mean["windSpeed"] * np.cos((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["std"] = df_mean["windSpeed"].std()
    df_mean.to_csv(f"lubw_{START}_utc.csv")
    print("done.")
