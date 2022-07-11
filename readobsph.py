import numpy as np
import pandas as pd

if __name__ == '__main__':
    read_file = "Czerny/DE_02_HD-Geo-Institut_Jul_Okt_2021.csv"
    df = pd.read_csv(read_file, encoding='ISO-8859-1',
                     usecols=["Station:", "DE_02_HD-Geo-Institut.3", "DE_02_HD-Geo-Institut.4"],
                     skiprows=[1, 2, 3])  # '', usecols=["dateObserved", "windSpeed", "windDirection"])
    df = df.rename(columns={"Station:": "dateObserved", "DE_02_HD-Geo-Institut.3": "windSpeed",
                            "DE_02_HD-Geo-Institut.4": "windDirection"})
    df["dateObserved"] = pd.to_datetime(df["dateObserved"], format="%d.%m.%Y %H:%M")
    df_mean = df.resample('H', on='dateObserved').mean()
    df_mean = df_mean.reset_index()
    # rearranging
    df_mean = df_mean[["dateObserved", "windSpeed", "windDirection"]]
    df_mean["u"] = (df_mean["windSpeed"] * np.sin((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["v"] = (df_mean["windSpeed"] * np.cos((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["std"] = df_mean["windSpeed"].std()
    df_mean.index = pd.to_datetime(df_mean.pop('dateObserved'), utc=True)
    df_mean.to_csv(f"czerny_190721_utc.csv")
    # Spalten noch umbenennen und mitteln
    read_file = "Gaiberg/DE_03_Gaiberg-Berghof_ZR_Jul_Okt_2021.csv"
    df = pd.read_csv(read_file, encoding='ISO-8859-1',
                     usecols=["Station:", "DE_03_Gaiberg-Berghof.4", "DE_03_Gaiberg-Berghof.5"], skiprows=[1, 2, 3])
    df = df.rename(columns={"Station:": "dateObserved", "DE_03_Gaiberg-Berghof.4": "windSpeed",
                            "DE_03_Gaiberg-Berghof.5": "windDirection"})
    df["dateObserved"] = pd.to_datetime(df["dateObserved"], format="%d.%m.%Y %H:%M")
    df_mean = df.resample('H', on='dateObserved').mean()
    df_mean = df_mean.reset_index()
    # rearranging
    df_mean = df_mean[["dateObserved", "windSpeed", "windDirection"]]
    df_mean["u"] = (df_mean["windSpeed"] * np.sin((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["v"] = (df_mean["windSpeed"] * np.cos((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["std"] = df_mean["windSpeed"].std()
    # Zeitzone konvertieren
    df_mean.index = pd.to_datetime(df_mean.pop('dateObserved'), utc=True)
    df_mean.to_csv(f"gaiberg_190721_utc.csv")
    print("done.")
