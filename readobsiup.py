import glob
import pandas as pd
from pathlib import Path
import numpy as np

FILE = "DatenFebruar/IUP/FebMaerz150921/*.dat"
FILE_csv = "DatenFebruar/IUP/FebMaerz150921/*.csv"
FILENAME = "DatenFebruar/IUP/FebMaerz150921/IUPdatenfebmaer21.csv"
START = "2021-07-22 00:00:00"
END = "2021-10-21 23:00:00"


def dateien_lesen(file):
    filenames = glob.glob(file)  # Liste der Dateien
    for fname in filenames:
        df = pd.read_csv(fname, skiprows=[i for i in range(0, 13)], sep=";", usecols=["IntvlStart", "WVel", "WDir"])
        df_new = df.rename(columns={"IntvlStart": "dateObserved", "WVel": "windSpeed", "WDir": "windDirection"})
        fname2 = Path(fname).stem
        df_new.to_csv(f"DatenFebruar/IUP/FebMaerz150921/{fname2}.csv")


def dateien_zusammenfuegen(file, savefilename):
    interesting_files = glob.glob(file)
    df_list = []
    for filename in sorted(interesting_files):
        df_list.append(pd.read_csv(filename, encoding='ISO-8859-1'))
    full_df = pd.concat(df_list)
    full_df.to_csv(savefilename)


def readDataIUP(file, start, end):
    df = pd.read_csv(file,
                     # encoding='ISO-8859-1',
                     # skiprows=(1, 2, 3),
                     usecols=["dateObserved", "windSpeed", "windDirection"]
                     )
    df["dateObserved"] = df["dateObserved"].astype('datetime64[ns]')
    df_mean = df.resample('H', on="dateObserved").mean()
    df_mean = df_mean.reset_index()
    df_mean = df_mean[(df_mean["dateObserved"] > start) & (df_mean["dateObserved"] < end)]

    df_mean["u"] = (df_mean["windSpeed"] * np.sin((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["v"] = (df_mean["windSpeed"] * np.cos((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["std"] = df_mean["windSpeed"].std()
    df_mean = df_mean.shift(-1, axis=0)
    df_mean.to_csv(f"IUP_gemittelt_{START}_UTC.csv")


if __name__ == '__main__':
    # IUP Daten ersten Zeilen entfernen und als CSV speichern
    # dateien_lesen(FILE)
    # IUP Dateien Dateine aus Ordner zusammenfügen
    dateien_zusammenfuegen("DatenFebruar/IUP/FebMaerz150921/*.csv",
                           "DatenFebruar/IUP/FebMaerz150921/IUPdatenfebmaer21.csv")
    readDataIUP(FILENAME, START, END)
    print("done.")
