import pandas as pd
import numpy as np
import pytz


STATIONCSVS_JUL = {
    "GHW": "messdaten/utc/grenzhoferweg_jul_okt_21_utc.csv",
    "HP": "messdaten/utc/hospital_jul_okt_21_utc.csv",
    "KOS": "messdaten/utc/koenigstuhl_jul_okt_21_utc.csv",
    "PT": "messdaten/utc/peterstal_jul_okt_21_utc.csv",
    "SB": "messdaten/utc/schlierbach_jul_okt_21_utc.csv",
    "THB": "messdaten/utc/theo_jul_okt_21_utc.csv",
    "WWR": "messdaten/utc/wasserwerk_jul_okt_21_utc.csv",
    "STB": "messdaten/utc/stadtbuecherei_jul_okt_21_utc.csv",
    "STW": "messdaten/utc/sternwarte_jul_okt_21_utc.csv",
    "IUP": "messdaten/utc/IUP_gemittelt_2021-06-30 23:00:00_utc.csv",
    "LUBW": "messdaten/utc/lubw_2021-07-01 23:00:00_utc.csv",
    "CZE": "messdaten/utc/czerny_190721_utc.csv",
    "KOE": "messdaten/utc/koepfel_jul_okt_21_utc.csv",
    "GAB": "messdaten/utc/gaiberg_190721_utc.csv",
}

def load_observed_jul(start, end, encoding='ISO-8859-1', n_stations=14, s=6):
    my_timezone = pytz.timezone("Europe/Berlin")
    ref = pd.date_range(start=start, end=end, freq="H", tz='utc')
    df = pd.DataFrame({"Date": ref})
    df_ghw = pd.read_csv(STATIONCSVS_JUL["GHW"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_ghw = df_ghw[(df_ghw["dateObserved"] > start) & (df_ghw["dateObserved"] < end)]
    df_ghw['dateObserved'] = pd.to_datetime(df_ghw['dateObserved'], utc=True)
    df_ghw = df.merge(right=df_ghw, left_on="Date", right_on="dateObserved", how="outer")
    del df_ghw["Date"]
    df_hp = pd.read_csv(STATIONCSVS_JUL["HP"], usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"],
                        encoding=encoding)
    df_hp = df_hp[(df_hp["dateObserved"] > start) & (df_hp["dateObserved"] < end)]
    df_hp['dateObserved'] = pd.to_datetime(df_hp['dateObserved'], utc=True)
    df_hp = df.merge(right=df_hp, left_on="Date", right_on="dateObserved", how="outer")
    del df_hp["Date"]
    df_kos = pd.read_csv(STATIONCSVS_JUL["KOS"], encoding=encoding,
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"])
    df_kos = df_kos[(df_kos["dateObserved"] > start) & (df_kos["dateObserved"] < end)]
    df_kos['dateObserved'] = pd.to_datetime(df_kos['dateObserved'], utc=True)
    df_kos = df.merge(right=df_kos, left_on="Date", right_on="dateObserved", how="outer")
    del df_kos["Date"]
    #
    df_stw = pd.read_csv(STATIONCSVS_JUL["STW"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_stw = df_stw[(df_stw["dateObserved"] > start) & (df_stw["dateObserved"] < end)]
    df_stw['dateObserved'] = pd.to_datetime(df_stw['dateObserved'], utc=True)
    df_stw = df.merge(right=df_stw, left_on="Date", right_on="dateObserved", how="outer")
    del df_stw["Date"]
    #
    df_pt = pd.read_csv(STATIONCSVS_JUL["PT"], usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"],
                        encoding=encoding)
    df_pt = df_pt[(df_pt["dateObserved"] > start) & (df_pt["dateObserved"] < end)]
    df_pt['dateObserved'] = pd.to_datetime(df_pt['dateObserved'], utc=True)
    df_pt = df.merge(right=df_pt, left_on="Date", right_on="dateObserved", how="outer")
    del df_pt["Date"]
    df_sb = pd.read_csv(STATIONCSVS_JUL["SB"], usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"],
                        encoding=encoding)
    df_sb = df_sb[(df_sb["dateObserved"] > start) & (df_sb["dateObserved"] < end)]
    df_sb['dateObserved'] = pd.to_datetime(df_sb['dateObserved'], utc=True)
    df_sb = df.merge(right=df_sb, left_on="Date", right_on="dateObserved", how="outer")
    del df_sb["Date"]
    df_stb = pd.read_csv(STATIONCSVS_JUL["STB"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_stb['dateObserved'] = pd.to_datetime(df_stb['dateObserved'], utc=True)

    df_stb = df_stb[(df_stb["dateObserved"] > start) & (df_stb["dateObserved"] < end)]
    df_stb = df.merge(right=df_stb, left_on="Date", right_on="dateObserved", how="outer")
    del df_stb["Date"]
    df_thb = pd.read_csv(STATIONCSVS_JUL["THB"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_thb = df_thb[(df_thb["dateObserved"] > start) & (df_thb["dateObserved"] < end)]
    df_thb['dateObserved'] = pd.to_datetime(df_thb['dateObserved'], utc=True)

    df_thb = df.merge(right=df_thb, left_on="Date", right_on="dateObserved", how="outer")
    del df_thb["Date"]
    #
    df_wwr = pd.read_csv(STATIONCSVS_JUL["WWR"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_wwr = df_wwr[(df_wwr["dateObserved"] > start) & (df_wwr["dateObserved"] < end)]
    df_wwr['dateObserved'] = pd.to_datetime(df_wwr['dateObserved'], utc=True)

    df_wwr = df.merge(right=df_wwr, left_on="Date", right_on="dateObserved", how="outer")
    del df_wwr["Date"]
    #
    df_koe = pd.read_csv(STATIONCSVS_JUL["KOE"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_koe = df_koe[(df_koe["dateObserved"] > start) & (df_koe["dateObserved"] < end)]
    df_koe['dateObserved'] = pd.to_datetime(df_koe['dateObserved'], utc=True)

    df_koe = df.merge(right=df_koe, left_on="Date", right_on="dateObserved", how="outer")
    del df_koe["Date"]

    df_lubw = pd.read_csv(STATIONCSVS_JUL["LUBW"],
                          usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_lubw = df_lubw[(df_lubw["dateObserved"] > start) & (df_lubw["dateObserved"] < end)]
    df_lubw['dateObserved'] = pd.to_datetime(df_lubw['dateObserved'], utc=True)

    df_lubw = df.merge(right=df_lubw, left_on="Date", right_on="dateObserved", how="outer")
    del df_lubw["Date"]

    df_iup = pd.read_csv(STATIONCSVS_JUL["IUP"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_iup = df_iup[(df_iup["dateObserved"] > start) & (df_iup["dateObserved"] < end)]
    df_iup['dateObserved'] = pd.to_datetime(df_iup['dateObserved'], utc=True)

    df_iup = df.merge(right=df_iup, left_on="Date", right_on="dateObserved", how="outer")
    del df_iup["Date"]

    df_cze = pd.read_csv(STATIONCSVS_JUL["CZE"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_cze['dateObserved'] = pd.to_datetime(df_cze['dateObserved'])
    df_cze = df_cze[(df_cze["dateObserved"] > start) & (df_cze["dateObserved"] < end)]

    df_cze = df.merge(right=df_cze, left_on="Date", right_on="dateObserved", how="outer")
    del df_cze["Date"]
    #
    df_gab = pd.read_csv(STATIONCSVS_JUL["GAB"],
                         usecols=["dateObserved", "windSpeed", "windDirection", "u", "v", "std"], encoding=encoding)
    df_gab = df_gab[(df_gab["dateObserved"] > start) & (df_gab["dateObserved"] < end)]
    df_gab['dateObserved'] = pd.to_datetime(df_gab['dateObserved'], utc=True)

    df_gab = df.merge(right=df_gab, left_on="Date", right_on="dateObserved", how="outer")
    del df_gab["Date"]
    h = len(df["Date"])  # what is this?

    # Spalten df:
    # Time, Wind_Geschwindigkeit, Wind_Richtung, u, v, std

    stations = np.zeros((h, s, n_stations)).astype(object)

    def to_np(df_station, station_index):
        station = df_station.to_numpy()
        stations[:, :, station_index] = station

    to_np(df_ghw, 0)
    to_np(df_hp, 1)
    to_np(df_kos, 2)
    to_np(df_stw, 3)
    to_np(df_pt, 4)
    to_np(df_sb, 5)
    to_np(df_stb, 6)
    to_np(df_thb, 7)
    to_np(df_wwr, 8)
    to_np(df_koe, 9)
    to_np(df_cze, 10)
    to_np(df_gab, 11)
    to_np(df_lubw, 12)
    to_np(df_iup, 13)

    return stations
