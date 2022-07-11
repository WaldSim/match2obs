import glob
import numpy as np
import pandas as pd

read_file = ["Grenzhoefer_Weg_Winterdienst/Grenzhoefer_Weg_Winterdienst_Daten2021/*.csv",
             "Hospital/Hospital_Daten2021/*.csv",
             "Koenigstuhl_Winterdienst/Koenigstuhl_Winterdienst_Daten2021/*.csv",
             "Landessternwarte_Koenigstuhl/Landessternwarte_Koenigstuhl_Daten2021/*.csv",
             "Peterstal_Winterdienst/Peterstal_Winterdienst_Daten2021/*.csv",
             "Schlierbach_Winterdienst/Schlierbach_Winterdienst_Daten2021/*.csv",
             "Stadtbuecherei/Stadtbuecherei_Daten2021/*.csv",
             "Theodor-Heuss-Bruecke/Theodor-Heuss-Bruecke_Daten2021/*.csv",
             "Wasserwerk_Rauschen/Wasserwerk_Rauschen_Daten2021/*.csv",
             "Ziegelhausen_Koepfel/Ziegelhausen_Koepfel_Daten2021/*.csv"
             ]
save_file = ["Grenzhoefer_Weg_Winterdienst/grenzhoferweg",
             "Hospital/hospital",
             "Koenigstuhl_Winterdienst/koenigstuhl",
             "Landessternwarte_Koenigstuhl/sternwarte",
             "Peterstal_Winterdienst/peterstal",
             "Schlierbach_Winterdienst/schlierbach",
             "Stadtbuecherei/stadtbuecherei",
             "Theodor-Heuss-Bruecke/theo",
             "Wasserwerk_Rauschen/wasserwerk",
             "Ziegelhausen_Koepfel/koepfel"
             ]
savefile = ["Grenzhoefer_Weg_Winterdienst/grenzhoferweg_jul_okt_21.csv",
            "Hospital/hospital_jul_okt_21.csv",
            "Koenigstuhl_Winterdienst/koenigstuhl_jul_okt_21.csv",
            "Landessternwarte_Koenigstuhl/sternwarte_jul_okt_21.csv",
            "Peterstal_Winterdienst/peterstal_jul_okt_21.csv",
            "Schlierbach_Winterdienst/schlierbach_jul_okt_21.csv",
            "Stadtbuecherei/stadtbuecherei_jul_okt_21.csv",
            "Theodor-Heuss-Bruecke/theo_jul_okt_21.csv",
            "Wasserwerk_Rauschen/wasserwerk_jul_okt_21.csv",
            "Ziegelhausen_Koepfel/koepfel_jul_okt_21.csv"
            ]


def dateien_zusammenfuegen(file, savefilename):
    interesting_files = glob.glob(file)
    df_list = []
    for filename in sorted(interesting_files):
        df_list.append(pd.read_csv(filename, encoding='ISO-8859-1'))
    full_df = pd.concat(df_list)
    full_df.to_csv(savefilename)


def read_data_dig(readfile, savefile, start, end):
    df = pd.read_csv(readfile, encoding='ISO-8859-1', usecols=["dateObserved", "windSpeed", "windDirection"])
    df["dateObserved"] = df["dateObserved"].astype('datetime64[ns]')
    df_mean = df.resample('H', on='dateObserved').mean()
    df_mean = df_mean.reset_index()
    # rearranging
    df_mean = df_mean[["dateObserved", "windSpeed", "windDirection"]]
    # df_mean = df_mean[(df_mean["dateObserved"] > start) & (df_mean["dateObserved"] < end)]
    df_mean["u"] = (df_mean["windSpeed"] * np.sin((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["v"] = (df_mean["windSpeed"] * np.cos((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["std"] = df_mean["windSpeed"].std()
    # df_mean.to_csv(f"{savefile}_{start}.csv")
    df_mean.to_csv(f"{savefile}_{start}_190721.csv")


def sample_data(readfile, savefile):
    df = pd.read_csv(readfile, encoding='ISO-8859-1', usecols=["dateObserved", "windSpeed", "windDirection"])
    df["dateObserved"] = df["dateObserved"].astype('datetime64[ns]')
    df_mean = df.resample('H', on='dateObserved').mean()
    df_mean = df_mean.reset_index()
    # rearranging
    df_mean = df_mean[["dateObserved", "windSpeed", "windDirection"]]
    # falls utc:
    df_mean.index = pd.to_datetime(df_mean.pop('dateObserved'), utc=True)
    df_mean = df_mean.reset_index()
    df_mean["u"] = (df_mean["windSpeed"] * np.sin((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["v"] = (df_mean["windSpeed"] * np.cos((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["std"] = df_mean["windSpeed"].std()
    df_mean.to_csv(f"{savefile}_jul_okt_21_utc.csv")


def sample_data_with_global(readfile, savefile):
    df = pd.read_csv(readfile, encoding='ISO-8859-1',
                     usecols=["dateObserved", "windSpeed", "windDirection", "solarRadiation"])
    df["dateObserved"] = df["dateObserved"].astype('datetime64[ns]')
    df_mean = df.resample('H', on='dateObserved').mean()
    df_mean = df_mean.reset_index()
    # rearranging
    df_mean = df_mean[["dateObserved", "windSpeed", "windDirection", "solarRadiation"]]
    # falls utc:
    df_mean.index = pd.to_datetime(df_mean.pop('dateObserved'), utc=True)
    df_mean = df_mean.reset_index()
    df_mean["u"] = (df_mean["windSpeed"] * np.sin((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["v"] = (df_mean["windSpeed"] * np.cos((df_mean["windDirection"] * 2 * np.pi) / 360))
    df_mean["std"] = df_mean["windSpeed"].std()
    df_mean.to_csv(f"{savefile}_jul_okt_21_utc.csv")


if __name__ == '__main__':
    for read, save, save2 in zip(read_file, save_file, savefile):
        dateien_zusammenfuegen(read, save2)
        sample_data(save2, save)

    # dateien_zusammenfuegen("Theodor-Heuss-Bruecke/Theodor-Heuss-Bruecke_Daten2021/*.csv", "Theodor-Heuss-Bruecke/theo_global")
    # sample_data_with_global("Theodor-Heuss-Bruecke/theo_global", "Theodor-Heuss-Bruecke/theo_global_all")

    print("done.")
