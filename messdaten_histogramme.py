import pandas as pd
from matplotlib import pyplot as plt
import glob

plt.style.use("bmh")

if __name__ == '__main__':
    path = "messdaten/utc/used"
    all_files = glob.glob(path + "/*.csv")
    dfs = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        dfs.append(df)

    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame2 = frame[frame.windSpeed > 0]
    frame2.hist("windSpeed", bins=20)  # , bins=40, color="red", alpha=0.5)
    plt.savefig("plots/hist_windspeed.png")
    plt.show()
    plt.close()

    frame3 = frame[(frame.windDirection > 0) & (frame.windDirection < 360)]
    all_speed = (frame2["windSpeed"]).count()
    low_speed = (frame2["windSpeed"] < 0.5).count()
    frame3.hist("windDirection", bins=40)
    plt.savefig("plots/hist_winddirection.png")
    plt.show()
    plt.close()

    df_glob = pd.read_csv("messdaten/utc/theo_global_all_jul_okt_21_utc.csv")
    df_lubw = pd.read_csv("messdaten/utc/lubw_2021-07-01 23:00:00_utc.csv")
    df_glob_wind_dir = df_glob[(df_glob.windDirection > 0) & (df_glob.windDirection < 360)]
    df_wind_dir_lubw = df_lubw[(df_lubw.windDirection > 0) & (df_lubw.windDirection < 360)]

    df_glob_wind_dir.hist("windDirection", bins=40)
    plt.savefig("plots/hist_winddirection_lubw_thb.png")
    plt.show()
    plt.close()

    df_glob_wind_speed = df_glob[df_glob.windSpeed > 0]
    df_lubw_wind_speed = df_lubw[df_lubw.windSpeed > 0]
    df_glob_wind_speed.hist("windSpeed", bins=40)
    df_lubw_wind_speed.hist("windSpeed", bins=40)
    plt.savefig("plots/hist_windspeed_lubw_thb.png")
    plt.show()
    plt.close()

    # solar radiation
    df_glob.hist("solarRadiation", bins=20, label=" THB global radiation [W/ms²]", color="purple")
    plt.legend()
    plt.savefig("plots/globalradiation.png", bbox_inches='tight')
    plt.show()
    plt.close()

    stab_a_b = df_glob[(df_glob["solarRadiation"] > 175) & (df_glob["windSpeed"] < 0.5)]
    high_glob = df_glob[(df_glob["solarRadiation"] > 175)]
    low_wind = df_glob[(df_glob["windSpeed"] < 0.5)]

    glob_lubw = pd.merge(left=df_lubw, right=df_glob, left_on="dateObserved", right_on="dateObserved")

    stab_a_b_lubw = glob_lubw[(glob_lubw["solarRadiation"] > 175) & (glob_lubw["windSpeed_x"] < 0.5)]
    stab_a_b_thb = glob_lubw[(glob_lubw["solarRadiation"] > 175) & (glob_lubw["windSpeed_y"] < 0.5)]

    print("done.")
