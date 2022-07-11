import numpy as np
import pandas as pd
from einops import rearrange
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from matchobs_functions import matching_winds, calculate_rmse, calculate_mean_bias, obshor, matchedhor, obsdir, \
    matcheddir, winkdiff, mbedir, mbedirstations, mbehor, mbehorstations, mbestattime, rmsedir, rmsedirstations, rmseuv, \
    rmseuvstations, rmsehor, rmsehorstations, rmsestattime, rmsetime, matchgralorgramm_jul
from readobs import load_observed_jul
from settings import USED_STATIONS, STATION_NAMES_JUL, START2, END2, LOAD_CATALOGUE, METRIK, USE_SMOOTHING, \
    GRAL, USE_WEIGHTING, N_CATALOGUE_USED, STATIONS_INDEX

plt.style.use("bmh")

if __name__ == '__main__':
    print(f"used_stations:{USED_STATIONS}")
    print(f"stations:{STATION_NAMES_JUL}")
    print(f"start:{START2}")
    print(f"end:{END2}")
    print(f"metrik:{METRIK}")
    print(f"n_cat:{N_CATALOGUE_USED}")
    print(f"smoothing:{USE_SMOOTHING}")
    print(f"load_cat:{LOAD_CATALOGUE}")
    print(f"gral:{GRAL}")
    print(f"use_weighting:{USE_WEIGHTING}")
    # load catalogue
    catalogue = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED)
    # load observations
    observations = load_observed_jul(START2, END2)
    # matching
    matched_winds = matching_winds(observations, catalogue, len(USED_STATIONS),
                                   USED_STATIONS, METRIK, USE_WEIGHTING,
                                   USE_SMOOTHING, N_CATALOGUE_USED)

    # horizontal wind speed
    obs_hor = obshor(observations)
    matched_hor = matchedhor(matched_winds)
    # wind direction
    obs_dir = obsdir(observations)
    matched_dir = matcheddir(matched_winds)
    # winkel differenz
    wink_diff = winkdiff(obs_dir, matched_dir)
    # statistical values
    # mb
    mbe_dir = mbedir(wink_diff)
    mbe_dir_stat = mbedirstations(wink_diff)
    mbe_hor = mbehor(obs_hor, matched_hor)
    mbe_hor_stat = mbehorstations(obs_hor, matched_hor)
    mbe_station_time = mbestattime(obs_hor, matched_hor)
    print("MBE:")
    print(f"mbe_dir:{mbe_dir}")
    print(f"mbe_dir_stat:{mbe_dir_stat}")
    print(f"mbe_hor:{mbe_hor}")
    print(f"mbe_hor_stat:{mbe_hor_stat}")
    # RMSE
    rmse_dir = rmsedir(wink_diff)
    rmse_dir_stat = rmsedirstations(wink_diff)
    rmse_uv = rmseuv(observations, matched_winds)
    rmse_uv_stat = rmseuvstations(observations, matched_winds)
    rmse_hor = rmsehor(obs_hor, matched_hor)
    rmse_hor_stat = rmsehorstations(obs_hor, matched_hor)
    rmse_station_time = rmsestattime(observations, matched_winds)
    rmse_time = rmsetime(observations, matched_winds)
    print("RMSE:")
    print(f"rmse_dir:{rmse_dir}")
    print(f"rmse_dir_stat:{rmse_dir_stat}")
    print(f"rmse_hor:{rmse_hor}")
    print(f"rmse_hor_stat:{rmse_hor_stat}")
    print(f"rmse_uv:{rmse_uv}")
    print(f"rmse_uv_stat:{rmse_uv_stat}")

    # sampeln nach:
    # Windgeschwindigkeit
    # Windrichtung
    # Stabilitätsklasse
    # Tag/Nacht (Tagesgang)
    rmse_time_all = []
    rmse_time_all.append(rmse_time)
    for i in range(0, len(STATION_NAMES_JUL)):
        rmse_time_all.append(rmse_station_time[i])

    rmse_stations_all_hor = []
    rmse_stations_all_hor.append(rmse_hor)
    for i in range(0, len(STATION_NAMES_JUL)):
        rmse_stations_all_hor.append(rmse_hor_stat[i])

    mbe_station_all = []
    mbe_station_all.append(mbe_hor)
    for i in range(0, len(STATION_NAMES_JUL)):
        mbe_station_all.append(mbe_hor_stat[i])

    eval2 = np.concatenate((rmse_station_time, obs_hor, observations[:, 2, :], matched_hor, matched_dir),
                           axis=1)
    eval2 = eval2.reshape((rmse_station_time.shape[0], 5, 14))
    eval = rearrange(eval2, "d s h  -> d h s ")
    eval = eval.reshape(((rmse_station_time.shape[0] * 14), 5))
    d_eval = {"RMSE": eval[:, 0],
              "obs_speed": eval[:, 1],
              "obs_dir": eval[:, 2],
              "matched_speed": eval[:, 3],
              "matched_dir": eval[:, 4], }

    df_eval = pd.DataFrame(d_eval)
    high_speed = df_eval["obs_speed"] >= 4.0
    eval_speed_high = df_eval.loc[high_speed]
    wind_no = (df_eval["obs_dir"] >= 75) & (df_eval["obs_dir"] <= 125)
    eval_wind_no = df_eval.loc[wind_no]
    # plt.scatter(eval_wind_no["obs_speed"], eval_wind_no["RMSE"])
    # plt.show()

    # cmap = plt.get_cmap("Spectral")
    cmap = cm.get_cmap('viridis')

    for i in range(0, 14):
        plt.scatter(eval2[:, 2, i], eval2[:, 1, i], c=eval2[:, 0, i], cmap=cmap, alpha=0.5,
                    label=f"{STATION_NAMES_JUL[i]}")
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.set_ylabel('RMSE [m/s]', rotation=270, labelpad=15, fontdict={"size": 14})
        plt.xlim(0, 360)
        plt.xlabel("wind direction [°]", fontsize=15)
        plt.ylabel("wind speed [m/s]", fontsize=15)
        plt.legend()
        plt.savefig(f"plots/windspeeddirrmse_{STATION_NAMES_JUL[i]}.png", bbox_inches='tight')
        #plt.show()
        plt.close()

    for i in range(0, 13):
        plt.scatter(eval2[:, 2, i], eval2[:, 0, i], c=eval2[:, 1, i], cmap=cmap, alpha=0.5,
                    label=f"{STATION_NAMES_JUL[i]}")
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.set_ylabel("wind speed [m/s]", rotation=270, labelpad=15, fontdict={"size": 14})
        plt.xlim(0, 360)
        plt.xlabel("wind direction [°]", fontsize=15)
        plt.ylabel('RMSE [m/s]', fontsize=15)
        plt.legend()
        plt.savefig(f"plots/winddirrmsewindspeed_{STATION_NAMES_JUL[i]}", bbox_inches='tight')
        #plt.show()
        plt.close()


    """
    # Liste an Stationen:
    # Talstationen:
    tal = ["WWR", "GHW", "IUP", "LUBW", "CZE", "STB", "HP", "THB", "SB"]
    # Bergstatinen:
    berg = ["KOE", "STW", "KOS", "GAB", "PT"]
    # nahe Neckar:
    neck = ["SB", "THB", "STB", "GHW", "WWR", "IUP", "LUBW", "CZE"]

    for t in tal:
        plt.plot(rmse_station_time[:, STATIONS_INDEX[t]], label=f"{t}")
    plt.legend()
    #plt.show()

    for b in berg:
        plt.plot(rmse_station_time[:, STATIONS_INDEX[b]], label=f"{b}")
    plt.legend()
    #plt.show()

    for s in USED_STATIONS:
        plt.plot(rmse_station_time[:, s], label=f"{STATION_NAMES_JUL[s]}")
    plt.legend()
    #plt.show()

    wint = ["GHW", "SB", "PT", "KOS"]
    for w in wint:
        plt.plot(rmse_station_time[:, STATIONS_INDEX[w]], label=f"{w}")
    plt.legend()
    #plt.show()"""

    print("Done.")
