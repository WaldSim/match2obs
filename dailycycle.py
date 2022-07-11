import pickle
import numpy as np
import pandas as pd
from einops import rearrange
from matplotlib import pyplot as plt
from windrose import WindroseAxes
from matchobs_functions import matching_winds, calculate_rmse, matchgralorgramm_jul, obshor, matchedhor, obsdir, \
    matcheddir, \
    winkdiff, mbedir, mbedirstations, mbehor, mbehorstations, rmsedir, rmsedirstations, rmseuv, rmseuvstations, \
    rmsehor, rmsehorstations, corrstations, corr, rmsestattime, mbestattime
from settings import USED_STATIONS, STATION_NAMES_JUL, START2, END2, LOAD_CATALOGUE, METRIK, USE_SMOOTHING, \
    GRAL, USE_WEIGHTING, N_CATALOGUE_USED
from readobs import load_observed_jul

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
    print("RMSE:")
    print(f"rmse_dir:{rmse_dir}")
    print(f"rmse_dir_stat:{rmse_dir_stat}")
    print(f"rmse_hor:{rmse_hor}")
    print(f"rmse_hor_stat:{rmse_hor_stat}")
    print(f"rmse_uv:{rmse_uv}")
    print(f"rmse_uv_stat:{rmse_uv_stat}")



    cmap = plt.get_cmap("Spectral")

    myDateRange = pd.date_range(START2, periods=obs_hor.shape[0], freq="H")
    df = ["df0", "df1", "df2", "df3", "df4", "df5", "df6", "df7", "df8", "df9",
          "df10", "df11", "df12", "df13"]

    data_all = []
    data_all_std = []
    daily_all_dir = []
    for s in range(0, 14):
        d = {"date": myDateRange,
             "RMSE": rmse_station_time[:, s],
             "obs_speed": obs_hor[:, s],
             "obs_dir": observations[:, 2, s],
             "matched_speed": matched_hor[:, s],
             "matched_dir": matched_dir[:, s],
             "mb": mbe_station_time[:, s]
             }
        df[s] = pd.DataFrame(d)
        df[s].head()

        conds = []
        time = {0: '00:00:00', 1: '01:00:00', 2: '02:00:00', 3: '03:00:00', 4: '04:00:00', 5: '05:00:00', 6: '06:00:00',
                7: '07:00:00', 8: '09:00:00', 9: '09:00:00', 10: '10:00:00', 11: '11:00:00', 12: '12:00:00',
                13: '13:00:00',
                14: '14:00:00', 15: '15:00:00', 16: '16:00:00', 17: '17:00:00', 18: '18:00:00', 19: '19:00:00',
                20: '20:00:00', 21: '21:00:00', 22: '22:00:00', 23: '23:00:00'}

        for i in range(0, 24):
            cond = (df[0]['date'].dt.strftime('%H:%M:%S') == time[i])
            conds.append(cond)

        data_dir = []
        data_speed = []
        df_hour = []
        df_hour_std = []
        df_hour_dir = []
        for i in range(0, 24):
            mask_hour = conds[i]
            df_i = df[s]
            # del df_i["date"]
            data = df_i.loc[mask_hour]
            dir = data["obs_dir"]
            speed = data["obs_speed"]
            # data_dir.append(dir)
            dir_arr = np.asarray(dir)
            data_dir.append(dir_arr)
            data_speed.append(speed)
            speed_arr = np.asarray(speed)
            mean = data.mean()
            std = data.std()

            df_hour.append(mean)
            df_hour_std.append(std[1:7]) # wenn über terminal ausgeführt wird std[0:7]: 1 -> 0
            # df_hour.dir.append(dir_arr)

        daily = np.asarray(df_hour)
        daily_std = np.asarray(df_hour_std)
        daily_dir = np.asarray(data_dir)
        data_all.append(daily)
        data_all_std.append(daily_std)
    daily_all_dir.append(daily_dir)
    data_all = np.asarray(data_all)
    data_all_std = np.asarray(data_all_std, dtype=float)
    daily_all_dir = np.asarray(daily_all_dir)

    ncols = 2
    nrows = 7
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15), sharex=True, sharey=True)
    fig.suptitle(f"Daily cycle of matched wind speed [m/s] and observed wind speed [m/s]", fontsize=26)
    k = 0
    for i in range(nrows):

        for j in range(ncols):
            try:
                x = np.arange(0, 24)
                axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")
                axes[i, j].plot(x, data_all[k, :, 3], label=f"matched speed", c="seagreen")
                axes[i, j].plot(x, data_all[k, :, 1], label=f"wind speed", c="purple")

                std1 = data_all[k, :, 3] + data_all_std[k, :, 3]
                std2 = data_all[k, :, 3] - data_all_std[k, :, 3]
                axes[i, j].plot(x, std1, c="seagreen", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 3] + data_all_std[k, :, 3], data_all[k, :, 3], alpha=0.05,
                                        color="seagreen")
                axes[i, j].plot(x, std2, c="seagreen", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 3] - data_all_std[k, :, 3], data_all[k, :, 3], alpha=0.05,
                                        color="seagreen")

                std1 = data_all[k, :, 1] + data_all_std[k, :, 1]
                std2 = data_all[k, :, 1] - data_all_std[k, :, 1]
                axes[i, j].plot(x, std1, c="purple", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 1] + data_all_std[k, :, 1], data_all[k, :, 1], alpha=0.05,
                                        color="purple")
                axes[i, j].plot(x, std2, c="purple", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 1] - data_all_std[k, :, 1], data_all[k, :, 1], alpha=0.05,
                                        color="purple")
                axes[i, j].set_title(f"{STATION_NAMES_JUL[k]}")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=7)
                fig.text(0.5, 0.07, 'time [h]', ha='center', fontsize=30)
                fig.text(0.07, 0.5, 'wind speed [m/s]', va='center', rotation='vertical', fontsize=30)
                k += 1
                plt.subplots_adjust(left=0.11,
                                    bottom=0.15,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.08,
                                    hspace=0.35)
            except IndexError:
                axes[i, j].axis("off")
    plt.savefig(f"plots/dailycycle_speed_obs_matched.png", bbox_inches='tight')
    plt.show()
    plt.close()

    ncols = 2
    nrows = 7
    fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15), sharex=True, sharey=True)
    fig2.suptitle(f"Daily cycle of RMSE [m/s] and wind speed [m/s]", fontsize=26)
    k = 0
    for i in range(nrows):

        for j in range(ncols):
            try:
                x = np.arange(0, 24)
                axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")
                axes[i, j].plot(x, data_all[k, :, 0], label=f"RMSE", c="seagreen")
                axes[i, j].plot(x, data_all[k, :, 1], label=f"wind speed", c="purple")

                std1 = data_all[k, :, 0] + data_all_std[k, :, 0]
                std2 = data_all[k, :, 0] - data_all_std[k, :, 0]
                axes[i, j].plot(x, std1, c="seagreen", alpha=0.3, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 0] + data_all_std[k, :, 0], data_all[k, :, 0], alpha=0.1,
                                        color="seagreen")
                axes[i, j].plot(x, std2, c="seagreen", alpha=0.5, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 0] - data_all_std[k, :, 0], data_all[k, :, 0], alpha=0.1,
                                        color="seagreen")
                std1 = data_all[k, :, 1] + data_all_std[k, :, 1]
                std2 = data_all[k, :, 1] - data_all_std[k, :, 1]
                axes[i, j].plot(x, std1, c="purple", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 1] + data_all_std[k, :, 1], data_all[k, :, 1], alpha=0.05,
                                        color="purple")
                axes[i, j].plot(x, std2, c="purple", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 1] - data_all_std[k, :, 1], data_all[k, :, 1], alpha=0.05,
                                        color="purple")
                # axes[i, j].plot(data_all[k, :, 3], label=f"sim wind speed")
                axes[i, j].set_title(f"{STATION_NAMES_JUL[k]}")
                # axes[i, j].legend(loc="upper right")
                fig2.text(0.5, 0.07, 'time [h]', ha='center', fontsize=30)
                fig2.text(0.07, 0.5, 'wind speed [m/s]', va='center', rotation='vertical', fontsize=30)
                k += 1
                plt.subplots_adjust(left=0.11,
                                    bottom=0.15,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.08,
                                    hspace=0.35)
            except IndexError:
                axes[i, j].axis("off")
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=7)
    plt.savefig(f"plots/dailycycle_speed_rmse.png", bbox_inches='tight')
    plt.show()
    plt.close()

    ncols = 2
    nrows = 7
    fig3, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15), sharex=True, sharey=True)
    fig.suptitle(f"Daily cycle of MB [m/s] and wind speed [m/s]", fontsize=26)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            try:
                x = np.arange(0, 24)
                axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")
                axes[i, j].plot(x, data_all[k, :, 5], label=f"MB", c="seagreen")
                axes[i, j].plot(x, data_all[k, :, 1], label=f"wind speed", c="purple")

                std1 = data_all[k, :, 5] + data_all_std[k, :, 5]
                std2 = data_all[k, :, 5] - data_all_std[k, :, 5]
                axes[i, j].plot(x, std1, c="seagreen", alpha=0.3, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 0] + data_all_std[k, :, 5], data_all[k, :, 0], alpha=0.1,
                                        color="seagreen")
                axes[i, j].plot(x, std2, c="seagreen", alpha=0.5, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 5] - data_all_std[k, :, 5], data_all[k, :, 0], alpha=0.1,
                                        color="seagreen")
                std1 = data_all[k, :, 1] + data_all_std[k, :, 1]
                std2 = data_all[k, :, 1] - data_all_std[k, :, 1]
                axes[i, j].plot(x, std1, c="purple", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 1] + data_all_std[k, :, 1], data_all[k, :, 1], alpha=0.05,
                                        color="purple")
                axes[i, j].plot(x, std2, c="purple", alpha=0.1, lw=np.pi / 2)
                axes[i, j].fill_between(x, data_all[k, :, 1] - data_all_std[k, :, 1], data_all[k, :, 1], alpha=0.05,
                                        color="purple")
                # axes[i, j].plot(data_all[k, :, 3], label=f"sim wind speed")
                axes[i, j].set_title(f"{STATION_NAMES_JUL[k]}")
                # axes[i, j].legend(loc="upper right")
                fig3.text(0.5, 0.07, 'time [h]', ha='center', fontsize=30)
                fig3.text(0.07, 0.5, 'wind speed [m/s]', va='center', rotation='vertical', fontsize=30)
                k += 1
                plt.subplots_adjust(left=0.11,
                                    bottom=0.15,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.08,
                                    hspace=0.35)
            except IndexError:
                axes[i, j].axis("off")
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=7)
    plt.savefig(f"plots/dailycycle_speed_mb.png", bbox_inches='tight')
    plt.show()
    plt.close()

    print("Done.")
