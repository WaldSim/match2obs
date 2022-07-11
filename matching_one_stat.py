import pickle
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from settings import START2, END2
from matchobs_functions import calculate_rmse, rmseuv, rmseuvstations
from readobs import load_observed_jul
from settings import STATION_NAMES_JUL

plt.style.use("bmh")
#####################################################################################
##################### Versuch nur für eine Station Daten darzustellen################
#####################################################################################


if __name__ == '__main__':
    # START2 = "2021-07-29 00:00:00"
    # END2 = "2021-08-08 23:00:00"
    obs = load_observed_jul(START2, END2)
    # winds = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE)
    # pkl_file = open('winds_cat1476_final.pkl', 'rb')
    pkl_file = open('winds_cat1008_final.pkl', 'rb')

    winds = pickle.load(pkl_file)
    pkl_file.close()
    METRIK = "l2vobsminmean"  # "test"
    N_CATALOGUE = 1424

    matched_winds = []
    RMSE_stations = []
    RMSE_all = []
    for k in range(len(STATION_NAMES_JUL)):
        diff = obs[:, 3:5, k, None] - winds[None, :-1, k,
                                      :]  # für u und v, differenz der beobachtung und simulation an ausgewählter Station k
        diff_hor_quad = (np.sum(diff ** 2, axis=1))  # differenz des horizontalen windvektors quadriert
        eta_h = np.argmin(diff_hor_quad, axis=1)  # minimalen Katalogneintrag ausgewählt für minimale differenz
        # print(f"{eta_h}for{k}")
        winds_matched = winds[:-1, k, eta_h]
        winds_hor = np.sqrt(
            np.sum(winds[:-1, k, :] ** 2, axis=0))  # horizontale windvektor der katalogeinträge an Station k
        winds_matched_hor = winds_hor[eta_h]
        obs_hor = np.sqrt(
            np.sum(obs[:, 3:5, k, None].astype(np.float32) ** 2, axis=1))  # horizontaler Windvektor beobachtung
        winds_u_v_matched = winds[:, k, eta_h]  # u, v an Messstation mit eta_h
        diff_station = np.sqrt(
            diff_hor_quad[:, eta_h].astype(np.float32))  # differenz horizontaler windvektor an station mit eta_h
        matched_winds2 = winds[:-1, :, eta_h]
        matched_winds3 = rearrange(matched_winds2, "d s h  -> h d s ")

        rmse_uv = rmseuv(obs, matched_winds3)
        RMSE_all.append(rmse_uv)
        stations_rmse = rmseuvstations(obs, matched_winds3)
        RMSE_stations.append(stations_rmse)
        # rmse für alle Stationen berechnen
        winds_hor_all = np.sqrt(np.sum(winds[:-1, :, :] ** 2, axis=0))
        winds_matched_all = winds_hor_all[:, eta_h]
        diff_match_all = obs_hor[:, :] - winds_matched_all.T
        matched_winds.append(matched_winds3)

    matched_winds = np.asarray(matched_winds)
    matched_winds_hor = np.sqrt(np.sum(matched_winds ** 2, axis=2))
    ncols = 2
    nrows = 7
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 15), sharex=True, sharey=True)
    k = 0
    for i in range(nrows):

        for j in range(ncols):
            try:

                axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")
                axes[i, j].plot(obs[700:800, 1, k], label="wind speed observed [m/s]", marker="", color="darkblue",
                                linestyle="--")

                axes[i, j].plot(matched_winds_hor[k, 700:800, k],
                                label=f"wind speed [m/s] only matched on station: {STATION_NAMES_JUL[k]}", marker="",
                                linestyle="-",
                                color="orange", alpha=0.8)
                axes[i, j].set_title(f"{STATION_NAMES_JUL[k]}")
                axes[i, j].legend(loc="upper right")
                axes[i, j].set_title(f"matched and observed wind for measuring station {STATION_NAMES_JUL[k]}",
                                     fontsize=24)
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
    plt.savefig(f"plots/windtimeseries_one_matched.png", bbox_inches='tight')
    plt.show()

    print("done.")
