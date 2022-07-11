import pandas as pd
from matplotlib import pyplot as plt

from matchobs_functions import matchgralorgramm_jul, obshor, matchedhor, obsdir, \
    matcheddir, \
    winkdiff, mbedir, mbedirstations, mbehor, mbehorstations, rmsedir, rmsedirstations, rmseuv, rmseuvstations, \
    rmsehor, rmsehorstations, corrstations, corr

from matchobs_functions import matching_winds
from plotting import  lin_reg_stations
from readobs import load_observed_jul

from settings import USED_STATIONS, STATION_NAMES_JUL, START2, END2, LOAD_CATALOGUE, METRIK, USE_SMOOTHING, \
    GRAL, USE_WEIGHTING, N_CATALOGUE_USED, STATION_COLORS

if __name__ == '__main__':
    # print all the settings
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
    print("RMSE:")
    print(f"rmse_dir:{rmse_dir}")
    print(f"rmse_dir_stat:{rmse_dir_stat}")
    print(f"rmse_hor:{rmse_hor}")
    print(f"rmse_hor_stat:{rmse_hor_stat}")
    print(f"rmse_uv:{rmse_uv}")
    print(f"rmse_uv_stat:{rmse_uv_stat}")

    rmse_stations_all = []
    rmse_stations_all.append(rmse_uv)
    for i in range(0, len(STATION_NAMES_JUL)):
        rmse_stations_all.append(rmse_uv_stat[i])

    rmse_stations_all_hor = []
    rmse_stations_all_hor.append(rmse_hor)
    for i in range(0, len(STATION_NAMES_JUL)):
        rmse_stations_all_hor.append(rmse_hor_stat[i])

    mbe_station_all_hor = []
    mbe_station_all_hor.append(mbe_hor)
    for i in range(0, len(STATION_NAMES_JUL)):
        mbe_station_all_hor.append(mbe_hor_stat[i])

    d = {"RMSE": rmse_stations_all,
         "RMSE_hor": rmse_stations_all_hor,
         "MBE_hor": mbe_station_all_hor,
         }
    df = pd.DataFrame(d, index=["all", "GHW", "HP", 'KOS', 'STW', 'PT', 'SB',
                                'STB', 'THB', 'WWR', 'KOE', "CZE", "GAB", "LUBW", "IUP"]
                      )

    fig, ax = plt.subplots()
    for i in range(0, 14):
        plt.scatter(mbe_hor_stat[i], rmse_hor_stat[i],
                    alpha=0.8, label=(f"{STATION_NAMES_JUL[i]}"), color=f"{STATION_COLORS[i]}")
        # plt.legend()
    plt.xlabel("MB (m/s)", fontsize=16)
    plt.ylabel("RMSE (m/s)", fontsize=16)
    ax.hlines(y=2.5, xmin=-1.5, xmax=1.5, linewidth=1, color='k')
    ax.hlines(y=2.0, xmin=-0.5, xmax=0.5, linewidth=1, color='gray')
    ax.hlines(y=0, xmin=-1.5, xmax=1.5, linewidth=1, color='k')
    ax.vlines(x=0.5, ymin=0, ymax=2.0, linewidth=1, color='gray')
    ax.vlines(x=1.5, ymin=0, ymax=2.5, linewidth=1, color='k')
    ax.vlines(x=-0.5, ymin=0, ymax=2.0, linewidth=1, color='gray')
    ax.vlines(x=-1.5, ymin=0, ymax=2.5, linewidth=1, color='k')
    plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
    plt.tight_layout()
    plt.title("")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.savefig("plots/rmse_mb_station_speed.png", bbox_inches="tight")
    plt.show()
    plt.close()

    # plt rmse und mean bias:

    fig, ax = plt.subplots()
    for i in range(0, 14):
        plt.scatter(mbe_hor_stat[i], rmse_hor_stat[i],
                    alpha=0.8, label=(f"{STATION_NAMES_JUL[i]}"), color=f"{STATION_COLORS[i]}")
    # plt.legend()
    plt.xlabel("MB (m/s)", fontsize=16)
    plt.ylabel("RMSE (m/s)", fontsize=16)
    ax.hlines(y=2.5, xmin=-1.5, xmax=1.5, linewidth=1, color='k')
    ax.hlines(y=2.0, xmin=-0.5, xmax=0.5, linewidth=1, color='darkgray')
    ax.hlines(y=0, xmin=-1.5, xmax=1.5, linewidth=1, color='k')
    ax.vlines(x=0.5, ymin=0, ymax=2.0, linewidth=1, color='darkgray')
    ax.vlines(x=1.5, ymin=0, ymax=2.5, linewidth=1, color='k')
    ax.vlines(x=-0.5, ymin=0, ymax=2.0, linewidth=1, color='darkgray')
    ax.vlines(x=-1.5, ymin=0, ymax=2.5, linewidth=1, color='k')
    plt.xlim(-2.0, 2.0)
    plt.ylim(0, 3.0)
    plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
    plt.tight_layout()
    plt.title("")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=7)
    plt.savefig("plots/rmse_mb_station_speed.png", bbox_inches="tight")
    plt.show()
    plt.close()


    # linear regression:

    fig, ax = plt.subplots(figsize=(14, 10))
    # ax.yaxis.grid(color="gray", alpha=0.5)
    for i in range(0, 14):
        lin_reg_stations(obs_hor[:, i], matched_hor[:, i],
                         ax=ax, color=f"{STATION_COLORS[i]}", alpha=0.4,
                         label=f"lin fit {STATION_NAMES_JUL[i]}", label_orig=f"data {STATION_NAMES_JUL[i]}")

        plt.ylabel("matched wind speed [m/s]", fontsize=15)
        plt.xlabel("observed wind speed [m/s]", fontsize=15)
        plt.show()
        # plt.savefig(f"plots/linregplots{STATION_NAMES_JUL[i]}.png")


    fig, ax = plt.subplots(figsize=(14, 10))
    ax.yaxis.grid(color="gray", linestyle="dashed")
    obs = obs_hor.flatten()
    mat = matched_hor.flatten()
    lin_reg_stations(obs, mat,
                     ax=ax, color="b", alpha=0.05, label=f"lin fit all stations",
                     label_orig=f"data all stations")
    # plt.xlim(0, 10)
    plt.show()
    plt.close()


    # korrelationskoeffizient
    corr_stat = corrstations(obs_hor, matched_hor)
    cor = corr(obs_hor, matched_hor)

print("done.")
