import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from settings import USED_STATIONS, STATION_NAMES_JUL, START2, END2, LOAD_CATALOGUE, METRIK, USE_SMOOTHING, \
    GRAL, USE_WEIGHTING, N_CATALOGUE_USED, STATIONDATA_JUL, STATION_COLORS
from matchobs_functions import matching_winds, matchgralorgramm_jul, \
    obshor, matchedhor, obsdir, matcheddir, \
    winkdiff, mbedir, mbedirstations, mbehor, mbehorstations, rmsedir, rmsedirstations, rmseuv, rmseuvstations, \
    rmsehor, rmsehorstations, rmsestattime, mbestattime
from readobs import load_observed_jul
plt.style.use('bmh')

if __name__ == "__main__":

    Stations = pd.DataFrame(STATIONDATA_JUL,
                            columns=["GHW",
                                     "HP",
                                     "KOS",
                                     "STW",
                                     "PT",
                                     "SB",
                                     "STB",
                                     "THB",
                                     "WWR",
                                     "KOE",
                                     "CZE",
                                     "GAB",
                                     "LUBW",
                                     "IUP"
                                     ])
    stations = Stations.to_numpy()
    STATION_HEIGHTS = stations[6] + stations[4]
    STATION_HEIGHTS_OVER_GROUND = stations[5]  # with buildings
    STATION_HEIGHTS_OVER_OVER_GROUND2 = stations[4]
    ### matching
    # load: Grid, Catalogue, Observations
    observations = load_observed_jul(START2, END2)
    # für GRAL
    catalogue = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED)
    obs_hor = obshor(observations)
    matched_winds = matching_winds(observations, catalogue, len(USED_STATIONS),
                                   USED_STATIONS, METRIK, USE_WEIGHTING,
                                   USE_SMOOTHING, N_CATALOGUE_USED)
    matched_hor = matchedhor(matched_winds)
    obs_dir = obsdir(observations)
    matched_dir = matcheddir(matched_winds)

    # winkel diff
    wink_diff = winkdiff(obs_dir, matched_dir)
    obs_hor_mean = np.nanmean(obs_hor, axis=0)

    # mb
    mbe_dir = mbedir(wink_diff)
    mbe_dir_stat = mbedirstations(wink_diff)
    mbe_hor = mbehor(obs_hor, matched_hor)
    mbe_station_hor = mbehorstations(obs_hor, matched_hor)
    mbe_station_time = mbestattime(obs_hor, matched_hor)
    print("MBE:")
    print(f"mbe_dir:{mbe_dir}")
    print(f"mbe_dir_stat:{mbe_dir_stat}")
    print(f"mbe_hor:{mbe_hor}")
    print(f"mbe_hor_stat:{mbe_station_hor}")
    # RMSE
    rmse_dir = rmsedir(wink_diff)
    rmse_station_dir = rmsedirstations(wink_diff)
    rmse_uv = rmseuv(observations, matched_winds)
    rmse_uv_stat = rmseuvstations(observations, matched_winds)
    rmse_hor = rmsehor(obs_hor, matched_hor)
    rmse_station_hor = rmsehorstations(obs_hor, matched_hor)
    rmse_station_time = rmsestattime(observations, matched_winds)
    print("RMSE:")
    print(f"rmse_dir:{rmse_dir}")
    print(f"rmse_dir_stat:{rmse_station_dir}")
    print(f"rmse_hor:{rmse_hor}")
    print(f"rmse_hor_stat:{rmse_station_hor}")
    print(f"rmse_uv:{rmse_uv}")
    print(f"rmse_uv_stat:{rmse_uv_stat}")

    RMSE_factor = rmse_uv / rmse_uv_stat
    print(RMSE_factor)

    n = ["GHW", "HP", "KOS", "STW", "PT", "SB", "STB", "THB", "WWR", "KOE", "CZE", "GAB", "LUBW", "IUP"]

    STATION_COLORS_OUTSIDE = {0: "seagreen", 1: "purple", 2: "seagreen", 3: "seagreen", 4: "seagreen", 5: "seagreen",
                              6: "purple", 7: "purple",
                              8: "seagreen", 9: "seagreen", 10: "purple", 11: "seagreen", 12: "purple", 13: "purple"}

    fig, ax = plt.subplots()
    for i, txt in enumerate(n):
        ax.scatter(STATION_HEIGHTS_OVER_GROUND[i], mbe_station_hor[i], color=f"{STATION_COLORS_OUTSIDE[i]}")
        ax.annotate(txt, (STATION_HEIGHTS_OVER_GROUND[i], mbe_station_hor[i]), color=f"{STATION_COLORS_OUTSIDE[i]}")
    plt.xlabel("height of the station over ground [m.a.s.l]", fontsize=16)
    plt.ylabel("MB [m/s]", fontsize=16)
    plt.title("MB vs. station height over ground", fontsize=18)
    plt.savefig("plots/mb_station_height_over_ground_inside.png")
    plt.show()

    fig2, ax = plt.subplots()
    ax.scatter(STATION_HEIGHTS, mbe_station_hor)
    for i, txt in enumerate(n):
        ax.annotate(txt, (STATION_HEIGHTS[i], mbe_station_hor[i]))
    plt.xlabel("height of the station[m.a.s.l]", fontsize=16)
    plt.ylabel("MB [m/s]", fontsize=16)
    plt.title("MB vs. station height ", fontsize=18)
    plt.savefig("plots/mb_station_height.png")
    plt.show()

    fig3, ax = plt.subplots()
    ax.scatter(STATION_HEIGHTS_OVER_GROUND, rmse_station_dir)
    for i, txt in enumerate(n):
        ax.annotate(txt, (STATION_HEIGHTS_OVER_GROUND[i], rmse_station_dir[i]))
    plt.xlabel("height of the station over ground [m.a.s.l]", fontsize=16)
    plt.ylabel("RMSE [°]", fontsize=16)
    plt.title("RMSE vs. station height over ground ", fontsize=18)
    plt.savefig("plots/mb_station_height.png")
    plt.show()

    k = ["HP", "LUBW", "STB", " CZE", "IUP"]

    fig4, ax = plt.subplots()
    ax.scatter(STATION_HEIGHTS, rmse_uv_stat)
    for i, txt in enumerate(n):
        ax.annotate(txt, (STATION_HEIGHTS[i], rmse_uv_stat[i]))
    plt.xlabel("height of the station [m.a.s.l]", fontsize=16)
    plt.ylabel("RMSE [m/s]", fontsize=16)
    plt.title("RMSE vs. station height", fontsize=18)
    plt.savefig("plots/station_heights_rmse.png")
    plt.show()

    fig5, ax = plt.subplots()
    ax.scatter(mbe_station_hor, STATION_HEIGHTS)
    for i, txt in enumerate(n):
        ax.annotate(txt, (mbe_station_hor[i], STATION_HEIGHTS[i]))
    plt.xlabel("mean bias [m/s]", fontsize=16)
    plt.ylabel("height [m]", fontsize=16)
    plt.title("mbe  vs. station height")
    #plt.savefig("plots/mbehorstationheight.png")
    plt.show()


    fig6, ax = plt.subplots()
    for i, txt in enumerate(n):
        ax.scatter(STATION_HEIGHTS[i], mbe_station_hor[i], color=f"{STATION_COLORS_OUTSIDE[i]}")
        ax.annotate(txt, (STATION_HEIGHTS[i], mbe_station_hor[i]), color=f"{STATION_COLORS_OUTSIDE[i]}")
    plt.xlabel("height [m]", fontsize=16)
    plt.ylabel("mean bias hor [m/s]", fontsize=16)
    plt.title("mbe hor vs. station height")
    #plt.savefig("plots/MBstationheight_outsidecity.png")
    plt.show()


    fig7, ax = plt.subplots()
    ax.scatter(mbe_station_hor, STATION_HEIGHTS_OVER_GROUND)
    for i, txt in enumerate(n):
        ax.annotate(txt, (mbe_station_hor[i], STATION_HEIGHTS_OVER_GROUND[i]))
    plt.xlabel("mbe hor", fontsize=16)
    plt.ylabel("height over ground [m]", fontsize=16)
    # plt.savefig("plots/mbehorstatonoverground.png")
    plt.show()


    fig8, ax = plt.subplots()
    ax.scatter(obs_hor_mean, rmse_uv_stat)
    for i, txt in enumerate(n):
        ax.annotate(txt, (obs_hor_mean[i], rmse_uv_stat[i]))
    plt.ylabel("RMSE [m/s]", fontsize=16)
    plt.xlabel("mean wind speed [m/s]", fontsize=16)
    plt.title("RMSE  vs. mean wind speed of the station", fontsize=18)
    # ax.legend(loc='upper center', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.savefig("plots/obshormeanrmseuv.png")
    plt.show()


    fig9, ax = plt.subplots()
    ax.scatter(mbe_station_hor, obs_hor_mean, )
    for i, txt in enumerate(n):
        ax.annotate(txt, (mbe_station_hor[i], obs_hor_mean[i]))
    plt.xlabel("mean bias [m/s]", fontsize=16)
    plt.ylabel("mean wind speed [m/s]", fontsize=16)
    plt.title("MB  vs. mean wind speed [m/s]")
    #plt.savefig("plots/mbestatobshormean.png")
    plt.show()


    fig10, ax = plt.subplots()
    for i in range(0, 14):
        ax.scatter(rmse_station_time[:, i], obs_hor[:, i], color=STATION_COLORS[i], label=f"{STATION_NAMES_JUL[i]}",
                   alpha=0.3)
    plt.xlabel("RMSE [m/s]", fontsize=16)
    plt.ylabel("wind speed [m/s]", fontsize=16)
    plt.title("wind speed vs.RMSE")
    plt.legend()
    # plt.savefig("plots/rmsetimeobshor.png")
    plt.show()


    fig11, ax = plt.subplots()
    for i in range(0, 14):
        ax.scatter(mbe_station_time[:, i], obs_hor[:, i], color=STATION_COLORS[i], label=f"{STATION_NAMES_JUL[i]}",
                   alpha=0.3)
    plt.xlabel("MB [m/s]", fontsize=16)
    plt.ylabel("wind speed [m/s]", fontsize=16)
    plt.title("wind speed vs.MB hor")
    plt.legend()
    #plt.savefig("plots/mbtimeobshor.png")
    plt.show()


    # form west to east
    fig12, ax = plt.subplots()
    ax.scatter(stations[2], rmse_uv_stat)
    for i, txt in enumerate(n):
        ax.annotate(txt, (stations[2][i], rmse_uv_stat[i]))
    plt.xlabel("West-East", fontsize=16)
    plt.ylabel("RMSE [m/s]", fontsize=16)
    plt.title("west_east vs RMSE", fontsize=18)
    # plt.savefig("plots/west_east_rmse.png")
    # plt.show()


    # form west to east
    fig13, ax = plt.subplots()
    ax.scatter(stations[2], mbe_station_hor)
    for i, txt in enumerate(n):
        ax.annotate(txt, (stations[2][i], mbe_station_hor[i]))
    plt.xlabel("West-East", fontsize=16)
    plt.ylabel("mbe hor [m/s]", fontsize=16)
    plt.title("west_east vs mbe hor")
    # plt.savefig("plots/west_east_mbe.png")
    plt.show()


    # form south to north
    fig14, ax = plt.subplots()
    ax.scatter(rmse_uv_stat, stations[3])
    for i, txt in enumerate(n):
        ax.annotate(txt, (rmse_uv_stat[i], stations[3][i]))
    plt.xlabel("RMSE [m/s]", fontsize=16)
    plt.ylabel("South- North", fontsize=16)
    plt.title("south-north vs RMSE", fontsize=18)
    plt.legend()
    # plt.savefig("plots/south_north_rmse.png")
    # plt.show()


    # form south to north
    fig15, ax = plt.subplots()
    ax.scatter(mbe_station_hor, stations[3])
    for i, txt in enumerate(n):
        ax.annotate(txt, (mbe_station_hor[i], stations[3][i]))
    plt.xlabel("mean bias hor [m/s]", fontsize=16)
    plt.ylabel("South- North", fontsize=16)
    plt.title("south-north vs mean bias hor")
    # plt.savefig("plots/south_north_mbe.png")
    plt.show()


    cmap = cm.get_cmap('viridis')

    fig16, ax = plt.subplots()
    ax.scatter(stations[2], stations[3], c=rmse_uv_stat, cmap=cmap)
    norm = plt.Normalize(np.min(rmse_uv_stat), np.max(rmse_uv_stat))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm)
    cbar.ax.set_title("rmse")
    for i, txt in enumerate(n):
        ax.annotate(txt, (stations[2][i], stations[3][i]))
    plt.xlabel("x-coordinate", fontsize=16)
    plt.ylabel("y-coordinate", fontsize=16)
    plt.title("RMSE in grid")
    # plt.savefig("plots/rmse_gitter.png")
    plt.show()


    fig17, ax = plt.subplots()
    ax.scatter(stations[2], stations[3], c=mbe_station_hor, cmap=cmap)
    norm = plt.Normalize(np.min(mbe_station_hor), np.max(mbe_station_hor))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm)
    cbar.ax.set_title("mbe hor [m/s]")
    for i, txt in enumerate(n):
        ax.annotate(txt, (stations[2][i], stations[3][i]))
    plt.xlabel("x-coordinate", fontsize=16)
    plt.ylabel("y-coordinate", fontsize=16)
    plt.title("mbe hor in grid")
    # plt.savefig("plots/mbe_gitter.png")
    plt.show()


    print("done.")
