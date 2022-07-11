import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from einops import rearrange
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from windrose import WindroseAxes
from matchobs_functions import choose_metric, smooth, min_score, matchedwinds, matchgralorgramm_jul
from math_functions import arctan, filter_nans
from readobs import load_observed_jul
from settings import END2, START2, STATION_NAMES_JUL, USED_STATIONS, METRIK, USE_WEIGHTING, USE_SMOOTHING, \
    N_CATALOGUE_USED, LOAD_CATALOGUE

def windrose(direction, speed, title):
    ax = WindroseAxes.from_ax()
    bins_range = np.arange(0, 8, 1)
    ax.bar(direction, speed, normed=True, bins=bins_range, opening=0.8, edgecolor="white")
    ax.set_title(title, fontsize=20)
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', ], fontsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=4)
    # ax.set_legend(fontsize=18)


STATIONDATA_lat_lon = {"IUP": [3476462, 5475506, 145, 49.416016, 8.67448],
                       "LUBW": [3476616.415, 5475898.736, 122, 49.42453, 8.67672],
                       "GAB": [3481423.178, 5469356.975, 331, 49.360900, 8.743220],
                       "CZE": [3476559.336, 5473792.047, 150, 49.400609, 8.676002],
                       "STW": [3480078.171, 5473526.270, 581, 49.39956, 8.72477],
                       "WWR": [3472735.935, 5476113.269, 116, 49.421320, 8.623167],
                       "STB": [3477329.729, 5474470.342, 134, 49.40906, 8.6864],
                       "GHW": [3472888.990, 5475770.394, 109, 49.418244, 8.625300],
                       "HP": [3477413.573, 5470824.362, 120, 49.373958, 8.687940],
                       "KOS": [3480342.295, 5474125.002, 565, 49.403737, 8.728104],
                       "PT": [3482028.905, 5479563.560, 353, 49.452689, 8.751095],
                       "SB": [3483945.193, 5474103.973, 119, 49.403654, 8.777746],
                       "THB": [3477747.053, 5475151.064, 117, 49.412873, 8.692289],
                       "KOE": [3481930.604, 5476005.216, 230, 49.420692, 8.749902],
                       }



def wrax(Main_ax, lon, lat):
    wrax = inset_axes(Main_ax,
                      width=1,
                      height=1,
                      loc="center",
                      bbox_to_anchor=(lon, lat),
                      bbox_transform=Main_ax.transData,
                      axes_class=windrose.WindroseAxes,
                      )
    return wrax


if __name__ == '__main__':

    observations = load_observed_jul(START2, END2)
    # GRAMM
    GRAL = False
    catalogue_gramm = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED)
    # GRAL
    pkl_file = open('winds_cat1476_final.pkl', 'rb')
    catalogue = pickle.load(pkl_file)
    pkl_file.close()

    Minlon, Maxlon, Minlat, Maxlat = (8.60, 8.785, 49.35, 49.453)
    Stations = pd.DataFrame(STATIONDATA_lat_lon,
                            columns=['GHW',
                                     'HP',
                                     'KOS',
                                     'STW',
                                     'PT',
                                     'SB',
                                     'STB',
                                     'THB',
                                     'WWR',
                                     'KOE',
                                     'CZE',
                                     'GAB',
                                     'LUBW',
                                     'IUP',
                                     ]
                            )

    # calculate matched winds
    eta_matrix = choose_metric(observations, catalogue, USED_STATIONS, METRIK, USE_WEIGHTING)
    score = smooth(eta_matrix, USE_SMOOTHING,
                   N_CATALOGUE_USED)  # , calculate_weight(observations,USED_STATIONS,USE_WEIGHTING))
    etah = min_score(score)
    # Calculate matched winds
    matched_winds = matchedwinds(catalogue, etah)  # , USED_STATIONS)
    matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
    catalogue_hor = np.sqrt(np.sum((catalogue[:-1, :, :].astype(np.float32)) ** 2, axis=0))
    matched_hor = catalogue_hor[:, etah]
    matched_hor2 = rearrange(matched_hor, "s h  -> h s ")
    matched_winds_hor = np.sqrt(np.sum(matched_winds2.astype(np.float32)[:, :, :] ** 2, axis=1))
    matched_winkel = arctan(matched_winds2[:, 1, :], matched_winds2[:, 0, :])

    for i in range(0, 14):
        ax = windrose(matched_winkel[:, i], matched_winds_hor[:, i], f"{STATION_NAMES_JUL[i]} simulated GRAL")
        plt.savefig(f"plots/windrose_matched_GRAL_{STATION_NAMES_JUL[i]}")
        plt.close()
        plt.show()

    for i in range(0, 14):
        observations_filtered = filter_nans(observations[:, 1, :].astype(float))
        ax = windrose(observations[:, 2, i], observations_filtered[:, i], f"{STATION_NAMES_JUL[i]} observed")
        plt.savefig(f"plots/windrose_observed_{STATION_NAMES_JUL[i]}")
        plt.show()
        plt.close()

    eta_matrix_gramm = choose_metric(observations, catalogue_gramm, USED_STATIONS, METRIK, USE_WEIGHTING)
    score_gramm = smooth(eta_matrix_gramm, USE_SMOOTHING,
                         N_CATALOGUE_USED)  # , calculate_weight(observations,USED_STATIONS,USE_WEIGHTING))
    etah_gramm = min_score(score_gramm)
    # Calculate matched winds
    matched_winds_gramm = matchedwinds(catalogue_gramm, etah_gramm)  # , USED_STATIONS)
    matched_winds2_gramm = rearrange(matched_winds_gramm, "d s h  -> h d s ")
    catalogue_hor_gramm = np.sqrt(np.sum((catalogue_gramm[:-1, :, :].astype(np.float32)) ** 2, axis=0))
    matched_hor_gramm = catalogue_hor_gramm[:, etah_gramm]
    matched_hor2_gramm = rearrange(matched_hor_gramm, "s h  -> h s ")
    matched_winds_hor_gramm = np.sqrt(np.sum(matched_winds2_gramm.astype(np.float32)[:, :, :] ** 2, axis=1))
    matched_winkel_gramm = arctan(matched_winds2_gramm[:, 1, :], matched_winds2_gramm[:, 0, :])

    for i in range(0, 14):
        ax = windrose(matched_winkel_gramm[:, i], matched_winds_hor_gramm[:, i],
                      f"{STATION_NAMES_JUL[i]} simulated GRAMM")
        plt.savefig(f"plots/windrose_matched_GRAMM_{STATION_NAMES_JUL[i]}")
        plt.show()
        plt.close()

    #test
    winkel = [360]
    speed = [1.]
    windrose(winkel, speed, f"simulated")
    plt.show()
    plt.close()

    print("done.")
