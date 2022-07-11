import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matchobs_functions import matching_winds
from math_functions import arctan
from plotting import lin_reg
from readobs import load_observed_jul
from settings import USED_STATIONS, STATION_NAMES_JUL, START2, END2, LOAD_CATALOGUE, METRIK, USE_SMOOTHING, \
    GRAL, USE_WEIGHTING, N_CATALOGUE_USED, STATIONS_INDEX
from matchobs_functions import matching_winds, calculate_rmse, calculate_mean_bias, obshor, matchedhor, obsdir, \
    matcheddir, winkdiff, mbedir, mbedirstations, mbehor, mbehorstations, mbestattime, rmsedir, rmsedirstations, rmseuv, \
    rmseuvstations, rmsehor, rmsehorstations, rmsestattime, rmsetime, matchgralorgramm_jul

# "
metric = ["l2vobs", "berchet", "l2_iup_thb_ghw_high", "l2vobsmin", "l2vobsmean", "l2vobsminmean", "dirwind+horwind"]
metric_name = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
METRIC_COLORS = {0: "maroon", 1: "darkorange", 2: 'gold', 3: 'darkolivegreen', 4: 'turquoise', 5: 'slategrey',
                 6: 'darkcyan'}

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

"""
observations= load_observed_jul(START2, END2)
#catalogue = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE)
pkl_file = open('winds_cat1476_final.pkl', 'rb')
catalogue = pickle.load(pkl_file)
pkl_file.close()
cat_hor = np.sqrt(np.sum(catalogue.astype(np.float32)[:-1, :, :] ** 2, axis=0))
cat_winkel = arctan(catalogue[1, :, :], catalogue[0, :, :])
cat_points = cat_hor * np.array([np.cos(cat_winkel), np.sin(cat_winkel)])
obs_points = observations[:, 1, :] * np.array([np.cos(observations[:, 2, :].astype(np.float32)), np.sin(observations[:, 2, :].astype(np.float32))])
obs = observations[:, 3:5, :].astype(np.float32)
obs_hor = np.sqrt(np.sum(observations[:, 3:5, :].astype(np.float32)**2, axis=1))
obs_winkel = observations[:, 2, :].astype(np.float32)"""
# matching
match_hor_all = []
diff_all = []
rmse_all = []
mean_bias_all = []
match_winkel_all = []
for METRIK in metric:
    # matching
    matched_winds = matching_winds(observations, catalogue, len(USED_STATIONS),
                                   USED_STATIONS, METRIK, USE_WEIGHTING,
                                   USE_SMOOTHING, N_CATALOGUE_USED)
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

    # RMSE
    rmse_dir = rmsedir(wink_diff)
    rmse_dir_stat = rmsedirstations(wink_diff)
    rmse_uv = rmseuv(observations, matched_winds)
    rmse_uv_stat = rmseuvstations(observations, matched_winds)
    rmse_hor = rmsehor(obs_hor, matched_hor)
    rmse_hor_stat = rmsehorstations(obs_hor, matched_hor)
    rmse_station_time = rmsestattime(observations, matched_winds)

    rmse_time = rmsetime(observations, matched_winds)
    match_hor_all.append(matched_hor)
    rmse_all.append(rmse_uv)
    mean_bias_all.append(mbe_hor)
    match_winkel_all.append(matched_dir)

    diff = obs_hor - matched_hor
    diff_all.append(diff)
    print(f"metric:{METRIK}")
    print("RMSE:")
    print(f"rmse_dir:{rmse_dir}")
    print(f"rmse_dir_stat:{rmse_dir_stat}")
    print(f"rmse_hor:{rmse_hor}")
    print(f"rmse_hor_stat:{rmse_hor_stat}")
    print(f"rmse_uv:{rmse_uv}")
    print(f"rmse_uv_stat:{rmse_uv_stat}")
    print("MBE:")
    print(f"mbe_dir:{mbe_dir}")
    print(f"mbe_dir_stat:{mbe_dir_stat}")
    print(f"mbe_hor:{mbe_hor}")
    print(f"mbe_hor_stat:{mbe_hor_stat}")

    d = {"RMSE": rmse_uv,
         "rmse_hor": rmse_hor,
         "mean_bias_hor": mbe_hor,
         }
    df = pd.DataFrame(d, index=["all"])

    df.index.names = ['Station']
    df.head()

match_hor_all = np.asarray(match_hor_all)
match_winkel_all = np.asarray(match_winkel_all)
diff_all = np.asarray(diff_all)
rmse_all = np.asarray(rmse_all)
mean_bias_all = np.asarray(mean_bias_all)
d_metric = {"RMSE": rmse_all,
            # "rmse_hor": rmse_hor,
            # "mean_bias_hor": mean_bias_hor,
            }
df_metric = pd.DataFrame(d_metric, index=metric)

df_metric.index.names = ['Station']
df_metric.head()
ncols = 2
nrows = 7
plt.figure(0)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 15), sharex=True, sharey=True)

k = 0
for i in range(nrows):

    for j in range(ncols):
        try:
            axes[i, j].grid(True, linewidth=0.5, color="black", linestyle="-")
            axes[i, j].plot(obs_hor[0:100, k], label="wind observed", marker="", color="darkblue", linestyle="--")
            for l in range(len(metric)):
                axes[i, j].plot(match_hor_all[l, 0:100, k], label=f"{metric_name[l]})", marker="", linestyle="-",
                                # color=f"{STATION_COLORS[l]}",
                                alpha=0.8)
            axes[i, j].set_title(f"{STATION_NAMES_JUL[k]}")
            plt.tight_layout()
            plt.title("")
            axes[i, j].set_title(f"matched and observed wind for measuring station {STATION_NAMES_JUL[k]}", fontsize=24)
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

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=8)
plt.savefig(f"plots/metric_test.png", bbox_inches='tight')
plt.show()
plt.close()

print("done.")
