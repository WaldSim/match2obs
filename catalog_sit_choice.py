import pickle

import numpy as np
from matplotlib import pyplot as plt

import settings
from matchobs_functions import matching_winds, calculate_rmse, matchgralorgramm_jul, obshor, matchedhor, obsdir, \
    matcheddir, \
    winkdiff, mbedir, mbedirstations, mbehor, mbehorstations, rmsedir, rmsedirstations, rmseuv, rmseuvstations, \
    rmsehor, rmsehorstations, corrstations, corr
from math_functions import arctan
from plotting import plot_matching_stations, plot_matching_stations_cats, lin_reg
from readobs import load_observed_jul
from settings import START2, END2, N_CATALOGUE_USED, USED_STATIONS, METRIK, USE_WEIGHTING, USE_SMOOTHING

# load cat 1
pkl_file = open('winds_cat1476_final.pkl', 'rb')
catalogue = pickle.load(pkl_file)
pkl_file.close()

# load cat 2
pkl_file = open('winds_cat1008_final.pkl', 'rb')
catalogue2 = pickle.load(pkl_file)
pkl_file.close()
# load observed data
observations = load_observed_jul(START2, END2)
obs_hor = obshor(observations)
obs_dir = obsdir(observations)
# matching with cat 1
N_CATALOGUE = 1476
matched_winds = matching_winds(observations, catalogue, len(USED_STATIONS), USED_STATIONS, METRIK, USE_WEIGHTING,
                               USE_SMOOTHING, N_CATALOGUE)
matched_hor = matchedhor(matched_winds)
matched_dir = matcheddir(matched_winds)
# matching with cat 2
N_CATALOGUE = 1008
matched_winds2 = matching_winds(observations, catalogue2, len(USED_STATIONS), USED_STATIONS, METRIK, USE_WEIGHTING,
                                USE_SMOOTHING, N_CATALOGUE)
matched_hor2 = matchedhor(matched_winds2)
matched_dir2 = matcheddir(matched_winds2)

# plotting cat 1, cat 2, observed for speed and direction
plot_matching_stations_cats(obs_hor, matched_hor, matched_hor2, settings.STATION_NAMES_JUL, settings.METRIK,
                            settings.USE_WEIGHTING, settings.USE_SMOOTHING, ncols=2, nrows=7)

# plotting bias, rmse cat 1 und cat 2 for speed and direction
# winkel differenz
wink_diff = winkdiff(obs_dir, matched_dir)
wink_diff2 = winkdiff(obs_dir, matched_dir2)

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
mbe_dir2 = mbedir(wink_diff2)
mbe_dir_stat2 = mbedirstations(wink_diff2)
mbe_hor2 = mbehor(obs_hor, matched_hor2)
mbe_hor_stat2 = mbehorstations(obs_hor, matched_hor2)
print("MBE2:")
print(f"mbe_dir2:{mbe_dir2}")
print(f"mbe_dir_stat2:{mbe_dir_stat2}")
print(f"mbe_hor2:{mbe_hor2}")
print(f"mbe_hor_stat2:{mbe_hor_stat2}")
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
rmse_dir2 = rmsedir(wink_diff2)
rmse_dir_stat2 = rmsedirstations(wink_diff2)
rmse_uv2 = rmseuv(observations, matched_winds2)
rmse_uv_stat2 = rmseuvstations(observations, matched_winds2)
rmse_hor2 = rmsehor(obs_hor, matched_hor2)
rmse_hor_stat2 = rmsehorstations(obs_hor, matched_hor2)
print("RMSE2:")
print(f"rmse_dir2:{rmse_dir2}")
print(f"rmse_dir_stat2:{rmse_dir_stat2}")
print(f"rmse_hor2:{rmse_hor2}")
print(f"rmse_hor_stat2:{rmse_hor_stat2}")
print(f"rmse_uv2:{rmse_uv2}")
print(f"rmse_uv_stat2:{rmse_uv_stat2}")

obshor = obs_hor.flatten()
obswink = obs_dir.flatten()
matchhor = matched_hor.flatten()
matchhor2 = matched_hor2.flatten()

lin_reg(obshor[:], matchhor[:], color='red')
plt.savefig("plots/linregcat1.png")
lin_reg(obshor[:], matchhor2[:], color='green')
plt.savefig("plots/linregcat2.png")
plt.show()
