import numpy as np
from matchobs_functions import matching_winds, calculate_rmse, matchgralorgramm_jul, obshor, matchedhor, obsdir, matcheddir, \
    winkdiff, mbedir, mbedirstations, mbehor, mbehorstations, rmsedir, rmsedirstations, rmseuv, rmseuvstations, \
    rmsehor, rmsehorstations, corrstations, corr

from math_functions import arctan
from plotting import plot_matching_stations
from readobs import load_observed_jul
from settings import USED_STATIONS, STATION_NAMES_JUL, START2, END2, LOAD_CATALOGUE, METRIK, USE_SMOOTHING, \
    GRAL, USE_WEIGHTING, N_CATALOGUE_USED

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
    # correlation
    corr_stat = corrstations(obs_hor, matched_hor)
    cor = corr(obs_hor, matched_hor)
    print("corr:")
    print(f"corr:{cor}")
    print(f"corr_stat:{corr_stat}")
    # plotting
    plot_matching_stations(obs_hor, matched_hor, STATION_NAMES_JUL, METRIK,
                           USE_WEIGHTING, USE_SMOOTHING, ncols=2, nrows=7)
    print("done.")
