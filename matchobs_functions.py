import pickle
import numpy as np
from einops import rearrange
from math_functions import custom_gaussian_filter, arctan
from readcat import make_gramm_grid_jul, load_catalogue_jul_1008, load_catalogue_jul_1476


def cat_jul_1008(load_cat):
    if load_cat:
        grid, z = make_gramm_grid_jul()
        print(f"z:{z}")
        catalogue = load_catalogue_jul_1008(grid, z)
        np.save("gramm_cat_1008", catalogue)
    else:
        catalogue = np.load("gramm_cat_1008.npy")
    return catalogue


def cat_jul_1476(load_cat):
    if load_cat:
        grid, z = make_gramm_grid_jul()
        print(f"z:{z}")
        catalogue = load_catalogue_jul_1476(grid, z)
        np.save("gramm_cat_1476", catalogue)
    else:
        catalogue = np.load("gramm_cat_1476.npy")
    return catalogue


def matchgralorgramm_jul(gral, load_cat, n_cat):
    assert n_cat in [1008, 1476]
    if n_cat == 1008:
        if gral:
            pkl_file = open('winds_cat1008_final.pkl', 'rb')
            catalogue = pickle.load(pkl_file)
            pkl_file.close()
        else:
            catalogue = cat_jul_1008(load_cat)
    elif n_cat == 1476:
        if gral:
            pkl_file = open("winds_cat1476_final.pkl", 'rb')
            catalogue = pickle.load(pkl_file)
            pkl_file.close()
        else:
            catalogue = cat_jul_1476(load_cat)
    return catalogue


def match_with_stations(observation, catalogue, stat_list,
                        use_weighting,
                        weight=1.,
                        use_smoothing=False,
                        metric="l2"):
    scores = choose_metric(observation, catalogue, stat_list, metric,
                           use_weighting)
    eta_h = min_score(scores)
    return eta_h


def matchedwinds_rmse(catalogue, etah):
    matchedwinds = catalogue[:-1, etah]

    return matchedwinds


def matching_winds(obs, cat, stations, used_stations, metrik, use_weighting, smoothing, ncat):
    eta_matrix = choose_metric(obs, cat, used_stations, metrik, use_weighting)
    score = smooth(eta_matrix, smoothing,
                   ncat)
    etah = min_score(score)
    matched_winds = matchedwinds(cat, etah)
    matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
    return matched_winds2


def calculate_weight(observations, station_list, use_weight):
    if use_weight:
        #  Gewichtungsfaktor berechnen
        if len(station_list) != 0:
            std = observations[:, 5, station_list]
            obs_min = np.median(observations[:, 1, station_list])
            obs_windspeed = observations[:, 1, station_list]
            obs_windspeed[obs_windspeed <= obs_min] = obs_min
            weight = std * obs_windspeed
            weight = weight[:, :, None]
        elif len(station_list) == 0:
            std = observations[:, 5]
            obs_min = np.median(observations[:, 1])
            obs_windspeed = observations[:, 1]
            obs_windspeed[obs_windspeed <= obs_min] = obs_min
            weight = std * obs_windspeed
            weight = weight[:, :, None]
            weight = std * obs_windspeed
    else:
        if len(station_list) != 0:
            weight = np.ones((observations.shape[0], len(station_list), 1))
        elif len(station_list) == 0:
            weight = np.ones((observations.shape[0], 1))
    return weight


def choose_metric(observation, catalogue, stat_list, metric,
                  use_weighting):
    assert metric in ["l2", "l2vobs", "berchet", "diffobsmin", "l2_iup_cze_wwr_high",
                      "l2_iup_thb_ghw_high", "dirwind+horwind", "l2vobsmean", "l2vobsmin", "l2vobsminmean"]
    if metric == "l2":
        if len(stat_list) != 0:
            diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
            diff_square = np.sqrt(np.sum(diff.astype(float) ** 2, axis=1))  # h n_match n_cat, summe über u, v
            weight = calculate_weight(observation, stat_list, use_weighting)
            eta_matrix = np.nansum(diff_square, axis=1)  # h n_cat, summe über stations
            # eta_matrix = np.nansum(diff_square / weight ** 2, axis=1)  # h n_cat, summe über stations
        elif len(stat_list) == 0:
            diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])
            diff_square = (np.sum(diff ** 2, axis=1))  # h n_match n_cat, summe über u, v
            weight = calculate_weight(observation, stat_list, use_weighting)
            eta_matrix = np.nansum(diff_square / weight[:, :, None] ** 2, axis=1)  # h n_cat, summe über stations
        eta_matrix[eta_matrix == 0] = np.nan
    elif metric == "l2vobs":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = np.sqrt((np.sum(diff.astype(float) ** 2, axis=1)))  # h n_match n_cat, summe über u, v
        obs_windspeed = observation[:, 1, stat_list].astype(float)
        epsilon = 10e-8
        weight = obs_windspeed + epsilon
        weight = weight[:, :, None]
        eta_matrix = np.nansum(diff_square / weight, axis=1)  # h n_cat, summe über stations
    elif metric == "berchet":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = (np.sum(diff ** 2, axis=1))  # h n_match n_cat, summe über u, v
        std = observation[:, 5, stat_list]
        obs_min = np.median(observation[:, 1, :])
        obs_windspeed = observation[:, 1, stat_list]
        obs_windspeed[obs_windspeed <= obs_min] = obs_min
        weight = std * obs_windspeed
        weight = weight[:, :, None]
        eta_matrix = np.nansum(diff_square / weight ** 2, axis=1)  # h n_cat, summe über stations
    elif metric == "l2std":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = np.sqrt((np.sum(diff.astype(np.float32) ** 2, axis=1)))  # h n_match n_cat, summe über u, v
        std = observation[:, 5, stat_list]
        obs_min = np.nanmedian(observation[:, 1, :].astype(np.float32))
        obs_windspeed = observation[:, 1, :]
        obs_windspeed[obs_windspeed <= obs_min] = obs_min
        weight = std[None, :]
        weight[weight == 0] = np.nan
        weight = weight[:, :, None]
        eta_matrix = np.nansum(diff_square / weight, axis=1)  # h n_cat, summe über stations
    elif metric == "l2_iup_cze_wwr_high":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = (np.nansum(diff ** 2, axis=1))  # h n_match n_cat, summe über u, v
        weight = calculate_weight(observation, stat_list, use_weighting)
        weight1 = weight[:, (0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12), :]
        # print(weight[0, 1, 0])
        weight2 = weight[:, 7, None]  # IUP
        weight3 = weight[:, 9, None]  # CZE
        weight4 = weight[:, 13, None]  # WWR
        diff_square2 = diff_square[:, 7, :]
        diff_square3 = diff_square[:, 9, :]
        diff_square4 = diff_square[:, 13, :]
        diff_square1 = (diff_square[:, (0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12),
                        :] / weight1 ** 2)  # h n_cat, summe über stations
        diff_square2 = (diff_square2[:, None, :] / (0.002 * weight2 ** 2))  # h n_cat, summe über stations
        diff_square3 = (diff_square3[:, None, :] / (0.02 * weight3 ** 2))  # h n_cat, summe über stations
        diff_square4 = (diff_square4[:, None, :] / (0.02 * weight4 ** 2))  # h n_cat, summe über stations
        diff_square_sum = np.concatenate((diff_square1, diff_square2, diff_square3, diff_square4), axis=1)
        eta_matrix = np.nansum(diff_square_sum, axis=1)
    elif metric == "l2_iup_thb_ghw_high":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = np.sqrt((np.nansum(diff.astype(float) ** 2, axis=1)))  # h n_match n_cat, summe über u, v
        weight = calculate_weight(observation, stat_list, use_weighting)
        weight1 = weight[:, (1, 2, 3, 4, 5, 6, 8, 9, 11, 12), :]
        # print(weight[0, 1, 0])
        weight2 = weight[:, 0, None]  # GHW
        weight3 = weight[:, 7, None]  # THB
        weight4 = weight[:, 9, None]  # CZE
        weight5 = weight[:, 13, None]  # IUP
        diff_square2 = diff_square[:, 0, :]
        diff_square3 = diff_square[:, 7, :]
        diff_square4 = diff_square[:, 9, :]
        diff_square5 = diff_square[:, 13, :]
        diff_square1 = (diff_square[:, (1, 2, 3, 4, 5, 6, 8, 9, 11, 12), :] / weight1)  # h n_cat, summe über stations
        diff_square2 = (diff_square2[:, None, :] / (0.002 * weight2))  # h n_cat, summe über stations
        diff_square3 = (diff_square3[:, None, :] / (0.002 * weight3))  # h n_cat, summe über stations
        diff_square4 = (diff_square4[:, None, :] / (0.002 * weight4))  # h n_cat, summe über stations
        diff_square5 = (diff_square5[:, None, :] / (0.002 * weight5))  # h n_cat, summe über stations
        diff_square_sum = np.concatenate((diff_square1, diff_square2, diff_square3, diff_square4, diff_square5), axis=1)
        eta_matrix = np.nansum(diff_square_sum, axis=1)
    elif metric == "l2vobsminmean":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = np.sqrt((np.sum(diff.astype(float) ** 2, axis=1)))  # h n_match n_cat, summe über u, v
        obs_min = np.median(observation[:, 1, stat_list, ])
        obs_windspeed = observation[:, 1, stat_list, ]
        obs_windspeed[obs_windspeed <= obs_min] = obs_min
        obs_windspeed_mean = (np.nanmean(obs_windspeed[:, :].astype(float), axis=0))
        obs_windspeedmean = obs_windspeed_mean[None, :, None]
        eta_matrix = np.nansum(diff_square / obs_windspeedmean, axis=1)  # h n_cat, summe über stations
        eta_matrix[eta_matrix == 0] = np.nan
    elif metric == "dirwind+horwind":
        cat_dir = arctan(catalogue[1, :, :].astype(np.float32), catalogue[0, :, :].astype(np.float32))
        diffwinkel = (observation[:, 2, :, None] - cat_dir[None, :, :]).astype(np.float32)
        diffwinkel2 = (diffwinkel + 180) % 360 - 180
        a = (diffwinkel2 * np.pi) / 180
        diff1 = np.cos(a)
        diff2 = np.sqrt((np.sqrt(np.sum(observation[:, 3:5, :, None].astype(float) ** 2, axis=1))
                         - np.sqrt(np.sum(catalogue[None, :-1, :, :].astype(float) ** 2, axis=1))) ** 2)
        diff_square = diff2
        obs_min = np.median(observation[:, 1, :, ])
        obs_windspeed = observation[:, 1, :, ]
        obs_windspeed[obs_windspeed <= obs_min] = obs_min
        obs_windspeed_mean = (np.nanmean(obs_windspeed[:, :].astype(float), axis=0))
        obs_windspeedmean = obs_windspeed_mean[None, :, None]
        l = 0.5
        eta_matrix = np.nansum(l * (diff_square / obs_windspeedmean) - (1 - l) * diff1,
                               axis=1)  # h n_cat, summe über stations
        eta_matrix[eta_matrix == 0] = np.nan
    elif metric == "l2vobsmean":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = np.sqrt((np.sum(diff.astype(float) ** 2, axis=1)))  # h n_match n_cat, summe über u, v
        obs_windspeed = observation[:, 1, stat_list]
        obs_windspeed_mean = (np.nanmean(obs_windspeed[:, :].astype(float), axis=0))
        obs_windspeedmean = obs_windspeed_mean[None, :, None]
        eta_matrix = np.nansum(diff_square / obs_windspeedmean, axis=1)  # h n_cat, summe über stations
        eta_matrix[eta_matrix == 0] = np.nan
    elif metric == "l2vobsmin":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])  # h, 2, n_match, n_cat
        diff_square = np.sqrt((np.sum(diff.astype(float) ** 2, axis=1)))  # h n_match n_cat, summe über u, v
        obs_min = np.median(observation[:, 1, stat_list])
        obs_windspeed = observation[:, 1, stat_list]
        obs_windspeed[obs_windspeed <= obs_min] = obs_min
        eta_matrix = np.nansum(diff_square / ((obs_windspeed[:, :, None])), axis=1)  # h n_cat, summe über stations
        eta_matrix[eta_matrix == 0] = np.nan
    elif metric == "diff":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])
        diff_abs = (np.sum(np.abs(diff.astype(float)), axis=1))
        eta_matrix = np.nansum((diff_abs), axis=1)
        eta_matrix[eta_matrix == 0] = np.nan
    elif metric == "diffobs":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])
        diff_abs = np.sum(np.abs(diff.astype(float)), axis=1)
        v_mean = (np.nanmean(observation[:, 1, stat_list].astype(float), axis=0))
        vmean = v_mean[None, :, None]
        diff_abs_vobs = diff_abs / vmean
        eta_matrix = np.nansum((diff_abs_vobs), axis=1)
        eta_matrix[eta_matrix == 0] = np.nan
    elif metric == "diffobsmin":
        diff = (observation[:, 3:5, stat_list, None] - catalogue[None, :-1, stat_list, :])
        diff_abs = np.sum(np.abs(diff.astype(float)), axis=1)
        obs_min = np.median(observation[:, 1, stat_list])
        obs_windspeed = observation[:, 1, stat_list]
        obs_windspeed[obs_windspeed <= obs_min] = obs_min
        obs_windspeed_mean = (np.nanmean(obs_windspeed[:, :].astype(float), axis=0))
        obs_windspeedmean = obs_windspeed_mean[None, :, None]
        diff_abs_vobs = diff_abs / obs_windspeedmean
        eta_matrix = np.nansum((diff_abs_vobs), axis=1)
        eta_matrix[eta_matrix == 0] = np.nan
    else:
        raise ValueError()
    return eta_matrix


def smooth(eta_matrix, use_smooth, n_cat):
    if use_smooth:
        smooth_etamatrix = []
        for i in range(n_cat):
            smooth_scores_one = custom_gaussian_filter(eta_matrix[:, i], sigma=1., truncate=4.)
            smooth_etamatrix.append(smooth_scores_one)
        smooth_etamatrix = np.asarray(smooth_etamatrix).astype(np.float32)
        score_smooth_before = rearrange(smooth_etamatrix, "n d  -> d n ")
        score_smooth = score_smooth_before.astype(np.float32)
    else:
        score_before = eta_matrix
        score_smooth = score_before.astype(np.float32)
    return score_smooth


def min_score(score):
    eta_h = np.argmin(score, axis=1)
    # np.save("arrays/etah.npy", eta_h)
    return eta_h


def matchedwinds(catalogue, etah):
    matchedwinds = catalogue[:-1, :, etah]
    return matchedwinds


def calculate_rmse(obs, match):
    rmse_all = np.sqrt(np.nanmean(
        np.sum((obs.astype(np.float32) - match.astype(np.float32)) ** 2, axis=1)))  # wurzel und quadrat heben sich auf
    rmse_station = np.sqrt(np.nanmean(np.sum((obs - match).astype(np.float32) ** 2, axis=1), axis=0))
    rmse_time = np.sqrt(np.nanmean(np.sum((obs - match).astype(np.float32) ** 2, axis=1), axis=1))
    rmse_station_time = np.sqrt((np.sum((obs - match).astype(np.float32) ** 2, axis=1)))
    obs_hor = np.sqrt(np.sum(obs.astype(np.float32) ** 2, axis=1))
    match_hor = np.sqrt(np.sum(match.astype(np.float32) ** 2, axis=1))
    rmse_all_hor = np.sqrt(np.nanmean(((obs_hor.astype(np.float32) - match_hor.astype(np.float32)) ** 2)))
    rmse_station_hor = np.sqrt(np.nanmean((obs_hor - match_hor).astype(np.float32) ** 2, axis=0))
    rmse_u = np.sqrt(np.nanmean((obs[:, 0, :].astype(np.float32) - match[:, 0, :].astype(np.float32)) ** 2))
    rmse_v = np.sqrt(np.nanmean((obs[:, 1, :].astype(np.float32) - match[:, 1, :].astype(np.float32)) ** 2))
    rmse_u_station = np.sqrt(
        np.nanmean((obs[:, 1, :].astype(np.float32) - match[:, 1, :].astype(np.float32)) ** 2, axis=0))
    rmse_v_station = np.sqrt(
        np.nanmean((obs[:, 1, :].astype(np.float32) - match[:, 1, :].astype(np.float32)) ** 2, axis=0))
    results_rmse = dict()
    results_rmse["rmse_time"] = rmse_time
    results_rmse["rmse_station_time"] = rmse_station_time
    results_rmse["rmse_all"] = rmse_all
    results_rmse["rmse_station"] = rmse_station
    results_rmse["rmse_all_hor"] = rmse_all_hor
    results_rmse["rmse_station_hor"] = rmse_station_hor
    results_rmse["rmse_u"] = rmse_u
    results_rmse["rmse_v"] = rmse_v
    results_rmse["rmse_station_u"] = rmse_u_station
    results_rmse["rmse_station_v"] = rmse_v_station
    return results_rmse


def calculate_rmse_per(obs, match):
    rmse_all = np.sqrt(np.nanmean(
        np.sum((obs.astype(np.float32) - match.astype(np.float32)) ** 2, axis=1)))  # wurzel und quadrat heben sich auf
    rmse_station = np.sqrt(np.nanmean(np.sum((obs - match).astype(np.float32) ** 2, axis=1), axis=0))
    rmse_time = np.sqrt(np.nanmean(np.sum((obs - match).astype(np.float32) ** 2, axis=1), axis=1))
    rmse_station_time = np.sqrt((np.sum((obs - match).astype(np.float32) ** 2, axis=1)))
    obs_hor = np.sqrt(np.sum(obs.astype(np.float32) ** 2, axis=1))
    match_hor = np.sqrt(np.sum(match.astype(np.float32) ** 2, axis=1))
    rmse_all_hor = np.sqrt(np.nanmean(((obs_hor.astype(np.float32) - match_hor.astype(np.float32)) ** 2)))
    rmse_station_hor = np.sqrt(np.nanmean((obs_hor - match_hor).astype(np.float32) ** 2, axis=0))
    results_rmse = dict()
    results_rmse["rmse_time"] = rmse_time
    results_rmse["rmse_station_time"] = rmse_station_time
    results_rmse["rmse_all"] = rmse_all
    results_rmse["rmse_station"] = rmse_station
    results_rmse["rmse_all_hor"] = rmse_all_hor
    results_rmse["rmse_station_hor"] = rmse_station_hor
    return results_rmse


def calculate_mean_bias(obs, match):
    mbe_all = np.nanmean(np.sqrt(np.sum((obs - match).astype(np.float32) ** 2, axis=1)))
    print(mbe_all)
    print(np.shape(mbe_all))
    mbe_station = np.nanmean((np.sqrt(np.sum((obs - match).astype(np.float32) ** 2, axis=1))), axis=0)
    mbe_time = (np.nanmean(np.sqrt((np.sum((obs - match).astype(np.float32) ** 2, axis=1))), axis=1))
    obs_hor = np.sqrt(np.sum(obs.astype(np.float32) ** 2, axis=1))
    match_hor = np.sqrt(np.sum(match.astype(np.float32) ** 2, axis=1))
    mbe_all_hor = (np.nanmean((obs_hor.astype(np.float32) - match_hor.astype(np.float32))))
    mbe_station_hor = (np.nanmean((obs_hor - match_hor).astype(np.float32), axis=0))
    results_mbe = dict()
    results_mbe["mbe_all"] = mbe_all
    results_mbe["mbe_station"] = mbe_station
    results_mbe["mbe_time"] = mbe_time
    results_mbe["mbe_all_hor"] = mbe_all_hor
    results_mbe["mbe_station_hor"] = mbe_station_hor
    return results_mbe


def calculate_mean_absolute(obs, match):
    mae_all = np.mean(np.sum((np.abs(obs - match)), axis=1))
    mae_station = np.nanmean(np.sum((np.abs(obs - match)), axis=1), axis=0)
    mae_time = np.sqrt(np.mean(np.nanmean(np.sum((np.abs(obs - match)), axis=1), axis=0), axis=0))
    results_mae = dict()
    results_mae["mae_all"] = mae_all
    results_mae["mae_station"] = mae_station
    results_mae["mae_time"] = mae_time
    return results_mae


def matchedwinds_rmse(catalogue, etah):
    matchedwinds = catalogue[:-1, etah]
    return matchedwinds


# MB definitions

def mbedir(winkeldiff):
    mbe_dir = np.nanmean((winkeldiff.astype(np.float32)))
    return mbe_dir


def mbedirstations(winkeldiff):
    mbe_dir_stations = np.nanmean((winkeldiff.astype(np.float32)), axis=0)
    return mbe_dir_stations


def mbehor(obshor, matchedhor):
    mbe_hor = np.nanmean(obshor - matchedhor)
    return mbe_hor


def mbehorstations(obshor, matchedhor):
    mbe_hor_stations = np.nanmean(obshor - matchedhor, axis=0)
    return mbe_hor_stations


def mbetime(obshor, matchhor):
    mbe_time = np.nanmean(((obshor - matchhor).astype(np.float32)), axis=1)
    return mbe_time


def mbestattime(obshor, matchhor):
    mbe_stat_time = (obshor - matchhor).astype(np.float32)
    return mbe_stat_time


# RMSE definitions

def rmsedir(winkeldiff):
    rmse_dir = np.sqrt(np.nanmean(winkeldiff.astype(np.float32) ** 2))
    return rmse_dir


def rmsedirstations(winkeldiff):
    rmse_dir_stations = np.sqrt(np.nanmean(winkeldiff.astype(np.float32) ** 2, axis=0))
    return rmse_dir_stations


def rmseuv(obs, match):
    rmse_uv = np.sqrt(np.nanmean(
        np.sum((obs[:, 3:5, :].astype(np.float32) - match.astype(np.float32)) ** 2, axis=1)))
    return rmse_uv


def rmseuvstations(obs, match):
    rmse_uv_stations = np.sqrt(np.nanmean(np.sum((obs[:, 3:5, :] - match).astype(np.float32) ** 2, axis=1), axis=0))
    return rmse_uv_stations


def rmsehor(obshor, matchhor):
    rmse_hor = np.sqrt(np.nanmean(((obshor.astype(np.float32) - matchhor.astype(np.float32)) ** 2)))
    return rmse_hor


def rmsehorstations(obshor, matchhor):
    rmse_hor_stations = np.sqrt(np.nanmean((obshor - matchhor).astype(np.float32) ** 2, axis=0))
    return rmse_hor_stations


def rmsetime(obs, matchuv):
    rmse_time = np.sqrt(np.nanmean(np.sum((obs[:, 3:5, :] - matchuv).astype(np.float32) ** 2, axis=1), axis=1))
    return rmse_time


def rmsestattime(obs, matchuv):
    rmse_station_time = np.sqrt((np.sum((obs[:, 3:5, :] - matchuv).astype(np.float32) ** 2, axis=1)))
    return rmse_station_time


# horizontal wind speed
def obshor(obs):
    obs_hor = np.sqrt(np.sum(obs[:, 3:5, :].astype(np.float32) ** 2, axis=1))
    return obs_hor


def matchedhor(matchedwinds):
    matched_hor = np.sqrt(
        np.sum(matchedwinds.astype(np.float32) ** 2, axis=1))
    return matched_hor


# wind direction

def obsdir(obs):
    obs_dir = obs[:, 2, :, ]
    return obs_dir


def matcheddir(matchedwinds):
    matched_dir = arctan(matchedwinds[:, 1, :], matchedwinds[:, 0, :])
    return matched_dir


# winkel differenz

def winkdiff(obsdir, matcheddir):
    a = obsdir - matcheddir
    winkeldiff = (a + 180) % 360 - 180
    return winkeldiff


# correlation
def corrstations(obshor, matchedhor):
    korrs = []
    for i in range(0, 14):
        obs = obshor[:, i].flatten()
        mat = matchedhor[:, i].flatten()
        bad = ~np.logical_or(np.isnan(obs), np.isnan(mat))
        obs1 = np.compress(bad, obs)
        mat1 = np.compress(bad, mat)
        kor = np.corrcoef(obs1, mat1)
        korrs.append(kor)
    return korrs


def corr(obshor, matchedhor):
    obs = obshor.flatten()
    mat = matchedhor.flatten()
    bad = ~np.logical_or(np.isnan(obs), np.isnan(mat))
    obs1 = np.compress(bad, obs)
    mat1 = np.compress(bad, mat)
    kor = np.corrcoef(obs1, mat1)
    return kor


def filter_nans(data):
    return np.ma.array(data, mask=np.isnan(data))
