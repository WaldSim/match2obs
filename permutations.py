import itertools
import pickle
import numpy as np
from einops import rearrange
from matchobs_functions import matchedwinds, calculate_rmse, calculate_rmse_per, matchgralorgramm_jul, \
    match_with_stations
from readobs import load_observed_jul
from settings import START2, END2, USE_WEIGHTING, GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED

if __name__ == "__main__":
    observations = load_observed_jul(START2, END2)
    catalogue = matchgralorgramm_jul(GRAL, LOAD_CATALOGUE, N_CATALOGUE_USED)
    all_stations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    n = 11  # number of stations
    keys = {"rmse_all",
            # "rmse_stations"
            }
    full_results = {
        key: {
            "mean": list()
        } for key in keys
    }
    rmses = [list() for _ in range(len(keys))]
    rmses_all = [list() for _ in range(len(keys))]
    stat_combi = []
    station_combinations = itertools.combinations(all_stations, n)
    for station_list in station_combinations:
        stat_combi.append(station_list)
        print(f"matching stations {station_list}")
        etah = match_with_stations(observations, catalogue, station_list, USE_WEIGHTING)
        matched_winds = matchedwinds(catalogue, etah)
        matched_winds2 = rearrange(matched_winds, "d s h  -> h d s ")
        results_rmse = calculate_rmse_per(observations[:, 3:5, station_list], matched_winds2[:, :, station_list])
        results_rmse_all = calculate_rmse_per(observations[:, 3:5, all_stations], matched_winds2[:, :, all_stations])

        for i, key in enumerate(keys):
            rmses[i].append(results_rmse[key])
            rmses_all[i].append(results_rmse_all[key])

    for i in range(len(rmses)):
        rmses[i] = np.array(rmses[i])
        rmses_all[i] = np.array(rmses_all[i])
    # for key, rmse in zip(keys, rmses):
    #    full_results[key]["mean"].append(rmse.mean())
    rmse = np.array(rmses)
    rmse_all = np.array(rmses_all)
    combi = np.array(stat_combi)
    np.save("arrays/rmse_permutations_test.npy", rmse)
    np.save("arrays/rmse_all_permutations_test.npy", rmse_all)
    np.save("arrays/combi_permutations_test.npy", combi)

    print("done.")
