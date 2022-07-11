import pickle
import struct
import numpy as np
import pandas as pd
from tqdm import trange
# station combination with 14 measuring stations
from settings import STATIONDATA_JUL

# numer of stations
N_STATIONS_JUL = 14
# number of catalogue entries
N_CATALOGUE = 1008
N_CATALOGUE_2 = 1476
FOLDER = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"  # GRAL and GRAMM input data folder for load catalogue, etc
FOLDER2 = "/mnt/data/users/swald/GRAMM-GRAL-final/HD_GRAL/"


def make_stations_jul():
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
    return stations


def make_gramm_grid_jul(folder=FOLDER):  # GRAMM.geb und ggeom.asc sind in diesem Fall für beide Kataloge gleich
    """ GRAMM-Grid bestimmen """
    file = open(folder + "GRAMM.geb", 'r')  # TODO: check reading utils, also below for topo
    GRAMMgeb = file.readlines()
    GRAMM_nx = int(GRAMMgeb[0].split(' ')[0])
    GRAMM_ny = int(GRAMMgeb[1].split(' ')[0])
    GRAMM_nz = int(GRAMMgeb[2].split(' ')[0])
    GRAMM_xmin = int(GRAMMgeb[3].split(' ')[0])
    GRAMM_xmax = int(GRAMMgeb[4].split(' ')[0])
    GRAMM_ymin = int(GRAMMgeb[5].split(' ')[0])
    GRAMM_ymax = int(GRAMMgeb[6].split(' ')[0])
    GRAMM_dx = (GRAMM_xmax - GRAMM_xmin) / GRAMM_nx
    GRAMM_dy = (GRAMM_ymax - GRAMM_ymin) / GRAMM_ny
    # Messstationen in Grid
    stations = make_stations_jul()
    station_grid = [(stations[2] - GRAMM_xmin) / GRAMM_dx,
                    (stations[3] - GRAMM_ymin) / GRAMM_dy]
    station_grid = list(np.float_(station_grid))  # TODO: what is this?
    station_grid = np.around(station_grid).astype(int)
    # Topographie
    nx, ny, nz = GRAMM_nx, GRAMM_ny, GRAMM_nz
    fic = 'ggeom.asc'
    f = open(folder + fic, 'r')
    data = f.readlines()
    f.close()
    # TODO: what it is this block for?
    tmp = data[1].split()
    topo_ind = 0
    topo = np.zeros((nx, ny), np.float)
    for i in range(ny):
        for j in range(nx):
            topo[j, i] = float(tmp[topo_ind])
            topo_ind += 1
    # TODO: what it is this block for?
    # z layers
    tmp = data[0].split()
    zalt = []
    for l in range(nz):
        zalt.append(float(tmp[-1 - l]))
    zalt = zalt.reverse()  # TODO: why unused?
    # make zgrid
    tmp = data[8].split()  # TODO: why at index 8?
    zgrid = np.zeros([nx, ny, nz], np.float)
    ind = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                zgrid[i, j, k] = float(tmp[ind])
                ind += 1
    ### z-layer über Abstand auswählen? Ja, da z immer die mitte des layers angibt
    layers = zgrid[station_grid[0], station_grid[1], :]
    # z_heights height stations
    z_heights = layers[:, 0] + stations[5]
    # z_stations aus topographie.py genommen
    dist = np.abs(z_heights[:, None] - layers)
    z = np.argmin(dist, axis=1)
    z_layer = z
    return station_grid, z_layer


def load_catalogue_jul_1008(station_grid, z_layer):
    cases = [f"{i + 1:05}" for i in np.arange(0, N_CATALOGUE)]
    # Katalogeinträge laden, Katalogeinträge an Messstationen auswählen
    winds = np.zeros((3, N_STATIONS_JUL, len(cases)))

    def fill_station(winds, dataarray, x, y, z, station_index, case_index):
        wind = dataarray[x, y, z, :] * 0.01  # convert to m/s
        winds[:, station_index, case_index] = wind
        return winds

    # für alle cases daten an messstationen
    for i in trange(len(cases), desc="Reading Calculated Catalogue Entries"):
        file = open(FOLDER + (cases[i]) + ".wnd", 'rb')
        data = file.read()
        file.close()
        nheader = 20  # header, ni,nj,nk,gridsiye -> 4*signed integer (=4*4) + float (4)
        header, nx, ny, nz, dx = struct.unpack('<iiiif', data[:nheader])
        dt = np.dtype(np.short)  # TODO: sure 16-bit?
        # simulierte Daten an Stelle im Grid, an der sich Messstation befindet
        datarr = np.frombuffer(data[nheader:], dtype=dt)
        datarr = np.reshape(datarr, [nx, ny, nz, 3])
        for j in range(N_STATIONS_JUL):
            winds = fill_station(winds, datarr, station_grid[0][j], station_grid[1][j], z_layer[j], j, i)
    return winds


def load_catalogue_jul_1476(station_grid, z_layer):
    cases = [f"{i + 1:05}" for i in np.arange(0, N_CATALOGUE_2)]
    # Katalogeinträge laden, Katalogeinträge an Messstationen auswählen
    winds = np.zeros((3, N_STATIONS_JUL, len(cases)))

    def fill_station(winds, dataarray, x, y, z, station_index, case_index):
        wind = dataarray[x, y, z, :] * 0.01  # convert to m/s
        winds[:, station_index, case_index] = wind
        return winds

    # für alle cases daten an messstationen
    for i in trange(len(cases), desc="Reading Calculated Catalogue Entries"):
        file = open(FOLDER2 + (cases[i]) + ".wnd", 'rb')
        data = file.read()
        file.close()
        nheader = 20  # header, ni,nj,nk,gridsiye -> 4*signed integer (=4*4) + float (4)
        header, nx, ny, nz, dx = struct.unpack('<iiiif', data[:nheader])
        dt = np.dtype(np.short)  # TODO: sure 16-bit?
        # simulierte Daten an Stelle im Grid, an der sich Messstation befindet
        datarr = np.frombuffer(data[nheader:], dtype=dt)
        datarr = np.reshape(datarr, [nx, ny, nz, 3])
        for j in range(N_STATIONS_JUL):
            winds = fill_station(winds, datarr, station_grid[0][j], station_grid[1][j], z_layer[j], j, i)
    return winds
