import numpy as np
import pandas as pd
import struct
import zipfile
import math
import pickle
import os
from settings import STATIONDATA_JUL

folder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"
FOLDER = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"

Stations_jul = pd.DataFrame(STATIONDATA_JUL, columns=["GHW", "HP", "KOS", "STW", "PT",
                                                      # "SB",
                                                      "STB", "THB", "WWR", "KOE", "CZE", "GAB", "LUBW", "IUP"])
nstations = Stations_jul.shape[1]
stations = Stations_jul.to_numpy()
## GRAL - Grid bestimmen
file = open(folder + "GRAL.geb", 'r')
GRALgeb = file.readlines()
file.close()
GRAL_dx = float(GRALgeb[0].split(' ')[0])
GRAL_dy = float(GRALgeb[1].split(' ')[0])
dz = float(GRALgeb[2].split(',')[0])
GRAL_nx = int(GRALgeb[3].split(' ')[0])
GRAL_ny = int(GRALgeb[4].split(' ')[0])
nslice = int(GRALgeb[5].split(' ')[0])
sourcegroups = list(map(int, GRALgeb[6].split(',')[0:-1]))
GRAL_xmin = int(GRALgeb[7].split(' ')[0])
GRAL_xmax = int(GRALgeb[8].split(' ')[0])
GRAL_ymin = int(GRALgeb[9].split(' ')[0])
GRAL_ymax = int(GRALgeb[10].split(' ')[0])
# Messstationen in Grid
station_grid_gral = [(stations[2] - GRAL_xmin) / GRAL_dx, (stations[3] - GRAL_ymin) / GRAL_dy]  ###
station_grid_gral = list(np.float_(station_grid_gral))
# station_grid_gral = np.around(station_grid_gral)
station_grid_gral = np.short(station_grid_gral)  # [x,y] Gitterpunkte in Grid für Messstationen
station_grid_gral[0][9] = 529


def read_GRAL_geometries(file):
    if zipfile.is_zipfile(file):
        gralgeom = zipfile.ZipFile(file, 'r')
        for filename in gralgeom.namelist():
            byte_list = gralgeom.read(filename)
            gralgeom.close()
    else:
        with open(file, mode='rb') as binfile:
            byte_list = binfile.read()
            binfile.close()
    nheader = 32
    header = byte_list[:nheader]
    nz, ny, nx, ikooagral, jkooagral, dzk, stretch, ahmin = struct.unpack('iiiiifff', header)
    blub = byte_list[nheader:]
    # float and int32 -> 4byte each
    datarr = np.zeros([nx + 1, ny + 1, 3])
    c = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            datarr[i, j, 0] = np.frombuffer(blub[c:(c + 4)], dtype=np.float32)
            datarr[i, j, 1] = np.frombuffer(blub[(c + 4):(c + 8)], dtype=np.int32)
            datarr[i, j, 2] = np.frombuffer(blub[(c + 8):(c + 12)], dtype=np.float32)
            c += 12
    ahk = datarr[:, :, 0]  # surface elevation
    bui_height = datarr[:, :, 2]  # building height
    kkart = datarr[:, :, 1].astype(int)  # index of gral surface
    return ahk, kkart, bui_height


ahk, kkart, buiheight = read_GRAL_geometries(folder + 'GRAL_geometries.txt')
folder = "HD_GRAL_new_topo_iup/"
cases = []
for i in np.arange(0, 5):
    c = i + 1
    case = "0000"
    if c > 9:
        case = "000"
        if c > 99:
            case = "00"
            if c > 999:
                case = "0"
                if c > 9999:
                    case = ""
    case = case + str(c)
    cases.append(case)


def fill_station(winds, dataarray, ux, uy, uz, station_index, case_index):
    wind = dataarray[ux, uy, uz, :] * 0.01
    winds[:, station_index, case_index] = wind
    return winds


winds = []
for i in range(0, len(cases)):
    if zipfile.is_zipfile(folder + cases[i] + ".gff"):
        gff = zipfile.ZipFile(folder + cases[i] + ".gff", 'r')
        try:
            byte_list = gff.read(gff.namelist()[0])
            gff.close()
            file = open(FOLDER + (cases[i]) + ".gff", 'rb')
            data = file.read()
            file.close()
            nheader = 32  # header, ni,nj,nk,gridsiye -> 4*signed integer (=4*4) + float (4)
            header = byte_list[:nheader]
            nz, ny, nx, direction, speed, akla, dxy, h = struct.unpack('iiiffifi', header)
            # convert direction to degree (strange, but seems to work)
            direction = 270. - math.degrees(direction)
            dt = np.dtype(np.short)
            count = (nx + 1) * (ny + 1) * (nz + 1) * 3
            data = np.fromstring(byte_list[nheader:len(byte_list)], dtype=dt,
                                 count=count, sep='')
            data = np.reshape(data, [nx + 1, ny + 1, nz + 1, 3])  # velocities stored in cm/s, convert to m/s
            wind = data[station_grid_gral[0][:], station_grid_gral[1][:], :, :] * 0.01  # convert to m/s
            winds.append(wind)
            print(f"catalogue number {i} appended")
        except FileNotFoundError:
            print("file not found! skipping...")
            print(f"pups_situation:{i}")
            an_array = np.empty((13, 401, 3))
            an_array[:] = np.NaN
            winds.append(an_array)
            print(f"catalogue number {i} appended")
    else:
        try:
            with open(os.path.join(folder, cases[i] + ".gff"), mode='rb') as binfile:
                byte_list = binfile.read()
                binfile.close()
        except FileNotFoundError:
            print(f"pups_situation:{i}")
            an_array = np.empty((13, 401, 3))
            an_array[:] = np.NaN
            winds.append(an_array)
            continue
winds = np.array(winds)
print(np.shape(winds))
# save winds
output = open('gff1008final_test.pkl', 'wb')
pickle.dump(winds, output)
output.close()
