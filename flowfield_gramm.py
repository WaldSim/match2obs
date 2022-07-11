import zipfile
import struct
import math
import copy
import numpy as np
import numpy.ma as ma
import string
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm, ListedColormap, LinearSegmentedColormap, BoundaryNorm, SymLogNorm

import zipfile
import struct
import math
import copy
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm, ListedColormap, LinearSegmentedColormap, BoundaryNorm, SymLogNorm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.patches as patches

cmap = plt.get_cmap('BrBG', 256)
GRAMM_folder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"
GRAL_folder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"

file = open(GRAMM_folder + "GRAMM.geb", 'r')
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

# angepasst am 11.12.20 Sanam Vardag
GRAMM_xcenter = np.linspace(GRAMM_xmin, GRAMM_xmax, num=GRAMM_nx) + 0.5 * GRAMM_dx
GRAMM_ycenter = np.linspace(GRAMM_ymin, GRAMM_ymax, num=GRAMM_ny) + 0.5 * GRAMM_dy
GRAMM_ycemesh, GRAMM_xcemesh = np.meshgrid(GRAMM_ycenter, GRAMM_xcenter, indexing='xy')

GRAMM_xcorner = np.linspace(GRAMM_xmin, GRAMM_xmax, num=GRAMM_nx)
GRAMM_ycorner = np.linspace(GRAMM_ymin, GRAMM_ymax, num=GRAMM_ny)
GRAMM_ycomesh, GRAMM_xcomesh = np.meshgrid(GRAMM_ycorner, GRAMM_xcorner, indexing='xy')
fic = GRAMM_folder + 'ggeom.asc'
f = open(fic, 'r')
data = f.readlines()
f.close()

tmp = data[1].split()
topo_ind = 0
topo = np.zeros((GRAMM_nx, GRAMM_ny), np.float)
for i in range(GRAMM_ny):
    for j in range(GRAMM_nx):
        topo[j, i] = float(tmp[topo_ind])
        topo_ind += 1

# Z Grid
tmp = data[8].split()
zgrid = np.zeros([GRAMM_nx, GRAMM_ny, GRAMM_nz], np.float)
ind = 0
for k in range(GRAMM_nz):
    for j in range(GRAMM_ny):
        for i in range(GRAMM_nx):
            zgrid[i, j, k] = float(tmp[ind])
            ind += 1

toplot = copy.deepcopy(topo)

##############
# Topography #
##############

fic = GRAMM_folder + 'ggeom.asc'
f = open(fic, 'r')
data = f.readlines()
f.close()

tmp = data[1].split()
topo_ind = 0
topo = np.zeros((GRAMM_nx, GRAMM_ny), np.float)
for i in range(GRAMM_ny):
    for j in range(GRAMM_nx):
        topo[j, i] = float(tmp[topo_ind])
        topo_ind += 1

landuse_ind = 0

heatcondu = data[1].split()
heatcondu = [float(i) for i in heatcondu]
heatcondum = np.zeros((GRAMM_nx, GRAMM_ny), np.float)
for i in range(GRAMM_ny):
    for j in range(GRAMM_nx):
        heatcondum[j, i] = float(heatcondu[landuse_ind])
        landuse_ind += 1

    # Z layers
tmp = data[0].split()
zalt = []
for l in range(GRAMM_nz):
    zalt.append(float(tmp[-1 - l]))
zalt.reverse()

# Z Grid
tmp = data[8].split()
zgrid = np.zeros([GRAMM_nx, GRAMM_ny, GRAMM_nz], np.float)
ind = 0
for k in range(GRAMM_nz):
    for j in range(GRAMM_ny):
        for i in range(GRAMM_nx):
            zgrid[i, j, k] = float(tmp[ind])
            ind += 1

# plot flow field
file = open(GRAL_folder + "GRAL.geb", 'r')
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

GRAL_xcenter = np.arange(GRAL_xmin, GRAL_xmax, GRAL_dx) + 0.5 * GRAL_dx
GRAL_ycenter = np.arange(GRAL_ymin, GRAL_ymax, GRAL_dy) + 0.5 * GRAL_dy
GRAL_ycemesh, GRAL_xcemesh = np.meshgrid(GRAL_ycenter, GRAL_xcenter, indexing='xy')

GRAL_xcorner = np.arange(GRAL_xmin, GRAL_xmax + GRAL_dx, GRAL_dx)
GRAL_ycorner = np.arange(GRAL_ymin, GRAL_ymax + GRAL_dy, GRAL_dy)
GRAL_ycomesh, GRAL_xcomesh = np.meshgrid(GRAL_ycorner, GRAL_xcorner, indexing='xy')

# choose folder

folder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"
wfolder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"

STATIONDATA_JUL = {"GHW": ["GHW1", "GHW2", 3472888.990, 5475770.394, 117],
                   "HP": ["HP1", "HP2", 3477413.573, 5470824.362, 120],
                   "KOS": ["KOS1", "KOS2", 3480342.295, 5474125.002, 566],
                   "STW": ["STW1", "STW2", 3480078.171, 5473526.270, 581],
                   "PT": ["PT1", "PT2", 3482028.905, 5479563.560, 354],
                   # "SB" : ["SB1","SB2", 3483945.193, 5474103.973, 120],
                   "STB": ["STB1", "STB2", 3477329.729, 5474470.342, 134],
                   "THB": ["THB1", "THB2", 3477783.586, 5475052.152, 122],
                   "WWR": ["WWR1", "WWR2", 3472735.935, 5476113.269, 116],
                   "KOE": ["KOE1", "KOE2", 3481930.604, 5476005.216, 230],
                   "CZE": ["CZE1", "CZE2", 3476559.336, 5473792.047, 150],
                   "GAB": ["GAB1", "GAB2", 3481423.178, 5469356.975, 331],
                   "LUBW": ["LUBW1", "LUBW2", 3476616.415, 5475898.736, 122],
                   "IUP": ["IUP1", "IUP2", 3476454.968, 5475644.404, 152], }

Stations = pd.DataFrame(STATIONDATA_JUL, columns=["GHW", "HP", "KOS", "STW", "PT",
                                                  # "SB",
                                                  "STB", "THB", "WWR", "KOE", "CZE", "GAB", "LUBW", "IUP"])

nstations = Stations.shape[1]
for i in np.arange(700, 701):
    c = i + 1
    outfolderw = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"
    infolderw = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"
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

    file = open(wfolder + case + ".wnd", 'rb')
    print(file)
    data = file.read()
    file.close()

    nheader = 20  # header, ni,nj,nk,gridsize -> 4*signed integer (=4*4) + float (4)
    header, nx, ny, nz, dx = struct.unpack('<iiiif', data[:nheader])

    dt = np.dtype(np.short)

    datarr = np.frombuffer(data[nheader:], dtype=dt)

    datarr = np.reshape(datarr, [nx, ny, nz, 3])
    wind_u = datarr[:, :, :, 0] * 0.01
    wind_v = datarr[:, :, :, 1] * 0.01
    wind_w = datarr[:, :, :, 2] * 0.01
    #   wind_u = np.flip(wind_u,2)
    #   wind_v = np.flip(wind_v,2)
    #   wind_w = np.flip(wind_w,2)
    umag = np.hypot(wind_u[:, :, :], wind_v[:, :, :])
    uabove = np.round(2 * umag.mean(axis=(0, 1))[12]) / 2
    qspace = 7
    level = 1
    zmin = np.max(zgrid[:, :, level])  # elevation of heighest level at which one wants to plot
    zindex = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            zindex[i, j] = np.argmin(np.abs(zgrid[i, j, :] - zmin))

    # ideally one should interpolate on the same height
    unx = np.int(np.ceil(GRAMM_nx / qspace))
    uny = np.int(np.ceil(GRAMM_ny / qspace))
    ufield = np.zeros([unx, uny])
    vfield = np.zeros([unx, uny])
    for i in range(unx):
        for j in range(uny):
            ii = (qspace * i)
            jj = (qspace * j)
            ufield[i, j] = wind_u[ii, jj, zindex[ii, jj]]
            vfield[i, j] = wind_v[ii, jj, zindex[ii, jj]]

    fig1, axs = plt.subplots(1, 1, figsize=(15, 15), facecolor='w', edgecolor='k')
    toplot = copy.deepcopy(topo)
    levs = np.array([120, 200, 300, 400, 500])
    cs = axs.contour(GRAMM_xcemesh[::1, ::1], GRAMM_ycemesh[::1, ::1], toplot[::1, ::1], levels=levs, colors='r')

    toplot = copy.deepcopy(heatcondum)
    levs = np.array([98])
    lab = 'Length = ' + str(uabove)
    q = axs.quiver(GRAMM_xcemesh[::qspace, ::qspace], GRAMM_ycemesh[::qspace, ::qspace], ufield, vfield,
                   scale=25 * np.log(1.5 + uabove))
    axs.quiverkey(q, X=0.8, Y=0.89, U=uabove, label=lab, labelpos='E', coordinates='figure')
    axs.clabel(cs, fmt='%d')
    axs.set_title('quiver [m s$^{-1}$]', fontsize=20)
    axs.tick_params(axis='x', rotation=20, labelsize=18)
    axs.tick_params(axis='y', rotation=70, labelsize=18)
    for station in Stations.columns:  ##### nachträglich eingesetzt
        plt.plot(Stations[station][2], Stations[station][3], 'mD', markersize=15)
        plt.text(Stations[station][2] + 200, Stations[station][3] - 220, station, fontsize=18, color='m')
    rect = patches.Rectangle((GRAL_xmin, GRAL_ymin), GRAL_xmax - GRAL_xmin, GRAL_ymax - GRAL_ymin, linewidth=3,
                             edgecolor='purple', fill=False)
    axs.add_patch(rect)

    # fig1.savefig(pltfolder+case+'_quiverabove_level'+str(level)+'.png', dpi=150)
    fig1.savefig(f"plots/{case}at{zmin}GRAMM_windfield.png", dpi=200, bbox_inches='tight')
    qspace = 1  # how sparce quivers should be
    unx = np.int(np.ceil(GRAMM_nx / qspace))
    uny = np.int(np.ceil(GRAMM_ny / qspace))
    ufield = np.zeros([unx, uny])
    vfield = np.zeros([unx, uny])

    for i in range(unx):
        for j in range(uny):
            ii = (qspace * i)
            jj = (qspace * j)
            ufield[i, j] = wind_u[ii, jj, zindex[ii, jj]]
            vfield[i, j] = wind_v[ii, jj, zindex[ii, jj]]

    fig1, axs = plt.subplots(1, 1, figsize=(15, 15), facecolor='w', edgecolor='k')
    toplot = copy.deepcopy(topo)
    levs = np.array([100, 200, 300, 400, 500, 600, 700])
    cs = axs.contour(GRAMM_xcemesh[::1, ::1], GRAMM_ycemesh[::1, ::1], toplot[::1, ::1], levels=levs, colors='r')

    toplot = copy.deepcopy(heatcondum)
    levs = np.array([98])
    cs2 = axs.contour(GRAMM_xcemesh, GRAMM_ycemesh, toplot, levels=levs, colors='b')

    for station in Stations.columns:
        plt.plot(Stations[station][2], Stations[station][3], 'mD', markersize=15)
        plt.text(Stations[station][2] + 200, Stations[station][3] - 220, station, fontsize=18, color='m')

    lab = 'Length = ' + str(uabove)
    q = axs.quiver(GRAMM_xcemesh[::qspace, ::qspace], GRAMM_ycemesh[::qspace, ::qspace], ufield, vfield,
                   scale=25 * np.log(1.5 + uabove))
    axs.quiverkey(q, X=0.8, Y=0.89, U=uabove, label=lab, labelpos='E', coordinates='figure')
    axs.clabel(cs, fmt='%d')
    axs.set_title('quiver [m s$^{-1}$]', fontsize=20)
    axs.tick_params(axis='x', rotation=20, labelsize=18)
    axs.tick_params(axis='y', rotation=70, labelsize=18)
    # plt.ylim((5465000,5485000))
    # plt.xlim((3466700 ,3486700))
    # plt.xlim(3476140.234, 3478306.675)
    # plt.ylim(5475899.796, 5474145.617)
    fig1.savefig(f"plots/{case}at{zmin}GRAMM_windfield.png", dpi=150)
    plt.show()

    zmin = 120

    qspace = 5  # how sparce quivers should be
    unx = np.int(np.ceil(GRAMM_nx / qspace))
    uny = np.int(np.ceil(GRAMM_ny / qspace))
    zmask = np.zeros([GRAMM_nx, GRAMM_ny], np.float)
    zminindex = np.zeros([GRAMM_nx, GRAMM_ny], np.int)
    ufield = np.zeros([unx, uny])
    vfield = np.zeros([unx, uny])

    for i in range(GRAMM_nx):
        for j in range(GRAMM_ny):
            zminindex[i, j] = np.argmin(np.abs(zgrid[i, j, :] - zmin))
            zmask[i, j] = topo[i, j] - zmin
    zmask = zmask > 20.

    for i in range(GRAMM_nx):
        for j in range(GRAMM_ny):
            blub = ma.masked_array(wind_u[:, :, zminindex[i, j]], mask=zmask)
            moep = ma.masked_array(wind_v[:, :, zminindex[i, j]], mask=zmask)

    for i in range(unx):
        for j in range(uny):
            ii = (qspace * i)
            jj = (qspace * j)
            ufield[i, j] = blub[ii, jj]
            vfield[i, j] = moep[ii, jj]

    fig2, axs = plt.subplots(1, 1, figsize=(15, 15), facecolor='w', edgecolor='k')
    toplot = copy.deepcopy(topo)
    levs = np.array([300, 400, 500, 600, 700, 800, 900])
    cs = axs.contour(GRAMM_xcemesh[::1, ::1], GRAMM_ycemesh[::1, ::1], toplot[::1, ::1], levels=levs, colors='r')

    toplot = copy.deepcopy(heatcondum)
    levs = np.array([98])
    cs2 = axs.contour(GRAMM_xcemesh, GRAMM_ycemesh, toplot, levels=levs, colors='b')
    for station in Stations.columns:
        if station == 'HEU':
            continue
        plt.plot(Stations[station][2], Stations[station][3], 'mD', markersize=15)
        plt.text(Stations[station][2] + 200, Stations[station][3] - 220, station, fontsize=18, color='m')
    rect = patches.Rectangle((GRAL_xmin, GRAL_ymin), GRAL_xmax - GRAL_xmin, GRAL_ymax - GRAL_ymin, linewidth=3,
                             edgecolor='purple', fill=False)
    axs.add_patch(rect)
    lab = 'Length = ' + str(np.ceil(uabove) / 2)
    q = axs.quiver(GRAMM_xcemesh[::qspace, ::qspace], GRAMM_ycemesh[::qspace, ::qspace], ufield, vfield,
                   scale=30 * np.log(1.5 + uabove))
    axs.quiverkey(q, X=0.8, Y=0.89, U=np.ceil(uabove) / 2, label=lab, labelpos='E', coordinates='figure')
    axs.clabel(cs, fmt='%d')
    axs.set_title('quiver [m s$^{-1}$]', fontsize=20)
    axs.tick_params(axis='x', rotation=20, labelsize=18)
    axs.tick_params(axis='y', rotation=70, labelsize=18)
    # plt.ylim((5465000,5485000))
    # plt.xlim((3466700 ,3486700))
    fig2.savefig(f"plots/{case}at{zmin}GRAMM_windfield.png", dpi=200, bbox_inches='tight')
    plt.show()


