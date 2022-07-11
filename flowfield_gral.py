import numpy as np
import copy
import pandas as pd
import struct
import matplotlib.pyplot as plt
import zipfile
import math

GRAMM_folder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"
GRAL_folder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"
wfolder = "/mnt/data/users/swald/GRAMM-GRAL/HD_GRAL_new_topo_iup/"

case = "00701"
case_gral = '00701'
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
stations = Stations.to_numpy()

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


def read_building(file):
    nbuild = sum(1 for line in open(file))
    print('reading ', nbuild, ' blocks (from buildings)')
    buildings = np.zeros((nbuild, 3))
    bh = np.zeros((GRAL_nx, GRAL_ny))
    with open(file, 'r+') as infile:
        line = infile.readline()
        cnt = 0
        while line:
            dat = line.split(', ')
            dat[-1] = dat[-1].strip()  # remove newline operator
            buildings[cnt, 0] = np.float(dat[0])
            buildings[cnt, 1] = np.float(dat[1])
            buildings[cnt, 2] = np.float(dat[3])

            i = np.int((np.float(dat[0]) - GRAL_dx / 2 - GRAL_xmin) / GRAL_dx)
            j = np.int((np.float(dat[1]) - GRAL_dy / 2 - GRAL_ymin) / GRAL_dy)
            bh[i, j] = np.float(dat[3])

            line = infile.readline()
            cnt += 1
            if cnt > nbuild:
                print('sth went wrong')
                break
    print('done')
    return buildings, bh


def extract_flowfield(file):
    if zipfile.is_zipfile(file):
        gff = zipfile.ZipFile(file, 'r')

        for filename in gff.namelist():
            byte_list = gff.read(filename)
            gff.close()

    else:
        with open(file, mode='rb') as binfile:
            byte_list = binfile.read()
            binfile.close()

    nheader = 32
    header = byte_list[:nheader]
    nz, ny, nx, direction, speed, akla, dxy, h = struct.unpack('iiiffifi', header)
    # convert direction to degree (strange, but seems to work)
    direction = 270. - math.degrees(direction)

    dt = np.dtype(np.short)

    count = (nx + 1) * (ny + 1) * (nz + 1) * 3
    data = np.fromstring(byte_list[nheader:len(byte_list)], dtype=dt,
                         count=count, sep='')

    data = np.reshape(data, [nx + 1, ny + 1, nz + 1, 3])  # velocities stored in cm/s, convert to m/s
    ux = data[:, :, :, 0] * 0.01
    vy = data[:, :, :, 1] * 0.01
    wz = data[:, :, :, 2] * 0.01

    return ux, vy, wz


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)


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

    return ahk, kkart, bui_height, ahmin


def extract_flowfield(file):
    if zipfile.is_zipfile(file):
        gff = zipfile.ZipFile(file, 'r')

        for filename in gff.namelist():
            byte_list = gff.read(filename)
            gff.close()

    else:
        with open(file, mode='rb') as binfile:
            byte_list = binfile.read()
            binfile.close()

    nheader = 32
    header = byte_list[:nheader]
    nz, ny, nx, direction, speed, akla, dxy, h = struct.unpack('iiiffifi', header)
    # convert direction to degree (strange, but seems to work)
    direction = 270. - math.degrees(direction)

    dt = np.dtype(np.short)

    count = (nx + 1) * (ny + 1) * (nz + 1) * 3
    data = np.fromstring(byte_list[nheader:len(byte_list)], dtype=dt,
                         count=count, sep='')

    data = np.reshape(data, [nx + 1, ny + 1, nz + 1, 3])  # velocities stored in cm/s, convert to m/s
    ux = data[:, :, :, 0] * 0.01
    vy = data[:, :, :, 1] * 0.01
    wz = data[:, :, :, 2] * 0.01

    return ux, vy, wz


# load gral and gramm data

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
umag = np.hypot(wind_u[:, :, :], wind_v[:, :, :])
# print(umag.mean(axis=(0, 1)))
uabove = np.round(2 * umag.mean(axis=(0, 1))[12]) / 2
qspace = 7
level = 1
# zmin=np.max(zgrid[:,:,level]) #elevation of heighest level at which one wants to plot
zmin = 150
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

buildings, bh = read_building(GRAL_folder + "buildings.dat")

u, v, w = extract_flowfield(GRAL_folder + case_gral + ".gff")
print('loaded wind')

ahk, kkart, buiheight, ahmin = read_GRAL_geometries(GRAL_folder + 'GRAL_geometries.txt')

# IUP größerer Berich
station_name = Stations.columns[12]
cmap = plt.get_cmap('BrBG', 256)

scale = 40
qgral = 8
qspace = 8
# lev = 16
zmin = [115, 120, 130, 150]
for k in range(0, len(zmin)):
    lev = (np.round(zmin[k] - ahmin) / 2).astype(int)
    umag = np.hypot(wind_u[:, :, :], wind_v[:, :, :])
    uabove = np.round(2 * umag.mean(axis=(0, 1))[12]) / 2
    zindex = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            zindex[i, j] = np.argmin(np.abs(zgrid[i, j, :] - zmin[k]))
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
    #toplot = copy.deepcopy(topo)
    #levs = np.array([300, 400, 500, 600, 700, 800, 900])
    #cs = axs.contour(GRAMM_xcemesh[::1, ::1], GRAMM_ycemesh[::1, ::1], toplot[::1, ::1], levels=levs, colors='r')

    # IUP

    fig = axs.figure
    toplot = copy.deepcopy(heatcondum)
    # buildings
    pcm = axs.pcolormesh(GRAL_xcomesh, GRAL_ycomesh, bh, vmin=0, vmax=120, cmap=cmap)
    ax = pcm.axes
    plt.plot(Stations[station_name][2], Stations[station_name][3], 'mD', markersize=10)
    plt.text(Stations[station_name][2] + 20, Stations[station_name][3] - 22, station_name, fontsize=15, color='m')
    lab = 'Length = ' + str(np.ceil(uabove) / 2)
    q = axs.quiver(GRAL_xcomesh[::qgral, ::qgral], GRAL_ycomesh[::qgral, ::qgral], u[::qgral, ::qgral, lev],
                   v[::qgral, ::qgral, lev], (w[::qgral, ::qgral, lev]), scale=scale, color="blue")
    # q=axs.quiver(GRAMM_xcemesh[::qspace,::qspace], GRAMM_ycemesh[::qspace,::qspace],ufield,vfield,scale=scale, color="yellow")
    # q = axs.quiver(Stations[station_name][2]-10, Stations[station_name][3]+10, 0.046, -0.749, scale=scale, color="b")
    axs.quiverkey(q, X=0.8, Y=0.89, U=np.ceil(uabove) / 2, label=lab, labelpos='E', coordinates='figure')
    #axs.clabel(cs, fmt='%d')
    axs.set_title('quiver [m s$^{-1}$]')
    axs.tick_params(axis='x', rotation=20)
    axs.tick_params(axis='y', rotation=70)
    plt.xlim(Stations[station_name][2] - 1500, Stations[station_name][2] + 1500)
    plt.ylim(Stations[station_name][3] - 1500, Stations[station_name][3] + 1500)
    plt.plot(Stations["IUP"][2], Stations["IUP"][3], 'mD', markersize=12, color="lightgrey")
    plt.text(Stations["IUP"][2] + 20, Stations["IUP"][3] - 30, "IUP", fontsize=17, color='lightgrey')
    plt.savefig(f"plots/IUPwindfelder{k}.png",bbox_inches='tight')
    plt.show()

# IUP
station_name = Stations.columns[12]
# cmap = plt.get_cmap('cubehelix_r', 256)

scale = 40
qgral = 1
qspace = 1
# lev = 16
zmin = [115, 120, 130, 150]
for k in range(0, len(zmin)):
    lev = (np.round(zmin[k] - ahmin) / 2).astype(int)
    umag = np.hypot(wind_u[:, :, :], wind_v[:, :, :])
    uabove = np.round(2 * umag.mean(axis=(0, 1))[12]) / 2
    zindex = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            zindex[i, j] = np.argmin(np.abs(zgrid[i, j, :] - zmin[k]))
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
    levs = np.array([300, 400, 500, 600, 700, 800, 900])
    cs = axs.contour(GRAMM_xcemesh[::1, ::1], GRAMM_ycemesh[::1, ::1], toplot[::1, ::1], levels=levs, colors='r')

    # IUP

    fig2 = axs.figure
    toplot = copy.deepcopy(heatcondum)
    cs2 = axs.contour(GRAMM_xcemesh, GRAMM_ycemesh, toplot, levels=levs, colors='r')
    # buildings
    pcm = axs.pcolormesh(GRAL_xcomesh, GRAL_ycomesh, bh, vmin=0, vmax=120, cmap=cmap)
    ax = pcm.axes
    plt.plot(Stations[station_name][2], Stations[station_name][3], 'mD', markersize=10)
    plt.text(Stations[station_name][2] + 20, Stations[station_name][3] - 22, station_name, fontsize=15, color='m')
    lab = 'Length = ' + str(np.ceil(uabove) / 2)
    q = axs.quiver(GRAL_xcomesh[::qgral, ::qgral], GRAL_ycomesh[::qgral, ::qgral], u[::qgral, ::qgral, lev],
                   v[::qgral, ::qgral, lev], (w[::qgral, ::qgral, lev]), scale=scale, color="g")
    # q=axs.quiver(GRAMM_xcemesh[::qspace,::qspace], GRAMM_ycemesh[::qspace,::qspace],ufield,vfield,scale=scale, color="r")
    q = axs.quiver(Stations[station_name][2] - 20, Stations[station_name][3] + 20, 0.046, -0.749, scale=scale,
                   color="b")
    axs.quiverkey(q, X=0.8, Y=0.89, U=np.ceil(uabove) / 2, label=lab, labelpos='E', coordinates='figure')
    axs.clabel(cs, fmt='%d')
    axs.set_title('quiver [m s$^{-1}$]')
    axs.tick_params(axis='x', rotation=20)
    axs.tick_params(axis='y', rotation=70)
    plt.xlim(Stations[station_name][2] - 100, Stations[station_name][2] + 100)
    plt.ylim(Stations[station_name][3] - 100, Stations[station_name][3] + 100)
    plt.savefig(f"plots/sit{case}_{zmin[k]}_iup_windfield.png", bbox_inches='tight')
    plt.show()

    # IUP zoom
    fig3 = axs.figure
    toplot = copy.deepcopy(heatcondum)
    cs2 = axs.contour(GRAMM_xcemesh, GRAMM_ycemesh, toplot, levels=levs, colors='r')
    # buildings
    pcm = axs.pcolormesh(GRAL_xcomesh, GRAL_ycomesh, bh, vmin=0, vmax=120, cmap=cmap)
    ax = pcm.axes
    plt.plot(Stations[station_name][2], Stations[station_name][3], 'mD', markersize=10)
    plt.text(Stations[station_name][2] + 20, Stations[station_name][3] - 22, station_name, fontsize=18, color='m')
    lab = 'Length = ' + str(np.ceil(uabove) / 2)
    q = axs.quiver(GRAL_xcomesh[::qgral, ::qgral], GRAL_ycomesh[::qgral, ::qgral], u[::qgral, ::qgral, lev],
                   v[::qgral, ::qgral, lev], (w[::qgral, ::qgral, lev]), scale=scale, color="g")
    # q=axs.quiver(GRAMM_xcemesh[::qspace,::qspace], GRAMM_ycemesh[::qspace,::qspace],ufield,vfield,scale=scale, color="r")
    q = axs.quiver(Stations[station_name][2] - 10, Stations[station_name][3] + 10, 0.046, -0.749, scale=scale,
                   color="b")
    axs.quiverkey(q, X=0.8, Y=0.89, U=np.ceil(uabove) / 2, label=lab, labelpos='E', coordinates='figure')
    axs.clabel(cs, fmt='%d')
    axs.set_title('quiver [m s$^{-1}$]', fontsize=20)
    axs.tick_params(axis='x', rotation=20, labelsize=18)
    axs.tick_params(axis='y', rotation=70, labelsize=18)
    plt.xlim(Stations[station_name][2] - 70, Stations[station_name][2] + 70)
    plt.ylim(Stations[station_name][3] - 70, Stations[station_name][3] + 70)
    plt.savefig(f"plots/sit{case}_{zmin[k]}_iup_windfield.png", bbox_inches='tight')
    plt.show()
