import numpy as np
import csv
import zipfile
import struct
import math

# defined functions
###

# round to significant digits
def round_sig(x, sig=2):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

#
def ReturnGRAMMGeometry(pathtogeometry):
    file = open(pathtogeometry,'r')
    GRAMMgeb=file.readlines()

    GRAMM_nx=int(GRAMMgeb[0].split(' ')[0])
    GRAMM_ny=int(GRAMMgeb[1].split(' ')[0])
    GRAMM_nz=int(GRAMMgeb[2].split(' ')[0])

    GRAMM_xmin=int(GRAMMgeb[3].split(' ')[0])
    GRAMM_xmax=int(GRAMMgeb[4].split(' ')[0])

    GRAMM_ymin=int(GRAMMgeb[5].split(' ')[0])
    GRAMM_ymax=int(GRAMMgeb[6].split(' ')[0])

    GRAMM_dy=(GRAMM_ymax-GRAMM_ymin)/GRAMM_ny
    GRAMM_dx=(GRAMM_xmax-GRAMM_xmin)/GRAMM_nx
    
    GRAMM_xcenter = np.arange(GRAMM_xmin, GRAMM_xmax, GRAMM_dx) + 0.5 * GRAMM_dx
    GRAMM_ycenter = np.arange(GRAMM_ymin, GRAMM_ymax, GRAMM_dy) + 0.5 * GRAMM_dy
    GRAMM_xcemesh, GRAMM_ycemesh = np.meshgrid(GRAMM_xcenter, GRAMM_ycenter, indexing='xy')

    GRAMM_xcorner = np.arange(GRAMM_xmin, GRAMM_xmax+GRAMM_dx, GRAMM_dx)
    GRAMM_ycorner = np.arange(GRAMM_ymin, GRAMM_ymax+GRAMM_dy, GRAMM_dy)
    GRAMM_xcomesh, GRAMM_ycomesh = np.meshgrid(GRAMM_xcorner, GRAMM_ycorner, indexing='xy')
    
    return GRAMM_nx, GRAMM_ny, GRAMM_nz, GRAMM_xmin, GRAMM_xmax, GRAMM_ymin, GRAMM_ymax, GRAMM_dx, GRAMM_dy, GRAMM_ycemesh, GRAMM_xcemesh, GRAMM_ycomesh, GRAMM_xcomesh

#
def ReturnTopography(pathtoggeom, GRAMM_nx, GRAMM_ny, GRAMM_nz):
    f = open(pathtoggeom,'r')
    data = f.readlines()
    f.close()
    tmp=data[1].split()
    topo_ind=0
    topo=np.zeros((GRAMM_nx,GRAMM_ny),np.float)
    for j in range(GRAMM_ny):
        for i in range(GRAMM_nx):
            topo[i,j]=float(tmp[topo_ind])
            topo_ind+=1

    # Z Grid
    tmp=data[8].split()
    zgrid=np.zeros([GRAMM_nx,GRAMM_ny,GRAMM_nz],np.float)
    ind=0
    for k in range(GRAMM_nz):
        for j in range(GRAMM_ny):
            for i in range(GRAMM_nx):
                zgrid[i,j,k]=float(tmp[ind])
                ind+=1
            
    return topo, zgrid


def ReturnLanduse(pathtoLU, GRAMM_nx, GRAMM_ny):
    f=open(pathtoLU,'r')
    data=f.readlines()
    f.close()

    thermalcondu=data[0].split()  #it is actually heatcondu/thermalcondu/900  ->  thermalcondu=1/(data[0]*900/data[1])
    thermalcondu = [float(i) for i in thermalcondu]

    heatcondu=data[1].split()
    heatcondu = [float(i) for i in heatcondu]

    thermalcondu=np.divide(1,np.divide(thermalcondu,heatcondu)*900)
    thermalcondu = [round_sig(i) for i in thermalcondu]

    roughness=data[2].split()
    roughness = [float(i) for i in roughness]

    moisture=data[3].split()
    moisture = [float(i) for i in moisture]

    emiss=data[4].split()
    emiss = [float(i) for i in emiss]

    albedo=data[5].split()
    albedo = [float(i) for i in albedo]

    landuse_ind=0
    thermalcondum=np.zeros((GRAMM_nx,GRAMM_ny),np.float)
    heatcondum=np.zeros((GRAMM_nx,GRAMM_ny),np.float)
    roughnessm=np.zeros((GRAMM_nx,GRAMM_ny),np.float)
    moisturem=np.zeros((GRAMM_nx,GRAMM_ny),np.float)
    emissm=np.zeros((GRAMM_nx,GRAMM_ny),np.float)
    albedom=np.zeros((GRAMM_nx,GRAMM_ny),np.float)

    for i in range(GRAMM_ny):
        for j in range(GRAMM_nx):
            thermalcondum[j,i]=float(thermalcondu[landuse_ind])
            heatcondum[j,i]=float(heatcondu[landuse_ind])
            roughnessm[j,i]=float(roughness[landuse_ind])
            moisturem[j,i]=float(moisture[landuse_ind])
            emissm[j,i]=float(emiss[landuse_ind])
            albedom[j,i]=float(albedo[landuse_ind])
            landuse_ind+=1  
    
    return thermalcondum, heatcondum, roughnessm, moisturem, emissm, albedom

        
        
# reads GRAL buildings.dat and creates an array and maskarray
# usage: Buildings, BuildingMask = ReturnBuildings(path+'GRAL/buildings.dat', geometry)
def ReturnBuildings(pathtobuildingsdat, geometry):
    GRAL_nx, GRAL_ny, GRAL_dx, GRAL_dy, GRAL_xmin, GRAL_ymin = geometry
    build=np.zeros([GRAL_nx,GRAL_ny])
    buildm=np.ones([GRAL_nx,GRAL_ny])
    with open(pathtobuildingsdat, newline='') as csvfile:
        buildings = csv.reader(csvfile, delimiter=',')
        for row in buildings:
            i=np.int((np.float(row[0])-GRAL_dx/2-GRAL_xmin)/GRAL_dx)
            j=np.int((np.float(row[1])-GRAL_dy/2-GRAL_ymin)/GRAL_dy)
            build[i,j]=np.float(row[3])
            buildm[i,j]=0
    build = np.ma.masked_array(build, mask=buildm)
    return build, buildm

def ReturnGRALconfig(pathtoindat):
    file = open(pathtoindat,'r')
    GRALin=file.readlines()

    file.close()

    particles=int(GRALin[0].split(' ')[0])
    pollutant = GRALin[8].split(' ')[0]
    slicethick = float(GRALin[10].split(' ')[0])

    slices = GRALin[9].split(',')[:]
    #test_list = [int(i) for i in test_list]
    for i,j in enumerate(slices):
        if j.isnumeric():
            slices[i]=float(j)
        else: 
            slices.remove(j)

    nslices = np.shape(slices)[0]
    
    return particles, pollutant, slices, nslices, slicethick

    
# read GRAL GRAL.geb do determine simulation domain
# usage: geometry, nslice, sourcegroups, GRAL_xmax, GRAL_ymax, GRAL_ycemesh, GRAL_xcemesh, GRAL_ycomesh, GRAL_xcomesh = ReturnGeometry(path+'GRAL.geb')
def ReturnGeometry(pathtogeometry):
    file = open(pathtogeometry,'r')
    GRALgeb=file.readlines()

    file.close()

    GRAL_dx=float(GRALgeb[0].split(' ')[0])
    GRAL_dy=float(GRALgeb[1].split(' ')[0])
    dz=float(GRALgeb[2].split(',')[0])
    GRAL_nx=int(GRALgeb[3].split(' ')[0])
    GRAL_ny=int(GRALgeb[4].split(' ')[0])
    nslice=int(GRALgeb[5].split(' ')[0])
    sourcegroups=list(map(int,GRALgeb[6].split(',')[0:-1]))
    GRAL_xmin=int(GRALgeb[7].split(' ')[0])
    GRAL_xmax=int(GRALgeb[8].split(' ')[0])
    GRAL_ymin=int(GRALgeb[9].split(' ')[0])
    GRAL_ymax=int(GRALgeb[10].split(' ')[0])
    xscale=GRAL_nx/GRAL_ny

    #GRAL output is actually 1 bigger than expected from domain declaration -> add one cell
    GRAL_xcenter = np.arange(GRAL_xmin, GRAL_xmax, GRAL_dx) + 0.5 * GRAL_dx
    GRAL_ycenter = np.arange(GRAL_ymin, GRAL_ymax, GRAL_dy) + 0.5 * GRAL_dy
    GRAL_xcemesh, GRAL_ycemesh = np.meshgrid(GRAL_xcenter, GRAL_ycenter, indexing='xy')

    GRAL_xcorner = np.arange(GRAL_xmin, GRAL_xmax+GRAL_dx, GRAL_dx)
    GRAL_ycorner = np.arange(GRAL_ymin, GRAL_ymax+GRAL_dy, GRAL_dy)
    GRAL_xcomesh, GRAL_ycomesh = np.meshgrid(GRAL_xcorner, GRAL_ycorner, indexing='xy')
    return [GRAL_nx, GRAL_ny, GRAL_dx, GRAL_dy, GRAL_xmin, GRAL_ymin], nslice, sourcegroups, GRAL_xmax, GRAL_ymax, GRAL_ycemesh, GRAL_xcemesh, GRAL_ycomesh, GRAL_xcomesh

# extracts flow field from GRAL .gff files
# usage: u, v, w = extract_flowfield(path+'00001.gff')
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
    #data = np.fromstring(byte_list[nheader:len(byte_list)], dtype=dt, count=count, sep='')
    data = np.frombuffer(byte_list[nheader:len(byte_list)], dtype=dt,
                         count=count)
    
    data = np.reshape(data, [nx + 1, ny + 1, nz + 1, 3])
    ux = data[:, :, :, 0] * 0.01
    vy = data[:, :, :, 1] * 0.01
    wz = data[:, :, :, 2] * 0.01
    
    print("done loading GRAL flowfields") 
    return ux, vy, wz

def extract_gramm_flowfield(file, nx, ny, nz):

    file = open(file,'rb')
    data=file.read()
    file.close()

    nheader = 20 #header, ni,nj,nk,gridsize -> 4*signed integer (=4*4) + float (4)
    header, nx, ny, nz, dx = struct.unpack('<iiiif',data[:nheader])

    dt = np.dtype(np.short) 

    datarr = np.frombuffer(data[nheader:], dtype=dt) 
    datarr = np.reshape(datarr, [nx, ny, nz, 3])     
    wind_u = datarr[:,:,:,0]*0.01
    wind_v = datarr[:,:,:,1]*0.01
    wind_w = datarr[:,:,:,2]*0.01
    umag=np.hypot(wind_u[:,:,:],wind_v[:,:,:])
    print("done loading GRAMM flowfields") 
    return wind_u, wind_v, wind_w, umag

# extracts data about the GRAL surface elevation
# usage: ahk, kkart, buiheight, oro = read_GRAL_geometries(path+'GRAL_geometries.txt')
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

    blub=byte_list[nheader:]
    #float and int32 -> 4byte each
    #somehow the array is padded with 0? Therefore it is 1 cell bigger in x- and y-dimension
    datarr=np.zeros([nx+1,ny+1,3])
    c=0
    for i in range(nx+1):
        for j in range(ny+1):
            datarr[i,j,0]=np.frombuffer(blub[c:(c+4)],dtype=np.float32)
            datarr[i,j,1]=np.frombuffer(blub[(c+4):(c+8)],dtype=np.int32)
            datarr[i,j,2]=np.frombuffer(blub[(c+8):(c+12)],dtype=np.float32)
            c+=12

    #remove the padding with zeroes at both ends
    ahk = datarr[:-1, :-1, 0]   #surface elevation
    kkart = datarr[:-1, :-1, 1].astype(int)  #index of gral surface
    bui_height = datarr[:-1, :-1, 2]  #building height
    oro = ahk - bui_height #orography / topography (without buildings!)

    return ahk, kkart, bui_height, oro

# read GRAL concentraions
# together with function load_con
# usage: con=load_con(sit=situation,folder=path,levs=[1],cats=cats)
def conc_field(file, geometry):
    '''
    dir_conc = con file directory
    field = filename of con file (excluding the *con)

    return array contaning the concentration at each cell
    '''
    GRAL_nx, GRAL_ny, GRAL_dx, GRAL_dy, GRAL_xmin, GRAL_ymin = geometry
    conc = np.zeros((GRAL_nx, GRAL_ny), np.float)

    with open(file, "rb") as f:

        # skip header
        f.read(4)

        while True:

            # read data
            data = f.read(12)

            if not data:
                break

            x, y, c = struct.unpack("iif", data)

            i = int(np.floor((x - GRAL_xmin) / GRAL_dx))
            j = int(np.floor((y - GRAL_ymin) / GRAL_dy))

            conc[i, j] = c

    return np.transpose(np.array(conc))

def load_con(sit,folder,levs,cats, geometry):
    GRAL_nx = geometry[0]
    GRAL_ny = geometry[1]
    ncat=np.shape(cats)[0]
    nlevs=np.shape(levs)[0] 
    if sit<10:
        sit="0000"+str(sit)
    elif sit<100:
        sit="000"+str(sit)
    elif sit<1000:
        sit="00"+str(sit)
    elif sit<10000:
        sit="0"+str(sit)
    else:
        sit=str(sit)
    
    dat=np.zeros((nlevs, ncat, GRAL_nx, GRAL_ny),dtype=float)
    print("loading ",nlevs*ncat," files")
    n = 0
    for ilev,lev in enumerate(levs):
        for icat,cat in enumerate(cats):           
            file=folder+str(sit)+"-"+str(lev)+cat+".con"
            print(file)
            dat[ilev,icat,:,:]=np.transpose(conc_field(file, geometry))
            n+=1
    print("done loading concentrations")                                                                        
    return dat