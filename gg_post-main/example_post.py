#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import importlib

import numpy as np

import datetime
#import time

import gg_post_utilities as ggpost
import gg_netcdf_utilities as ggnc
import gg_plot_utilities as ggplot

importlib.reload(ggpost)
importlib.reload(ggnc)
importlib.reload(ggplot)

#plt.rcParams["figure.figsize"] = (15,15)
print("finished module import")


# In[2]:


###############################################################################################################################################################
#Arrays are loaded as (x,y) or (x,y,z)
#Most plotting routines assume (y,x), therefore the arrays are permuted befor plotting
#netcdf are usually in the shape (z,y,x), therefore the arrays are permuted before saving as netcdf
###############################################################################################################################################################


# In[3]:


###############################################################################################################################################################
### Paths #####################################################################################################################################################
###############################################################################################################################################################
pathGRAMM  = 'G:/toNETCDF/GRAMM/'   #GRAMM 
pathGRAL   = 'G:/toNETCDF/GRAL/'   #GRAL
pathGRAMMu = 'G:/toNETCDF/GRAMM/'   #GRAMM wind fields
pathGRALu  = 'G:/toNETCDF/GRAL/'   #GRAL wind fields
pathGRALc  = 'G:/toNETCDF/GRAL/'   #GRAL concentration fields
pathNCout  = 'G:/toNETCDF/GRAL/'


# In[5]:


###############################################################################################################################################################
### Read GRAMM parameters #####################################################################################################################################
###############################################################################################################################################################
# read GRAMM GRAMM.geb to determine simulation domain
GRAMM_nx, GRAMM_ny, GRAMM_nz, GRAMM_xmin, GRAMM_xmax, GRAMM_ymin, GRAMM_ymax, GRAMM_dx, GRAMM_dy, GRAMM_ycemesh, GRAMM_xcemesh, GRAMM_ycomesh, GRAMM_xcomesh = ggpost.ReturnGRAMMGeometry(pathGRAMM+'GRAMM.geb')

# extract GRAMM topography and vertical grid from ggeom.asc
topo, zgrid = ggpost.ReturnTopography(pathGRAMM+'ggeom.asc', GRAMM_nx, GRAMM_ny, GRAMM_nz)

# extract GRAMM landuse data from landuse.asc
thermalcondum, heatcondum, roughnessm, moisturem, emissm, albedom = ggpost.ReturnLanduse(pathGRAMM+'landuse.asc', GRAMM_nx, GRAMM_ny)

print('done loading GRAMM parameters')
###############################################################################################################################################################
### Read GRAL parameters #####################################################################################################################################
###############################################################################################################################################################

# read GRAL GRAL.geb to determine simulation domain
# GRAL_nx, GRAL_ny, GRAL_dx, GRAL_dy, GRAL_xmin, GRAL_ymin = geometry
geometry, nslice, sourcegroups, GRAL_xmax, GRAL_ymax, GRAL_ycemesh, GRAL_xcemesh, GRAL_ycomesh, GRAL_xcomesh = ggpost.ReturnGeometry(pathGRAL+'GRAL.geb')

# read GRAl in.dat 
particles, pollutant, slices, nslices, slicethick = ggpost.ReturnGRALconfig(pathGRAL+'in.dat')

# reads GRAL buildings.dat and creates an array and maskarray
Buildings, BuildingMask = ggpost.ReturnBuildings(pathGRAL+'buildings.dat', geometry)

# extracts data about the GRAL surface 
# surface elevation, index of gral surface, building height, orography (topography without buildings!)
ahk, kkart, buiheight, oro = ggpost.read_GRAL_geometries(pathGRAL+'GRAL_geometries.txt')

print('done loading GRAL parameters')


# In[6]:


###############################################################################################################################################################
### Plot input data ###########################################################################################################################################
###############################################################################################################################################################

# plot GRAMM landuse
#ggplot.plotlanduse(topo,thermalcondum,heatcondum,roughnessm,moisturem,emissm,albedom,GRAMM_xcemesh,GRAMM_ycemesh,GRAMM_xcomesh,GRAMM_ycomesh)
# plot GRAMM and GRAL topography
# currently this calls ReturnGeometry() and ReturnGRAMMGeometry() internally, since the list of arguments to pass was so long
ggplot.plottopography(topo, oro, pathGRAMM, pathGRAL)


# In[7]:


###############################################################################################################################################################
### Read output data ##########################################################################################################################################
###############################################################################################################################################################

# extract GRAMM flowfield
wind_u, wind_v, wind_w, umag = ggpost.extract_gramm_flowfield(pathGRAMMu+'00001.wnd', nx=GRAMM_nx, ny=GRAMM_ny, nz=GRAMM_nz)

# read GRAL concentraions
con=ggpost.load_con(sit=1,folder=pathGRALc,levs=[1,2,3,4,5],cats=['01'], geometry=geometry)

# extract GRAL flowfield, 1 bigger than domain in each dimension!
u, v, w = ggpost.extract_flowfield(pathGRALu+'00001.gff')
GRAL_nz=np.shape(u)[2] #hardcoded?



# In[9]:


###############################################################################################################################################################
### Plot wind vectors #########################################################################################################################################
### Treatment of vertical levels differs between GRAMM and GRAL################################################################################################
###############################################################################################################################################################

# GRAMM wind fields are on levels above ground. Therefore, for all x,y the level closes to max(zrid[:,:,level]) is selected -> roughly same altitude
# spacing quivers forther apart allows us to see more clearly what is going on, but is not necesseraily a good representation of the data
ggplot.plotGRAMMwind(zgrid, GRAMM_nx,GRAMM_ny,GRAMM_xcemesh,GRAMM_ycemesh,wind_u,wind_v,topo,heatcondum,qspace=4, level=15)


# GRAL wind fields are on a level above sea -> a given altitude. Therefore, at a given level all cells that are below the topography will just contain zeroes 
# kkart is the surface level and kkart+1 is the first level of the atmosphere for every x,y
# level, xmin, xmax, ymin, ymax
geom=[20, 680, 780, 620, 700]
ggplot.plotGRALwind(geom,u,v,GRAL_xcemesh,GRAL_ycemesh,GRAL_xcomesh,GRAL_ycomesh,kkart,Buildings)


# In[ ]:


###############################################################################################################################################################
### Write NETCDFs #########################################################################################################################################
###############################################################################################################################################################

# Write concentration field
NETCDFdata={"title": 'Concentration fields computed by GRAL model',
"institute": 'EMPA, Dubendorf, Switzerland',
"source": 'generated with full computation of buildings',
"history": 'Created on ' + datetime.datetime.now().strftime('%Y/%m/%d'),
"comment": 'Data packed into int format and compressed',
"convention": 'CF-1.7',
"x": geometry[0],
"y": geometry[1],           
"z": nslices,           
"group": 1,
"zlib": True,
'complevel': 4,
"type": 'conc',
"units": 'microgram / m3',   
"name": 'NOx',
"long_name": 'NOx concentration',
"standard_name": 'mass_concentration_of_nox_expressed_as_nitrogen_in_air',
"scale_factor": 0.1,
"add_offset": 0,
"dtype": 'u2',
"automaskandscale": True,
"dimensions" : ('group', 'layer', 'y', 'x'),
"chunksize" : (1, 1, geometry[1], geometry[0]),
"addgrid" : True,
"gridx" : GRAL_xcemesh[1,:],
"gridy" : GRAL_ycemesh[:,1],
"gridz" : slices}

ggnc.writeNETCDF_latlon(inarray=np.transpose(con,axes=[0,1,3,2]),outpath=pathNCout+'conc_latlon.nc',NETCDFdata=NETCDFdata)


# In[ ]:


# Write GRAMM winds

NETCDFdata={"title": 'Wind fields computed by GRAMM model',
"institute": 'EMPA, Dubendorf, Switzerland',
"source": 'generated with full computation of buildings',
"history": 'Created on ' + datetime.datetime.now().strftime('%Y/%m/%d'),
"comment": 'Data packed into int format and compressed',
"convention": 'CF-1.7',
"x": GRAMM_nx,
"y": GRAMM_ny,           
"z": GRAMM_nz,           
"group": 1,
"zlib": True,
'complevel': 4,
"type": 'vel',
"units": 'm / s',   
"name" : ['eastward_wind', 'northward_wind', 'upward_air_velocity'],
"long_name": ['u-velocity component','v-velocity component','w-velocity component'],
"standard_name": ['eastward_wind', 'northward_wind', 'upward_air_velocity'],
"scale_factor": 0.01,
"add_offset": 0,
"dtype": 'i2',
"automaskandscale": True,
"dimensions" : ('z', 'y', 'x'),
"chunksize" : (1, GRAMM_ny, GRAMM_nx),
"addgrid" : True,
"gridx" : GRAMM_xcemesh[1,:],
"gridy" : GRAMM_ycemesh[:,1],
"gridz" : np.arange(0, GRAMM_nz, 1)}

ggnc.writeNETCDF3_latlon(inarray1=np.transpose(wind_u,[2,1,0]),inarray2=np.transpose(wind_v,[2,1,0]),inarray3=np.transpose(wind_w,[2,1,0]),outpath=pathNCout+'uvw_latlon.nc',NETCDFdata=NETCDFdata)


# In[ ]:


# Write GRAL winds

NETCDFdata={"title": 'Wind fields computed by GRAL model',
"institute": 'EMPA, Dubendorf, Switzerland',
"source": 'generated with full computation of buildings',
"history": 'Created on ' + datetime.datetime.now().strftime('%Y/%m/%d'),
"comment": 'Data packed into int format and compressed',
"convention": 'CF-1.7',
"x": geometry[0]+1,
"y": geometry[1]+1,           
"z": GRAL_nz,           
"group": 1,
"zlib": True,
'complevel': 4,
"type": 'vel',
"units": 'm / s',   
"name" : ['u', 'v', 'w'],
"long_name": ['u-velocity component','v-velocity component','w-velocity component'],
"standard_name": ['eastward_wind', 'northward_wind', 'upward_air_velocity'],
"scale_factor": 0.01,
"add_offset": 0.0,
"dtype": 'i2',
"automaskandscale": True,
"dimensions" : ('z', 'y', 'x'),
"chunksizes" : (1, geometry[1]+1, geometry[0]+1),
"addgrid" : True,
"gridx" : GRAL_xcomesh[1,:], 
"gridy" : GRAL_ycomesh[:,1],
"gridz" : np.arange(0, GRAL_nz, 1)}
inarray1=np.transpose(u[:,:,:],[2,1,0])
inarray2=np.transpose(v[:,:,:],[2,1,0])
inarray3=np.transpose(w[:,:,:],[2,1,0])
ggnc.writeNETCDF3_latlon(inarray1=inarray1,inarray2=inarray2,inarray3=inarray3,outpath=pathNCout+'/GRAL-uvw.nc',NETCDFdata=NETCDFdata)

