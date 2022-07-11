#import netCDF4 as nc
from netCDF4 import Dataset
from pyproj import CRS, Transformer

import numpy as np

def writeNETCDF(inarray,outpath,NETCDFdata):
    
    fout = Dataset(outpath, 'w', format = 'NETCDF4')

    # Global Attributes
    fout.setncattr('title',NETCDFdata['title'])
    fout.setncattr('institute',NETCDFdata['institute'])
    fout.setncattr('source',NETCDFdata['source'])
    fout.setncattr('history',NETCDFdata['history'])
    fout.setncattr('comment',NETCDFdata['comment'])
    fout.setncattr('Conventions',NETCDFdata['convention'])

    # Creating dimensions
    fout.createDimension('x',NETCDFdata['x'])
    fout.createDimension('y',NETCDFdata['y'])
    fout.createDimension('group',NETCDFdata['group'])
    if 'layer' in NETCDFdata['dimensions']:
        fout.createDimension('layer',NETCDFdata['z'])
    elif 'z' in NETCDFdata['dimensions']:
            print('have to add z, not yet implemented')

    if NETCDFdata['addgrid']:
        xs = fout.createVariable('x', 'f4', ('x',))
        fout.variables['x'].setncattr('units','m')
        fout.variables['x'].setncattr('long_name','Easting')
        fout.variables['x'].setncattr('standard_name','projection_x_coordinate')
        ys = fout.createVariable('y', 'f4', ('y',))
        fout.variables['y'].setncattr('units','m')
        fout.variables['y'].setncattr('long_name','Northing')
        fout.variables['y'].setncattr('standard_name','projection_y_coordinate')
        xs[:] = NETCDFdata['gridx']
        ys[:] = NETCDFdata['gridy']
        
        if 'layer' in NETCDFdata['dimensions']:
            zs = fout.createVariable('layer', 'f4', ('layer',))
            fout.variables['layer'].setncattr('units','m')
            fout.variables['layer'].setncattr('long_name','Elevation above ground')
            zs[:] = NETCDFdata['gridz']
        elif 'z' in NETCDFdata['dimensions']:
            print('have to add z, not yet implemented')
    
    vartype=NETCDFdata['name']
    # Create variables - Concentrations
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = 9, chunksizes = NETCDFdata['chunksize'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])

    # Filling variables
    fout.variables[vartype][:] = inarray


    
    fout.close()

def writeNETCDF_latlon(inarray,outpath,NETCDFdata):
    
    fout = Dataset(outpath, 'w', format = 'NETCDF4')

    # Global Attributes
    fout.setncattr('title',NETCDFdata['title'])
    fout.setncattr('institute',NETCDFdata['institute'])
    fout.setncattr('source',NETCDFdata['source'])
    fout.setncattr('history',NETCDFdata['history'])
    fout.setncattr('comment',NETCDFdata['comment'])
    fout.setncattr('Conventions',NETCDFdata['convention'])

    # Creating dimensions
    fout.createDimension('x',NETCDFdata['x'])
    fout.createDimension('y',NETCDFdata['y'])
  
    
    fout.createDimension('group',NETCDFdata['group'])
    if 'layer' in NETCDFdata['dimensions']:
        fout.createDimension('layer',NETCDFdata['z'])
    elif 'z' in NETCDFdata['dimensions']:
            print('have to add z, not yet implemented')

    if NETCDFdata['addgrid']:
        xs = fout.createVariable('x', 'f4', ('x',))
        fout.variables['x'].setncattr('units','m')
        fout.variables['x'].setncattr('long_name','Easting')
        fout.variables['x'].setncattr('standard_name','projection_x_coordinate')
        ys = fout.createVariable('y', 'f4', ('y',))
        fout.variables['y'].setncattr('units','m')
        fout.variables['y'].setncattr('long_name','Northing')
        fout.variables['y'].setncattr('standard_name','projection_y_coordinate')
        xs[:] = NETCDFdata['gridx']
        ys[:] = NETCDFdata['gridy']
        
        lat = fout.createDimension('lat',NETCDFdata['y'])
        lon = fout.createDimension('lon',NETCDFdata['x'])
        lat = fout.createVariable('lat', 'f4', ('y', 'x',))
        lat.standard_name = "latitude"
        lat.units = "degrees_north"
        lon = fout.createVariable('lon', 'f4', ('y', 'x',))
        lon.standard_name = "longitude"
        lon.units = "degrees_east"  


        
        if 'layer' in NETCDFdata['dimensions']:
            zs = fout.createVariable('layer', 'f4', ('layer',))
            fout.variables['layer'].setncattr('units','m')
            fout.variables['layer'].setncattr('long_name','Elevation above ground')
            zs[:] = NETCDFdata['gridz']
        elif 'z' in NETCDFdata['dimensions']:
            print('have to add z, not yet implemented')
    
    vartype=NETCDFdata['name']
    # Create variables - Concentrations
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])
    fout.variables[vartype].setncattr('coordinates', "lat lon")
    fout.variables[vartype].setncattr('grid_mapping', "crs: y x")
    
    # Filling variables
    fout.variables[vartype][:] = inarray
    
    
    # coordinate system
    crs = fout.createVariable('crs', 'i4')
    crs_def = CRS("EPSG:21781")
    cf_grid_mapping = crs_def.to_cf()

    for key, value in cf_grid_mapping.items():
        setattr(crs, key, value)
    
    transformer=Transformer.from_crs("EPSG:21781","EPSG:4326")

    xg,yg = np.meshgrid(xs,ys)

    latt, lonn = transformer.transform(xg,yg)
     
    lon[:,:] = lonn
    lat[:,:] = latt

    fout.close()

def writeNETCDF3(inarray1,inarray2,inarray3,outpath,NETCDFdata):
    
    fout = Dataset(outpath, 'w', format = 'NETCDF4')

    # Global Attributes
    fout.setncattr('title',NETCDFdata['title'])
    fout.setncattr('institute',NETCDFdata['institute'])
    fout.setncattr('source',NETCDFdata['source'])
    fout.setncattr('history',NETCDFdata['history'])
    fout.setncattr('comment',NETCDFdata['comment'])
    fout.setncattr('Conventions',NETCDFdata['convention'])

    # Creating dimensions
    fout.createDimension('x',NETCDFdata['x'])
    fout.createDimension('y',NETCDFdata['y'])
    fout.createDimension('group',NETCDFdata['group'])
    if 'layer' in NETCDFdata['dimensions']:
        fout.createDimension('layer',NETCDFdata['z'])
    elif 'z' in NETCDFdata['dimensions']:
        fout.createDimension('z',NETCDFdata['z'])

    if NETCDFdata['addgrid']:
        xs = fout.createVariable('x', 'f4', ('x',))
        fout.variables['x'].setncattr('units','m')
        fout.variables['x'].setncattr('long_name','Easting')
        fout.variables['x'].setncattr('standard_name','projection_x_coordinate')
        ys = fout.createVariable('y', 'f4', ('y',))
        fout.variables['y'].setncattr('units','m')
        fout.variables['y'].setncattr('long_name','Northing')
        fout.variables['y'].setncattr('standard_name','projection_y_coordinate')
        xs[:] = NETCDFdata['gridx']
        ys[:] = NETCDFdata['gridy']
               
        if 'layer' in NETCDFdata['dimensions']:
            zs = fout.createVariable('layer', 'f4', ('layer',))
            fout.variables['layer'].setncattr('units','m')
            fout.variables['layer'].setncattr('long_name','Elevation above ground')
            zs[:] = slices
        elif 'z' in NETCDFdata['dimensions']:
            zs = fout.createVariable('z', 'i4', ('z',))
            fout.variables['z'].setncattr('units','-')
            fout.variables['z'].setncattr('long_name','vertical grid level')
            fout.variables['z'].setncattr('positive','up')
            zs[:] = NETCDFdata['gridz']
    
    vartype=NETCDFdata['name'][0]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'], chunksizes = NETCDFdata['chunksize'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][0])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][0])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])

    fout.variables[vartype][:] = inarray1

    
    vartype=NETCDFdata['name'][1]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'], chunksizes = NETCDFdata['chunksize'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][1])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][1])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])

    fout.variables[vartype][:] = inarray2
    
    vartype=NETCDFdata['name'][2]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'], chunksizes = NETCDFdata['chunksize'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][2])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][2])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])

    fout.variables[vartype][:] = inarray3
    
    fout.close()
    

def writeNETCDF3_latlon(inarray1,inarray2,inarray3,outpath,NETCDFdata):
    
    fout = Dataset(outpath, 'w', format = 'NETCDF4')

    # Global Attributes
    fout.setncattr('title',NETCDFdata['title'])
    fout.setncattr('institute',NETCDFdata['institute'])
    fout.setncattr('source',NETCDFdata['source'])
    fout.setncattr('history',NETCDFdata['history'])
    fout.setncattr('comment',NETCDFdata['comment'])
    fout.setncattr('Conventions',NETCDFdata['convention'])

    # Creating dimensions
    fout.createDimension('x',NETCDFdata['x'])
    fout.createDimension('y',NETCDFdata['y'])
    fout.createDimension('group',NETCDFdata['group'])
    if 'layer' in NETCDFdata['dimensions']:
        fout.createDimension('layer',NETCDFdata['z'])
    elif 'z' in NETCDFdata['dimensions']:
        fout.createDimension('z',NETCDFdata['z'])

    if NETCDFdata['addgrid']:
        xs = fout.createVariable('x', 'f4', ('x',))
        fout.variables['x'].setncattr('units','m')
        fout.variables['x'].setncattr('long_name','Easting')
        fout.variables['x'].setncattr('standard_name','projection_x_coordinate')
        ys = fout.createVariable('y', 'f4', ('y',))
        fout.variables['y'].setncattr('units','m')
        fout.variables['y'].setncattr('long_name','Northing')
        fout.variables['y'].setncattr('standard_name','projection_y_coordinate')
        xs[:] = NETCDFdata['gridx']
        ys[:] = NETCDFdata['gridy']
        
        lat = fout.createDimension('latitude',NETCDFdata['y'])
        lon = fout.createDimension('longitude',NETCDFdata['x'])
        lat = fout.createVariable('latitude', 'f4', ('y', 'x',))
        lat.standard_name = "latitude"
        lat.long_name = "Latitude"
        lat.units = "degrees_north"
        lon = fout.createVariable('longitude', 'f4', ('y', 'x',))
        lon.standard_name = "longitude"
        lon.long_name = "Longitude"
        lon.units = "degrees_east"  
        
        if 'layer' in NETCDFdata['dimensions']:
            zs = fout.createVariable('layer', 'f4', ('layer',))
            fout.variables['layer'].setncattr('units','m')
            fout.variables['layer'].setncattr('long_name','Elevation above ground')
            zs[:] = slices
        elif 'z' in NETCDFdata['dimensions']:
            zs = fout.createVariable('z', 'i4', ('z',))
            fout.variables['z'].setncattr('units','m')
            fout.variables['z'].setncattr('standard_name','altitude')
            fout.variables['z'].setncattr('long_name','height above reference')
            fout.variables['z'].setncattr('positive','up')
            zs[:] = NETCDFdata['gridz']
    
    vartype=NETCDFdata['name'][0]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][0])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][0])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])
    fout.variables[vartype].setncattr('coordinates', "latitude longitude")
    fout.variables[vartype].setncattr('grid_mapping', "crs: y x")
    
    fout.variables[vartype][:] = inarray1

    
    vartype=NETCDFdata['name'][1]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][1])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][1])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])
    fout.variables[vartype].setncattr('coordinates', "latitude longitude")
    fout.variables[vartype].setncattr('grid_mapping', "crs: y x")
    
    fout.variables[vartype][:] = inarray2
    
    vartype=NETCDFdata['name'][2]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = NETCDFdata['dimensions'], zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][2])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][2])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])
    fout.variables[vartype].setncattr('coordinates', "latitude longitude")
    fout.variables[vartype].setncattr('grid_mapping', "crs: y x")
    
    fout.variables[vartype][:] = inarray3
    
    
    crs = fout.createVariable('crs', 'i4')
    crs_def = CRS("EPSG:21781")
    cf_grid_mapping = crs_def.to_cf()

    for key, value in cf_grid_mapping.items():
        setattr(crs, key, value)
    
    transformer=Transformer.from_crs("EPSG:21781","EPSG:4326")

    xg,yg = np.meshgrid(xs,ys)

    latt, lonn = transformer.transform(xg,yg)
     
    lon[:,:] = lonn
    lat[:,:] = latt
    
    fout.close()
    

def writeNETCDF3_onlatlon(inarray1,inarray2,inarray3,outpath,NETCDFdata):
    
    fout = Dataset(outpath, 'w', format = 'NETCDF4')

    # Global Attributes
    fout.setncattr('title',NETCDFdata['title'])
    fout.setncattr('institute',NETCDFdata['institute'])
    fout.setncattr('source',NETCDFdata['source'])
    fout.setncattr('history',NETCDFdata['history'])
    fout.setncattr('comment',NETCDFdata['comment'])
    fout.setncattr('Conventions',NETCDFdata['convention'])

    # Creating dimensions
    fout.createDimension('x',NETCDFdata['x'])
    fout.createDimension('y',NETCDFdata['y'])
    fout.createDimension('group',NETCDFdata['group'])
    if 'layer' in NETCDFdata['dimensions']:
        fout.createDimension('layer',NETCDFdata['z'])
    elif 'z' in NETCDFdata['dimensions']:
        fout.createDimension('z',NETCDFdata['z'])

    if NETCDFdata['addgrid']:
        xs = fout.createVariable('x', 'f4', ('x',))
        fout.variables['x'].setncattr('units','m')
        fout.variables['x'].setncattr('long_name','Easting')
        fout.variables['x'].setncattr('standard_name','projection_x_coordinate')
        ys = fout.createVariable('y', 'f4', ('y',))
        fout.variables['y'].setncattr('units','m')
        fout.variables['y'].setncattr('long_name','Northing')
        fout.variables['y'].setncattr('standard_name','projection_y_coordinate')
        xs[:] = NETCDFdata['gridx']
        ys[:] = NETCDFdata['gridy']
        
        time = fout.createDimension('time',1)
        time = fout.createVariable('time', 'i4', ('time'))
        fout.variables['time'].setncattr('units','hours since 2015-00-01T00:00:00Z')
        fout.variables['time'].setncattr('long_name','forecast time')
        fout.variables['time'].setncattr('standard_name','time')
        time[:] = 0
        
        lat = fout.createDimension('latitude',NETCDFdata['y'])
        lon = fout.createDimension('longitude',NETCDFdata['x'])
        lat = fout.createVariable('latitude', 'f4', ('y'))
        lat.standard_name = "latitude"
        lat.long_name = "Latitude"
        lat.units = "degrees_north"
        lat.ioos_category = "Location"
        lat.axis = "Y"
        lat._CoordinateAxisType = "Lat"
        lon = fout.createVariable('longitude', 'f4', ('x'))
        lon.standard_name = "longitude"
        lon.long_name = "Longitude"
        lon.units = "degrees_east"  
        lon.ioos_category = "Location"
        lon.axis = "X"
        lon._CoordinateAxisType = "Lon"        
        
       
        transformer=Transformer.from_crs("EPSG:21781","EPSG:4326")

        lat1, lon1 = transformer.transform(xs[0],ys[0])
        lat2, lon2 = transformer.transform(xs[0],ys[-1])
        lat3, lon3 = transformer.transform(xs[-1],ys[-1])
        lat4, lon4 = transformer.transform(xs[-1],ys[0])
    
        lat[:] = np.linspace((lat1+lat4)/2,(lat2+lat3)/2,NETCDFdata['y'])
        lat.actual_range = ((lat1+lat4)/2, (lat2+lat3)/2)
        lon[:] = np.linspace((lon1+lon2)/2,(lon3+lon4)/2,NETCDFdata['x'])
        lon.actual_range = ((lon1+lon2)/2, (lon3+lon4)/2)
        
        if 'layer' in NETCDFdata['dimensions']:
            zs = fout.createVariable('layer', 'f4', ('layer',))
            fout.variables['layer'].setncattr('units','m')
            fout.variables['layer'].setncattr('long_name','Elevation above ground')
            zs[:] = slices
        elif 'z' in NETCDFdata['dimensions']:
            zs = fout.createVariable('z', 'i4', ('z',))
            fout.variables['z'].setncattr('units','m')
            fout.variables['z'].setncattr('standard_name','altitude')
            fout.variables['z'].setncattr('long_name','height above reference')
            fout.variables['z'].setncattr('positive','up')
            zs[:] = NETCDFdata['gridz']
    
    vartype=NETCDFdata['name'][0]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = ('time', 'z', 'latitude', 'longitude'), zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][0])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][0])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])
    
    fout.variables[vartype][:] = inarray1

    
    vartype=NETCDFdata['name'][1]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = ('time', 'z', 'latitude', 'longitude'), zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][1])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][1])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])
    
    fout.variables[vartype][:] = inarray2
    
    vartype=NETCDFdata['name'][2]
    fout.createVariable(vartype, NETCDFdata['dtype'], dimensions = ('time', 'z', 'latitude', 'longitude'), zlib = NETCDFdata['zlib'], complevel = NETCDFdata['complevel'])

    fout.variables[vartype].setncattr('units',NETCDFdata['units'])
    fout.variables[vartype].setncattr('long_name',NETCDFdata['long_name'][2])
    fout.variables[vartype].setncattr('standard_name',NETCDFdata['standard_name'][2])
    fout.variables[vartype].setncattr('scale_factor', NETCDFdata['scale_factor'])
    fout.variables[vartype].setncattr('add_offset',   NETCDFdata['add_offset'])
    fout.variables[vartype].set_auto_maskandscale(NETCDFdata['automaskandscale'])
    
    fout.variables[vartype][:] = inarray3
    
    
    fout.close()