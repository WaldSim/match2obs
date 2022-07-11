import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, LogNorm, ListedColormap, LinearSegmentedColormap,BoundaryNorm, SymLogNorm
import matplotlib.colors as colors

import numpy as np

import gg_post_utilities as ggpost


# round to significant digits
def round_sig(x, sig=2):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

# format ticker in scientific notation
def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)

# colorbar next to plot
def colorbar(mappable,ticks=None):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, format=ticker.FuncFormatter(fmt), ticks=ticks)

def plottopography(topo, oro, pathGRAMM, pathGRAL):
    GRAMM_nx, GRAMM_ny, GRAMM_nz, GRAMM_xmin, GRAMM_xmax, GRAMM_ymin, GRAMM_ymax, GRAMM_dx, GRAMM_dy, GRAMM_ycemesh, GRAMM_xcemesh, GRAMM_ycomesh, GRAMM_xcomesh = ggpost.ReturnGRAMMGeometry(pathGRAMM+'GRAMM.geb')
    geometry, nslice, sourcegroups, GRAL_xmax, GRAL_ymax, GRAL_ycemesh, GRAL_xcemesh, GRAL_ycomesh, GRAL_xcomesh = ggpost.ReturnGeometry(pathGRAL+'GRAL.geb')

    # Generate a colormap
    toplot = np.transpose(topo)
    BrBG_cmap = plt.get_cmap('terrain', 1. * np.ptp(toplot))
    BrBG_cvals = BrBG_cmap(np.arange(160,int(np.ptp(toplot))))
    cmap2 = matplotlib.colors.ListedColormap(BrBG_cvals)

    # plot GRAMM topography
    fig, axs = plt.subplots(1,1, figsize=(12, 12), facecolor='w', edgecolor='k')

    minv=np.min(toplot)
    maxv=0.9 * np.max(toplot)
    pcm = axs.pcolormesh(GRAMM_xcemesh,GRAMM_ycemesh,toplot, \
               vmin = minv, vmax = maxv,\
               cmap = cmap2,\
               )
    lev=np.linspace(300,800,6)
    cs = axs.contour(GRAMM_xcemesh,GRAMM_ycemesh,toplot,levels=lev,colors='k')
    cb = colorbar(pcm, ticks=lev)
    cb.set_label('elevation [m]')
    plt.savefig("grammtopo.png", bbox_inches='tight')
    plt.show()
    
    # plot GRAMM topography in GRAL domain
    ggx=np.floor((geometry[4]-GRAMM_xmin)/GRAMM_dx).astype('int')
    ggy=np.floor((geometry[5]-GRAMM_ymin)/GRAMM_dy).astype('int')
    ggux=np.ceil((GRAL_xmax-GRAMM_xmax)/GRAMM_dx).astype('int')
    gguy=np.ceil((GRAL_ymax-GRAMM_ymax)/GRAMM_dy).astype('int')

    fig, axs = plt.subplots(1,1, figsize=(12, 12), facecolor='w', edgecolor='k')
    pcm = axs.pcolormesh(GRAMM_xcemesh[ggy:gguy,ggx:ggux],GRAMM_ycemesh[ggy:gguy,ggx:ggux],toplot[ggy:gguy,ggx:ggux], \
               vmin = minv, vmax = maxv,\
               cmap = cmap2,\
               )
    cs = axs.contour(GRAMM_xcemesh[ggy:gguy,ggx:ggux],GRAMM_ycemesh[ggy:gguy,ggx:ggux],toplot[ggy:gguy,ggx:ggux],levels=lev,colors='k')
    cb = colorbar(pcm, ticks=lev)
    cb.set_label('elevation [m]')
    plt.savefig("grammgraltopo.png", bbox_inches='tight')
    #plt.show()
    
    # plot GRAL topography
    toplot=np.transpose(oro)

    fig, axs = plt.subplots(1,1, figsize=(12, 12), facecolor='w', edgecolor='k')
    pcm = axs.pcolormesh(GRAL_xcemesh,GRAL_ycemesh,toplot, \
               vmin = minv, vmax = maxv,\
               cmap = cmap2,\
               )
    cs = axs.contour(GRAL_xcemesh,GRAL_ycemesh,toplot,levels=lev,colors='k')
    cb = colorbar(pcm, ticks=lev)
    cb.set_label('elevation [m]')
    plt.savefig("graltopo.png", bbox_inches='tight')
    #plt.show()
    
    # plot difference between the two
    # shift x by 1?
    topodiff=np.zeros(np.shape(oro))
    ratiox=int(GRAMM_dx/geometry[2])
    ratioy=int(GRAMM_dy/geometry[3])
    ggx=np.floor((geometry[4]-GRAMM_xmin)/GRAMM_dx).astype('int')
    ggy=np.floor((geometry[5]-GRAMM_ymin)/GRAMM_dy).astype('int')

    for i in range(geometry[0]):
        for j in range(geometry[1]):
            ii=i//ratiox+ggx+1  
            jj=j//ratioy+ggy
            topodiff[i,j]=oro[i,j]-topo[ii,jj]
   
    # Generate a colormap
    toplot=np.transpose(topodiff)

    fig, axs = plt.subplots(1,1, figsize=(12, 12), facecolor='w', edgecolor='k')

    pcm = axs.pcolormesh(GRAL_xcomesh,GRAL_ycomesh,toplot, \
               vmin = -15, vmax = 15,\
               cmap = 'twilight_shifted',\
               )
    lev=np.linspace(-15,15,15)
    cb = colorbar(pcm, ticks=lev)
    cb.set_label('elevation [m]')

    plt.show()
   

# def plotGRALtopo():
#     xl=0 #0
#     xu=1367 #1367
#     yl=0 #0
#     yu=1295 #1295

#     nx=xu-xl
#     ny=yu-yl
#     up=np.zeros((nx,ny))
#     vp=np.zeros((nx,ny))
#     fig, axs = plt.subplots(1,1, figsize=(25*np.sqrt(nx/ny), 25/np.sqrt(nx/ny)), facecolor='w', edgecolor='k')

#     pcm = axs.pcolormesh(GRAL_xcomesh[yl:yu,xl:xu],GRAL_ycomesh[yl:yu,xl:xu],np.transpose(oro[xl:xu,yl:yu]), \
#                vmin=np.min(oro[xl:xu,yl:yu]), vmax=np.max(oro[xl:xu,yl:yu]),\
#                cmap = 'terrain',\
#                )
#     z = np.transpose(np.ma.masked_array(Buildings[xl:xu,yl:yu], Buildings[xl:xu,yl:yu] < 1))
#     pcm2 = axs.pcolormesh(GRAL_xcomesh[yl:yu,xl:xu],GRAL_ycomesh[yl:yu,xl:xu], z , \
#                norm=LogNorm(vmin=np.min(z), vmax=np.max(z)),\
#                cmap = 'inferno_r',\
#                )

#     axs.tick_params(axis='y', rotation=90)

#     ax = pcm.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("left", size="5%", pad=0.6)
#     cb=fig.colorbar(pcm, cax=cax)
#     cax.yaxis.set_ticks_position('left')
#     cb.ax.tick_params(labelsize=18)

#     cax2 = divider.append_axes("right", size="5%", pad=0.05)
#     cb2=fig.colorbar(pcm2, cax=cax2)
#     cb2.ax.tick_params(labelsize=18)

#     ax.tick_params(axis='both', which='major', labelsize=18)


#     #plt.savefig(str(sit)+'oro.png', dpi=300, bbox_inches='tight')


def plotlanduse(topo,thermalcondum,heatcondum,roughnessm,moisturem,emissm,albedom,GRAMM_xcemesh,GRAMM_ycemesh,GRAMM_xcomesh,GRAMM_ycomesh):
    fig1, axs = plt.subplots(3,3, figsize=(15, 15), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    toplot=np.transpose(topo)

    # Generate a colormap
    BrBG_cmap = plt.get_cmap('terrain', 1. * np.ptp(toplot))
    BrBG_cvals = BrBG_cmap(np.arange(160,int(np.ptp(toplot))))
    cmap2 = matplotlib.colors.ListedColormap(BrBG_cvals)

    pcm = axs[1].pcolormesh(GRAMM_xcomesh,GRAMM_ycomesh,toplot, \
               vmin = np.min(toplot), vmax = 0.9 * np.max(toplot),\
               cmap = cmap2,\
               )
    axs[1].tick_params(axis='x', rotation=20)
    axs[1].tick_params(axis='y', rotation=70)
    axs[0].axis('off')
    axs[2].axis('off')
    levs=np.linspace(300,900,7)
    cs = axs[1].contour(GRAMM_xcemesh,GRAMM_ycemesh,toplot,levels=levs,colors='k')
    #CB = fig.colorbar(pcm, orientation='horizontal', shrink=0.8)
    cb=colorbar(pcm)



    toplot=np.transpose(thermalcondum)
    #ticks=np.concatenate((np.arange(1),np.unique(toplot)))
    ticks=np.unique(toplot)
    pcm = axs[3].pcolormesh(GRAMM_xcomesh,GRAMM_ycomesh,toplot, \
               vmin = np.min(toplot), vmax = 1 * np.max(toplot),\
               cmap = 'OrRd',\
               )
    axs[3].tick_params(axis='x', rotation=20)
    axs[3].tick_params(axis='y', rotation=70)    
    axs[3].set_title('thermal conductivity [m$^{2}$ s$^{-1}$]')
    ax = pcm.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(pcm, cax=cax, format=ticker.FuncFormatter(fmt),ticks=ticks)


    toplot=np.transpose(heatcondum)
    ticks=np.unique(toplot)

    pcm = axs[4].pcolormesh(GRAMM_xcomesh,GRAMM_ycomesh,toplot, \
               vmin = np.min(toplot), vmax = 1 * np.max(toplot),\
               cmap = 'coolwarm',\
               norm=LogNorm(),\
               )
    axs[4].tick_params(axis='x', rotation=20)
    axs[4].tick_params(axis='y', rotation=70)
    axs[4].set_title('Heat conductivity [W m$^{-1}$ K$^{-1}$]')
    ax = pcm.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(pcm, cax=cax, format=ticker.FuncFormatter(fmt),ticks=ticks)


    toplot=np.transpose(roughnessm)
    ticks=np.unique(toplot)

    pcm = axs[5].pcolormesh(GRAMM_xcomesh,GRAMM_ycomesh,toplot, \
               vmin = np.min(toplot), vmax = 1 * np.max(toplot),\
               cmap = 'RdGy',\
               norm=LogNorm()
               )
    axs[5].tick_params(axis='x', rotation=20)
    axs[5].tick_params(axis='y', rotation=70)
    axs[5].set_title('roughness [m]')
    ax = pcm.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(pcm, cax=cax, format=ticker.FuncFormatter(fmt),ticks=ticks)


    toplot=np.transpose(moisturem)
    ticks=np.unique(toplot)

    pcm = axs[6].pcolormesh(GRAMM_xcomesh,GRAMM_ycomesh,toplot, \
               vmin = np.min(toplot), vmax = 1 * np.max(toplot),\
               cmap = 'YlGnBu',\
               )
    axs[6].set_title('soil moisture [kg kg$^{-1}$]')
    axs[6].tick_params(axis='x', rotation=20)
    axs[6].tick_params(axis='y', rotation=70)
    ax = pcm.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(pcm, cax=cax, format=ticker.FuncFormatter(fmt),ticks=ticks)


    toplot=np.transpose(emissm)
    ticks=np.unique(toplot)

    pcm = axs[7].pcolormesh(GRAMM_xcomesh,GRAMM_ycomesh,toplot, \
               vmin = np.min(toplot), vmax = 1 * np.max(toplot),\
               cmap = 'cividis',\
               )
    axs[7].set_title('emissivity [-]')
    axs[7].tick_params(axis='x', rotation=20)
    axs[7].tick_params(axis='y', rotation=70)
    ax = pcm.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(pcm, cax=cax, format=ticker.FuncFormatter(fmt),ticks=ticks)

    toplot=np.transpose(albedom)
    ticks=np.unique(toplot)

    pcm = axs[8].pcolormesh(GRAMM_xcomesh,GRAMM_ycomesh,toplot, \
               vmin = np.min(toplot), vmax = 1 * np.max(toplot),\
               cmap = 'bone',\
               )
    axs[8].set_title('albedo [-]')
    axs[8].tick_params(axis='x', rotation=20)
    axs[8].tick_params(axis='y', rotation=70)
    ax = pcm.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(pcm, cax=cax, format=ticker.FuncFormatter(fmt),ticks=ticks)
    plt.savefig("landuse.png", bbox_inches='tight')

    #plt.show()
    
    
def plotGRAMMwind(zgrid,GRAMM_nx,GRAMM_ny,GRAMM_xcemesh,GRAMM_ycemesh,wind_u,wind_v,topo,heatcondum,qspace, level):
    #remember that first dimension is x, but that's the row index in an array -> y in a plot
    #transpose before quivers
    
    zmin=np.max(zgrid[:,:,level])
    zindex=np.zeros((GRAMM_nx,GRAMM_ny),dtype=int)
    for i in range(GRAMM_nx):
        for j in range(GRAMM_ny):
            zindex[i,j]=np.argmin(np.abs(zgrid[i,j,:]-zmin))
    
    unx=np.int(np.ceil(GRAMM_nx/qspace))
    uny=np.int(np.ceil(GRAMM_ny/qspace))
    ufield=np.zeros([unx,uny])
    vfield=np.zeros([unx,uny])
    for i in range(unx):
        for j in range(uny):
            ii=(qspace*i)
            jj=(qspace*j)
            ufield[i,j]=wind_u[ii,jj,zindex[ii,jj]]
            vfield[i,j]=wind_v[ii,jj,zindex[ii,jj]]


    fig1, axs = plt.subplots(1,1, figsize=(15, 15), facecolor='w', edgecolor='k')
    toplot=np.transpose(topo)
    levs=np.array([300,400,500,600,700,800,900])
    cs = axs.contour(GRAMM_xcemesh[::1,::1],GRAMM_ycemesh[::1,::1],toplot[::1,::1],levels=levs,colors='r')

    toplot=np.transpose(heatcondum) #to determine lakes
    levs=np.array([98])
    cs2 = axs.contour(GRAMM_xcemesh,GRAMM_ycemesh,toplot,levels=levs,colors='b')
    lab='Length = '+str(1)
    q=axs.quiver(GRAMM_xcemesh[::qspace,::qspace], GRAMM_ycemesh[::qspace,::qspace],np.transpose(ufield),np.transpose(vfield),scale=15/1)
    axs.quiverkey(q, X=0.8, Y=0.89, U=1, label=lab, labelpos='E',coordinates='figure')
    axs.clabel(cs,fmt='%d')
    axs.set_title('quiver [m s$^{-1}$]')
    axs.tick_params(axis='x', rotation=20)
    axs.tick_params(axis='y', rotation=70)
    plt.show()
    
def plotcon():
    cmap = plt.get_cmap('jet', 256)
    field=np.sum(con[0,:,:,:],axis=0)

    fig, axs = plt.subplots(1,1, figsize=(20, 20), facecolor='w', edgecolor='k')
    i=0  #which level (1 is actually the second one in python, as usual)
    j=0  #which SourceGroup

    pcm = axs.pcolormesh(GRAL_xcomesh,GRAL_ycomesh,np.transpose(con[0,0,:,:]), \
               norm=colors.LogNorm(vmin=30, vmax=130),\
               cmap = cmap,\
               )
    pcm2 = axs.pcolormesh(GRAL_xcomesh,GRAL_ycomesh,np.transpose(Buildings), \
               vmin=0, vmax=1,\
               cmap = 'binary',\
               )
    ax = pcm.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(pcm, cax=cax, format=ticker.FuncFormatter(fmt))

    axs.set_title('con')
    
def plotGRALwind(geom,u,v,GRAL_xcemesh,GRAL_ycemesh,GRAL_xcomesh,GRAL_ycomesh,kkart,Buildings):
    #plot smaller section
    #klev = 20 #for one given level "above sea", this means it is not the same level above ground everywehre -> no wind inside buildings / hills
    #you might have to chose the first level above ground everywhere instead (that won't give any good looking vectorfields though!)
    #xl=680 #0
    #xu=780 #1367
    #yl=620 #0
    #yu=700 #1295
    klev = geom[0]
    xl = geom[1]
    xu = geom[2]
    yl = geom[3]
    yu = geom[4]

    #
    nx=xu-xl
    ny=yu-yl
    up=np.zeros((nx,ny))
    vp=np.zeros((nx,ny))
    for ii in range(nx):
        for jj in range(ny):
            i=ii+xl
            j=jj+yl
            k=klev
            #up[ii,jj]=u[i,j,k]
            #vp[ii,jj]=v[i,j,k]
            #k=kkart[i,j]+1   #I think this is 1st level above ground
            up[ii,jj]=u[i,j,k]  
            vp[ii,jj]=v[i,j,k] 

    umag=np.hypot(up,vp)  #calculate wind speed from the 2 components
    fig, axs = plt.subplots(1,1, figsize=(25*np.sqrt(nx/ny), 25/np.sqrt(nx/ny)), facecolor='w', edgecolor='k')

    #color background where surface is below the desired level "above sea" (wind)
    z = np.transpose(np.ma.masked_array(kkart[xl:xu,yl:yu], kkart[xl:xu,yl:yu] > klev))
    pcm = axs.pcolormesh(GRAL_xcomesh[yl:yu+1,xl:xu+1],GRAL_ycomesh[yl:yu+1,xl:xu+1],z, \
               vmin=np.min(kkart[xl:xu,yl:yu]), vmax=np.max(kkart[xl:xu,yl:yu]),\
               cmap = 'YlGn',\
               )

    #color background where surface is above the desired level "above sea" (no wind)
    z = np.transpose(np.ma.masked_array(kkart[xl:xu,yl:yu], kkart[xl:xu,yl:yu] <= klev))
    pcm2 = axs.pcolormesh(GRAL_xcomesh[yl:yu+1,xl:xu+1],GRAL_ycomesh[yl:yu+1,xl:xu+1],z, \
               vmin=np.min(kkart[xl:xu,yl:yu]), vmax=np.max(kkart[xl:xu,yl:yu]),\
               cmap = 'copper_r',\
              #cmap = 'YlOrRd'
               )

    #put borders around buildings
    z = np.transpose(np.ma.masked_array(Buildings[xl:xu,yl:yu], Buildings[xl:xu,yl:yu] < 1))
    pcm3 = axs.pcolormesh(GRAL_xcomesh[yl:yu+1,xl:xu+1],GRAL_ycomesh[yl:yu+1,xl:xu+1],z , \
               vmin=0, vmax=0.1,\
               facecolor='none', edgecolors='k', \
               cmap = 'binary',\
               )

    i=1
    q=axs.quiver(GRAL_xcemesh[yl:yu:i,xl:xu:i], GRAL_ycemesh[yl:yu:i,xl:xu:i],np.transpose(up[::i,::i]),np.transpose(vp[::i,::i]),scale=70,color='royalblue',width=0.0012)

    axs.tick_params(axis='both', which='major', labelsize=18)

    lab='Length = '+str(5.5)
    axs.quiverkey(q, X=0.8, Y=0.89, U=5.5, label=lab, labelpos='E',coordinates='figure',fontproperties={"size": 15})
    axs.set_title('quiver [m s$^{-1}$]', fontsize=15, fontweight="bold")
    axs.tick_params(axis='x', rotation=20)
    axs.tick_params(axis='y', rotation=70)