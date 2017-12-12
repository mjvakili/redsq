import numpy as np
import ezgal
import matplotlib.pyplot as plt
import pyfits as pf
import pandas as pd
import seaborn as sns 
import itertools
import util

sns.set_style("white")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
sns.set_palette(sns.color_palette(["#9b59b6",
                                   "#e74c3c", "#3498db", 
                                   "#34495e", "#2ecc71"]))
palette = itertools.cycle(sns.color_palette())


def mstar(z):
    

    return 22.44+3.36*np.log(z)+0.273*np.log(z)**2-0.0618*np.log(z)**3-0.0227*np.log(z)**4

       

def mag_function(zs, znorm,zf,Z):
    model = ezgal.model("/net/delft/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_"+str(Z)+"_chab.model")
    model.add_filter("/net/vuntus/data1/vakili/easy/u.dat" , "u" , units = "nm")
    model.add_filter("/net/vuntus/data1/vakili/easy/g.dat" , "g" , units = "nm")
    model.add_filter("/net/vuntus/data1/vakili/easy/r.dat" , "r" , units = "nm")
    model.add_filter("/net/vuntus/data1/vakili/easy/i.dat" , "i" , units = "nm")
    kcorr_sloan = model.get_kcorrects(zf=3.0 , zs = znorm , filters = "sloan_i")
    model.set_normalization("sloan_i" , znorm , mstar(znorm)-kcorr_sloan, vega=False, apparent=True)
    mu = model.get_apparent_mags(zf=zf , filters = "u" , zs= zs)
    mg = model.get_apparent_mags(zf=zf , filters = "g" , zs= zs)
    mr = model.get_apparent_mags(zf=zf , filters = "r" , zs= zs)
    mi = model.get_apparent_mags(zf=zf , filters = "i" , zs= zs)
    return mu, mg, mr, mi

def color_function(zs, znorm,zf,Z):
    model = ezgal.model("/net/delft/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_"+str(Z)+"_chab.model")
    model.add_filter("/net/vuntus/data1/vakili/easy/u.dat" , "u" , units = "nm")
    model.add_filter("/net/vuntus/data1/vakili/easy/g.dat" , "g" , units = "nm")
    model.add_filter("/net/vuntus/data1/vakili/easy/r.dat" , "r" , units = "nm")
    model.add_filter("/net/vuntus/data1/vakili/easy/i.dat" , "i" , units = "nm")
    kcorr_sloan = model.get_kcorrects(zf=3.0 , zs = znorm , filters = "sloan_i")
    model.set_normalization("sloan_i" , znorm , mstar(znorm)-kcorr_sloan, vega=False, apparent=True)
    mu = model.get_apparent_mags(zf=zf , filters = "u" , zs= zs)
    mg = model.get_apparent_mags(zf=zf , filters = "g" , zs= zs)
    mr = model.get_apparent_mags(zf=zf , filters = "r" , zs= zs)
    mi = model.get_apparent_mags(zf=zf , filters = "i" , zs= zs)
    return np.vstack([mu-mg, mg-mr, mr-mi, mi])

def gama_reduction():
    
    data = gama
    
    data = data[data['IMAFLAGS_ISO_U']&01010111==0]
    data = data[data['IMAFLAGS_ISO_G']&01010111==0]
    data = data[data['IMAFLAGS_ISO_R']&01010111==0]
    data = data[data['IMAFLAGS_ISO_I']&01010111==0]
    
    data = data[data['MAGERR_GAAP_U']>0]
    data = data[data['MAGERR_GAAP_G']>0]
    data = data[data['MAGERR_GAAP_R']>0]
    data = data[data['MAGERR_GAAP_I']>0]
    
    data = data[data['MAG_GAAP_u_CALIB']<25.4]
    data = data[data['MAG_GAAP_g_CALIB']<25.6]
    data = data[data['MAG_GAAP_r_CALIB']<24.7]
    data = data[data['MAG_GAAP_i_CALIB']<24.5]
    
    u = data['MAG_GAAP_u_CALIB']
    g = data['MAG_GAAP_g_CALIB']
    r = data['MAG_GAAP_r_CALIB']
    i = data['MAG_GAAP_i_CALIB']
    
    uerr = data['MAGERR_GAAP_U']
    gerr = data['MAGERR_GAAP_G']
    rerr = data['MAGERR_GAAP_R']
    ierr = data['MAGERR_GAAP_I']

    Z = data['Z']
    RA = data['RA']
    DEC = data['DEC']

    u_g = data['COLOR_GAAPHOM_U_G']
    g_r = data['COLOR_GAAPHOM_G_R']
    r_i = data['COLOR_GAAPHOM_R_I']
    
    col = np.vstack([u,g,r,i,uerr,gerr,rerr,ierr,u_g,g_r,r_i,Z,RA,DEC])
    
    return col

def sdss_reduction():
    
    data = sdss
    
    data = data[data['IMAFLAGS_ISO_U']&01010111==0]
    data = data[data['IMAFLAGS_ISO_G']&01010111==0]
    data = data[data['IMAFLAGS_ISO_R']&01010111==0]
    data = data[data['IMAFLAGS_ISO_I']&01010111==0]
    
    data = data[data['MAGERR_GAAP_U']>0]
    data = data[data['MAGERR_GAAP_G']>0]
    data = data[data['MAGERR_GAAP_R']>0]
    data = data[data['MAGERR_GAAP_I']>0]

    u = data['MAG_GAAP_U'] + data['ZPT_OFFSET_U'] - data['EXT_SFD_U']
    g = data['MAG_GAAP_G'] + data['ZPT_OFFSET_G'] - data['EXT_SFD_G']
    r = data['MAG_GAAP_R'] + data['ZPT_OFFSET_R'] - data['EXT_SFD_R']
    i = data['MAG_GAAP_I'] + data['ZPT_OFFSET_I'] - data['EXT_SFD_I']
    
    mask = (u<25.4)&(g<25.6)&(r<24.7)&(i<24.5) 
    
    data = data[mask]
    
    u = data['MAG_GAAP_U'] + data['ZPT_OFFSET_U'] - data['EXT_SFD_U']
    g = data['MAG_GAAP_G'] + data['ZPT_OFFSET_G'] - data['EXT_SFD_G']
    r = data['MAG_GAAP_R'] + data['ZPT_OFFSET_R'] - data['EXT_SFD_R']
    i = data['MAG_GAAP_I'] + data['ZPT_OFFSET_I'] - data['EXT_SFD_I']
    
    uerr = data['MAGERR_GAAP_U']
    gerr = data['MAGERR_GAAP_G']
    rerr = data['MAGERR_GAAP_R']
    ierr = data['MAGERR_GAAP_I']

    Z = data['Z']
    RA = data['RA']
    DEC = data['DEC']

    u_g = data['COLOR_GAAPHOM_U_G']
    g_r = data['COLOR_GAAPHOM_G_R']
    r_i = data['COLOR_GAAPHOM_R_I']

    col = np.vstack([u,g,r,i,uerr,gerr,rerr,ierr,u_g,g_r,r_i,Z,RA,DEC])
    
    return col

def red_mock():

    from scipy.interpolate import spline

    mock_z = np.linspace(0.05,0.8,len(col[-3,:]))
    mock_color = spline(zs, mock_median.T, mock_z)
    mock_colorerr = spline(zs, mock_std.T, mock_z)
    mock_dispersion = np.random.normal(loc=0.0, scale=mock_colorerr, size=None)
    mock_color += mock_dispersion

    fig , ax = plt.subplots(nrows=4,ncols=1 , figsize=(5,15))

    for ind in range(4):
  
      ax[ind].scatter(mock_z , mock_color[:,ind] , s = 0.01 , color = next(palette))
      ax[ind].set_xlim(0,0.8)
      ax[ind].set_xlabel(r"$z$" , fontsize = 15)

    ax[0].set_ylim(0,4)
    ax[1].set_ylim(0,2.5)
    ax[2].set_ylim(0,1.5)
    ax[3].set_ylim(14,24)
    ax[0].set_ylabel(r"$u-g$" , fontsize = 15)
    ax[1].set_ylabel(r"$g-r$" , fontsize = 15)
    ax[2].set_ylabel(r"$r-i$" , fontsize = 15)
    ax[3].set_ylabel(r"$m_i$" , fontsize = 15)
    plt.legend(loc = 'best')
    plt.savefig(util.fig_dir()+'bc03_mock_uniformz.png') 

    mock_z = col[-3,:]
    mock_color = spline(zs, mock_median.T, mock_z)
    mock_colorerr = spline(zs, mock_std.T, mock_z)
    mock_dispersion = np.random.normal(loc=0.0, scale=mock_colorerr, size=None)
    mock_color += mock_dispersion

    fig , ax = plt.subplots(nrows=4,ncols=1 , figsize=(5,15))

    for ind in range(4):
  
      ax[ind].scatter(mock_z , mock_color[:,ind] , s = 0.01 , color = next(palette))
      ax[ind].set_xlim(0,0.8)
      ax[ind].set_xlabel(r"$z$" , fontsize = 15)

    ax[0].set_ylim(0,4)
    ax[1].set_ylim(0,2.5)
    ax[2].set_ylim(0,1.5)
    ax[3].set_ylim(14,24)
    ax[0].set_ylabel(r"$u-g$" , fontsize = 15)
    ax[1].set_ylabel(r"$g-r$" , fontsize = 15)
    ax[2].set_ylabel(r"$r-i$" , fontsize = 15)
    ax[3].set_ylabel(r"$m_i$" , fontsize = 15)
    plt.legend(loc = 'best')
    plt.savefig(util.fig_dir()+'bc03_mock_specz.png')           
    
    z_init = 0.1 
    for iz in range(35):
      zmask = (mock_z>z_init+iz*0.02)&(mock_z<z_init+(iz+1)*0.02)
      fig , ax = plt.subplots(nrows=3,ncols=1 , figsize=(5,15))
      for ind in range(3):

        ax[ind].scatter(mock_color[zmask,-1] , mock_color[zmask,ind] , s = 1 , color = next(palette), label = str(z_init+iz*0.02)+"<z<"+str(z_init+(iz+1)*0.02))
        #ax[ind].set_xlim([15,24])
        ax[ind].set_xlabel(r"$m_i$" , fontsize = 15)
      ax[0].set_ylim(0,4)
      ax[1].set_ylim(0,2.5)
      ax[2].set_ylim(0,1.5)
      ax[0].set_ylabel(r"$u-g$" , fontsize = 15)
      ax[1].set_ylabel(r"$g-r$" , fontsize = 15)
      ax[2].set_ylabel(r"$r-i$" , fontsize = 15)
      

      plt.legend(loc = 'best')
      plt.savefig(util.fig_dir()+'bc03_mock_cm'+str(iz)+'.png')     
    return None     

if __name__ == '__main__':

  gama = pf.open("data/KiDSxGAMAequ+G23.fits")[1].data
  sdss = pf.open("data/KiDS.DR3.x.SDSS.DR13.galaxies.fits")[1].data
  mask = np.where(np.in1d(sdss['ID'] , gama['ID'])==False)
  sdss = sdss[mask]
  col = np.hstack([gama_reduction() , sdss_reduction()])
  zs = np.linspace(0.05,0.8,100)
  mock1 = color_function(zs,0.25,1.5,0.008)
  mock2 = color_function(zs,0.25,1.5,0.02)  
  mock3 = color_function(zs,0.25,3,0.008)
  mock4 = color_function(zs,0.25,3,0.02)
  mock_median = np.zeros_like(mock1)
  mock_std = np.zeros_like(mock1)

  for ind in range(4):

      mock_median[ind] = np.median(np.vstack([mock1[ind], mock2[ind], mock3[ind], mock4[ind]]),axis=0)
      mock_std[ind] = np.std(np.vstack([mock1[ind], mock2[ind], mock3[ind], mock4[ind]]),axis=0)
  
  fig , ax = plt.subplots(nrows=3,ncols=1 , figsize=(5,15))

  for ind in range(3):
  
    ax[ind].scatter(col[11,:] , col[ind+8,:] , s = 0.001 , color = next(palette))
    ax[ind].plot(zs , mock1[ind] , label = "zf=1.5 , Z=0.008")
    ax[ind].plot(zs , mock2[ind] , label = "zf=1.5 , Z=0.02")
    ax[ind].plot(zs , mock3[ind] , label = "zf=3 , Z=0.008")
    ax[ind].plot(zs , mock4[ind] , label = "zf=3 , Z=0.02")
    ax[ind].set_xlim(0,0.8)
    ax[ind].set_xlabel(r"$z$" , fontsize = 15)
  ax[0].set_ylim(0,4)
  ax[1].set_ylim(0,2.5)
  ax[2].set_ylim(0,1.5)
  ax[0].set_ylabel(r"$u-g$" , fontsize = 15)
  ax[1].set_ylabel(r"$g-r$" , fontsize = 15)
  ax[2].set_ylabel(r"$r-i$" , fontsize = 15)
  plt.legend(loc = 'best')
  plt.savefig(util.fig_dir()+'bc03_kids.png')  

  red_mock()
