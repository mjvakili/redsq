import pyfits as pf
import matplotlib.pyplot as plt
import multiprocessing
import emcee
import numpy as np
from astroML.density_estimation import XDGMM
from matplotlib.patches import Ellipse
from astroML.plotting.tools import draw_ellipse
plt.switch_backend("Agg")
import pandas as pd
import seaborn as sns 
import itertools
sns.set_style("white")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
sns.set_palette(sns.color_palette(["#9b59b6", "#95a5a6", 
                                   "#e74c3c", "#3498db", 
				   "#34495e", "#2ecc71"]))
from astroML.density_estimation import XDGMM
from scipy.stats import norm
import scipy as sc
import scipy.linalg as linalg
import scipy.optimize as op


def filter(style):
    '''
    There are two options for filenames: KiDSxGAMA matched catalog
                                         KiDS GAMA-like sample
    filename = "KiDSxGAMAequ+G23.fits
    filename = "KiDS_DR3_GAMA-like_rAUTOless20.3_NOMASKING_forMV.fits"
    '''
    if style == 'GAMA-MATCHED': filename = "KiDSxGAMAequ+G23.fits"
    elif style == 'GAMA-LIKE' : filename = "KiDS_DR3_GAMA-like_rAUTOless20.3_NOMASKING_forMV.fits"

    a = pf.open(filename)[1].data
    
    a = a[a['IMAFLAGS_ISO_U']&01010111==0]
    a = a[a['IMAFLAGS_ISO_G']&01010111==0]
    a = a[a['IMAFLAGS_ISO_R']&01010111==0]
    a = a[a['IMAFLAGS_ISO_I']&01010111==0]

    a = a[a['MAGERR_GAAP_U']>0]
    a = a[a['MAGERR_GAAP_G']>0]
    a = a[a['MAGERR_GAAP_R']>0]
    a = a[a['MAGERR_GAAP_I']>0]

    a = a[a['MAG_GAAP_u_CALIB']<25.4]
    a = a[a['MAG_GAAP_g_CALIB']<25.6]
    a = a[a['MAG_GAAP_r_CALIB']<24.7]
    a = a[a['MAG_GAAP_i_CALIB']<24.5]

    return a

def sdss_reduction():
    
    gama = pf.open("data/KiDSxGAMAequ+G23.fits")[1].data
    sdss = pf.open("data/KiDS.DR3.x.SDSS.DR13.galaxies.fits")[1].data
    mask = np.where(np.in1d(sdss['ID'] , gama['ID'])==False)
    sdss = sdss[mask]
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
    
def gama_reduction():


    gama = pf.open("data/KiDSxGAMAequ+G23.fits")[1].data
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

def catalog_combinator():
    '''
    combines sdss and gama catalogs 
    '''
    combined_cat = np.hstack([gama_reduction() , sdss_reduction()])

    return combined_cat


def catalog_slicer(zmin, zmax, component):

     
    z = combined_cat[11,:]
    mask = (z>zmin) & (z<zmax)
    reduced_cat = combined_cat[:,mask]
    
    color = reduced_cat[8:11,:]
    color_err = np.zeros_like(color)
    
    color_err = np.zeros((3,3,color.shape[1]))

    color_err[0,0,:] = reduced_cat[4,:]**2. + reduced_cat[5,:]**2.
    color_err[1,1,:] = reduced_cat[5,:]**2. + reduced_cat[6,:]**2.
    color_err[2,2,:] = reduced_cat[6,:]**2. + reduced_cat[7,:]**2.
    color_err[0,1,:] = -1.* reduced_cat[5,:]**2.
    color_err[1,0,:] = -1.* reduced_cat[5,:]**2.
    color_err[1,2,:] = -1.* reduced_cat[6,:]**2.
    color_err[2,1,:] = -1.* reduced_cat[6,:]**2.

    zspec = reduced_cat[11,:]
    x = reduced_cat[3,:]    #mi the reference magnitude
    xerr = reduced_cat[7,:] #ierr
    
    return zspec, x, xerr, color,  color_err

def mixture_fitting(args):
    '''
    component = 0 : u-g, 1: g-r, 2: r-i
    '''
    zmin, zmax, component = args
    zspec, x, xerr, color, color_err = catalog_slicer(zmin, zmax, component)
    
    Y_xd = np.vstack([x,color[component,:]]).T
    Yerr_xd = np.zeros((Y_xd.shape[0] , 2 , 2))
    Yerr_xd[:,0,0] = xerr
    Yerr_xd[:,1,1] = color_err[component,component,:]
    #fitting a two component GMM to (mi , color(component) space in the redshift bin)
    clf_in = XDGMM(2, n_iter=400)
    clf_in.fit(Y_xd, Yerr_xd)
    # mixture component associated with the red population
    red_index = np.where(clf_in.mu[:,1] == clf_in.mu[:,1].max())[0] 
    mu_red , V_red= clf_in.mu[red_index] , clf_in.V[red_index][0]
    red_line = mu_red[0,1] + V_red[0,1]*(Y_xd[:,0] - mu_red[0,0])/V_red[0,0]
    red_scatter = V_red[1,1] - V_red[0,1]**2./V_red[0,0]
    chi_red = (Y_xd[:,1] - red_line)**2. / (red_scatter + Yerr_xd[:,1,1])
    mask = chi_red < 2
    ##UPDATE : I have converged on using g-r for masking purposes!!
    # at this point we don't care which color component was used for masking
    # we keep the masked galaxies (chisq<2) and fit a linear line to the i-colors.
    # this step is agnostic about the color component used for masking 
    # note that we ahve used mu_red[0,0] (the first component of the center of the red galaxies) as m_ref
    x_xd = x[mask]
    xerr_xd = x[mask]

    Y_xd = np.vstack([color[0,mask], color[1,mask], color[2,mask]]).T
    Yerr_xd = np.zeros((Y_xd.shape[0] , 3 , 3))
    for i in xrange(3):
        for j in xrange(3):
            Yerr_xd[:,i,j] = color_err[i,j,mask]
    # fitting a two component GMM to the remainder of galaxies in the three dimensional colorspace
    clf_fi = XDGMM(2, n_iter=400)
    clf_fi.fit(Y_xd, Yerr_xd)
    pure_index = np.where(clf_fi.mu[:,1] == clf_fi.mu[:,1].max())

        
    
    return clf_fi.V[pure_index][0].flatten()

if __name__ == '__main__':

   
   combined_cat = catalog_combinator()

   
   z_init , z_fin = 0.1 , 0.8
   
   Nthreads = 42
   znods = np.linspace(z_init, z_fin, Nthreads+1)

   import multiprocessing
   from multiprocessing import Pool
   import h5py
   
   pool = Pool(Nthreads)
   mapfn = pool.map
   arglist = [None] * Nthreads
  
   for i in range(Nthreads):

       zmin , zmax = znods[i], znods[i+1]
       arglist[i] = (zmin, zmax , 1)
       
   result = list(mapfn(mixture_fitting, [ars for ars in arglist]))
   arr = np.array(result)
   """
   arr = result[0][1]
   mref = np.zeros((Nthreads))
   mref[0] = result[0][0]

   for i in range(1, Nthreads):

       arr = np.vstack([arr, result[i][1]])
       mref[i] = result[i][0]
   """
   red_file = h5py.File("scatter_prior.hdf5" , 'w')
   red_file.create_dataset("cov",(arr.shape[0], arr.shape[1]), data = np.zeros((arr.shape[0], arr.shape[1])))
   red_file["cov"][:] = arr
   red_file.close()
   pool.close()
