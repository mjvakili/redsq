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

    x = reduced_cat[3,:] #mi the reference magnitude
    xerr = reduced_cat[7,:] #ierr
    ##y = color[component, :]  #u-g , g-r , r-i
    ##yerr = color_err[component , component, :] #corresponding errors

    return x, xerr, color,  color_err

def mixture_fitting(zmin, zmax, component):
    '''
    component = 0 : u-g, 1: g-r, 2: r-i
    '''
    x, xerr, color, color_err = catalog_slicer(zmin, zmax, component)
    
    Y_xd = np.vstack([x,color[component,:]]).T
    Yerr_xd = np.zeros((Y_xd.shape[0] , 2 , 2))
    Yerr_xd[:,0,0] = xerr
    Yerr_xd[:,1,1] = color_err[component,component,:]
    
    #fitting GMM to (i , color(component))
    clf = XDGMM(2, n_iter=400)
    clf.fit(Y_xd, Yerr_xd)
    
    # mixture component associated with the red population
    red_index = np.where(clf.mu[:,1] == clf.mu[:,1].max())[0] 
    mu_red , V_red= clf.mu[red_index] , clf.V[red_index][0]
   
    ### computing the redsq membership probability and keeping points with chisq < 2
    ##dY_red = Y - mu_red
    ##V_red_inv = np.linalg.inv(V_red)
    ##VdY = np.tensordot(V_red_inv, dY_red , axes=(1,1))
    ##chi = np.sum(dY_red.T * VdY , axis = 0)
    
    red_line = mu_red[0,1] + V_red[0,1]*(Y[:,0] - mu_red[0,0])/V_red[0,0]
    red_scatter = V_red[1,1] - V_red[0,1]**2./V_red[0,0]
    mask = (Y[:,1]>red_line - red_scatter**.5)&(Y[:,1]<red_line + red_scatter**.5)

    # at this point we don't care which color component was used for masking
    # we keep the masked galaxies (chisq<2) and fit a linear line to the i-colors.
    # this step is agnostic about the color component used for masking 
    # note that we ahve used mu_red[0,0] (the first component of the center of the red galaxies) as m_ref

    nll = lambda *args: -lnlike(*args)
    
    result = op.minimize(nll, [0.0, mu_red[0,1], np.log(V_red[1,1]**.5)], args=(Y[mask,0], color[0,mask], color_err[0,0,mask]**.5 ,mu_red[0,0]))
    m_ug, b_ug, lnf_ug = result["x"]

    result = op.minimize(nll, [0.0, mu_red[0,1], np.log(V_red[1,1]**.5)], args=(Y[mask,0], color[1,mask], color_err[1,1,mask]**.5 ,mu_red[0,0]))
    m_gr, b_gr, lnf_gr = result["x"]

    result = op.minimize(nll, [0.0, mu_red[0,1], np.log(V_red[1,1]**.5)], args=(Y[mask,0], color[2,mask], color_err[2,2,mask]**.5 ,mu_red[0,0]))
    m_ri, b_ri, lnf_ri = result["x"]

    # now that we have the slope, intercept of lines, we need o find scatter. Note that the scatter we have found from the line fitting 
    # is the scatter after marginalizing over other color components so we are not particularly interested in that!

    #in order to find the intrinsic red-sequence scatter, we fit XD model to the ensemble of masked (red) galaxies in 
    #the three dimensional color space. we also use each  galaxy's 3x3 observed color uncertainties.


    #fitting GMM to three-dimensional color
    clf = XDGMM(1, n_iter=400)
    
    Y_xd = np.vstack([color[0,mask], color[1,mask], color[2,mask]]).T
    Yerr_xd = np.zeros((Y_xd.shape[0] , 3 , 3))
    for i in xrange(3):
        for j in xrange(3):
            Yerr_xd[:,i,j] = color_err[i,j,mask]
    clf.fit(Y_xd, Yerr_xd)
    var_int = clf.V[0] #intrinsic variance

    return m_ug, b_ug, m_gr, b_gr, m_ri, b_ri, var_int

def lnlike(theta, x, y, yerr ,xref):
    
    m, b, lnf = theta
    model = m * (x-xref) + b
    inv_sigma2 = 1.0/(yerr**2 + np.exp(2*lnf))

    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

if __name__ == '__main__':

   
   combined_cat = catalog_combinator()

   Niter = 5 
   z_init = 0.1
   Nthreads = 1

   import multiprocessing
   from multiprocessing import Pool
   
   pool = Pool(Nthreads)
   mapfn = pool.map
   arglist = [None] * Nthreads
   for i in range(Nthreads):

       zmin = z_init + i*0.02
       zmax = zmin + 0.02
       x, y, yerr = catalog_slicer(zmin , zmax , 1)
       
       ##filename = "results3/gr_result_z_"+str(zmin)+"_"+str(zmax)+".txt"
       ##pmem = np.loadtxt(filename)[12:]
       ##x , y , yerr = x[pmem>0.8] , y[pmem>0.8] , yerr[pmem>0.8]
       
       arglist[i] = (zmin, zmax, 1, x, y, yerr)  
   result = list(mapfn(mcmc, [ars for ars in arglist]))
   for i in range(Nthreads):
       zmin = z_init + i*0.01
       zmax = zmin + 0.01
       x, y, yerr = test(zmin , zmax , 2)
       arglist[i] = (zmin, zmax , 8, x, y , yerr)  
   result = list(mapfn(mcmc, [ars for ars in arglist]))
   
   for t in range(Nthreads):
       zmin = z_init + t * 0.01
       zmax = zmin + 0.01
       np.savetxt("results/ri_result_z_"+str(zmin)+"_"+str(zmax)+".txt" , np.array(result[t]))
   pool.close()