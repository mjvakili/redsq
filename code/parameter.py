'''
piece of code 
for plotting 
the best-fit c-m relation 
parameters
'''

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
import multiprocessing

plt.switch_backend("Agg")

import pandas as pd
import seaborn as sns 
import itertools
sns.set_style("white")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
sns.set_palette(sns.color_palette(["#9b59b6", "#95a5a6", 
                                   "#e74c3c", "#3498db", 
				   "#34495e", "#2ecc71"]))


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


def test(zmin , zmax):
    '''
    test returns x = mi 
                 y = g-r
              yerr = (g-r)-err

    update :  1) make this return u-g , g-r , r-i
              1) this will require covariance between color components.
              2) return xerr = mi_err. Is it even worth it to include x errors?
    '''

    gals = pf.open("groups/G3CGalv07.fits")[1].data   #galaxy group catalog
    gals = gals[gals['GroupID']!=0]              #keeping galaxies that are in groups  
    match = filter('GAMA-MATCHED') 
    
    mask = np.in1d(match['CATAID'] , gals[gals['GroupID']!=0]['CATAID'])
    matched_gals = match[mask]

    ug = matched_gals['COLOR_GAAPHOM_U_G'] 
    gr = matched_gals['COLOR_GAAPHOM_G_R'] 
    ri = matched_gals['COLOR_GAAPHOM_R_I']
    z = matched_gals['Z']
    mi = matched_gals['MAG_GAAP_i_CALIB']
    colors = np.array([ug , gr , ri])
    
    u_err = matched_gals['MAGERR_GAAP_U']
    g_err = matched_gals['MAGERR_GAAP_G']
    i_err = matched_gals['MAGERR_GAAP_I']
    r_err = matched_gals['MAGERR_GAAP_R']
    
    c_err = np.array([u_err**2. + g_err**2.,
                      g_err**2. + r_err**2.,
		      r_err**2. + i_err**2.])**0.5 

    mi = mi[(zmin<z)&(z<zmax)]
    redshift_mask = np.where((zmin<z)&(z<zmax))[0]
    colors = colors[:, redshift_mask]
    c_err = c_err[: , redshift_mask]
    z = z[redshift_mask]
    
    return z , mi , colors.T , c_err.T


def scatter_redsq():

    z_init = 0.2
    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True)
    
    
    for zbin in range(20):
        
        zmin = z_init + zbin * 0.01
	zmax  = zmin + 0.01

        z , mi , color , colorerr = test(zmin , zmax)
        filename = "results/gr_result_z_"+str(zmin)+"_"+str(zmax)+".txt"
        pmem = np.loadtxt(filename)[12:]
        pmem = pmem / np.sum(pmem)
	print pmem.max()

        scatter_kwargs = {"zorder":100}
	#error_kwargs = {"lw":.5, "zorder":0}
	plt.scatter(z,color[:,1],c=pmem, s = 0.0007,vmin = 0 , vmax = 0.01, **scatter_kwargs)
	#errorbar(X,Y,yerr=ERR,fmt=None, marker=None, mew=0,**error_kwargs )

	#axs[i].errorbar(z, colors[i+1,:], yerr = c_err[i+1,:],
	#                c= next(palette), fmt = 'o')
        #axs[i].legend(loc='lower right', ncol=2)
        #axs[i].set_ylabel(labels[i+1])
	#axs[0].set_yscale('log')
        #axs[i].set_xlim([16.5, 21.5])
    #axs[0].set_ylim([-0.1, 2.1])
    #axs[1].set_ylim([-0.1, 1.1])
    #axs[1].set_xlabel(r'$m_{i}$')
    plt.colorbar() 
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/global.png")
    plt.close()
    
    return None

def plot_redsq(color_component):

    if color_component == "ug": 
       ext = "ug"
    elif color_component == "gr": 
       ext = "gr"
    elif color_component == "ri": 
       ext = "ri"

    z_init = 0.2
    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters

    bestfit = np.zeros((10,3))
    error = np.zeros((10,3))

    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True)
    """
    for zbin in range(10):
        
        zmin = z_init + zbin * 0.02
	zmax  = zmin + 0.02
        filename = "results3/"+ext+"_result_z_"+str(zmin)+"_"+str(zmax)+".txt"
        est = np.loadtxt(filename)[:12]
        bestfit[zbin,:] = est[:3]   
        error[zbin,:] = est[6:9]   
    z = np.linspace(z_init , 0.4 , 10) 
    for i in range(3):

        axs[i].plot(z, bestfit[:,i], c= next(palette), lw = 1.0 , ls = "--")

    palette = itertools.cycle(sns.color_palette())
    """
    for zbin in range(10):
        
        zmin = z_init + zbin * 0.02
	zmax  = zmin + 0.02
        filename = "results4/"+ext+"_result_z_"+str(zmin)+"_"+str(zmax)+".txt"
        est = np.loadtxt(filename)[:12]
        bestfit[zbin,:] = est[:3]   
        error[zbin,:] = est[6:9]   
     
    z = np.linspace(z_init , 0.4 , 10) 

    for i in range(3):

	axs[i].plot(z, bestfit[:,i], c= next(palette), lw = 2.0)
	#axs[i].errorbar(z, bestfit[:,i], yerr = error[:,i] , c= next(palette), lw = 2.0)
    axs[2].set_xlabel(r'$z$' , fontsize = 20)
    axs[0].set_ylabel(r'$a_{r-i}$' , fontsize = 20)
    axs[1].set_ylabel(r'$b_{r-i}$' , fontsize = 20)
    axs[2].set_ylabel(r'$\delta_{r-i}$' , fontsize = 20)
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/global_solution_"+ext+".png")
    plt.close()
    
    return None
     
if __name__ == '__main__':

    plot_redsq("ri")
