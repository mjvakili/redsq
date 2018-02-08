import pyfits as pf
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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
import multiprocessing
from multiprocessing import Pool
import h5py

def nod_init3d(args, method):
    
    xref , theta_init , Y= args
    
    nll = lambda *args: -lnlike3d(*args)
    
    z = Y[:,0]
    mi = Y[:,1] 
    colors = Y[:,3:6]
    colorerrs = Y[:,6:]
    colorerrs  = colorerrs.reshape(colorerrs.shape[0], 3, 3)
   
    colorerrs = colorerrs[z<0.75,:,:]
    mi = mi[z<0.75]
    colors = colors[z<0.75,:]
    z = z[z<0.75]


    if method == 'TNC':
        result = op.minimize(nll, theta_init, args=(z, mi, colors, colorerrs, xref), method = 'TNC', bounds = bnds, options={'disp':True})
    
    if method == 'BFGS':
        result = op.minimize(nll, theta_init, args=(z, mi, colors, colorerrs, xref), method = 'L-BFGS-B', bounds = bnds,  options={'disp':True,'maxiter':150})
    opt_arr = result["x"]
    print opt_arr

    return opt_arr

def lnprior(theta):

    for i in range(3*(Nm+Nb-2), len(theta_init)):
       if bnds[i][0]<theta[i]:
          return 0.0
       else: 
          return -np.inf


def lnlike3d(theta, z, x, y, Cerr ,xref):
    
    m = theta[0:3*(Nm-1)].reshape(Nm-1,3) #array of m-nodes
    b = theta[3*(Nm-1):3*(Nm+Nb-2)].reshape(Nb-1,3) #array of b-nodes
    lnf = theta[3*(Nm+Nb-2):].reshape(Nf-1,3) #array of lnf-nodes
    ######################## EXCLUDING THE LAST NODES##########
    m = m[:-1,:]
    b = b[:-1,:]
    lnf = lnf[:-1,:]
    ###############################################################
    bz = CubicSpline(bnod , b)(z)
    mz = CubicSpline(mnod , m)(z)
    lnfz = CubicSpline(fnod , lnf)(z)
    xrefz = CubicSpline(xrefnod , xref)(z)

    Cint = np.zeros((len(z), 3, 3))
    Cint[:,0,0] = np.exp(2.* lnfz[:,0])
    Cint[:,1,1] = np.exp(2.* lnfz[:,1])
    Cint[:,2,2] = np.exp(2.* lnfz[:,2])
    #Cint[:,0,0] = lnfz[:,0]**2
    #Cint[:,1,1] = lnfz[:,1]**2
    #Cint[:,2,2] = lnfz[:,2]**2
    Ctot = Cerr + Cint
    res = mz * (x - xrefz)[:,None] + bz - y
    chi = np.sum(np.einsum('mn,mn->n', res, np.einsum('ijk,ik->ij',np.linalg.inv(Ctot),res))) + np.sum(np.log(np.linalg.det(Ctot)))
    
    return -0.5*chi

if __name__ == '__main__':
 
   Nb, Nm, Nf = 15, 8, 6
   z_init , z_fin = 0.1, 0.8

   bnod = np.linspace(z_init,z_fin,Nb) #spacing of .05
   mnod = np.linspace(z_init,z_fin,Nm) #spacing of .1
   fnod = np.linspace(z_init,z_fin,Nf) #spacing of .14
   xrefnod = np.linspace(z_init,z_fin,20) #spacing of .05 

   bnod = .5*(bnod[1:]+bnod[:-1])
   fnod = .5*(fnod[1:]+fnod[:-1])
   mnod = .5*(mnod[1:]+mnod[:-1])

   bnod = bnod[:-1]
   mnod = mnod[:-1]
   fnod = fnod[:-1]
   #m = m[:-1,:]
   #b = b[:-1,:]
   #lnf = lnf[:-1,:]

   red_file = h5py.File("red_cat.hdf5" , 'r')
   red_sample = red_file['red'][:]
   mrefs = red_file['mref'][:]
   red_file.close()
   Nthreads = 35
   znods = np.linspace(z_init, z_fin, Nthreads+1)
   znods = .5 * (znods[1:]+znods[:-1]) 
   mrefs = mrefs[znods<0.75]
   znods = znods[znods<0.75]
   
   xref = CubicSpline(znods, mrefs)(xrefnod)

   """
   m_init = h5py.File("red_par3d_mnods.hdf5","r")["red"][:].reshape(Nm-1,9)[:,0:3]
   b_init = h5py.File("red_par3d_bnods.hdf5","r")["red"][:].reshape(Nb-1,9)[:,3:6]
   lnf_init = h5py.File("red_par3d_lnfnods.hdf5","r")["red"][:].reshape(Nf-1,9)[:,6:]
   lnf_init = np.log(lnf_init)

   print "b=" , b_init 
   print "m=" , m_init 
   print "lnf=" , lnf_init

   theta_init = np.hstack([m_init.flatten(), b_init.flatten(), lnf_init.flatten()])
   """
   #theta_init = np.loadtxt('opt_theta.txt')
   # we already have a good estimate of theta, so we start the optimization by what we have found previously
   theta_init = np.loadtxt('opt_theta.txt')
   
   bnds = [(None, None) for i in range(len(theta_init))]
   l_bnds = np.loadtxt("lower_bnd_scatter.txt").flatten()
   u_bnds = np.loadtxt("upper_bnd_scatter.txt").flatten()

   for i in range(3*(Nm+Nb-2), len(theta_init)):
        theta_init[i] = .5 * np.log(l_bnds[i-3*(Nm+Nb-2)]) + .5 * np.log(u_bnds[i-3*(Nm+Nb-2)])
        bnds[i] = (np.log(l_bnds[i-3*(Nm+Nb-2)]) , np.log(u_bnds[i-3*(Nm+Nb-2)]))
        print (theta_init[i]<bnds[i][1])&(theta_init[i]>bnds[i][0]) 
   print "parameter bounds for prior are = " , bnds
   
   optimized_theta = nod_init3d([xref , theta_init, red_sample] , method = 'BFGS')

   #np.savetxt("opt_theta_new.txt", optimized_theta) #with hardcoded prior of 0.02 for the intrinsic scatters
   np.savetxt("opt_theta_bfgs_bounded3.txt", optimized_theta) #with hardcoded priors from taking medians of GMM fitting
