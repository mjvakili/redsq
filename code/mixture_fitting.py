import pyfits as pf
import matplotlib.pyplot as plt

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

def node_init(args):

    zmin, zmax , inod = args
    xref = mrefs[inod]
    mask = (red_sample[:,0]>zmin)&(red_sample[:,0]<zmax)
    Y = red_sample[mask,:]
    
    nll = lambda *args: -lnlike(*args)
   
    result = op.minimize(nll, [0.01, np.median(Y[:,3]), .05], args=(Y[:,1], Y[:,3], Y[:,6]**.5 , xref), method = 'BFGS')#, method = 'CG',options={'gtol': 1e-05, 'eps': 1.4901161193847656e-08})
    m_ug, b_ug, lnf_ug = result["x"]
    
    result = op.minimize(nll, [0.01, np.median(Y[:,4]), .05], args=(Y[:,1], Y[:,4], Y[:,10]**.5 , xref))
    m_gr, b_gr, lnf_gr = result["x"]
    
    result = op.minimize(nll, [0.01, np.median(Y[:,5]), .05], args=(Y[:,1], Y[:,5], Y[:,14]**.5 , xref))
    m_ri, b_ri, lnf_ri = result["x"]
    
    return np.array([m_ug, m_gr, m_ri, b_ug, b_gr, b_ri, np.exp(lnf_ug), np.exp(lnf_gr), np.exp(lnf_ri)])

def node_init3d(args):
    
    zmin, zmax , inod = args
    xref = mrefs[inod]
    mask = (red_sample[:,0]>zmin)&(red_sample[:,0]<zmax)
    Y = red_sample[mask,:]
    
    nll = lambda *args: -lnlike3d(*args)
     
    mi = Y[:,1] 
    colors = Y[:,3:6]
    colorerrs = Y[:,6:]
    colorerrs  = colorerrs.reshape(colorerrs.shape[0], 3, 3)
    result = op.minimize(nll, [0.01, 0.01, 0.01, np.median(colors, axis=0)[0], np.median(colors, axis=0)[1], np.median(colors, axis=0)[2], .05, 0.05, 0.05], args=(mi, colors, colorerrs, xref), method = 'BFGS')
    opt_arr = result["x"]
    opt_arr[6:] = np.exp(opt_arr[6:])

    return opt_arr

def lnlike(theta, x, y, yerr ,xref):
    
    m, b, lnf = theta
    model = m * (x-xref) + b
    inv_sigma2 = 1.0/(yerr**2 + np.exp(2*lnf))

    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnlike3d(theta, x, y, Cerr ,xref):
    
    m = theta[:3]
    b = theta[3:6]
    lnf = theta[6:]
    Cint = np.diag(np.exp(2.*np.array([lnf[0],lnf[1],lnf[2]])))
    Ctot = Cerr + Cint[None,:,:]
    res = m[None,:]*(x-xref)[:,None] + b[None,:] - y
    chi = np.sum(np.einsum('mn,mn->n', res, np.einsum('ijk,ik->ij',np.linalg.inv(Ctot),res))) + np.sum(np.log(np.linalg.det(Ctot)))
    
    return -0.5*chi

if __name__ == '__main__':
   
   red_file = h5py.File("red_cat.hdf5" , 'r')
   red_sample = red_file['red'][:]
   mrefs = red_file['mref'][:]
   red_file.close()
   
   z_init , z_fin = 0.1 , 0.8
   Nthreads = 35
   znods = np.linspace(z_init, z_fin, Nthreads+1)
   Nthreads = 5
   sidenods = np.linspace(z_init, z_fin, Nthreads+1)
   from scipy.interpolate import spline
   mrefs = spline(.5*(znods[1:]+znods[:-1]), mrefs, sidenods)
   znods = sidenods
   pool = Pool(Nthreads)
   mapfn = pool.map
   arglist = [None] * Nthreads
  
   for inod in range(Nthreads):

       zmin , zmax = znods[inod], znods[inod+1]
       arglist[inod] = (zmin, zmax , inod)
   opt = list(mapfn(node_init3d, [ars for ars in arglist]))
    
   arr = opt[0]   
   for i in range(1, Nthreads):
       arr = np.vstack([arr, opt[i]])
   red_par = h5py.File("red_par3d.hdf5" , 'w')
   red_par.create_dataset("red",(arr.shape[0], arr.shape[1]), data = np.zeros((arr.shape[0], arr.shape[1])))
   red_par["red"][:] = arr
   red_par.close()
   pool.close()
