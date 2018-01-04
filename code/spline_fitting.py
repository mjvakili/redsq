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

def node_init3d(args):
    
    xref , theta_init = args
    
    nll = lambda *args: -lnlike3d(*args)
     
    mi = Y[:,1] 
    colors = Y[:,3:6]
    colorerrs = Y[:,6:]
    colorerrs  = colorerrs.reshape(colorerrs.shape[0], 3, 3)

    result = op.minimize(nll, theta_init, args=(mi, colors, colorerrs, xref), method = 'BFGS')
    opt_arr = result["x"]
    print opt_arr

    return opt_arr

def lnlike3d(theta, z, x, y, Cerr ,xref):
    
    m = theta[0] #array of m-nodes
    b = theta[1] #array of b-nodes
    lnf = theta[2] #array of lnf-nodes

    bz = CubicSpline(bnod , b)(z)
    mz = CubicSpline(mnod , m)(z)
    lnfz = CubicSpline(fnod , lnf)(z)
    xrefz = CubicSpline(xrefnod , xref)(z)

    Cint = np.zeros((len(z), 3, 3))
    Cint[:,0,0] = np.exp(2.* lnfz[:,0])
    Cint[:,1,1] = np.exp(2.* lnfz[:,1])
    Cint[:,2,2] = np.exp(2.* lnfz[:,2])

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
   xrefnod = np.linspace(z_init,z_fin,Nb) #spacing of .05 

   red_file = h5py.File("red_cat.hdf5" , 'r')
   red_sample = red_file['red'][:]
   mrefs = red_file['mref'][:]
   red_file.close()
   
   z_init , z_fin = 0.1 , 0.8
   Nthreads = 35
   znods = np.linspace(z_init, z_fin, Nthreads+1)
   xref = CubicSpline(.5*(znods[1:]+znods[:-1]), mrefs, xrefnod)
