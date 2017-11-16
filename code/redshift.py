'''
class for estimating
p(z|m,c) \propto p(c|m,z)p(m|z)p(z)

p(c|m,z) constructed by spline 
interpolation of the best-fit 
c-m relation parameters


p(m|z) is given by the 
schecter fuction with m_char
derived from BC03, KIDS 
bandpass, and EZGAL fella

p(z) = dV_{comoving}/dz
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import CubicSpline
import ezgal
import cosmolopy.distances as cd
import util

cosmo = {'omega_M_0:0.3', 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':1.0}

def prior(z):
    '''
    dv/dz to impose uniformity
    in redshift
    '''
    d_a = cd.angular_diameter_distance(z, **cosmo) #angular diameter distance
    h = cd.e_z(z, **cosmo) #hubble parameter
    dvdz = (1+z)**2. * d_a **2  * h **-1. #dv/dz
    #for the sake of numerical stability we devide dvdz by 10000.0
    return dvdz / 10000.0

def redmapper_mstar(z):
    '''
    Rykoff+12 fitting formula for characteristic sloan iband mags 
    of redmapper galaxy cluster members at z<0.4.
    Used to normalize the Ezgal model
    '''
    return 22.44 + 3.36*np.log(z) + 0.273*np.log(z)**2 - 0.0618*np.log(z)**3 - 0.0227*np.log(z)**4

def kids_mstar(zs):
    '''
    returns mstar of redsequence galaxies 
    as observed by kids i band
    '''
    model = ezgal.model("/net/delft/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_0.02_chab.model")
    model.add_filter("/net/delft/data2/vakili/easy/i.dat" , "kids" , units = "nm")
    kcorr_sloan = model.get_kcorrects(zf=3.0 , zs = 0.25 , filters = "sloan_i")
    model.set_normalization("sloan_i" , 0.25 , mstar(0.25)-kcorr_sloan, vega=False, apparent=True)
    zf = 3.0 #HARDCODED
    #kcorr = model.get_kcorrects(zf=3.0, zs=zs , filters = "kids") #WE SHALL KCORRECT THE MAGS IF NEEDED
    mags = model.get_apparent_mags(zf=zf , filters = "kids" , zs= zs)

    return mags

def schecter(m,z):
    '''
    magnitude distribution
    as a function of redshift
    '''
    mchar = kids_mstar(z)
    dm = m - mchar
    exparg = 10. ** (-0.4 * dm)

    return (exparg ** 2.) * np.exp(-1.*exparg) 

def luminosity(m,z):
    '''
    L/Lstar of redsequence galaxies
    '''
    mchar = kids_mstar(z)
    dm = m - mchar
    exparg = 10. ** (-0.4 * dm)

    return exparg

class estimate(object):

    def __init__(self, c, m, Cerr, zmin, zmax, dz)
 
        self.c = c       #color 3-dimensional color vector :[u-g , g-r , r-i]
        self.m = m       #observed magnitude in the i band : mi
        self.Cerr = Cerr #observational color errorbars: 3x3 matrix
        self.zmin = zmin #minimum redshift
        self.zmax = zmax #maximum redshift
        self.dz = dz     #width of redshift bins

        cm_dir = util.cm_dir()
        Nbin = (self.zmax - self.zmin) / self.dz
        cov , ug , gr , ri = [] , [] , [] , []
        cm_dir = util.cm_dir()

        for i in xrange(Nbin):

            z1 = self.zmin + self.dz * i
            z2 = self.zmin + self.dz * (i+1)
            file_ug = cm_dir+"ug_result_z"+str(z1)+"_"+str(z2)+".txt"     
            file_gr = cm_dir+"gr_result_z"+str(z1)+"_"+str(z2)+".txt"   
            file_ri = cm_dir+"ri_result_z"+str(z1)+"_"+str(z2)+".txt"     
            file_cov = cm_dir+"covariance_z"+str(z1)+"_"+str(z2)+".txt"     
            ug.append(np.loadtxt(file_ug)[:2])
            gr.append(np.loadtxt(file_gr)[:2])
            ri.append(np.loadtxt(file_ri)[:2])
            cov.append(np.loadtxt(file_cov))

        self.cov , self.ug , self.gr , self.ri = np.array(cov) , np.array(ug) , \
                                                 np.array(gr) , np.array(ri)
        return None

    def interpolate(self , z):
        """
        interpolate a , b , c arrays from 
        nodes to a given z
        """
        Nbin = (self.zmax - self.zmin)/self.dz
        z_input = np.linspace(self.zmin , self.zmax , Nbin)
        cov_z = CubicSpline(z_input, self.cov)(z)
        ug_z = CubicSpline(z_input, self.ug)(z)
        gr_z = CubicSpline(z_input, self.gr)(z)
        ri_z = CubicSpline(z_input, self.ri)(z)

        return cov_z , ug_z , gr_z , ri_z
   
    def lnredsq(self, z ,mag ,c ,cerr):

        cov_z, ug_z, gr_z , ri_z = self.interpolate(z)
        cov_tot = cov_z + cerr
        slope  = np.array([ug_z[0],gr_z[0],ri_z[0]])
        incpt  = np.array([ug_z[1],gr_z[1],ri_z[1]])

        cmod = slope * (mag - 19) + intcp
        dc = c - cmod
        lnred = -0.5 * np.dot(dc, np.linalg.solve(cov_tot, dc))

        return lnred 

