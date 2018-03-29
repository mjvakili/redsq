
import numpy as np
import h5py
from scipy.interpolate import CubicSpline
import ezgal
import cosmolopy.distance as cd
import util
import emcee
import scipy.optimize as op

def vc(z):
    '''
    dv/dz to impose uniformity
    in redshift
    '''
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':1.0}
 
    return cd.comoving_volume(z,**cosmo)

def luminosity(m,z):
    '''
    L/Lstar of redsequence galaxies
    '''
    mchar = kids_mstar(z)
    dm = m - mchar
    exparg = 10. ** (-0.4 * dm)

    return exparg
    
def kids_mstar(zs):
    '''
    returns mstar of redsequence galaxies 
    as observed by kids i band
    '''
    #kcorr = model.get_kcorrects(zf=3.0, zs=zs , filters = "kids") #WE SHALL KCORRECT THE MAGS IF NEEDED
    mags = model.get_apparent_mags(zf=zf , filters = "kids" , zs= zs)

    return mags
class calibrate(object):

    def __init__(self, zmin, zmax, dz, Nb, Nm, Nf, 
                 Nhist, Nab, Nchi , lmin, nbar, 
		 z, l, chi, colors, color_errs, mis, 
		 calib):
        
        # selection summary
	self.lmin = lmin
        self.nbar = nbar

        # ID, color, mag of all gals
        self.ID = ID
        self.colors = colors
        self.color_errs = color_errs
        self.mis = mis
        
        #value added properties of all gals
        self.z = z
	self.l = l
	self.chi = chi

        # ID and z of calib gals
        self.calib = calib   # the whole calibration sample consisting of ID, zrm, chi, l, zspec. the same as calib_sample defined below

        # initial estimates of chi, l, z
        ###self.chi_init = chi_init
        ###self.l_init = l_init
        ###self.z_init = z_init

        # sum numbers for making an initial cut
        ###self.chi_abs = chi_abs
        ###self.zmin_abs = zmin_abs
        ###self.zmax_abs = zmax_abs
        ###self. l_abs = l_abs

        self.zmin = zmin #minimum redshift
        self.zmax = zmax #maximum redshift
        self.dz = dz     #width of redshift bins
        self.Nb = Nb
	self.Nm = Nm
	self.Nf = Nf
        
        self.bnod = np.linspace(self.zmin,self.zmax,self.Nb) #spacing of .05
        self.mnod = np.linspace(self.zmin,self.zmax,self.Nm) #spacing of .1
        self.fnod = np.linspace(self.zmin,self.zmax,self.Nf) #spacing of .14
        self.xrefnod = np.linspace(self.zmin,self.zmax,20) #spacing of .05 
        self.bnod = .5*(self.bnod[1:]+self.bnod[:-1])
        self.fnod = .5*(self.fnod[1:]+self.fnod[:-1])
        self.mnod = .5*(self.mnod[1:]+self.mnod[:-1])

        self.Nhist = Nhist  #the number of bins for making hist of data
        self.Nab = Nab    #the number of spline nodes for afterburner
        self.Nchi = Nchi  #the number of spline nodes for parameterizing chimax
        self.hisnod = np.linspace(self.zmin,self.zmax,self.Nhist) #the nods a which we make hist of data 
        self.abnod = np.linspace(self.zmin,self.zmax,self.Nab) # the afterburner nods
        self.chinod = np.linspace(self.zmin,self.zmax,self.Nchi) #the chimax nods

        # the inferred c-m relation parameters
        self.theta = np.loadtxt("opt_theta_bfgs_bounded2.txt")
        self.m = self.theta[0:3*(self.Nm-1)].reshape(self.Nm-1,3) #array of m-nodes
        self.b = self.theta[3*(self.Nm-1):3*(self.Nm+self.Nb-2)].reshape(self.Nb-1,3) #array of b-nodes
        self.lnf = self.theta[3*(self.Nm+self.Nb-2):].reshape(self.Nf-1,3) #array of lnf-nodes

        ####################################HACKY ###############################
        self.bnod = self.bnod[:-1]
        self.mnod = self.mnod[:-1]
        self.fnod = self.fnod[:-1]
        self.m = self.m[:-1,:]
        self.b = self.b[:-1,:]
        self.lnf = self.lnf[:-1,:]
        ####################################################################
        red_file = h5py.File("red_cat.hdf5" , 'r')
        red_sample = red_file['red'][:]
        mrefs = red_file['mref'][:]
        red_file.close()
        znods = np.linspace(self.zmin, self.zmax, 36)
	znods = .5*(znods[1:]+znods[:-1])
	#####################################HACKY#############################
        mrefs = mrefs[znods<0.7]
        znods = znods[znods<0.7]
	#######################################################################
        self.xref = CubicSpline(znods, mrefs)(self.xrefnod)
        
	#######################
        self.dz_theta_zero = 0.01

        return None

    def interpolate(self , z):
        """
        interpolate a , b , c arrays from 
        nodes to a given z
        """

        bz = CubicSpline(self.bnod , self.b)(z)
        mz = CubicSpline(self.mnod , self.m)(z)
        lnfz = CubicSpline(self.fnod , self.lnf)(z)
        xrefz = CubicSpline(self.xrefnod , self.xref)(z)
        
        return mz, bz, lnfz, xrefz

    def chi3d(self):
        """
	this recalculates chi3d for every galaxy in the survey for which we have a zred 
	regardless of l and chi
        fast calculation of all the chis for a set of zs 
        z = the current estimate of reshifts
        """
        bz = CubicSpline(self.bnod , self.b)(self.z)
        mz = CubicSpline(self.mnod , self.m)(self.z)
        lnfz = CubicSpline(self.fnod , self.lnf)(self.z)
        xrefz = CubicSpline(self.xrefnod , self.xref)(self.z)

        Cint = np.zeros((len(self.z), 3, 3))
        Cint[:,0,0] = np.exp(2.* lnfz[:,0])
        Cint[:,1,1] = np.exp(2.* lnfz[:,1])
        Cint[:,2,2] = np.exp(2.* lnfz[:,2])
        Ctot = self.color_errs + Cint
        res = mz * (self.mis - xrefz)[:,None] + bz - self.colors
        chis = np.einsum('mn,mn->n', res, np.einsum('ijk,ik->ij',np.linalg.inv(Ctot),res))
    
        return chis
    
    def l3d(self):
        """
	this recalculates l3d for every galaxy in the survey for which we have a zred
        regardless of its chi and l
	fast calculation of all the luminosity ratios for a set of zs 
        z = the current estimate of reshifts
        """
        ls = luminosity(self.mis, self.z)
                 
        return ls
    
    def ab_lnlike(self, dz_theta, mask):
        """
        dz_theta = estimate of dz at ab spline nods
        mask = some mask on chi and l of spec red gals
        """
        x , y = self.calib[mask,1], self.calib[mask,1]-self.calib[mask,-1]
        dz_pred = CubicSpline(self.abnod , dz_theta)(x)
        chisq = np.sum(np.abs(y - dz_pred))
    
        return chisq
     
    def solve_ab_lnlike(self, mask):
        """
	solves ab_lnlike
	dz_theta_zero = initial guess for dz_theta
	mask = some mask on chi and l of spec red gals
        """
        nll = lambda *args: self.ab_lnlike(*args)
	result = op.minimize(nll, self.dz_theta_zero*np.ones((self.Nab)), args=(mask))
	
	return result["x"]
    
 
    def chimax_lnlike(self, chimax_theta):

	#first have to run the ab
        #alt chinods , norm =  np.exp(chimax_theta[:-1]), np.exp(chimax_theta[-1]) 
        chinods =  np.exp(chimax_theta)
        chimax_calib = CubicSpline(self.chinod ,chinods)(self.calib[:,1])
	mask = (self.calib[:,2] < chimax_calib)&(self.calib[:,3] > self.lmin)
        dz_ab = self.solve_ab_lnlike(mask)
	print "CURRENT ESTIMATE OF AB = " , dz_ab
        # calibrate the calib-zs and rezs
        self.calib[:,1] = self.calib[:,1] - CubicSpline(self.abnod, dz_ab)(self.calib[:,1]) #calib gals
	self.z = self.z - CubicSpline(self.abnod , dz_ab)(self.z) #all gals
        #calibrate the red-chis and red-ls
        self.chi = self.chi3d()
	##print "DONE WITH THE HARD"
	self.l = self.l3d()
        #mask the red-ls that are larger than self.lmin (=0.5 or 1)
        mask = self.l > self.lmin
      
    	#chinods , norm = np.exp(chimax_theta[:-1]), np.exp(chimax_theta[-1])  
    	chinods = np.exp(chimax_theta)
    	norm = self.nbar * 360.3 / (41252.96)
    	
	chi_maxes = CubicSpline(self.chinod ,chinods)(self.z)
    	sample3 = self.z[self.catalog[:,2]<chi_maxes]
	hist , edges = np.histogram(sample3[:,1], bins = self.Nhist)
    	bins = .5*(edges[1:]+edges[:-1])
    	dbin = edges[1] - edges[0]
    	dvbin = vc(edges[1:]) - vc(edges[:-1])
    	dvbin = dvbin * 360.3 / (41252.96)
    	hist = hist / dvbin

    	chisq = np.sum((hist - self.nbar)**2./ (self.nbar* dvbin**-1.))
    
    	return chisq

    def solve_chimax_lnlike(self):
        """
	solves chi_max_lnlike
        """

    	chinods_0 = 2 + np.zeros((self.Nchi))
        print "chinods_0" , chinods_0    	
	nll = lambda *args: self.chimax_lnlike(*args)
        bnds = []
	for h in range(len(chinods_0)):
	    bnds.append((np.log(1.),np.log(3)))
    	result = op.minimize(nll, np.log(chinods_0) , method='SLSQP', bounds = bnds, options={'disp': True ,'eps' : .001})
    	print "esimated chis" , np.exp(result["x"])

        return None
         

if __name__ == '__main__':

   ############ THE ENTIRE KIDS WITH COLORS AND MAGS #############
   reduced_kids = h5py.File("reduced_kids.h5" , "r")

   ID = reduced_kids['ID'][:34290650]
   RA  = reduced_kids['RA'][:34290660]
   DEC = reduced_kids['DEC'][:34290660]
   DEC = reduced_kids['DEC'][:34290660]
   mi = reduced_kids['mi'][:34290660]
   redshift = reduced_kids['redshift'][:34290660]
   colors = reduced_kids['colors'][:34290660]
   color_errs = reduced_kids['color_errs'][:34290660]
   
   ########## THE ENTIRE KIDS WITH UNCALIB Z l CHI #############
   red_sample = h5py.File("red_photometric_sample_v2.h5" , "r")
   red_sample = red_sample["opt"][:34290660]

   ######### KiDS GALAXIES WITH SPECZ FOR CALIBRATION ######## 
   spec_kids = h5py.File("reduced_speccalibkids.h5" , 'r')
   spec_ID = spec_kids["ID"][:]
   spec_z = spec_kids["redshift"][:]
   
   ######## MATCHED SPECZ & REDZ ARRAY FOR CALIBRATION #######
   
   spec_ID , spec_index = np.unique(spec_kids["ID"][:] , return_index = True)
   spec_z = spec_kids["redshift"][:][spec_index]
   mask_one = np.where(np.in1d(ID,spec_ID)==True)[0]
   mask_two = np.where((np.in1d(spec_ID,ID[mask_one])==True))[0]
   #specz = spec_z[mask_two]
   arg_one = np.argsort(ID[mask_one])
   arg_two = np.argsort(spec_ID[mask_two])

   ##print (ID[mask_one][arg_one] == spec_ID[mask_two][arg_two]) #sanity check!

   spec_red_sample = red_sample[mask_one][arg_one]
   mask_chil = (spec_red_sample[:,0]==1)
   spec_specz = spec_z[mask_two][arg_two]
   
   calib_sample = np.vstack([spec_red_sample.T , spec_specz]).T
   
   ##print calib_sample #sanity check
   

   ########## INITIALIZATION OF THE EZGAL MODEL ########

   model = ezgal.model("/net/vuntus/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_0.02_salp.model")
   zf = 3.0 #HARDCODED
   model.add_filter("/net/vuntus/data2/vakili/easy/i.dat" , "kids" , units = "nm")
   model.set_normalization("sloan_i" , 0.2 , 17.85, vega=False, apparent=True)
   cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':1.0}

   ######## calibration ##########

   zmin, zmax, dz = 0.1, 0.8, 0.02
   Nb, Nm, Nf = 15, 8, 6
   Nhis, Nab, Nchi = 10, 7, 6   
   lmin, nbar = 0.5, 0.001
   z, l, chi = red_sample[:,1], red_sample[:,2], red_sample[:,2]
   calib = calib_sample
   colors = colors
   color_errs = color_errs
   mis = mi

   cal = calibrate(zmin, zmax, dz, Nb, Nm, Nf, Nhis, Nab, Nchi , lmin, nbar, z, l, chi, colors, color_errs, mis, calib)

   cal.solve_chimax_lnlike()
