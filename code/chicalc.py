
import numpy as np
import h5py
from scipy.interpolate import CubicSpline
import ezgal
import cosmolopy.distance as cd
import util
import emcee

def dvdz(z):
    '''
    dv/dz to impose uniformity
    in redshift
    '''
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72}
    d_a = cd.angular_diameter_distance(z, **cosmo) #angular diameter distance
    h = cd.e_z(z, **cosmo) #hubble parameter
    dvdz = (1+z)**2. * d_a **2  * h **-1. #dv/dz
    return dvdz

def luminosity(m,z):
    '''
    L/Lstar of redsequence galaxies
    '''
    mchar = kids_mstar(z)
    dm = m - mchar
    exparg = 10. ** (-0.4 * dm)

    return exparg
    
class calibrate(object):

    def __init__(self, zmin, zmax, dz, Nb, Nm, Nf, Nhis, Nab, Nchi):
       
        # the inferred c-m relation parameters
        self.theta = theta

        # ID, color, mag of all gals
        self.ID = ID
        self.colors = colors
        self.color_errs = color_errs
        self.mis = mis
        
        # ID and z of calib gals
        self.ID_calib = ID_calib
        self.z_calib = z_calib
       
        #CROSS MATCH BETWEEN ALL and calib gals

        self.mask_calib = np.where(np.in1d(self.ID_calib , self.ID)==True)

        # initial estimates of chi, l, z
        self.chi_init = chi_init
        self.l_init = l_init
        self.z_init = z_init

        # sum numbers for making an initial cut
        self.chi_abs = chi_abs
        self.zmin_abs = zmin_abs
        self.zmax_abs = zmax_abs
        self. l_abs = l_abs


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
        znods = znods[znods<0.7]
        mrefs = mrefs[znods<0.7]
	#######################################################################
        self.xref = CubicSpline(znods, mrefs)(self.xrefnod)
        

        #######################

        self.dz_theta_0 = dz_theta_0
        self.chimax_theta_0 = chimax_theta_0 

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

    def chi3d(self,z):
        """
	this recalculates chi3d for every galaxy in the survey for which we have a zred 
	regardless of l and chi
        fast calculation of all the chis for a set of zs 
        z = the current estimate of reshifts
        """
        bz = CubicSpline(self.bnod , self.b)(z)
        mz = CubicSpline(self.mnod , self.m)(z)
        lnfz = CubicSpline(self.fnod , self.lnf)(z)
        xrefz = CubicSpline(self.xrefnod , self.xref)(z)

        Cint = np.zeros((len(self.z), 3, 3))
        Cint[:,0,0] = np.exp(2.* lnfz[:,0])
        Cint[:,1,1] = np.exp(2.* lnfz[:,1])
        Cint[:,2,2] = np.exp(2.* lnfz[:,2])
        Ctot = self.color_errs + Cint
        res = mz * (self.mis - xrefz)[:,None] + bz - self.colors
        chis = np.einsum('mn,mn->n', res, np.einsum('ijk,ik->ij',np.linalg.inv(Ctot),res))
    
        return chis
    
    def l3d(self, z):
        """
	this recalculates l3d for every galaxy in the survey for which we have a zred
        regardless of its chi and l
	fast calculation of all the luminosity ratios for a set of zs 
        z = the current estimate of reshifts
        """
        ls = luminosity(self.mis, z)
                 
        return ls
    
    def ab_lnlike(self, dz_theta, mask):
        """
        dz_theta = estimate of dz at ab spline nods
        mask = some mask on chi and l of spec red gals
        """
        x , y = self.calib_sample[mask,1], self.calib_sample[mask,1]-self.calib_sample[mask,-1]
        dz_pred = CubicSpline(self.abnod , dz_theta)(x)
        chisq = np.sum(np.abs(y - dz_pred))
    
        return chisq
     
    def solve_ab_lnlike(self, mask):
        """
	solves ab_lnlike
	dz_theta_0 = initial guess for dz_theta
	mask = some mask on chi and l of spec red gals
        """
        nll = lambda *args: self.ab_lnlike(*args)
	result = op.minimize(nll, self.dz_theta_zero*np.ones((self.Nab)), args=(mask))
	
	return result["x"]
    
    def chimax_lnlike(self, chimax_theta):
        """
        temp estimate of chimax_theta at the loc of chi nods
	"""

	#first have to run the ab
        #alt chinods , norm =  np.exp(chimax_theta[:-1]), np.exp(chimax_theta[-1]) 
        chinods =  np.exp(chimax_theta)
	chi_maxes = CubicSpline(nods ,chinods)(self.calib[:,1])
	mask = self.calib[:,2] < chimax_calib
        dz_ab =  solve_ab_lnlike(self, mask)
	print "CURRENT ESTIMATE OF AB = " , dz_a

	# calibrate the calib-zs and rezs
	self.calib[:,1] = self.calib[:,1] - CubicSpline(self.abnod, dz_ab)(self.z) #calib gals
	self.z = self.z - CubicSpline(self.abnod , dz_ab)(self.z) #all gals
        #calibrate the red-chis and red-ls
        self.chi = self.chi3d(z)
	self.l = self.l3d(z)
        #mask the red-ls that are larger than self.lmin (=0.5 or 1)
	mask = self.l > self.lmin
      
    	#chinods , norm = np.exp(chimax_theta[:-1]), np.exp(chimax_theta[-1])  
    	chinods = np.exp(chimax_theta)
    	
	chi_maxes = CubicSpline(nods ,chinods)(self.z[mask])
    	sample_z = self.z[self.chi < chi_maxes]
    	hist , edges = np.histogram(sample_z, bins = self.Nhist)
    	bins = .5*(edges[1:]+edges[:-1])
    	dbin = edges[1] - edges[0]
    	dvdbin = dvdz(bins)
    	dvbin = dvdbin * dbin
    	chisq = np.sum((hist - self.norm*dvbin)**2./(hist+ self.norm*dvbin)) 
        
	return chisq
 

    def solve_chimax_lnlike(self):
        """
	solves chi_max_lnlike
        """

    	chinods_0 = 2 + np.zeros((self.Nhist))

    	nll = lambda *args: chimax_lnlike(*args)
    
    	if prior_tmp == 'forced':
       	   bnds = []
       	   for i in range(len(nods)): 
               bnds.append((np.log(1.),np.log(4.0)))
       	       norm_bound = np.log(target_nbar / (processed.shape[0]* 360.3 / (41252.96 * 34290660)))
       			print norm_bound
       			bnds.append((-1.7, -1.6))
    	if prior_tmp == 'free':
       	bnds = []
       	for i in range(len(nods)): 
           bnds.append((np.log(1.),np.log(3)))
       	norm_bound = np.log(target_nbar / (processed.shape[0]* 360.3 / (41252.96 * 34290660)))
       	print norm_bound
       	#bnds.append((norm_bound - 1.0, norm_bound+1))
       	#bnds.append((norm_bound - 10, norm_bound+10.))
       	bnds.append((-10., -1.))
    	result = op.minimize(nll, np.append(np.log(chinods_0), np.log(norm_0)), args=(sample2, nods), method='SLSQP', bounds = bnds, options={'disp': True ,'eps' : .001})
    	print "number density" , np.exp(result["x"])[-1] * processed.shape[0]* 360.3 / (41253 * 34290660)
    	print "processed fraction" , processed.shape[0] / 34290660.0
    	print "result" , np.exp(result["x"])
    	density = np.exp(result["x"])[-1] * processed.shape[0]* 360.3 / (41253 * 34290660)  

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

   print (ID[mask_one][arg_one] == spec_ID[mask_two][arg_two]) #sanity check!

   spec_red_sample = red_sample[mask_one][arg_one]
   mask_chil = (spec_red_sample[:,0]==1)
   spec_specz = spec_z[mask_two][arg_two]
   
   calib_sample = np.vstack([spec_red_sample.T , spec_specz]).T
   
   print calib_sample #sanity check
   

   ########## INITIALIZATION OF THE EZGAL MODEL ########

   model = ezgal.model("/net/delft/data2/vakili/easy/ezgal_models/www.baryons.org/ezgal/models/bc03_burst_0.1_z_0.02_salp.model")
   zf = 3.0 #HARDCODED
   model.add_filter("/net/delft/data2/vakili/easy/i.dat" , "kids" , units = "nm")
   #kcorr_sloan = model.get_kcorrects(zf=3.0 , zs = 0.25 , filters = "sloan_i")
   model.set_normalization("sloan_i" , 0.2 , 17.85, vega=False, apparent=True)
   cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':1.0}
