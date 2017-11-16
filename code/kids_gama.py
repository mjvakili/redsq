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

def match_index(args):
    s , e = args[0] , args[1]
    holder = []
    for index in range(s,e):
        index_prime = np.where(like['ID'] == match['ID'][index])[0]
	#print index_prime
        if len(index_prime) == 0:
	   zaan = -1.
	else: 
	   zaan = like['zphot_ANNz2'][index_prime[0]]
	holder.append(zaan)   
    
    return holder

def plot_zphot_zspec(Nthreads):

    from multiprocessing import Pool

    match = filter('GAMA-MATCHED')
    n_samples = match['ID'].shape[0]

    pool = Pool(Nthreads)
    mapfn = pool.map
    Nchunk = np.ceil(1. / Nthreads * n_samples).astype(np.int)
    arglist = [None] * Nthreads
    for i in range(Nthreads):
	s = int(i * Nchunk)
	e = int(s + Nchunk)
	if i == Nthreads - 1 : e = 203024
	print s , e
	arglist[i] = (s, e)  
    result = list(mapfn(match_index, [ars for ars in arglist]))
    result = np.concatenate(result)
    #print result.flatten()
    np.savetxt("zphot_matched.txt" , result.flatten())
    pool.close()
    pool.terminate()
    pool.join()

    return None

def zphot_zspec():


    zphot = np.loadtxt("zphot_matched.txt")
    zspec = match['Z'][:-1]

    zspec = zspec[zphot != -1.]
    zphot = zphot[zphot != -1.]

    print zphot.max()
    print zspec.max()

    plt.figure(figsize = (10 , 10))
    plt.scatter(zphot , zspec , s = 0.0001)
    plt.xlabel("photo-z")
    plt.ylabel("spec-z")
    plt.xlim([0,1.])
    plt.ylim([0,1.])
   
    plt.savefig("/home/vakili/public_html/files/z.png")
    plt.close() 


    plt.figure(figsize = (10 , 10))
    plt.scatter(zphot , zspec , c = "r" , s = 0.1)
    plt.xlabel("photo-z")
    plt.ylabel("spec-z")
    plt.xlim([0,4.])
    plt.ylim([0,4.])
   
    plt.savefig("/home/vakili/public_html/files/z_extended_range.png")
    plt.close() 



    plt.figure(figsize = (10 , 10))
    plt.scatter(zphot , (zphot-zspec)/(1. + zspec) , s = 0.0001)
    plt.xlabel("photo-z")
    plt.ylabel(r"$\delta z / (1+z)$")
    plt.xlim([0,1.])
    plt.ylim([-0.1,0.1])
   
    plt.savefig("/home/vakili/public_html/files/dz.png")

    return None 

def visual():

    templatetypesnb = (1, 2, 5) # nb of ellipticals, spirals, and starburst used in the 8-template library.

    ellipticals = ['El_B2004a.sed'][0:templatetypesnb[0]]
    spirals = ['Sbc_B2004a.sed','Scd_B2004a.sed'][0:templatetypesnb[1]]
    irregulars = ['Im_B2004a.sed','SB3_B2004a.sed','SB2_B2004a.sed',
               'ssp_25Myr_z008.sed','ssp_5Myr_z008.sed'][0:templatetypesnb[2]]
    template_names = [nm.replace('.sed','') for nm in ellipticals+spirals+irregulars]



    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
    for i, template_name in enumerate(template_names):
        data = np.genfromtxt('./seds/'+template_name+'.sed')
        wavelength, template_sed = data[:,0], data[:,1] * data[:,0]**2
	fnorm = np.interp(7e3, wavelength, template_sed)
	axs[0].plot(wavelength, (template_sed / fnorm), label=template_names[i], 
		                c=next(palette), lw=2)
        axs[0].legend(loc='lower right', ncol=2)
        axs[0].set_ylabel(r'$L_\nu(\lambda)$')
	axs[0].set_yscale('log')
        axs[0].set_ylim([1e-3, 1e2])

    ab_filters = ['u', 'g', 'r', 'i']
    palette = itertools.cycle(sns.cubehelix_palette(5, light=0.6))
    filters = [np.genfromtxt('filters/'+band+'.dat') for band in ab_filters]
    for f, data in zip(ab_filters, filters):
      axs[1].plot(10. * data[:,0], data[:,1], label="KiDS "+f, c=next(palette), lw=3)
      axs[1].set_xlim([1e3, 1.1e4])
      axs[1].set_ylim([0, 0.99])
      axs[1].set_xlabel(r'$\lambda$  [Angstrom]')
      axs[1].set_ylabel(r'$R(\lambda)$')
      axs[1].legend(loc='upper center', ncol=5)
    
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/filters.png")

    return None 

def gals_in_groups():


    gals = pf.open("groups/G3CGalv07.fits")[1].data   #galaxy group catalog
    #gal = gal[gal['GroupID']!=0]              #keeping galaxies that are in groups  
    
    match = filter('GAMA-MATCHED') 

    mask = np.in1d(match['CATAID'] , gals[gals['GroupID']!=0]['CATAID'])

    zphot = np.loadtxt("zphot_matched.txt")

    match = match[mask]
    zphot = zphot[mask]

    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
    palette = itertools.cycle(sns.cubehelix_palette(1, light=0.6))
    axs.scatter(zphot , match['Z'] , color = next(palette) , s = 0.001)

    axs.set_xlim([0,1.])
    axs.set_ylim([0,1.])

    axs.set_xlabel(r'$zphot$')
    axs.set_ylabel(r'$zspec$')

    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/photoz_galaxies_ingroups.png")
    plt.close()


    fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
    palette = itertools.cycle(sns.cubehelix_palette(1, light=0.6))
    axs.scatter(zphot , (zphot-match['Z'])/(1. + match['Z']), color = next(palette), s = 0.001)
    axs.set_xlabel("photo-z")
    axs.set_ylabel(r"$\delta z / (1+z)$")
    axs.set_xlim([0,1.])
    axs.set_ylim([-0.1,0.1])
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/dz_galaxies_ingroups.png")

    return None


def BPZ_in_groups():


    gals = pf.open("groups/G3CGalv07.fits")[1].data   #galaxy group catalog
    match = filter('GAMA-MATCHED') 
    mask = np.in1d(match['CATAID'] , gals[gals['GroupID']!=0]['CATAID'])
    match_group = match[mask]


    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
    palette = itertools.cycle(sns.cubehelix_palette(1, light=0.6))
    axs.scatter(match['Z_B_BPZ'] , match['Z'] , color = next(palette) , s = 0.1)
    axs.scatter(match_group['Z_B_BPZ'] , match_group['Z'] , color = next(palette) , s = 1.0)
    axs.set_xlim([0,1.])
    axs.set_ylim([0,1.])
    axs.set_xlabel(r'$Z_B$')
    axs.set_ylabel(r'$zspec$')
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/BPZ_performance.png")
    plt.close()

    return None


def plot_color_redshift():


    gals = pf.open("groups/G3CGalv07.fits")[1].data   #galaxy group catalog
    gals = gals[gals['GroupID']!=0]              #keeping galaxies that are in groups  
    match = filter('GAMA-MATCHED') 
    
    mask = np.in1d(match['CATAID'] , gals[gals['GroupID']!=0]['CATAID'])

    matched_gals = match[mask]

    ug = matched_gals['COLOR_GAAPHOM_U_G'] 
    gr = matched_gals['COLOR_GAAPHOM_G_R'] 
    ri = matched_gals['COLOR_GAAPHOM_R_I']
    colors = np.array([ug , gr , ri])
    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    z = matched_gals['Z']
    
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True)
    
    for i in range(3):
	axs[i].scatter(z, colors[i,:], color = next(palette), s = 0.001)
        axs[i].legend(loc='lower right', ncol=2)
        axs[i].set_ylabel(labels[i])
	#axs[0].set_yscale('log')
        axs[i].set_xlim([0.0, 1])
    axs[0].set_ylim([-.1, 4.1])
    axs[1].set_ylim([-0.1, 2.1])
    axs[2].set_ylim([-0.1, 1.1])
    axs[2].set_xlabel(r'$z$')
    
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/colors_in_groups.png")
    plt.close()


    ug = match['COLOR_GAAPHOM_U_G'] 
    gr = match['COLOR_GAAPHOM_G_R'] 
    ri = match['COLOR_GAAPHOM_R_I']
    colors = np.array([ug , gr , ri])
    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    z = match['Z']
    
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True)
    
    for i in range(3):
	axs[i].scatter(z, colors[i,:], color = next(palette), s = 0.001)
        axs[i].legend(loc='lower right', ncol=2)
        axs[i].set_ylabel(labels[i])
	#axs[0].set_yscale('log')
        axs[i].set_xlim([0.0, 1])
    axs[0].set_ylim([-.1, 4.1])
    axs[1].set_ylim([-0.1, 2.1])
    axs[2].set_ylim([-0.1, 1.1])
    axs[2].set_xlabel(r'$z$')
    
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/colors_in_all.png")
    plt.close()

    return None 
    

def plot_color_magnitude(zmin , zmax):


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

    mi = mi[(zmin<z)&(z<zmax)]
    redshift_mask = np.where((zmin<z)&(z<zmax))[0]
    colors = colors[:, redshift_mask]

    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True)
    
    for i in range(3):
	axs[i].scatter(mi, colors[i,:], color = next(palette), s = 0.1)
        axs[i].legend(loc='lower right', ncol=2)
        axs[i].set_ylabel(labels[i])
	#axs[0].set_yscale('log')
        axs[i].set_xlim([15.5, 22.5])
    axs[0].set_ylim([-0.1, 4.1])
    axs[1].set_ylim([-0.1, 2.1])
    axs[2].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel(r'$m_{i}$')
    
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/colors_magnitude_groups_"+str(zmin)+"_"+str(zmax)+".png")
    plt.close()


    ug = match['COLOR_GAAPHOM_U_G'] 
    gr = match['COLOR_GAAPHOM_G_R'] 
    ri = match['COLOR_GAAPHOM_R_I']
    z = match['Z']
    mi = match['MAG_GAAP_i_CALIB']
    colors = np.array([ug , gr , ri])

    mi = mi[(zmin<z)&(z<zmax)]
    redshift_mask = np.where((zmin<z)&(z<zmax))[0]
    colors = colors[:, redshift_mask]

    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 1, figsize=(5.5, 6), sharex=True)
    
    for i in range(3):
	axs[i].scatter(mi, colors[i,:], color = next(palette), s = 0.1)
        axs[i].legend(loc='lower right', ncol=2)
        axs[i].set_ylabel(labels[i])
	#axs[0].set_yscale('log')
        axs[i].set_xlim([15.5, 22.5])
    axs[0].set_ylim([-0.1, 4.1])
    axs[1].set_ylim([-0.1, 2.1])
    axs[2].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel(r'$m_{i}$')
    
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/colors_magnitude_all_"+str(zmin)+"_"+str(zmax)+".png")

    return None 

def redshift_distribution():

    """
    gal_type = "all" or "grp"
    """
    gals = pf.open("groups/G3CGalv07.fits")[1].data   #galaxy group catalog
    #gals = gals[gals['GroupID']!=0]              #keeping galaxies that are in groups  
    match = filter('GAMA-MATCHED')


    mask = np.in1d(match['CATAID'] , gals[gals['GroupID']!=0]['CATAID'])
    matched_gals = match[mask]

    z_grp = matched_gals['Z']
    z_all = match['Z']

    figure = plt.figure(figsize = (10,10))
    sns.distplot(z_grp , kde = False , hist = True , norm_hist = True, label = "group galaxies")
    sns.distplot(z_all , kde = False , hist = True , norm_hist = True, label = "all galaxies")
    plt.legend(loc = 'best')
    plt.xlim([0.0 , 0.8])
    plt.xlabel(r"$z$" , fontsize = 10)
    plt.savefig("/home/vakili/public_html/files/redshift_distributions.png")
    plt.close()


    mi_grp = matched_gals['MAG_GAAP_i_CALIB']
    mi_all = match['MAG_GAAP_i_CALIB']

    figure = plt.figure(figsize = (10,10))
    sns.distplot(mi_grp , kde = False , hist = True , norm_hist = True, label = "group galaxies")
    sns.distplot(mi_all , kde = False , hist = True , norm_hist = True, label = "all galaxies")
    plt.legend(loc = 'best')
    plt.xlim([14 , 23])
    plt.xlabel(r"$m_{\rm i}$" , fontsize = 10)
    plt.savefig("/home/vakili/public_html/files/magnitude_distributions.png")
    plt.close()


def reference_magnitude():

    
    match = filter('GAMA-MATCHED')                          #filtered matched kidsxgama catalog
    groups = pf.open("groups/G3CFoFGroupv06.fits")[1].data  #groups
    groups = groups[groups['Nfof'] > 5]                      #keeping groups woth Nfof>2
    
    mask = np.in1d(match['CATAID'] , groups['BCGCATAID'])   #finding BCGS in the filtered matched kidsxgama catalog
    matched_gals = match[mask]

    print matched_gals.shape

    mi_bcg = matched_gals['MAG_GAAP_i_CALIB']
    g_r = matched_gals['COLOR_GAAPHOM_G_R']
    z_bcg = matched_gals['Z']

    total_bins = 20    # number of redshift bins
    bins = np.linspace(0.1,0.4, total_bins) #this divides groups to 20 redshift bins between 0.05 and 0.45
    delta = bins[1]-bins[0]
    idx  = np.digitize(z_bcg,bins)
    median = [np.median(mi_bcg[idx==k]) for k in range(total_bins)]
    print median
    print len(median)

    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    axs[0].scatter(z_bcg , mi_bcg , s = 0.2 , color = "black")
    axs[0].plot(bins - delta/2 , median , 'r-' , lw = 2.0)
    axs[0].set_xlim([0 , 1])
    axs[0].set_ylim([14 , 23])
    axs[0].set_ylabel(r"$m_{\rm i}$" , fontsize = 10)
    
    axs[1].scatter(z_bcg , g_r , s = 0.2 , color = "black")
    axs[1].set_xlim([0 , 1])
    axs[1].set_ylim([0 , 2])
    axs[1].set_ylabel(r"$g-r$" , fontsize = 10)
    axs[1].set_xlabel(r"$z$" , fontsize = 10)
    
    plt.savefig("/home/vakili/public_html/files/BCG_mi_z.png")
    plt.close()

    return bins - delta/2  , median

def test():


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

    mi = mi[(0.25<z)&(z<0.26)]
    redshift_mask = np.where((0.25<z)&(z<0.26))[0]
    colors = colors[:, redshift_mask]

    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
    
    for i in range(2):
	axs[i].scatter(mi, colors[i+1,:], color = next(palette), s = 0.1)
        axs[i].legend(loc='lower right', ncol=2)
        axs[i].set_ylabel(labels[i+1])
	#axs[0].set_yscale('log')
        axs[i].set_xlim([16.5, 21.5])
    axs[0].set_ylim([-0.1, 2.1])
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel(r'$m_{i}$')
    
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/test_groups.png")
    plt.close()


    ug = match['COLOR_GAAPHOM_U_G'] 
    gr = match['COLOR_GAAPHOM_G_R'] 
    ri = match['COLOR_GAAPHOM_R_I']
    z = match['Z']
    mi = match['MAG_GAAP_i_CALIB']
    colors = np.array([ug , gr , ri])

    mi = mi[(0.25<z)&(z<0.26)]
    redshift_mask = np.where((0.25<z)&(z<0.26))[0]
    colors = colors[:, redshift_mask]

    labels = [r'$u-g$',r'$g-r$',r'$r-i$']
    # Load and plot the templates and filters
    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True)
    
    for i in range(2):
	axs[i].scatter(mi, colors[i+1,:], color = next(palette), s = 0.1)
        axs[i].legend(loc='lower right', ncol=2)
        axs[i].set_ylabel(labels[i+1])
	#axs[0].set_yscale('log')
        axs[i].set_xlim([16.5, 21.5])
    axs[0].set_ylim([-0.1, 2.1])
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel(r'$m_{i}$')
    
    fig.tight_layout()
    plt.savefig("/home/vakili/public_html/files/test_all.png")

    return None 


if __name__ == '__main__':

   #reference_magnitude(0.25, 0.3)
   #redshift_distribution()
   #test()
   #BPZ_in_groups()
   plot_color_magnitude(0.25 , 0.3)
   #plot_color_redshift()
   #gals_in_groups()

   ##match = filter('GAMA-MATCHED')
   ##like = filter('GAMA-LIKE')

   ##plot_zphot_zspec(20)
   ##zphot_zspec()
   ##visual()
