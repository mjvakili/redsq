import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.io.ascii as asciitable


#table = asciitable.read("test_phot.dat_clusters_0.86_1_phot.dat")
m1 = np.loadtxt("zf3.0_z0.02")
m2 = np.loadtxt("zf2.0_z0.001")


plt.figure(figsize = (10, 10))
#ax = plt.gca()


plt.plot(m1[:,0] , m1[:,2] , color = "r" , lw = 2.0 , label = r"$z_{f}=3, Z= 0.2 $" )
plt.plot(m2[:,0] , m2[:,2] , color = "b" , lw = 2.0 , label = r"$z_{f}=2, Z= 0.002 $")
plt.xlim([0.0, 1.0])
plt.xlabel(r"$z$" , fontsize = 10)
plt.ylabel(r"$m_{i}$" , fontsize = 10)
plt.legend(loc = 'best')
#plt.axis('scaled')
plt.savefig("/home/vakili/public_html/files/unnormalized_apparent_magnitudes.png")
