import numpy as np
import matplotlib.pyplot as plt
import sys

myfile = sys.argv[1]

data = np.loadtxt(myfile, dtype = float)


x = (180./np.pi)*np.linspace(0,np.pi,50)
y = (180./np.pi)*np.linspace(0,2.*np.pi,100)

X,Y = np.meshgrid(y,x)

z = data[:,2].reshape((x.size,y.size))

fig, ax = plt.subplots(figsize=(9, 6))

im = ax.pcolor(X,Y,z, cmap='rainbow')#
im2 = ax.contour(X,Y,z, levels=10, colors='black', linestyles='solid')

fig.colorbar(im, label=r'$\Psi(\theta,\phi)$ [eV]')

#fig.savefig('RES/potential_colormap.png', dpi=300,facecolor='w', edgecolor='w', orientation='portrait', papertype='a4', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()


