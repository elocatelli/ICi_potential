import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib.pylab import *
font = matplotlib.font_manager.FontProperties(size=28)
plt.rcParams['text.usetex'] = True

myfile = sys.argv[1]
outfile = sys.argv[2]

data = np.loadtxt(myfile, dtype = float)

x = (180./np.pi)*np.linspace(0,np.pi,50)
y = (180./np.pi)*np.linspace(0,2.*np.pi,100)

X,Y = np.meshgrid(y,x)

z = data[:,2].reshape((x.size,y.size))

fig, ax = plt.subplots(figsize=(9, 6))

im = ax.pcolor(X,Y,z, cmap='rainbow', shading='auto')#
im2 = ax.contour(X,Y,z, levels=10, colors='black', linestyles='solid')

cb = fig.colorbar(im, format='%.0e')
cb.ax.set_ylabel(r'$e \Psi({\theta},\phi)$ [eV]')
text = cb.ax.yaxis.label
text.set_font_properties(font)
cb.ax.tick_params(axis='y', which='major', labelsize=20)


ax.set_xlabel('$\phi$')
text = ax.xaxis.label
text.set_font_properties(font)
 
ax.set_ylabel(r'${\theta}$')
text = ax.yaxis.label
text.set_font_properties(font)

plt.xticks(np.arange(0, 361, 90))
plt.yticks(np.arange(0, 181, 45))
ax.tick_params(axis='both', which='major', labelsize=20)

fig.savefig(outfile, dpi=300,facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1)
#plt.show()


