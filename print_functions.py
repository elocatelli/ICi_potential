import numpy as np

import matplotlib.pyplot as plt

#def gen_colormap():
#
#    for f in np.linspace(0,1,100):
#        a=(1-f)/0.2;
#        X=Math.floor(a);
#        Y=Math.floor(255*(a-X));

#        case 0: r=255;g=Y;b=0;break;
#        case 1: r=255-Y;g=255;b=0;break;
#        case 2: r=0;g=255;b=Y;break;
#        case 3: r=0;g=255-Y;b=255;break;
#        case 4: r=Y;g=0;b=255;break;
#        case 5: r=255;g=0;b=255;break;

def print_potential_colormap(data):

    x = (180./np.pi)*np.linspace(0,np.pi,50)
    y = (180./np.pi)*np.linspace(0,2*np.pi,100)

    X,Y = np.meshgrid(y,x)

    z = data[:,2].reshape((x.size,y.size))

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.pcolor(X,Y,z, cmap='rainbow')#
    im2 = ax.contour(X,Y,z, levels=10, colors='black', linestyles='solid')

    fig.colorbar(im, label='V(th,ph)')
    fig.savefig('RES/potential_colormap.png', dpi=300,facecolor='w', edgecolor='w', orientation='portrait', papertype='a4', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1)
    ##plt.show()


def print_lorenzo(_data, Np, same_patch, delta, box, fname):

    f = open(fname, 'w')

    if type(box) is float:
        f.write(".Box: %.5f , %.5f , %.5f\n" % (box,box,box))
    else:
        f.write(".Box: %.5f , %.5f , %.5f\n" % (box[0],box[1],box[2]))

    if same_patch:
        for i in range(_data.shape[0]):
            if i%(Np+1) == 0:
                f.write("%.5f %.5f %.5f @ %.5f C[%s]\n" % (_data[i,0], _data[i,1], _data[i,2], _data[i,3], "192,192,192"))
                f.write("%.5f %.5f %.5f @ %.5f C[%s]\n" % (_data[i,0], _data[i,1], _data[i,2], _data[i,3]+delta/2.,"192,192,192,0.25"))
            else:
                f.write("%.5f %.5f %.5f @ %.5f C[%s]\n" % (_data[i,0], _data[i,1], _data[i,2], _data[i,3], "252,233,3"))
    else:
        for i in range(_data.shape[0]):
            if i%(Np+1) == 0:
                f.write("%.5f %.5f %.5f @ %.5f C[%s]\n" % (_data[i,0], _data[i,1], _data[i,2], _data[i,3], "192,192,192"))
                f.write("%.5f %.5f %.5f @ %.5f C[%s]\n" % (_data[i,0], _data[i,1], _data[i,2], _data[i,3]+delta/2.,"192,192,192,0.25"))
            else:
                f.write("%.5f %.5f %.5f @ %.5f C[%s]\n" % (_data[i,0], _data[i,1], _data[i,2], _data[i,3], "252,233,3"))

    f.close()

#def print_lammpstrj(_data, Np, same_patch, delta, box, fname):


def print_lammps_init(_particle, box, outfile):

    f = open(outfile,"w")

    f.write("LAMMPS data file\n\n")
    f.write("%ld atoms\n" % (myN))
    f.write("%ld bonds\n" % (myNpol*(myLpol-1)) )
    f.write("0 angles\n0 dihedrals\n0 impropers\n\n")

    f.write("%d atom types\n1 bond types\n0 angle types\n0 dihedral types\n0 improper types\n\n" % (ptypes+1))

    f.write("0.0 %.1f xlo xhi\n0.0 %.1f ylo yhi\n0.0 %.1f zlo zhi\n\n" % (mybox, mybox, mybox))

    f.write("Masses\n\n1 1\n" ) #2 %.1f \n3 %.1f\n4 %.1f\n\n" % (massstar0,massstar1,massstar2))
    for i in range(2,ptypes+2):
        f.write("%d %.3f \n" % (i, 1))

    f.write("\n")

    f.write("Atoms\n\n")
    j = 0
    for i in range(myN):
        f.write("%ld 1 1 %.5f %.5f %.5f\n" % (i+1, myx, myy, myz))

    f.write("Velocities\n\n")

    for i in range(myN):
        temp = np.sqrt(T)*np.random.randn(3)
        f.write("%ld %.5f %.5f %.5f\n" % (i+1 , temp[0], temp[1], temp[2] ) )

    f.write("\n")


def snapshot_orientations():
    
    to_orient = myIPC.generate_orientations()

    box = [10.,10.,10.]; 
    startp = np.array([1,1,8]); startpx = np.copy(startp)
    toadd = np.array([0.5,0,0]); 
    toshiftx = np.array([3., 0, 0]); toshifty = np.array([0.,0.,-3.]) 

    particle = np.empty((12*len(to_orient),4))

    w = 0; z = 0; k=0
    for j in range(len(to_orient)):

        for a in [[0,1]]:
    
            particle[w,:3] = startpx-toadd; particle[w,3] = 0.5; w+=1
            for i in range(myIPC.npatch):
                particle[w,:3] = particle[z,:3] + myIPC.ecc[i]*np.asarray(to_orient[j][a[0]][i]); particle[w,3] = myIPC.patch_sigma[i]; w+=1
            z += myIPC.npatch+1
    
            particle[w,:3] = startpx+toadd; particle[w,3] = 0.5; w+=1
            for i in range(myIPC.npatch):
                particle[w,:3] = particle[z,:3] + myIPC.ecc[i]*np.asarray(to_orient[j][a[1]][i]); particle[w,3] = myIPC.patch_sigma[i]; w+=1
            z += myIPC.npatch+1

            startpx = startpx + toshiftx; k+=1
            if k == 4:
               startp = startp + toshifty
               startpx = np.copy(startp)
               k = 0

    print_lorenzo(particle, myIPC.npatch, myIPC.samepatch, 0.2, box, 'test.mgl')


#def print_pymol():
