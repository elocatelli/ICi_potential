# ICi_potential
Coarse graining potential generator for ICi particles

## Usage

The code is to be intended as a library of functions; the script "ICi_potential.py" provides an example of how they can be used.

To use ICi_potential.py, one should type in the terminal

python3 ICi_potential.py PATH_TO_FOLDER PATH_TO_PARAMETER_FILE PATH_TO_TOPOLOGY_FILE

where the 3 parameters are all optional, that is, if not given the default values will be used (if valid).

PATH_TO_FOLDER is the path to the folder where the results will be saved (defaults to current folder)

PATH_TO_PARAMETER_FILE is the path to the mandatory file that contains the model parameters (see below - defaults to 'params.dat' in the current folder)

PATH_TO_TOPOLOGY_FILE is the path to the optional file that contains the topology of the ICi particle (truly optional - defaults to the polar topology)

For an example of the parameter file, see "params_template.dat"; for an example of the topology file, see "topology_template.dat".

### Description of the parameters 
(for the physical meaning of some of them, see the paper "Anisotropic DLVO-like interaction for charge patchiness in colloids and proteins" by Gnidovec et al.):

nr_patches : number of off-centre charges

same_patch : the off-centre charges have the same value or not (boolean)

patch_charge : the value of the off-centre charge (if the two are not equal provide two values, separated by a comma)

colloid_charge : the value of the central charge

screening_constant : the value of the quantity kR

eccentricity : the parameter a (in units of the diameter 2R!)

lmax = 30 : the maximum number of modes in the expansion of the electric potential

real_size = 3E-7  : the target size of the experimental system one has in mind

pathway_distance = 1. : centre-centre distance at which the pathway is computed

energy_units = False  :  whether some quantities are printed in real energy units (eV) or in units of kB T

## Relevant attributes:

the code contains a Python class, ICi, with several attributes.

the most useful for the end user are:
print_potential_surface('2patch', nph=100): computes and print in a text file the electric potential at contact as a function of the polar angle (from the one pole to the other - the first pole corresponds to the first off-centre charge, in case the two charges are not equal). The file name is hard-coded, but can be changed inside the function (function in file: ICi_sp_potentials.py)

do_effective_potential('2patch', '2patch', Np=1): computes and prints in a text file the pair energy at contact in the three main orientations (PP, EE, EP). For Np>1, a curve with the radial dependence of the pair energy in these three configurations will be produced. The file name is hard-coded, but can be changed inside the function (function in file: ICi_effective_potentials.py) 

pathway(myICi.ICidict['folder']+'/test_pathway_ipc.dat'): computes and prints in a text file the pair energy along the rotational pathway (see the paper mentioned above for a description of the pathway); the distance is fixed by the parameter "pathway_distance". The function accepts a string, that is the path of the file where the pathway will be saved. In the example, it will be saved the file "test_pathway_ipc.dat" in the same folder specified by command line. The number of sample points is hard coded in the paramter "path_N", that can be found in the file "ICi_definitions.py"

Other attributes can be found (commented) in the file "ICi_potential.py"
