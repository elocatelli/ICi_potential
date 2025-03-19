# ICi_potential
Coarse graining potential generator for ICi particles

## Usage

The code is to be intended as a library of functions; the script "ICi_potential.py" provides an example of how they can be used.

To use ICi_potential.py, one should type in the terminal

python3 ICi_potential.py PATH_TO_FOLDER PATH_TO_PARAMETER_FILE PATH_TO_TOPOLOGY_FILE

where the 3 parameters are all optional, that is, if not given the default values will be used (if valid).

PATH_TO_FOLDER is the path to the folder where the results will be saved (defaults to current folder). Note: this sets one of the attributes of the python class "ICi" (see below). Barring changes to the code, all files produced by the script will be saved in this folder

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

lmax : the maximum number of modes in the expansion of the electric potential

real_size : the target size of the experimental system one has in mind

pathway_distance : centre-centre distance at which the pathway is computed

energy_units :  whether some quantities are printed in real energy units (eV) or in units of kB T (boolean)

## Relevant attributes:

the code contains a Python class, ICi, with several methods.

To create an instance of the class, simply type 

myICI = ICi()

The most useful methods for the end user are:

set_params(): sets all the attributes of the class ICi, reading the parameter file

set_charge_topology(): sets and check the arrangement of the off-centre charges 

check_params(): prints the parameters on the file "parameters_check.dat" for a first check

print_potential_surface(filename, Np=100): computes and saves in a text file the electric potential at contact as a function of the polar angle (from the one pole to the other - the first pole corresponds to the first off-centre charge, in case the two charges are not equal).

do_effective_potential(filename, Np=100): computes and saves in a text file the pair energy at contact in the three main orientations (PP, EE, EP). For Np>1, the radial dependence of the pair energy (between 1 and 3 units of length) in these three configurations will be computed; setting Np=1 will only calculate and save only the contact value. 

pathway(filename): computes and saves in text file the pair energy along the rotational pathway (see the paper mentioned above for a description of the pathway); the distance is fixed by the parameter "pathway_distance". The number of sample points is hard coded in the paramter "path_N", that can be found in the file "ICi_definitions.py" 

logfile_mf(): prints a few important information on the system. This method is called by default with "do_effective_potential"

compute_potential_zero(): estimates where the single particle potential crosses zero, which , as reported in the reference paper, can be identified with the patch size. If the single particle potential has already been computed, the zero is computed as long as the number of stored points is larger than 10 (e.g. when calling the method "print_potential_surface",  Np> 10). The estimated value is stored as the attribute "sp_zero". This method is called by default with "logfile_mf" 

When relevant, 'filename' is a string, that will set the file name where the data will be saved.  

A few other methods can be found (commented) in the file "ICi_potential.py".


## Brief description of each file:

ICi_potential : example of a script for computing single particle/pair potentials and other quantities

ICi_definition : all the attributes are defined here

ICi_orientations : a collection of all the relevant orientations (e.g. PP, EE, EP)

ICi_particle : sets particle-based properties (such as the off-charge arrangement)

ICi_sp_potential : methods for computing and storing/saving the single particle effective potential

ICi_effective_potential : methods for computing and storing/saving the pair effective potential energy

ICi_pathway : methods for constructing the rotational pathway

utils : miscellaneous useful functions

