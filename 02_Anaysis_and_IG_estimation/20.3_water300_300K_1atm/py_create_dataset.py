import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import MDAnalysis as mda
from scipy.spatial import cKDTree
import pickle
from tqdm import tqdm
import argparse
import sys

def cut_after(dists, d0, d_width=0.1):
    return 1. / (1. + np.exp((dists-d0) / d_width))

def cut_before(dists, d0, d_width=0.1):
    return 1. / (1. + np.exp(-(dists-d0) / d_width))

def compute_weights_1stshell(dists, cutoff_1st, d_width):
    return cut_after(dists, d0=cutoff_1st, d_width=d_width)

def compute_weights_2ndshell(dists, cutoff_1st, cutoff_2nd, d_width):
    return (-1. + cut_before(dists, d0=cutoff_1st, d_width=d_width)
               + cut_after(dists, d0=cutoff_2nd, d_width=d_width))

def cut_cos_front(coss, cos0 = 0, cos_width=0.1):
    return 1. / (1. + np.exp(-(coss - cos0) / cos_width))

def compute_weights_1stshell_front(dists, coss, cutoff_1st, d_width, cos_width):
    return (cut_after(dists, d0=cutoff_1st, d_width=d_width) 
            * cut_cos_front(coss, cos_width=cos_width)
            )

def compute_weights_1stshell_back(dists, coss, cutoff_1st, d_width, cos_width):
    return ( cut_after(dists, d0=cutoff_1st, d_width=d_width) 
            * (1. - cut_cos_front(coss, cos_width=cos_width))
            )

def compute_weights_2ndshell_front(dists, coss, cutoff_1st, cutoff_2nd, d_width, cos_width):
    return  (compute_weights_2ndshell(dists, cutoff_1st, cutoff_2nd, d_width)
            * cut_cos_front(coss, cos_width=cos_width)
            )

def compute_weights_2ndshell_back(dists, coss, cutoff_1st, cutoff_2nd, d_width, cos_width):
    return  (compute_weights_2ndshell(dists, cutoff_1st, cutoff_2nd, d_width)
            * (1. - cut_cos_front(coss, cos_width=cos_width))
            )

def main():
    stepsize = 1 

    # read random seed and number of trajectories (from 1 to 1962)
    parser = argparse.ArgumentParser()
    #parser.add_argument("--traj_number", dest="traj_number", default=None, type=int)
    parser.add_argument("--seed", dest="seed", default=None, type=int)
    #parser.add_argument("--central", dest="central", default=None, type=int)
    args = parser.parse_args()
    if args.seed is None:
        sys.exit("Error: seed number must be given by user with --seed")
    seed = args.seed
    # read trajectory with MDAnalysis
    u = mda.Universe(f'conf.gro', 
                     f'300w_300K_1atm.xtc')
    frame_numbers = np.loadtxt("uniform_starts.dat", dtype = int)
    center_indexs = np.loadtxt("central_oxygens.dat", dtype = int)
    water = u.select_atoms("type O or type H")
    natoms = water.n_atoms
    Nframes = len(u.trajectory)

    print(f"Number of atoms: {natoms}\nNumber of frames: {Nframes}")
    assert natoms % 3 == 0, "System does not contain only water molecules"

    # set arrays
    indices_oxygens = np.arange(0,natoms,3, dtype=int)
    indices_hydrogens = np.column_stack((np.arange(1,natoms,3, dtype=int),
                                         np.arange(2,natoms,3, dtype=int)))
    asserterror = "Number of O and H atoms does not match"
    assert indices_oxygens.shape[0] == indices_hydrogens.shape[0], asserterror

    # set cutoff distances defining first and second shell (in angstrom)
    cutoff_1st = 3.3
    cutoff_2nd = 5.7
    d_width = 0.1 # this is only to set the width of the switching function
    cos_width = 0.1 # this is to set the width of the cos switching function
    #print(indices_oxygens)
    #selections = [i for i in range(1,2000)]œ
    for traj_number, central in zip(frame_numbers, center_indexs):
        print(f"0_frame: {traj_number} , Random_central_oxygen_index : {central} ") 
        np.random.seed(seed)
        indices_reference_oxygens = np.random.choice(indices_oxygens, replace=True, size=2000)
        i_central_oxygen = indices_reference_oxygens[central] # traj_number goes from 1 to 2000
        
        density_1stshell = np.zeros((Nframes,))
        density_2ndshell = np.zeros((Nframes,))

        print("boostrap length= ", len(u.trajectory[traj_number:traj_number+frame_numbers[0]])*10*0.001, "ps")

        for ts in tqdm((u.trajectory[traj_number:traj_number+frame_numbers[0]])[::stepsize]):

            coordinates = water.positions # updated automatically in the loop
            box = water.dimensions[:3] # 3 unit cell dimensions (orthogonal box)
            delta = 0.415

        ######################## COMPUTE SHELLS ######################## 
            
            coord_diff = coordinates[i_central_oxygen,np.newaxis] - coordinates[indices_oxygens] # only oxygens considered
            coord_diff = coord_diff - np.around(coord_diff / box) * box # apply pbc
            dist_iO_from_allO = np.sqrt(np.sum(coord_diff*coord_diff, axis=1))
            

        # select 45 neighbors to be sure that all waters in second shell are included
            indices_neighbors = np.argpartition(dist_iO_from_allO, np.arange(45))[1:45]
            dist_iO_from_allO = dist_iO_from_allO[indices_neighbors]
            
            weights_1stshell = compute_weights_1stshell(
                dists = dist_iO_from_allO,
                cutoff_1st = cutoff_1st, 
                d_width = d_width,
            )

            weights_2ndshell = compute_weights_2ndshell(
                dists=dist_iO_from_allO,
                cutoff_1st=cutoff_1st,
                cutoff_2nd=cutoff_2nd,
                d_width=d_width,
            )

            density_1stshell[ts.frame] = weights_1stshell.sum()
            density_2ndshell[ts.frame] = weights_2ndshell.sum()
        #print(0.1*dipole_central[ts.frame])
        #print(dipole_1stshell.shape,dipole_2ndshell.shape)
        pickle.dump([density_1stshell[traj_number:traj_number+frame_numbers[0]][::stepsize], density_2ndshell[traj_number:traj_number+frame_numbers[0]][::stepsize] ], 
                    open(f"./pickles_local-density/density_cent_1st_2nd_seed{seed}_traj{traj_number}.p", "wb"))      # to nm * e (gromacs)

    return
if __name__ == "__main__":
    main()
