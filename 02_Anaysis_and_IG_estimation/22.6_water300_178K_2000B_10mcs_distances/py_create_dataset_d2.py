import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import MDAnalysis as mda
from scipy.spatial import cKDTree
import pickle
from tqdm import tqdm
import argparse
import sys
from copy import deepcopy

def cut_after(dists, d0, d_width=0.1):
    return 1. / (1. + np.exp((dists-d0) / d_width))

def cut_before(dists, d0, d_width=0.1):
    return 1. / (1. + np.exp(-(dists-d0) / d_width))

def compute_weights_1stshell(dists, cutoff_1st, d_width):
    return cut_after(dists, d0=cutoff_1st, d_width=d_width)

def compute_weights_2ndshell(dists, cutoff_1st, cutoff_2nd, d_width):
    return (-1. + cut_before(dists, d0=cutoff_1st, d_width=d_width)
               + cut_after(dists, d0=cutoff_2nd, d_width=d_width))

def compute_weights_out_1stshell(dists, cutoff_1st, d_width):
    return cut_before(dists, d0=cutoff_1st, d_width=d_width)

def compute_weights_out_2ndshell(dists, cutoff_2nd, d_width):
    return cut_before(dists, d0=cutoff_2nd, d_width=d_width)

def main():
        #### PARAMETERS TO MODIFY
    Main_Traj_Timestep = 10.0   # ps
    stepjump           = 1 # (na)  Number of step to jumps from main traj to data trajs
    DataTraj_Timestep  = Main_Traj_Timestep * stepjump # 10ps
    Ndatatraj = 2000 # Number of pieces of traj to generate
    Frst_frame  = 1 # 200ns   # Number of inital Frame of the Main Traj to Ingnore for convergeance
    DataTraj_length = 5000 # Size of the pieces to optain 
    DataTraj_timelength = DataTraj_length * DataTraj_Timestep # 50 ns

    # read random seed and number of trajectories (from 1 to 1962)
    parser = argparse.ArgumentParser()
    #parser.add_argument("--traj_number", dest="traj_number", default=None, type=int)
    parser.add_argument("--seed", dest="seed", default=None, type=int)
    parser.add_argument("--WORKD", dest="WORKD", default=".", type=str)

    #parser.add_argument("--central", dest="central", default=None, type=int)
    args = parser.parse_args()
    if args.seed is None:
        sys.exit("Error: seed number must be given by user with --seed")
    seed = args.seed
    WORKD = args.WORKD
    # Read trajectory with MDAnalysis
    u = mda.Universe(f'conf.gro',
                     f'300w_300K_1atm.xtc')

    # Calculate bootstraping parameters
    Nframes = len(u.trajectory)

    Datatraj_Steps = (Nframes - DataTraj_length - Frst_frame) // Ndatatraj
    Datatraj_Size = DataTraj_length
    inital_frames_numbers = np.arange(Ndatatraj) * Datatraj_Steps + Frst_frame

    center_indexs = np.arange(Ndatatraj)
    water = u.select_atoms("type O or type H")
    natoms = water.n_atoms

    print(f"Number of atoms: {natoms}\nNumber of frames: {Nframes}")
    assert natoms % 3 == 0, "System does not contain only water molecules"

    # set arrays
    indices_oxygens = np.arange(0,natoms,3, dtype=int)
    indices_hydrogens = np.column_stack((np.arange(1,natoms,3, dtype=int),
                                         np.arange(2,natoms,3, dtype=int)))
    assert indices_oxygens.shape[0] == indices_hydrogens.shape[0], "Number of O and H atoms does not match"

    # set cutoff distances defining first and second shell (in angstrom)
    cutoff_1st = 3.3
    cutoff_2nd = 5.7
    d_width = 0.1 # this is only to set the width of the switching function
    #print(indices_oxygens)
    #selections = [i for i in range(1,2000)]

    ##### PRINT USEFULL DATA FOR NEXT STEPS #########
    print("datatrajs lengths = ", Datatraj_Size // stepjump , "step ; ", (Datatraj_Size // stepjump) * DataTraj_Timestep, "ps" )
    print("datatrajs timesteps =", DataTraj_Timestep)
    Target_Traj_Length = Datatraj_Size // stepjump

    ##### INITALISE SEED #######

    np.random.seed(seed)
    indices_reference_oxygens = np.random.choice(indices_oxygens, replace=True, size=Ndatatraj)
    frame_numbers = inital_frames_numbers 
    stepsize = stepjump
    for traj_number, central in tqdm(zip(frame_numbers, center_indexs)):
        print(f"0_frame: {traj_number} , Random_central_oxygen_index : {central} ") 
        i_central_oxygen = indices_reference_oxygens[central] # traj_number goes from 1 to 2000
        d_1stshell = np.zeros(Nframes)
        d_2ndshell = np.zeros(Nframes)
        d_out_1stshell = np.zeros(Nframes)
        d_out_2ndshell = np.zeros(Nframes)
        d_central = np.zeros(Nframes)
        d_total = np.zeros(Nframes)
        d_total_corrected = np.zeros(Nframes)
        shell_mol_sum= np.zeros((Nframes,2))
        #print("boostrap length= ", len(u.trajectory[traj_number:traj_number+frame_numbers[0]])*10*0.001, "ps")
        i = 0
        for ts in (u.trajectory[traj_number:traj_number+DataTraj_length])[::stepsize]:
            if (i == 0) : # read only for timstep = traj_number
                inipos = water.positions.copy()
                i = 1
            coordinates = water.positions # updated automatically in the loop
            box = water.dimensions[:3] # 3 unit cell dimensions (orthogonal box)
            #print("box = ", box)
            delta = 0.415

        ######################### COMPUTE ALL DIPOLES #########################
        
            displacment_vector = coordinates[indices_oxygens] - inipos[indices_oxygens]
            displacment_vector -= np.round(displacment_vector / box) * box  # apply pbc
            #print("maxdiff = ", np.max(np.abs(displacment_vector)))
            displacement_d = np.linalg.norm(displacment_vector, axis=1)

            #print("displacement_d.shape = ", displacement_d.shape)

        ######################## COMPUTE SHELL DIPOLES ######################## 

            coord_diff = coordinates[i_central_oxygen,np.newaxis] - coordinates[indices_oxygens] # only oxygens considered
            coord_diff = coord_diff - np.round(coord_diff / box) * box # apply pbc
            dist_iO_from_allO = np.sqrt(np.sum(coord_diff*coord_diff, axis=1))

        # select 45 neighbors to be sure that all waters in second shell are included
            indices_neighbors = np.argpartition(dist_iO_from_allO, np.arange(45))[1:45]
            dist_iO_from_allO = dist_iO_from_allO[indices_neighbors]
            
            d_central[ts.frame] = (displacement_d[int(i_central_oxygen/3)]**2)
            #print(d_central[ts.frame])
        pickle.dump([d_central[traj_number:traj_number+DataTraj_length][::stepsize], 
                     ],
                    open(f"{WORKD}/pickles_d2/d2_cent_seed{seed}_traj{traj_number}.p", "wb"))
        #print("trajectory written") # to nm * e (gromacs)
        pickle.dump([shell_mol_sum[traj_number:traj_number+frame_numbers[0]][::stepsize]], open(f"{WORKD}/pickles_shell_mols/sum_shellsmol_1st_2nd_seed{seed}_traj{traj_number}.p", "wb"))

    return
if __name__ == "__main__":
    main()
