import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import MDAnalysis as mda
from scipy.spatial import cKDTree
import pickle
from tqdm import tqdm

import argparse
import sys
colors = ["#FF595E","#8AC926","#1982C4","black"]


from crosscorrelation import crosscorrelation_1d

def cut_after(dists, d0, d_width=0.1):
    return 1. / (1. + np.exp((dists-d0) / d_width))

def cut_before(dists, d0, d_width=0.1):
    return 1. / (1. + np.exp(-(dists-d0) / d_width))

def compute_weights_1stshell(dists, cutoff_1st, d_width):
    return cut_after(dists, d0=cutoff_1st, d_width=d_width)

def compute_weights_2ndshell(dists, cutoff_1st, cutoff_2nd, d_width):
    return (-1. + cut_before(dists, d0=cutoff_1st, d_width=d_width)
               + cut_after(dists, d0=cutoff_2nd, d_width=d_width))

def main():

    # read random seed and number of trajectories (from 1 to 1962)

    # read trajectory with MDAnalysis
    u = mda.Universe(f'conf.gro',
                     f'300w_300K_1atm.xtc')
    water = u.select_atoms("type O or type H")
    natoms = water.n_atoms
    Nframes = len(u.trajectory)
    Nseed = 50

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

    corelation_time = 100 # ps
    ingnoreddata = 90 # %
    timesetp = 10 # fs
    stepjump = 10
    Correlation_size =  int(corelation_time * 1000 / timesetp / stepjump)

    indices_reference_oxygens = np.random.permutation(indices_oxygens)[:Nseed]
    dipole_central = np.zeros((Nframes, Nseed ,3))
    dipole_1stshell = np.zeros((Nframes, Nseed ,3))
    dipole_2ndshell = np.zeros((Nframes, Nseed ,3))

    for ts in tqdm(u.trajectory[Nframes*ingnoreddata//100:]):
        coordinates = water.positions # updated automatically in the loop
        box = water.dimensions[:3] # 3 unit cell dimensions (orthogonal box)
        delta = 0.415
            ######################### COMPUTE ALL DIPOLES #########################

        vectors_HO = coordinates[indices_oxygens,np.newaxis] - coordinates[indices_hydrogens]
        vectors_HO[:,0] = vectors_HO[:,0] - np.around(vectors_HO[:,0] / box[np.newaxis,:]) * box[np.newaxis,:]
        vectors_HO[:,1] = vectors_HO[:,1] - np.around(vectors_HO[:,1] / box[np.newaxis,:]) * box[np.newaxis,:]
        dipoles = delta * vectors_HO.sum(axis=1)

        for i, i_central_oxygen in enumerate(indices_reference_oxygens):
            coord_diff = coordinates[i_central_oxygen,np.newaxis] - coordinates[indices_oxygens] # only oxygens considered
            coord_diff = coord_diff - np.around(coord_diff / box) * box # apply pbc
            dist_iO_from_allO = np.sqrt(np.sum(coord_diff*coord_diff, axis=1))

        # select 45 neighbors to be sure that all waters in second shell are included
            indices_neighbors = np.argpartition(dist_iO_from_allO, np.arange(45))[1:45]
            dist_iO_from_allO = dist_iO_from_allO[indices_neighbors]

            weights_1stshell = compute_weights_1stshell(
                dists=dist_iO_from_allO,
                cutoff_1st=cutoff_1st,
                d_width=d_width
            )

            weights_2ndshell = compute_weights_2ndshell(
                dists=dist_iO_from_allO,
                cutoff_1st=cutoff_1st,
                cutoff_2nd=cutoff_2nd,
                d_width=d_width
            )

            dipole_1stshell[ts.frame,i] = (dipoles[indices_neighbors] * weights_1stshell[:,np.newaxis]).sum(axis=0)
            dipole_2ndshell[ts.frame,i] = (dipoles[indices_neighbors] * weights_2ndshell[:,np.newaxis]).sum(axis=0)
            dipole_central[ts.frame,i] = dipoles[int(i_central_oxygen/3)]

    dipole_central_1stshell_crosscor = np.zeros((Correlation_size, Nseed))
    dipole_1stshell_2ndshell_crosscor = np.zeros((Correlation_size, Nseed))
    dipole_central_2ndshell_crosscor = np.zeros((Correlation_size, Nseed))


    print(f"length before reshape : {len(dipole_central[:,0,0])}")

    dipole_central = dipole_central[Nframes*ingnoreddata//100:,:,:]
    dipole_1stshell = dipole_1stshell[Nframes*ingnoreddata//100:,:,:]
    dipole_2ndshell = dipole_2ndshell[Nframes*ingnoreddata//100:,:,:]
    
    pickle.dump(dipole_central, open(f"pickles_dipoles_central.p", "wb"))
    pickle.dump(dipole_1stshell, open(f"pickles_dipoles_1stshell.p", "wb"))
    pickle.dump(dipole_2ndshell, open(f"pickles_dipoles_2ndshell.p", "wb"))

    print(f"length after reshape : {len(dipole_central[:,0,0])}")

    for seed in tqdm(range(Nseed)):
        time, dipole_central_1stshell_crosscor[:,seed] = crosscorrelation_1d( 
                dipole_central[:,seed][::stepjump], 
                dipole_1stshell[:,seed][::stepjump], 
                timestep = timesetp * stepjump, 
                cutoff= Correlation_size)
        time, dipole_1stshell_2ndshell_crosscor[:,seed] = crosscorrelation_1d( 
                dipole_1stshell[:,seed][::stepjump], 
                dipole_2ndshell[:,seed][::stepjump], 
                timestep = timesetp * stepjump, 
                cutoff= Correlation_size)
        time, dipole_central_2ndshell_crosscor[:,seed] = crosscorrelation_1d( 
                dipole_2ndshell[:,seed][::stepjump], 
                dipole_central[:,seed][::stepjump], 
                timestep = timesetp * stepjump, 
                cutoff= Correlation_size)

    dipole_central_1stshell_crosscor_mean = dipole_central_1stshell_crosscor.mean(axis=1)
    dipole_1stshell_2ndshell_crosscor_mean = dipole_1stshell_2ndshell_crosscor.mean(axis=1)
    dipole_central_2ndshell_crosscor_mean = dipole_central_2ndshell_crosscor.mean(axis=1)

    np.savetxt( "dipole_central_1stshell_crosscor_mean_dumb.txt", dipole_central_1stshell_crosscor_mean)
    np.savetxt( "dipole_1stshell_2ndshell_crosscor_mean_dumb.txt", dipole_1stshell_2ndshell_crosscor_mean)
    np.savetxt( "dipole_central_2ndshell_crosscor_mean_dumb.txt", dipole_central_2ndshell_crosscor_mean)

    dipole_central_1stshell_crosscor_std = dipole_central_1stshell_crosscor.std(axis=1)/np.sqrt(50)
    dipole_1stshell_2ndshell_crosscor_std = dipole_1stshell_2ndshell_crosscor.std(axis=1)/np.sqrt(50)
    dipole_central_2ndshell_crosscor_std = dipole_central_2ndshell_crosscor.std(axis=1)/np.sqrt(50)

    time_and_dipole_central_1stshell_crosscor_mean = np.array([time, dipole_central_1stshell_crosscor_mean, dipole_central_1stshell_crosscor_std]).T
    time_and_dipole_1stshell_2ndshell_crosscor_mean = np.array([time, dipole_1stshell_2ndshell_crosscor_mean, dipole_1stshell_2ndshell_crosscor_std]).T
    time_and_dipole_central_2ndshell_crosscor_mean = np.array([time, dipole_central_2ndshell_crosscor_mean, dipole_central_2ndshell_crosscor_std]).T

    np.savetxt( "dipole_central_1stshell_crosscor_mean_v3.txt", time_and_dipole_central_1stshell_crosscor_mean)
    np.savetxt( "dipole_1stshell_2ndshell_crosscor_mean_v3.txt", time_and_dipole_1stshell_2ndshell_crosscor_mean)
    np.savetxt( "dipole_central_2ndshell_crosscor_mean_v3.txt", time_and_dipole_central_2ndshell_crosscor_mean)
#    fig, ax = plt.subplots()

    return

if __name__ == "__main__":
    main()
   
