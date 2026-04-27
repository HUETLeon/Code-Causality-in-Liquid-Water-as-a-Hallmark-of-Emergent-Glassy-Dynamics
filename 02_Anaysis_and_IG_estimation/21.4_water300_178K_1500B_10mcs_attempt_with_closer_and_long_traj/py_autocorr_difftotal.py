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


from autocorrelation import autocorrelation_1d

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

    corelation_time = 100000000 # ps
    ingnoreddata = 0 # %
    timesetp = 10000 # fs
    stepjump = 100
    Correlation_size =  int(corelation_time * 1000 / timesetp / stepjump)
    
    print("Correlation_size: ", Correlation_size)

    indices_reference_oxygens = np.random.permutation(indices_oxygens)[:Nseed]
    dipole_out_1stshell = np.zeros((Nframes, Nseed ,3))
    dipole_out_2ndshell = np.zeros((Nframes, Nseed ,3))

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

            weights_out_1stshell = compute_weights_out_1stshell(
                dists=dist_iO_from_allO,
                cutoff_1st=cutoff_1st,
                d_width=d_width
            )

            weights_out_2ndshell = compute_weights_out_2ndshell(
                dists=dist_iO_from_allO,
                cutoff_2nd=cutoff_2nd,
                d_width=d_width
            )

            dipole_out_1stshell[ts.frame,i] = (dipoles * weights_out_1stshell[:,np.newaxis]).sum(axis=0)
            dipole_out_2ndshell[ts.frame,i] = (dipoles * weights_out_2ndshell[:,np.newaxis]).sum(axis=0)

    dipole_out_1stshell_cor = np.zeros((Correlation_size, Nseed))
    dipole_out_2ndshell_cor = np.zeros((Correlation_size, Nseed))
    
    dipole_out_1stshell = dipole_out_1stshell[Nframes*ingnoreddata//100:,:,:]
    dipole_out_2ndshell = dipole_out_2ndshell[Nframes*ingnoreddata//100:,:,:]

    pickle.dump(dipole_out_1stshell, open("./pickles_dipoles_out_1stshell.p", 'wb'))
    pickle.dump(dipole_out_2ndshell, open("./pickles_dipoles_out_2ndshell.p", 'wb'))

    print("shapes before autocorelation: ", dipole_out_1stshell_cor.shape, dipole_out_1stshell.shape)

    for seed in tqdm(range(Nseed)):
        time, dipole_out_1stshell_cor[:,seed] = autocorrelation_1d( dipole_out_1stshell[:,seed][::stepjump], timestep = timesetp * stepjump, cutoff= Correlation_size)
        time, dipole_out_2ndshell_cor[:,seed] = autocorrelation_1d( dipole_out_2ndshell[:,seed][::stepjump], timestep = timesetp * stepjump, cutoff= Correlation_size)

    dipole_out_1stshell_cor_mean = dipole_out_1stshell_cor.mean(axis=1)
    dipole_out_2ndshell_cor_mean = dipole_out_2ndshell_cor.mean(axis=1)

    np.savetxt( "dipole_out_1stshell_cor_mean_dumb.txt", dipole_out_1stshell_cor_mean)
    np.savetxt( "dipole_out_2ndshell_cor_mean_dumb.txt", dipole_out_2ndshell_cor_mean)

    dipole_out_1stshell_cor_std = dipole_out_1stshell_cor.std(axis=1)/np.sqrt(Nseed)
    dipole_out_2ndshell_cor_std = dipole_out_2ndshell_cor.std(axis=1)/np.sqrt(Nseed)

    time_and_dipole_out_1stshell_cor_mean = np.array([time, dipole_out_1stshell_cor_mean, dipole_out_1stshell_cor_std]).T
    time_and_dipole_out_2ndshell_cor_mean = np.array([time, dipole_out_2ndshell_cor_mean, dipole_out_2ndshell_cor_std]).T

    np.savetxt( "dipole_out_1stshell_cor_mean_v3.txt", time_and_dipole_out_1stshell_cor_mean)
    np.savetxt( "dipole_out_2ndshell_cor_mean_v3.txt", time_and_dipole_out_2ndshell_cor_mean)
#    fig, ax = plt.subplots()

    return

if __name__ == "__main__":
    main()
   
