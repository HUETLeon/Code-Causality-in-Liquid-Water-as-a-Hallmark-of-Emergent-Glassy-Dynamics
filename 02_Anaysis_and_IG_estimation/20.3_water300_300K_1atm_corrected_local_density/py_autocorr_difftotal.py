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
    Nseed = 1

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
    ingnoreddata = 00 # %
    timesetp = 10 # fs
    stepjump = 10
    Correlation_size =  int(corelation_time * 1000 / timesetp / stepjump)

    indices_reference_oxygens = np.random.permutation(indices_oxygens)[:Nseed]
    dipole_ref = np.zeros((Nframes, Nseed ,3))
    dipole_1stshell = np.zeros((Nframes, Nseed ,3))
    dipole_2ndshell = np.zeros((Nframes, Nseed ,3))
    dipole_out_1stshell = np.zeros((Nframes, Nseed ,3))
    dipole_out_2ndshell = np.zeros((Nframes, Nseed ,3))
    dipole_out_ref = np.zeros((Nframes, Nseed ,3))

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
            
            dipole_1stshell[ts.frame] = (dipoles * weights_1stshell[:,np.newaxis]).sum(axis=0)
            dipole_2ndshell[ts.frame] = (dipoles * weights_2ndshell[:,np.newaxis]).sum(axis=0)
            dipole_ref[ts.frame] = dipoles[int(i_central_oxygen/3)]

            dipole_out_1stshell[ts.frame,i] = (dipoles * weights_out_1stshell[:,np.newaxis]).sum(axis=0)
            dipole_out_2ndshell[ts.frame,i] = (dipoles * weights_out_2ndshell[:,np.newaxis]).sum(axis=0)
            dipole_out_ref[ts.frame,i] = dipoles.sum(axis=0) - dipoles[i_central_oxygen//3,:]

    dipole_1stshell = dipole_1stshell[Nframes*ingnoreddata//100:,:,:]
    dipole_2ndshell = dipole_2ndshell[Nframes*ingnoreddata//100:,:,:]
    dipole_ref = dipole_ref[Nframes*ingnoreddata//100:,:,:]

    dipole_out_1stshell = dipole_out_1stshell[Nframes*ingnoreddata//100:,:,:]
    dipole_out_2ndshell = dipole_out_2ndshell[Nframes*ingnoreddata//100:,:,:]
    dipole_out_ref = dipole_out_ref[Nframes*ingnoreddata//100:,:,:]

    
    pickle.dump(dipole_ref, open(f"pickles_dipoles_ref_oneseed.p", "wb"))
    pickle.dump(dipole_1stshell, open(f"pickles_dipoles_1stshell_oneseed.p", "wb"))
    pickle.dump(dipole_2ndshell, open(f"pickles_dipoles_2ndshell_oneseed.p", "wb"))

    pickle.dump(dipole_out_1stshell, open(f"pickles_dipoles_out_1stshell_oneseed.p", "wb"))
    pickle.dump(dipole_out_2ndshell, open(f"pickles_dipoles_out_2ndshell_oneseed.p", "wb"))
    pickle.dump(dipole_out_ref, open(f"pickles_dipoles_out_central_oneseed.p", "wb"))
    
#    fig, ax = plt.subplots()

    return

if __name__ == "__main__":
    main()
   
