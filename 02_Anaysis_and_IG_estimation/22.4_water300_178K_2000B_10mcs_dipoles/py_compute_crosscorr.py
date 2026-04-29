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

    corelation_time = 10000000 # ps
    ingnoreddata = 0 # %
    timestep = 10000 # fs
    stepjump = 100
    Correlation_size =  int(corelation_time * 1000 / timestep / stepjump)

    print("Correlation_size: ", Correlation_size)

    dipole_central_1stshell_crosscor = np.zeros((Correlation_size, Nseed))
    dipole_1stshell_2ndshell_crosscor = np.zeros((Correlation_size, Nseed))
    dipole_central_2ndshell_crosscor = np.zeros((Correlation_size, Nseed))

    dipole_central =  pickle.load(open(f"pickles_dipoles_central.p", "rb"))
    dipole_1stshell = pickle.load(open(f"pickles_dipoles_1stshell.p", "rb"))
    dipole_2ndshell = pickle.load(open(f"pickles_dipoles_2ndshell.p", "rb"))
    dipole_out_1stshell = pickle.load(open(f"pickles_dipoles_out_1stshell.p", "rb"))
    dipole_out_2ndshell = pickle.load(open(f"pickles_dipoles_out_2ndshell.p", "rb"))

    for seed in tqdm(range(Nseed)):
        time, dipole_central_1stshell_crosscor[:,seed] = crosscorrelation_1d(
                dipole_central[:,seed][::stepjump],
                dipole_1stshell[:,seed][::stepjump],
                timestep = timestep * stepjump,
                cutoff= Correlation_size)
        time, dipole_1stshell_2ndshell_crosscor[:,seed] = crosscorrelation_1d(
                dipole_1stshell[:,seed][::stepjump],
                dipole_2ndshell[:,seed][::stepjump],
                timestep = timestep * stepjump,
                cutoff= Correlation_size)
        time, dipole_central_2ndshell_crosscor[:,seed] = crosscorrelation_1d(
                dipole_2ndshell[:,seed][::stepjump],
                dipole_central[:,seed][::stepjump],
                timestep = timestep * stepjump,
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


    return

if __name__ == "__main__":
    main()
   
