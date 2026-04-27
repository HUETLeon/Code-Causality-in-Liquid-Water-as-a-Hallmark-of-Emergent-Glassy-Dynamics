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

    corelation_time = 10000000 # ps
    ingnoreddata = 0 # %
    timestep = 10000 # fs
    stepjump = 100
    Correlation_size =  int(corelation_time * 1000 / timestep / stepjump)

    print("Correlation_size: ", Correlation_size)

    dipole_ref_cor = np.zeros((Correlation_size, Nseed, 3))
    dipole_1stshell_cor = np.zeros((Correlation_size, Nseed, 3))
    dipole_2ndshell_cor = np.zeros((Correlation_size, Nseed, 3))
    dipole_out_ref_cor = np.zeros((Correlation_size, Nseed, 3))
    dipole_out_1stshell_cor = np.zeros((Correlation_size, Nseed, 3))
    dipole_out_2ndshell_cor = np.zeros((Correlation_size, Nseed, 3))

    dipole_ref_1stshell_crosscor =  np.zeros((Correlation_size, Nseed, 3))
    dipole_ref_2ndshell_crosscor =  np.zeros((Correlation_size, Nseed, 3))
    dipole_1stshell_2ndshell_crosscor = np.zeros((Correlation_size, Nseed, 3))
    dipole_ref_out_ref_crosscor = np.zeros((Correlation_size, Nseed, 3))
    dipole_1stshell_out_1stshell_crosscor = np.zeros((Correlation_size, Nseed, 3))
    dipole_2ndshell_out_2ndshell_crosscor = np.zeros((Correlation_size, Nseed, 3))

    
    dipole_ref =  pickle.load(open(f"pickles_dipoles_ref_oneseed.p", "rb"))
    dipole_1stshell = pickle.load(open(f"pickles_dipoles_1stshell_oneseed.p", "rb"))
    dipole_2ndshell = pickle.load(open(f"pickles_dipoles_2ndshell_oneseed.p", "rb"))
    dipole_out_ref = pickle.load(open(f"pickles_dipoles_out_ref_oneseed.p", "rb"))
    dipole_out_1stshell = pickle.load(open(f"pickles_dipoles_out_1stshell_oneseed.p", "rb"))
    dipole_out_2ndshell = pickle.load(open(f"pickles_dipoles_out_2ndshell_oneseed.p", "rb"))


    print(f"shape dipole_ref : {dipole_ref.shape}")
    print(f"shape dipole_ref_cor : {dipole_ref_cor.shape}")

    for seed in tqdm(range(Nseed)):
        ##corr
        (dipole_ref_cor[:,seed,0], 
        dipole_ref_cor[:,seed,1], 
        dipole_ref_cor[:,seed,2]) = autocorrelation_1d( dipole_ref[:,seed][::stepjump], 
                                                        timestep = timestep * stepjump, 
                                                        cutoff= Correlation_size
                                                        )
        (dipole_1stshell_cor[:,seed,0], 
        dipole_1stshell_cor[:,seed,1], 
        dipole_1stshell_cor[:,seed,2]) = autocorrelation_1d( dipole_1stshell[:,seed][::stepjump], 
                                                            timestep = timestep * stepjump, 
                                                            cutoff= Correlation_size
                                                            )
        (dipole_2ndshell_cor[:,seed,0], 
        dipole_2ndshell_cor[:,seed,1], 
        dipole_2ndshell_cor[:,seed,2]) = autocorrelation_1d( dipole_2ndshell[:,seed][::stepjump], 
                                                            timestep = timestep * stepjump, 
                                                            cutoff= Correlation_size
                                                            )
        (dipole_out_ref_cor[:,seed,0], 
        dipole_out_ref_cor[:,seed,1], 
        dipole_out_ref_cor[:,seed,2]) = autocorrelation_1d( dipole_out_ref[:,seed][::stepjump], 
                                                            timestep = timestep * stepjump, 
                                                            cutoff= Correlation_size
                                                            )
        (dipole_out_1stshell_cor[:,seed,0], 
        dipole_out_1stshell_cor[:,seed,1], 
        dipole_out_1stshell_cor[:,seed,2]) = autocorrelation_1d( dipole_out_1stshell[:,seed][::stepjump], 
                                                                timestep = timestep * stepjump, 
                                                                cutoff= Correlation_size
                                                                )
        (dipole_out_2ndshell_cor[:,seed,0], 
        dipole_out_2ndshell_cor[:,seed,1], 
        dipole_out_2ndshell_cor[:,seed,2]) = autocorrelation_1d( dipole_out_2ndshell[:,seed][::stepjump], 
                                                                timestep = timestep * stepjump, 
                                                                cutoff= Correlation_size
                                                                )
        ##crosscorr
        (dipole_ref_1stshell_crosscor[:,seed,0], 
        dipole_ref_1stshell_crosscor[:,seed,1], 
        dipole_ref_1stshell_crosscor[:,seed,2]) = crosscorrelation_1d( dipole_ref[:,seed][::stepjump], 
                                                                    dipole_1stshell[:,seed][::stepjump], 
                                                                    timestep = timestep * stepjump, 
                                                                    cutoff= Correlation_size
                                                                    )
        (dipole_1stshell_2ndshell_crosscor[:,seed,0], 
        dipole_1stshell_2ndshell_crosscor[:,seed,1], 
        dipole_1stshell_2ndshell_crosscor[:,seed,2]) = crosscorrelation_1d( dipole_1stshell[:,seed][::stepjump], 
                                                                        dipole_2ndshell[:,seed][::stepjump], 
                                                                        timestep = timestep * stepjump, 
                                                                        cutoff= Correlation_size
                                                                        )
        (dipole_ref_2ndshell_crosscor[:,seed,0], 
        dipole_ref_2ndshell_crosscor[:,seed,1], 
        dipole_ref_2ndshell_crosscor[:,seed,2]) = crosscorrelation_1d( dipole_ref[:,seed][::stepjump], 
                                                                    dipole_2ndshell[:,seed][::stepjump], 
                                                                    timestep = timestep * stepjump, 
                                                                    cutoff= Correlation_size
                                                                    )
        (dipole_ref_out_ref_crosscor[:,seed,0], 
        dipole_ref_out_ref_crosscor[:,seed,1], 
        dipole_ref_out_ref_crosscor[:,seed,2]) = crosscorrelation_1d( dipole_ref[:,seed][::stepjump], 
                                                                    dipole_out_ref[:,seed][::stepjump], 
                                                                    timestep = timestep * stepjump, 
                                                                    cutoff= Correlation_size
                                                                    )
        (dipole_1stshell_out_1stshell_crosscor[:,seed,0],
        dipole_1stshell_out_1stshell_crosscor[:,seed,1], 
        dipole_1stshell_out_1stshell_crosscor[:,seed,2]) = crosscorrelation_1d( dipole_out_1stshell[:,seed][::stepjump], 
                                                                            dipole_1stshell[:,seed][::stepjump], 
                                                                            timestep = timestep * stepjump, 
                                                                            cutoff= Correlation_size
                                                                            )
        (dipole_2ndshell_out_2ndshell_crosscor[:,seed,0], 
        dipole_2ndshell_out_2ndshell_crosscor[:,seed,1], 
        dipole_2ndshell_out_2ndshell_crosscor[:,seed,2]) = crosscorrelation_1d( dipole_out_2ndshell[:,seed][::stepjump], 
                                                                                dipole_2ndshell[:,seed][::stepjump], 
                                                                                timestep = timestep * stepjump, 
                                                                                cutoff= Correlation_size
                                                                                )

    dipole_ref_cor_mean = dipole_ref_cor.mean(axis=1)
    dipole_1stshell_cor_mean = dipole_1stshell_cor.mean(axis=1)
    dipole_2ndshell_cor_mean = dipole_2ndshell_cor.mean(axis=1)
    dipole_out_ref_cor_mean = dipole_out_ref_cor.mean(axis=1)
    dipole_out_1stshell_cor_mean = dipole_out_1stshell_cor.mean(axis=1)
    dipole_out_2ndshell_cor_mean = dipole_out_2ndshell_cor.mean(axis=1)

    
    dipole_ref_1stshell_crosscor_mean = dipole_ref_1stshell_crosscor.mean(axis=1)
    dipole_1stshell_2ndshell_crosscor_mean = dipole_1stshell_2ndshell_crosscor.mean(axis=1)
    dipole_ref_2ndshell_crosscor_mean = dipole_ref_2ndshell_crosscor.mean(axis=1)
    dipole_ref_out_ref_crosscor_mean = dipole_ref_out_ref_crosscor.mean(axis=1)
    dipole_1stshell_out_1stshell_crosscor_mean = dipole_1stshell_out_1stshell_crosscor.mean(axis=1)
    dipole_2ndshell_out_2ndshell_crosscor_mean = dipole_2ndshell_out_2ndshell_crosscor.mean(axis=1)

    np.savetxt( "dipole_ref_cor_mean_v3.txt", dipole_ref_cor_mean)
    np.savetxt( "dipole_1st_cor_mean_v3.txt", dipole_1stshell_cor_mean)
    np.savetxt( "dipole_2nd_cor_mean_v3.txt", dipole_2ndshell_cor_mean)
    np.savetxt( "dipole_outref_cor_mean_v3.txt", dipole_out_ref_cor_mean)
    np.savetxt( "dipole_out1st_cor_mean_v3.txt", dipole_out_1stshell_cor_mean)
    np.savetxt( "dipole_out2nd_cor_mean_v3.txt", dipole_out_2ndshell_cor_mean)

    np.savetxt( "dipole_ref_1st_crosscor_mean_v3.txt", dipole_ref_1stshell_crosscor_mean)
    np.savetxt( "dipole_1st_2nd_crosscor_mean_v3.txt", dipole_1stshell_2ndshell_crosscor_mean)
    np.savetxt( "dipole_ref_2nd_crosscor_mean_v3.txt", dipole_ref_2ndshell_crosscor_mean)
    np.savetxt( "dipole_ref_outref_crosscor_mean_v3.txt", dipole_ref_out_ref_crosscor_mean)
    np.savetxt( "dipole_1st_out1st_crosscor_mean_v3.txt", dipole_1stshell_out_1stshell_crosscor_mean)
    np.savetxt( "dipole_2nd_out2nd_crosscor_mean_v3.txt", dipole_2ndshell_out_2ndshell_crosscor_mean)


#    fig, ax = plt.subplots()

    return

if __name__ == "__main__":
    main()
   
