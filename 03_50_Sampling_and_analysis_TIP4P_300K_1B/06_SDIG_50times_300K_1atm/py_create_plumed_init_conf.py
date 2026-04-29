import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import MDAnalysis as mda
from scipy.spatial import cKDTree
import pickle
from tqdm import tqdm
import argparse
import sys


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
    assert indices_oxygens.shape[0] == indices_hydrogens.shape[0], "Number of O and H atoms does not match"

    #selections = [i for i in range(1,2000)]
    np.random.seed(seed)
    at = u.atoms
    with mda.Writer('../PLUMED_INIT_CONF/frames_seed{seed}.xtc', at.n_atoms, apprend=TRUE) as w:
        frames = u.trajectory
        for ts in [frames[i] for i in frame_numbers] :
            w.write(at)


    return
if __name__ == "__main__":
    main()
