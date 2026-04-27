from threadpoolctl import threadpool_limits
import numpy as np
from joblib import Parallel, delayed
import pickle
from dadapy.metric_comparisons import MetricComparisons
from tqdm import tqdm
import argparse
import sys

def read_files_dipole(namefile):

    d_central, d_1stshell, d_2ndshell = pickle.load(open(namefile, 'rb'))

    return np.array([d_central, 
                        d_1stshell, 
                        d_2ndshell, 
                        ])

def construct_X_Y(Ntrajs, seed, traj_frames, njobs):

    (d_central, d_1stshell, d_2ndshell ) = (
            np.swapaxes(Parallel(n_jobs=njobs)(delayed(read_files_dipole)
            (namefile=f'./pickles_d/d_cent_1st_2nd_seed{seed}_traj{traj_frame}.p') for traj_frame in traj_frames), axis1=0, axis2=1)
    )
    d_central = d_central.reshape((Ntrajs,-1))
    d_1stshell = d_1stshell.reshape((Ntrajs,-1))
    d_2ndshell = d_2ndshell.reshape((Ntrajs,-1))
    print(d_central)
    return d_central, d_1stshell, d_2ndshell


def main():

    # set parameters to compute imbalances
    Nframes = 5000
    Ntrajs = 2000
    njobs = 56
    nseed = 50
    mean_mean = np.zeros((Nframes,3))
    std_mean = np.zeros((Nframes,3))
    mean = np.zeros((Nframes,nseed,3))
    traj_frames=np.loadtxt("Datatrajs_frame_numbers.txt", dtype='int')
    print(traj_frames)
    for seed in tqdm(range(1,51)):
        (d_central, d_1stshell, d_2ndshell 
            )= (
                construct_X_Y(Ntrajs=Ntrajs, seed=seed, traj_frames=traj_frames, njobs=njobs))
        print("seed: " , seed)
        mean[:,seed-1,0] = d_central.mean(axis=0)
        mean[:,seed-1,1] = d_1stshell.mean(axis=0)
        mean[:,seed-1,2] = d_2ndshell.mean(axis=0)
   
    print(mean)
    mean_mean = mean.mean(axis=1)
    std_mean = mean.std(axis=1)
    np.savetxt("pickles_mean/central_mean_and_std.txt", np.array([mean_mean[:,0], std_mean[:,0]]).T)
    np.savetxt("pickles_mean/1stshell_mean_and_std.txt", np.array([mean_mean[:,1], std_mean[:,1]]).T)
    np.savetxt("pickles_mean/2ndshell_mean_and_std.txt", np.array([mean_mean[:,2], std_mean[:,2]]).T)
    A = np.random.randint(low = 0, high = Ntrajs, size = 3) 
    
    np.savetxt("pickles_mean/central_exemple.txt",  d_central[A,:] )
    np.savetxt("pickles_mean/1stshell_exemple.txt", d_1stshell[A,:] )
    np.savetxt("pickles_mean/2ndshell_exemple.txt", d_2ndshell[A,:] )
    print( d_central[A,:])
    return

if __name__ == "__main__":
    main()
