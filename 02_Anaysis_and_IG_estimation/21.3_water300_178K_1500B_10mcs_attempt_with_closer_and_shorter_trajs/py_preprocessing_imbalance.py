from threadpoolctl import threadpool_limits
import numpy as np
from joblib import Parallel, delayed
import pickle
from dadapy.metric_comparisons import MetricComparisons
from tqdm import tqdm
import argparse
import sys
import warnings
import matplotlib.pyplot as plt
from dadapy import data
from sklearn.datasets import make_swiss_roll
from imbalance_gain_windows import *

def compute_ids_scaling(X, range_max=2048, n_min=20):
    "instantiate data class"
    _data = data.Data(coordinates=X, maxk=100)
    "compute ids scaling gride"
    ids_gride, ids_err_gride, rs_gride = _data.return_id_scaling_gride(
        range_max=range_max
    )
    "compute ids with twoNN + decimation"
    ids_twoNN, ids_err_twoNN, rs_twoNN = _data.return_id_scaling_2NN(n_min=n_min)
    return ids_gride, ids_twoNN, rs_gride, rs_twoNN


def read_files_dipole(namefile, t, E, tau_e, stds):

    dipoles_center_temp, dipoles_1stshell_temp, dipoles_2ndshell_temp = pickle.load(open(namefile, 'rb'))

    embed_times = np.arange(t, t+E, tau_e)
    dipoles_center_t =   +dipoles_center_temp[embed_times]
    dipoles_1stshell_t = +dipoles_1stshell_temp[embed_times]
    dipoles_2ndshell_t = +dipoles_2ndshell_temp[embed_times]

    dipoles_center_t = ((dipoles_center_t) / (stds[0].max()) / np.sqrt(3))
    dipoles_1stshell_t = ((dipoles_1stshell_t) / (stds[0].max()) / np.sqrt(3))
    dipoles_2ndshell_t = ((dipoles_2ndshell_t) / (stds[0].max()) / np.sqrt(3))

    return np.array([dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t])


def compute_scaling_factors(namefile):

    dipoles_center, dipoles_1stshell, dipoles_2ndshell = pickle.load(open(namefile, 'rb'))
    
    stds = [dipoles_center.std(axis=0, keepdims=True),
            dipoles_1stshell.std(axis=0, keepdims=True),
            dipoles_2ndshell.std(axis=0, keepdims=True)]

    return stds


def construct_Xt_Yt(Ntrajs, seed, t, E, tau_e, njobs, stds, frame_numbers):

    assert len(stds) == 3, "Error: wrong number of stds"
    (dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t) = (
        np.swapaxes(Parallel(n_jobs=njobs)(delayed(read_files_dipole)
            (namefile=f'./pickles_dipoles/dipoles_cent_1st_2nd_seed{seed}_traj{(traj_number)}.p', 
                t=t, E=E, tau_e=tau_e, stds=stds) for traj_number in frame_numbers), axis1=0, axis2=1)
    )
    dipoles_center_t =  dipoles_center_t.reshape((Ntrajs,-1))
    dipoles_1stshell_t = dipoles_1stshell_t.reshape((Ntrajs,-1))
    dipoles_2ndshell_t = dipoles_2ndshell_t.reshape((Ntrajs,-1))

    return dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t


def main():

    # read random seed and time tau
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", dest="seed", default=None, type=int)
    parser.add_argument("--E", dest="E", default=None, type=int)
    args = parser.parse_args()
    if args.seed is None:
        sys.exit("Error: random seed must be given by user with --seed")
    if args.E is None:
        warnings.warn("Warning: Embending dimension must be specified, assuming E=1")
        args.E = 1

    
    SwRrlls, t =  make_swiss_roll(10000)
    # read external parameters from dataset creation output
    frame_numbers = np.int64(np.loadtxt("Datatrajs_frame_numbers.txt"))
    data_traj_timesteps = np.loadtxt("Datatrajs_Steps_in_ps.txt", )

    Nframes = len(data_traj_timesteps)
    Ntrajs = len(frame_numbers) 
    
    # set parameters to compute imbalances
    tau_e = 1
    E = args.E
    t0 = 1 ## OUTDATED ## assertion discard first 5 ps, could contain artifacts (nvt -> nve)
    k = 2
    njobs = 56
    eval_jump = 10
    
    
    tend = Nframes - t0 - E*tau_e
    taus = np.arange(0,tend, eval_jump) # evaluation times, timestep = 10ps)
    beta = np.linspace(0.,1.,200,endpoint=False)
    alphas = beta/(1-beta)

    # compute standardization parameters on reference trajectory
    stds = compute_scaling_factors(f'./pickles_dipoles/dipoles_cent_1st_2nd_seed1_traj{frame_numbers[0]}.p')

    

    # construct X0 and Y0
    dipoles_center_t0, dipoles_1stshell_t0, dipoles_2ndshell_t0 = (
        construct_Xt_Yt(
            Ntrajs = Ntrajs, 
            seed = args.seed, 
            t = t0, 
            E = E, 
            tau_e = tau_e, 
            njobs = njobs, 
            stds = stds, 
            frame_numbers = frame_numbers
            ) 
        )

    print("\nBegin ID estimation\n")

    _data = data.Data(dipoles_center_t0)
    results = _data.compute_id_2NN()
    with open(f"IDS/dipoles_center_t0_id_E{E}.txt", 'a') as file:
        file.write(f"{results[0]}\t{results[1]}\t{results[2]}\n")

    _data = data.Data(dipoles_1stshell_t0)
    results = _data.compute_id_2NN()
    with open(f"IDS/dipoles_1stshell_t0_id_E{E}.txt", 'a') as file:
        file.write(f"{results[0]}\t{results[1]}\t{results[2]}\n")

    _data = data.Data(dipoles_2ndshell_t0)
    results = _data.compute_id_2NN()
    with open(f"IDS/dipoles_2ndshell_t0_id_E{E}.txt", 'a') as file:
        file.write(f"{results[0]}\t{results[1]}\t{results[2]}\n")
    
    print("\nBegin rank mapping\n")
    
    for i in range(1,10+1):
        discard_close_ind =i
        nn_indices_center = return_nn_indices(dipoles_center_t0, discard_close_ind=discard_close_ind)
        nn_indices_1st = return_nn_indices(dipoles_1stshell_t0, discard_close_ind=discard_close_ind)
        nn_indices_2nd = return_nn_indices(dipoles_2ndshell_t0, discard_close_ind=discard_close_ind)

        fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(12,3))
        indices = np.arange(len(nn_indices_center))
        ax1.plot(indices, nn_indices_center, '.')
        ax2.plot(indices, nn_indices_1st, '.')
        ax3.plot(indices, nn_indices_2nd, '.')
        ax1.set(xlabel="Point index", ylabel="nn index")
        ax2.set(xlabel="Point index")
        ax3.set(xlabel="Point index")
        plt.savefig(f"RANKS/Rank_map_E{E}_Disc{discard_close_ind}_seed.jpg")

        plt.close()

    return

if __name__ == "__main__":
    main()
