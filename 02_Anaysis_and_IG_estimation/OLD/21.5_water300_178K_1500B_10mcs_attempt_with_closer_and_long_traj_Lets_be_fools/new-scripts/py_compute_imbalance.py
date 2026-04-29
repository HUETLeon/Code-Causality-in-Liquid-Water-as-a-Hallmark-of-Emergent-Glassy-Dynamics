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
    parser.add_argument("--dci", dest="dci", default=0, type=int)
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
    discard_close_ind = args.dci
    k = 80
    njobs = 56
    eval_jump = 50
    
    
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

    print("\nBegin IB estimation\n")

    

    for tau in tqdm(taus):
            
        # construct Xtau and Ytau
        dipoles_center_tau, dipoles_1stshell_tau, dipoles_2ndshell_tau = (
            construct_Xt_Yt(
                Ntrajs = Ntrajs, 
                seed = args.seed, 
                t = t0+tau, 
                E = E, 
                tau_e = tau_e, 
                njobs = njobs, 
                stds = stds,
                frame_numbers = frame_numbers
                ) 
        )

        d = MetricComparisons(maxk=Ntrajs-1, n_jobs=njobs)

        # center <-> 1st shell
        info_imbalances_center_to_1st = scan_alphas(
            cause_present=dipoles_center_t0, 
            effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, 
            alphas=alphas, 
            discard_close_ind=discard_close_ind,
            k=k,
            n_jobs=njobs)
        info_imbalances_1st_to_center = scan_alphas(
            cause_present=dipoles_1stshell_t0, 
            effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, 
            alphas=alphas, 
            discard_close_ind=discard_close_ind,
            k=k,
            n_jobs=njobs)

        # center <-> 2nd shell
        info_imbalances_center_to_2nd = scan_alphas(
            cause_present=dipoles_center_t0, 
            effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, 
            alphas=alphas,
            discard_close_ind=discard_close_ind,
            k=k,
            n_jobs=njobs)
        info_imbalances_2nd_to_center = scan_alphas(
            cause_present=dipoles_2ndshell_t0, 
            effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, 
            alphas=alphas, 
            discard_close_ind=discard_close_ind,
            k=k,
            n_jobs=njobs)

        # 1st shell <-> 2nd shell
        info_imbalances_1st_to_2nd = scan_alphas(
            cause_present=dipoles_1stshell_t0, 
            effect_present=dipoles_2ndshell_t0, 
            effect_future=dipoles_2ndshell_tau, 
            alphas=alphas, 
            discard_close_ind=discard_close_ind,
            k=k,
            n_jobs=njobs)
        info_imbalances_2nd_to_1st = scan_alphas(
            cause_present=dipoles_2ndshell_t0, 
            effect_present=dipoles_1stshell_t0, 
            effect_future=dipoles_1stshell_tau, 
            alphas=alphas, 
            discard_close_ind=discard_close_ind,
            k=k,
            n_jobs=njobs)

        # save pickle
        pickle.dump([info_imbalances_center_to_1st, 
            info_imbalances_1st_to_center, 
            info_imbalances_center_to_2nd, 
            info_imbalances_2nd_to_center, 
            info_imbalances_1st_to_2nd, 
            info_imbalances_2nd_to_1st,
            ],
            open(f"./pickles_imb_E{E}_dci{discard_close_ind}/imb_seed{args.seed}_tau{tau}_k{k}.p", "wb"))

    return

if __name__ == "__main__":
    main()
