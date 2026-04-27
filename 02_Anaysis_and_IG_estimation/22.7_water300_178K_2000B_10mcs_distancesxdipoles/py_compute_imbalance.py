from threadpoolctl import threadpool_limits
import numpy as np
from joblib import Parallel, delayed
import pickle
from dadapy.metric_comparisons import MetricComparisons
from tqdm import tqdm
import argparse
import sys

def read_files_d(namefile, t, E, tau_e, means, stds):

    d_center_temp, d_1stshell_temp, d_2ndshell_temp = pickle.load(open(namefile, 'rb'))

    #print("input shapes: ", d_center_temp.shape, d_total_temp.shape, d_totalc_temp.shape)

    embed_times = np.arange(t, t+E, tau_e)
    d_center_temp -= means[0]
    d_1stshell_temp -= means[1]
    d_2ndshell_temp -= means[2]

    d_center_t =   +d_center_temp[embed_times]
    d_1stshell_t = +d_1stshell_temp[embed_times]
    d_2ndshell_t = +d_2ndshell_temp[embed_times]

    d_center_t = ((d_center_t) / stds[0].max())
    d_1stshell_t = ((d_1stshell_t) / stds[1].max())
    d_2ndshell_t = ((d_2ndshell_t) / stds[2].max())

    return np.array([d_center_t, 
                        d_1stshell_t, 
                        d_2ndshell_t, 
                        ])

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

def compute_scaling_factors_d(namefile, means):

    d_center, d_1stshell, d_2ndshell = pickle.load(open(namefile, 'rb'))
    
    d_center -= means[0]
    d_1stshell -= means[1]
    d_2ndshell -= means[2]

    stds = [d_center.std(axis=0, keepdims=True),
            d_1stshell.std(axis=0, keepdims=True),
            d_2ndshell.std(axis=0, keepdims=True), 
            ]

    return stds

def compute_scaling_factors_dipole(namefile):

    dipoles_center, dipoles_1stshell, dipoles_2ndshell = pickle.load(open(namefile, 'rb'))

    stds = [dipoles_center.std(axis=0, keepdims=True),
            dipoles_1stshell.std(axis=0, keepdims=True),
            dipoles_2ndshell.std(axis=0, keepdims=True)]

    return stds


def construct_Xt_Yt_d(Ntrajs, seed, t, E, tau_e, njobs, traj_numbers, means, stds):

    (d_center_t, d_1stshell_t, d_2ndshell_t ) = (
        np.swapaxes(Parallel(n_jobs=njobs)(delayed(read_files_d)
            (namefile=f'./pickles_d/d_cent_1st_2nd_seed{seed}_traj{traj_number}.p', 
                t=t, E=E, tau_e=tau_e, means=means, stds=stds) for traj_number in traj_numbers), axis1=0, axis2=1)
    )
    d_center_t = d_center_t.reshape((Ntrajs,-1))
    d_1stshell_t = d_1stshell_t.reshape((Ntrajs,-1))
    d_2ndshell_t = d_2ndshell_t.reshape((Ntrajs,-1))

    return d_center_t, d_1stshell_t, d_2ndshell_t

def construct_Xt_Yt_dipole(Ntrajs, seed, t, E, tau_e, njobs, stds, frame_numbers):

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
    parser.add_argument("--E", dest="E", default=1,  type=int)
    args = parser.parse_args()
    if args.seed is None:
        sys.exit("Error: random seed must be given by user with --seed")

    # set parameters to compute imbalances
    Nframes = 5000
    Ntrajs = 2000
    tau_e = 1
    E = args.E
    E_d = E
    E_dipole = E
    t0 = 0 ## OUTDATED assertion discard first 5 ps, could contain artifacts (nvt -> nve)
    tend = Nframes - t0 - E*tau_e
    taus = np.arange(0,tend,20) # 20 frames = 200 fs = 0.2 ps
    k = 20
    betas = np.linspace(0., 1., 200, endpoint=False)
    alphas = betas / (1-betas)
    njobs = 56
    traj_numbers = np.loadtxt("Datatrajs_frame_numbers.txt", dtype='int')
    
    #get means values:
    means = np.zeros((3,Nframes))

    means[0] = np.loadtxt("pickles_mean/central_mean_and_std.txt")[:,0]
    means[1] = np.loadtxt("pickles_mean/1stshell_mean_and_std.txt")[:,0]
    means[2] = np.loadtxt("pickles_mean/2ndshell_mean_and_std.txt")[:,0]

    # compute standardization parameters on reference trajectory
    stds_d = compute_scaling_factors_d(f'./pickles_d/d_cent_1st_2nd_seed1_traj1.p', means=means)
    stds_dipole = compute_scaling_factors_dipole(f'./pickles_dipoles/dipoles_cent_1st_2nd_seed1_traj{traj_numbers[0]}.p')
    # construct X0 and Y0
    (d_center_t0, 
            d_1stshell_t0, 
            d_2ndshell_t0, 
            )= (
        construct_Xt_Yt_d(Ntrajs=Ntrajs, seed=args.seed, t=t0, E=E, tau_e=tau_e, njobs=njobs, traj_numbers=traj_numbers, means=means, stds=stds_d) 
    )

    dipoles_center_t0, dipoles_1stshell_t0, dipoles_2ndshell_t0 = (
        construct_Xt_Yt_dipole(
            Ntrajs = Ntrajs,
            seed = args.seed,
            t = t0,
            E = E,
            tau_e = tau_e,
            njobs = njobs,
            stds = stds_dipole,
            frame_numbers = traj_numbers
            )
        )



    for tau in tqdm(taus):
            
        # construct Xtau and Ytau
        (d_center_tau, 
                d_1stshell_tau, 
                d_2ndshell_tau , 
                )= (
            construct_Xt_Yt_d(Ntrajs=Ntrajs, seed=args.seed, t=t0+tau, E=E, tau_e=tau_e, njobs=njobs, traj_numbers=traj_numbers, means=means, stds=stds_d) 
        )

        dipoles_center_tau, dipoles_1stshell_tau, dipoles_2ndshell_tau = (
            construct_Xt_Yt_dipole(
                Ntrajs = Ntrajs,
                seed = args.seed,
                t = t0+tau,
                E = E,
                tau_e = tau_e,
                njobs = njobs,
                stds = stds_dipole,
                frame_numbers = traj_numbers
                )
        )

        d = MetricComparisons(maxk=Ntrajs-1, n_jobs=njobs)
        
        #print("input shapes: ", d_center_tau.shape, d_total_tau.shape, d_total_corrected_tau.shape)
        #print("inputs : \n", d_center_tau, d_total_tau)
        
        # 1st_shell_d <-> central dipôles
        info_imbalances_1std_to_centraldipole = d.return_inf_imb_causality(
            cause_present=d_1stshell_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)
        info_imbalances_centraldipole_to_1std = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=d_1stshell_t0,
            effect_future=d_1stshell_tau, weights=alphas, k=k)

        #  1st_shell_d <->  1stshell dipôles
        info_imbalances_1std_to_1stdipole = d.return_inf_imb_causality(
            cause_present=d_1stshell_t0, effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)
        info_imbalances_1stdipole_to_1std = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=d_1stshell_t0,
            effect_future=d_1stshell_tau, weights=alphas, k=k)

        # central_d <-> central dipôles
        info_imbalances_centrald_to_centraldipole = d.return_inf_imb_causality(
            cause_present=d_center_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)
        info_imbalances_centraldipole_to_centrald = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=d_center_t0,
            effect_future=d_center_tau, weights=alphas, k=k)

        #  central_d <->  1stshell dipôles
        info_imbalances_centrald_to_1stdipole = d.return_inf_imb_causality(
            cause_present=d_center_t0, effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)
        info_imbalances_1stdipole_to_centrald = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=d_center_t0,
            effect_future=d_center_tau, weights=alphas, k=k)

        #  1st_shell_d <->  2ndshell dipôles
        info_imbalances_1std_to_2nddipole = d.return_inf_imb_causality(
            cause_present=d_1stshell_t0, effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nddipole_to_1std = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=d_1stshell_t0,
            effect_future=d_1stshell_tau, weights=alphas, k=k)

        #  central_d <->  2ndshell dipôles
        info_imbalances_centrald_to_2nddipole = d.return_inf_imb_causality(
            cause_present=d_center_t0, effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nddipole_to_centrald = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=d_center_t0,
            effect_future=d_center_tau, weights=alphas, k=k)

        # save pickle
        pickle.dump([
                info_imbalances_1std_to_centraldipole, info_imbalances_centraldipole_to_1std,
                info_imbalances_1std_to_1stdipole, info_imbalances_1stdipole_to_1std,
                info_imbalances_centrald_to_centraldipole, info_imbalances_centraldipole_to_centrald,
                info_imbalances_centrald_to_1stdipole, info_imbalances_1stdipole_to_centrald,
                info_imbalances_1std_to_2nddipole, info_imbalances_2nddipole_to_1std,
                info_imbalances_centrald_to_2nddipole, info_imbalances_2nddipole_to_centrald
                ],
                open(f"./pickles_imb_completion/imb_seed{args.seed}_E{E}_tau{tau}_k{k}.p", "wb"))

    return

if __name__ == "__main__":
    main()
