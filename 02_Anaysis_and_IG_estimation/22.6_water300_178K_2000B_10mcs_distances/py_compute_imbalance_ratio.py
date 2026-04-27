from threadpoolctl import threadpool_limits
import numpy as np
from joblib import Parallel, delayed
import pickle
from dadapy.metric_comparisons import MetricComparisons
from tqdm import tqdm
import argparse
import sys

def read_files_dipole(namefile, t, E, tau_e, means, stds):

    d_center_temp, d_1stshell_temp, d_2ndshell_temp = pickle.load(open(namefile, 'rb'))

    #print("input shapes: ", d_center_temp.shape, d_total_temp.shape, d_totalc_temp.shape)

    embed_times = np.arange(t, t+E, tau_e)
    d_center_temp[0]=1
    d_1stshell_temp[0]=1
    d_2ndshell_temp[0]=1
    d_center_temp[1:] /= means[0][1:]
    d_1stshell_temp[1:] /= means[1][1:]
    d_2ndshell_temp[1:] /= means[2][1:]

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


def compute_scaling_factors(namefile, means):

    d_center, d_1stshell, d_2ndshell = pickle.load(open(namefile, 'rb'))
    
    d_center -= means[0]
    d_1stshell -= means[1]
    d_2ndshell -= means[2]

    stds = [d_center.std(axis=0, keepdims=True),
            d_1stshell.std(axis=0, keepdims=True),
            d_2ndshell.std(axis=0, keepdims=True), 
            ]

    return stds


def construct_Xt_Yt(Ntrajs, seed, t, E, tau_e, njobs, means, stds):

    (d_center_t, d_1stshell_t, d_2ndshell_t ) = (
        np.swapaxes(Parallel(n_jobs=njobs)(delayed(read_files_dipole)
            (namefile=f'./pickles_d/d_cent_1st_2nd_seed{seed}_traj{(traj_number*10000)+5000}.p', 
                t=t, E=E, tau_e=tau_e, means=means, stds=stds) for traj_number in range(Ntrajs)), axis1=0, axis2=1)
    )
    d_center_t = d_center_t.reshape((Ntrajs,-1))
    d_1stshell_t = d_1stshell_t.reshape((Ntrajs,-1))
    d_2ndshell_t = d_2ndshell_t.reshape((Ntrajs,-1))

    return d_center_t, d_1stshell_t, d_2ndshell_t


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
    t0 = 0 ## OUTDATED assertion discard first 5 ps, could contain artifacts (nvt -> nve)
    tend = Nframes - t0 - E*tau_e
    taus = np.arange(0,tend,20) # 20 frames = 200 fs = 0.2 ps
    k = 20
    betas = np.linspace(0., 1., 200, endpoint=False)
    alphas = betas / (1-betas)
    njobs = 56
    
    #get means values:
    means = np.zeros((3,Nframes))

    means[0] = np.loadtxt("pickles_mean/central_mean_and_std.txt")[:,0]
    means[1] = np.loadtxt("pickles_mean/1stshell_mean_and_std.txt")[:,0]
    means[2] = np.loadtxt("pickles_mean/2ndshell_mean_and_std.txt")[:,0]

    # compute standardization parameters on reference trajectory
    stds = compute_scaling_factors(f'./pickles_d/d_cent_1st_2nd_seed1_traj5000.p', means=means)

    # construct X0 and Y0
    (d_center_t0, 
            d_1stshell_t0, 
            d_2ndshell_t0, 
            )= (
        construct_Xt_Yt(Ntrajs=Ntrajs, seed=args.seed, t=t0, E=E, tau_e=tau_e, njobs=njobs, means=means, stds=stds) 
    )

    for tau in tqdm(taus):
            
        # construct Xtau and Ytau
        (d_center_tau, 
                d_1stshell_tau, 
                d_2ndshell_tau , 
                )= (
            construct_Xt_Yt(Ntrajs=Ntrajs, seed=args.seed, t=t0+tau, E=E, tau_e=tau_e, njobs=njobs, means=means, stds=stds) 
        )

        d = MetricComparisons(maxk=Ntrajs-1, n_jobs=njobs)
        
        #print("input shapes: ", d_center_tau.shape, d_total_tau.shape, d_total_corrected_tau.shape)
        #print("inputs : \n", d_center_tau, d_total_tau)
        
        # center <-> 1st shell
        info_imbalances_center_to_1st = d.return_inf_imb_causality(
            cause_present=d_center_t0, effect_present=d_1stshell_t0,
            effect_future=d_1stshell_tau, weights=alphas, k=k)
        info_imbalances_1st_to_center = d.return_inf_imb_causality(
            cause_present=d_1stshell_t0, effect_present=d_center_t0,
            effect_future=d_center_tau, weights=alphas, k=k)

        # center <-> 2nd shell
        info_imbalances_center_to_2nd = d.return_inf_imb_causality(
            cause_present=d_center_t0, effect_present=d_2ndshell_t0,
            effect_future=d_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nd_to_center = d.return_inf_imb_causality(
            cause_present=d_2ndshell_t0, effect_present=d_center_t0,
            effect_future=d_center_tau, weights=alphas, k=k)

        # 1st shell <-> 2nd shell
        info_imbalances_1st_to_2nd = d.return_inf_imb_causality(
            cause_present=d_1stshell_t0, effect_present=d_2ndshell_t0, 
            effect_future=d_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nd_to_1st = d.return_inf_imb_causality(
            cause_present=d_2ndshell_t0, effect_present=d_1stshell_t0, 
            effect_future=d_1stshell_tau, weights=alphas, k=k)

        # save pickle
        pickle.dump([
                info_imbalances_center_to_1st   , info_imbalances_1st_to_center   ,
                info_imbalances_center_to_2nd   , info_imbalances_2nd_to_center   , 
                info_imbalances_1st_to_2nd      , info_imbalances_2nd_to_1st      ,
                ],
                open(f"./pickles_imb_ratio/imb_seed{args.seed}_E{E}_tau{tau}_k{k}.p", "wb"))

    return

if __name__ == "__main__":
    main()
