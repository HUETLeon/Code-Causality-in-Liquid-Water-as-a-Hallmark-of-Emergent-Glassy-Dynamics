from threadpoolctl import threadpool_limits
import numpy as np
from joblib import Parallel, delayed
import pickle
from dadapy.metric_comparisons import MetricComparisons
from tqdm import tqdm
import argparse
import sys

def read_files_dipole(namefile, t, E, tau_e, stds):

    dipoles_center_temp, dipoles_1stshell_temp, dipoles_2ndshell_temp, dipoles_total_temp, dipoles_totalc_temp, dipoles_out_1stshell_temp, dipoles_out_2ndshell_temp = pickle.load(open(namefile, 'rb'))

    #print("input shapes: ", dipoles_center_temp.shape, dipoles_total_temp.shape, dipoles_totalc_temp.shape)

    embed_times = np.arange(t, t+E, tau_e)
    dipoles_center_t =   +dipoles_center_temp[embed_times]
    dipoles_1stshell_t = +dipoles_1stshell_temp[embed_times]
    dipoles_2ndshell_t = +dipoles_2ndshell_temp[embed_times]
    dipoles_total_t = +dipoles_total_temp[embed_times]
    dipoles_totalc_t = +dipoles_totalc_temp[embed_times]
    dipoles_out_1stshell_t = +dipoles_out_1stshell_temp[embed_times]
    dipoles_out_2ndshell_t = +dipoles_out_2ndshell_temp[embed_times]

    dipoles_center_t = ((dipoles_center_t) / stds[0].max())
    dipoles_1stshell_t = ((dipoles_1stshell_t) / stds[1].max())
    dipoles_2ndshell_t = ((dipoles_2ndshell_t) / stds[2].max())
    dipoles_total_t = ((dipoles_total_t) / stds[3].max())
    dipoles_totalc_t = ((dipoles_totalc_t) / stds[4].max())
    dipoles_out_1stshell_t = ((dipoles_out_1stshell_t)/ stds[5].max())
    dipoles_out_2ndshell_t = ((dipoles_out_2ndshell_t)/ stds[6].max())

    return np.array([dipoles_center_t, 
                        dipoles_1stshell_t, 
                        dipoles_2ndshell_t, 
                        dipoles_total_t, 
                        dipoles_totalc_t,
                        dipoles_out_1stshell_t,
                        dipoles_out_2ndshell_t
                        ])


def compute_scaling_factors(namefile):

    dipoles_center, dipoles_1stshell, dipoles_2ndshell , dipoles_total, dipoles_totalc, dipoles_out_1stshell, dipoles_out_2ndshell  = pickle.load(open(namefile, 'rb'))
    
    stds = [dipoles_center.std(axis=0, keepdims=True),
            dipoles_1stshell.std(axis=0, keepdims=True),
            dipoles_2ndshell.std(axis=0, keepdims=True), 
            dipoles_total.std(axis=0, keepdims=True),
            dipoles_totalc.std(axis=0, keepdims=True),
            dipoles_out_1stshell.std(axis=0, keepdims=True),
            dipoles_out_2ndshell.std(axis=0, keepdims=True)
            ]

    return stds


def construct_Xt_Yt(Ntrajs, seed, t, E, tau_e, njobs, stds):

    assert len(stds) == 7, "Error: wrong number of stds"
    (dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t, dipoles_total_t, dipoles_total_corrected_t, dipole_out_1stshell_t, dipole_out_2ndshell_t ) = (
        np.swapaxes(Parallel(n_jobs=njobs)(delayed(read_files_dipole)
            (namefile=f'./pickles_dipoles/dipoles_cent_1st_2nd_total_totalc_out1st_out2nd_seed{seed}_traj{(traj_number*10000)+5000}.p', 
                t=t, E=E, tau_e=tau_e, stds=stds) for traj_number in range(Ntrajs)), axis1=0, axis2=1)
    )
    dipoles_center_t = dipoles_center_t.reshape((Ntrajs,-1))
    dipoles_1stshell_t = dipoles_1stshell_t.reshape((Ntrajs,-1))
    dipoles_2ndshell_t = dipoles_2ndshell_t.reshape((Ntrajs,-1))
    dipoles_total_t = dipoles_total_t.reshape((Ntrajs,-1))
    dipoles_total_corrected_t = dipoles_total_corrected_t.reshape((Ntrajs,-1))
    dipole_out_1stshell_t = dipole_out_1stshell_t.reshape((Ntrajs,-1))
    dipole_out_2ndshell_t = dipole_out_2ndshell_t.reshape((Ntrajs,-1))

    return dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t, dipoles_total_t, dipoles_total_corrected_t, dipole_out_1stshell_t, dipole_out_2ndshell_t


def main():

    # read random seed and time tau
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", dest="seed", default=None, type=int)
    args = parser.parse_args()
    if args.seed is None:
        sys.exit("Error: random seed must be given by user with --seed")

    # set parameters to compute imbalances
    Nframes = 5000
    Ntrajs = 2000
    E = 50
    tau_e = 1
    E = 50
    t0 = 0 ## OUTDATED assertion discard first 5 ps, could contain artifacts (nvt -> nve)
    tend = Nframes - t0 - E*tau_e
    taus = np.arange(0,tend,20) # 20 frames = 200 fs = 0.2 ps
    k = 20
    betas = np.linspace(0., 1., 200, endpoint=False)
    alphas = betas / (1-betas)
    njobs = 56

    # compute standardization parameters on reference trajectory
    stds = compute_scaling_factors(f'./pickles_dipoles/dipoles_cent_1st_2nd_total_totalc_out1st_out2nd_seed1_traj5000.p')

    # construct X0 and Y0
    (dipoles_center_t0, 
            dipoles_1stshell_t0, 
            dipoles_2ndshell_t0, 
            dipoles_total_t0, 
            dipoles_total_corrected_t0,
            dipoles_out_1stshell_t0,
            dipoles_out_2ndshell_t0
            )= (
        construct_Xt_Yt(Ntrajs=Ntrajs, seed=args.seed, t=t0, E=E, tau_e=tau_e, njobs=njobs, stds=stds) 
    )

    for tau in tqdm(taus):
            
        # construct Xtau and Ytau
        (dipoles_center_tau, 
                dipoles_1stshell_tau, 
                dipoles_2ndshell_tau , 
                dipoles_total_tau, 
                dipoles_total_corrected_tau,
                dipoles_out_1stshell_tau,
                dipoles_out_2ndshell_tau
                )= (
            construct_Xt_Yt(Ntrajs=Ntrajs, seed=args.seed, t=t0+tau, E=E, tau_e=tau_e, njobs=njobs, stds=stds) 
        )

        d = MetricComparisons(maxk=Ntrajs-1, n_jobs=njobs)
        
        #print("input shapes: ", dipoles_center_tau.shape, dipoles_total_tau.shape, dipoles_total_corrected_tau.shape)
        #print("inputs : \n", dipoles_center_tau, dipoles_total_tau)
        
        # center <-> 1st shell
        info_imbalances_center_to_1st = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)
        info_imbalances_1st_to_center = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)

        # center <-> 2nd shell
        info_imbalances_center_to_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nd_to_center = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)

        # 1st shell <-> 2nd shell
        info_imbalances_1st_to_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=dipoles_2ndshell_t0, 
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nd_to_1st = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=dipoles_1stshell_t0, 
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)

        # center <-> total
        info_imbalances_center_to_total = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=dipoles_total_t0,
            effect_future=dipoles_total_tau, weights=alphas, k=k)
        info_imbalances_total_to_center = d.return_inf_imb_causality(
            cause_present=dipoles_total_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)
        
        # 1st shell <-> total
        info_imbalances_1st_to_total = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=dipoles_total_t0,
            effect_future=dipoles_total_tau, weights=alphas, k=k)
        info_imbalances_total_to_1st = d.return_inf_imb_causality(
            cause_present=dipoles_total_t0, effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)

        # 2nd shell <-> total
        info_imbalances_2nd_to_total = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=dipoles_total_t0,
            effect_future=dipoles_total_tau, weights=alphas, k=k)
        info_imbalances_total_to_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_total_t0, effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)

        # center <-> totalc
        info_imbalances_center_to_totalc = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=dipoles_total_corrected_t0,
            effect_future=dipoles_total_corrected_tau, weights=alphas, k=k)
        info_imbalances_totalc_to_center = d.return_inf_imb_causality(
            cause_present=dipoles_total_corrected_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)

        # 1st shell <-> totalc
        info_imbalances_1st_to_totalc = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=dipoles_total_corrected_t0,
            effect_future=dipoles_total_corrected_tau, weights=alphas, k=k)
        info_imbalances_totalc_to_1st = d.return_inf_imb_causality(
            cause_present=dipoles_total_corrected_t0, effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)

         # 2nd shell <-> totalc
        info_imbalances_2nd_to_totalc = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=dipoles_total_corrected_t0,
            effect_future=dipoles_total_corrected_tau, weights=alphas, k=k)
        info_imbalances_totalc_to_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_total_corrected_t0, effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)

        # 1st shell <-> out 1st shell
        info_imbalances_1st_to_out_1st = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=dipoles_out_1stshell_t0,
            effect_future=dipoles_out_1stshell_tau, weights=alphas, k=k)
        info_imbalances_out_1st_to_1st = d.return_inf_imb_causality(
            cause_present=dipoles_out_1stshell_t0, effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)

        # 2nd shell <-> out 2nd shell
        info_imbalances_2nd_to_out_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=dipoles_out_2ndshell_t0,
            effect_future=dipoles_out_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_out_2nd_to_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_out_2ndshell_t0, effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)

        # save pickle
        pickle.dump([
                info_imbalances_center_to_1st   , info_imbalances_1st_to_center   ,
                info_imbalances_center_to_2nd   , info_imbalances_2nd_to_center   , 
                info_imbalances_1st_to_2nd      , info_imbalances_2nd_to_1st      ,
                info_imbalances_center_to_total , info_imbalances_total_to_center ,
                info_imbalances_1st_to_total    , info_imbalances_total_to_1st    ,
                info_imbalances_2nd_to_total    , info_imbalances_total_to_2nd    ,
                info_imbalances_center_to_totalc, info_imbalances_totalc_to_center,
                info_imbalances_1st_to_totalc   , info_imbalances_totalc_to_1st   ,
                info_imbalances_2nd_to_totalc   , info_imbalances_totalc_to_2nd   ,
                info_imbalances_1st_to_out_1st  , info_imbalances_out_1st_to_1st  ,
                info_imbalances_2nd_to_out_2nd  , info_imbalances_out_2nd_to_2nd  ,
                ],
                open(f"./pickles_imb/imb_seed{args.seed}_tau{tau}_k{k}.p", "wb"))

    return

if __name__ == "__main__":
    main()
