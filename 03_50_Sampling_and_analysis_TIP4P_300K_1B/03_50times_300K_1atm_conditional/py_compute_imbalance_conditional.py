from threadpoolctl import threadpool_limits
import numpy as np
from joblib import Parallel, delayed
import pickle
from dadapy.metric_comparisons import MetricComparisons
from tqdm import tqdm
import argparse
import sys

def read_files_dipole(namefile, t, E, tau_e, stds):

    dipoles_center_temp, dipoles_1stshell_temp, dipoles_2ndshell_temp, dipole_total, dipole_total_c = pickle.load(open(namefile, 'rb'))

    embed_times = np.arange(t, t+E, tau_e)
    dipoles_center_t =   +dipoles_center_temp[embed_times]
    dipoles_1stshell_t = +dipoles_1stshell_temp[embed_times]
    dipoles_2ndshell_t = +dipoles_2ndshell_temp[embed_times]

    dipoles_center_t = ((dipoles_center_t) / stds[0])
    dipoles_1stshell_t = ((dipoles_1stshell_t) / stds[1])
    dipoles_2ndshell_t = ((dipoles_2ndshell_t) / stds[2])

    return np.array([dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t])


def compute_scaling_factors(namefile):

    dipoles_center, dipoles_1stshell, dipoles_2ndshell, dipole_total , dipole_total_c = pickle.load(open(namefile, 'rb'))
    
    stds = [dipoles_center.std(axis=0, keepdims=True),
            dipoles_1stshell.std(axis=0, keepdims=True),
            dipoles_2ndshell.std(axis=0, keepdims=True)]

    return stds


def construct_Xt_Yt(Ntrajs, seed, t, E, tau_e, njobs, stds):

    assert len(stds) == 3, "Error: wrong number of stds"
    (dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t) = (
        np.swapaxes(Parallel(n_jobs=njobs)(delayed(read_files_dipole)
            (namefile=f'./pickles_dipoles/dipoles_cent_1st_2nd_total_totalc_seed{seed}_traj{(traj_number*10000)+5000}.p', 
                t=t, E=E, tau_e=tau_e, stds=stds) for traj_number in range(Ntrajs)), axis1=0, axis2=1)
    )
    dipoles_center_t =  dipoles_center_t.reshape((Ntrajs,-1))
    dipoles_1stshell_t = dipoles_1stshell_t.reshape((Ntrajs,-1))
    dipoles_2ndshell_t = dipoles_2ndshell_t.reshape((Ntrajs,-1))

    return dipoles_center_t, dipoles_1stshell_t, dipoles_2ndshell_t


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
    tau_e = 1
    E = 1
    t0 = 1 ## OUTDATED assertion discard first 5 ps, could contain artifacts (nvt -> nve)
    tend = Nframes - t0 - E*tau_e
    taus = np.arange(0,tend,20) # 20 frames = 200 fs = 0.2 ps
    k = 20
    alphas = np.linspace(0.,1.5,150)
    njobs = 56

    # compute standardization parameters on reference trajectory
    stds = compute_scaling_factors(f'./pickles_dipoles/dipoles_cent_1st_2nd_total_totalc_seed1_traj5000.p')

    # construct X0 and Y0
    dipoles_center_t0, dipoles_1stshell_t0, dipoles_2ndshell_t0 = (
        construct_Xt_Yt(Ntrajs=Ntrajs, seed=args.seed, t=t0, E=E, tau_e=tau_e, njobs=njobs, stds=stds) 
    )

    

    print(dipoles_center_t0.shape)
    
    oxygen_indexs = np.loadtxt(f"oxygen_indexs/seed_{args.seed}_indexs.txt", dtype=int)

    defects_in = np.loadtxt(f"defects/defects_in_aw_seed{args.seed}.txt", dtype=int)
    defects_out = np.loadtxt(f"defects/defects_out_aw_seed{args.seed}.txt", dtype=int)
    
    mask = np.zeros_like(oxygen_indexs, dtype=int)

    for i in range(0,len(mask)):
        def_in = defects_in[i,oxygen_indexs[i]//3]
        def_out = defects_out[i,oxygen_indexs[i]//3]
        if (def_in -2 != 0) | (def_out -2 != 0):
            mask[i] = 1

    indexs_defects = np.array([ i for i, m in enumerate(mask) if m == 1])
    indexs_defectless = np.array([ i for i, m in enumerate(mask) if m == 0])
    
    Ntraj_defects = len(indexs_defects)
    Ntraj_defectless = len(indexs_defectless)

    proportion_of_defects = len(indexs_defects) / (2000)

    print("defect rate = ", proportion_of_defects)

    dipoles_center_t0_defects = dipoles_center_t0[indexs_defects,:]
    dipoles_1stshell_t0_defects = dipoles_1stshell_t0[indexs_defects,:]
    dipoles_2ndshell_t0_defects = dipoles_2ndshell_t0[indexs_defects,:]

    dipoles_center_t0_defectless = dipoles_center_t0[indexs_defectless,:]
    dipoles_1stshell_t0_defectless = dipoles_1stshell_t0[indexs_defectless,:]
    dipoles_2ndshell_t0_defectless = dipoles_2ndshell_t0[indexs_defectless,:]

    for tau in tqdm(taus):
            
        # construct Xtau and Ytau
        dipoles_center_tau, dipoles_1stshell_tau, dipoles_2ndshell_tau = (
            construct_Xt_Yt(Ntrajs=Ntrajs, seed=args.seed, t=t0+tau, E=E, tau_e=tau_e, njobs=njobs, stds=stds) 
        )

        
        dipoles_center_tau_defects = dipoles_center_tau[indexs_defects,:]
        dipoles_1stshell_tau_defects = dipoles_1stshell_tau[indexs_defects,:]
        dipoles_2ndshell_tau_defects = dipoles_2ndshell_tau[indexs_defects,:]
        
        dipoles_center_tau_defectless = dipoles_center_tau[indexs_defectless,:]
        dipoles_1stshell_tau_defectless = dipoles_1stshell_tau[indexs_defectless,:]
        dipoles_2ndshell_tau_defectless = dipoles_2ndshell_tau[indexs_defectless,:]

        d = MetricComparisons(maxk = Ntrajs - 1, n_jobs = njobs)

        d_defects = MetricComparisons(maxk = Ntraj_defects - 1, n_jobs = njobs)
        d_defectless = MetricComparisons(maxk = Ntraj_defectless - 1, n_jobs = njobs)


        # center <-> 1st shell
        info_imbalances_center_to_1st = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=dipoles_1stshell_t0,
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)
        info_imbalances_1st_to_center = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)
        
        info_imbalances_center_to_1st_defects = d_defects.return_inf_imb_causality(
            cause_present=dipoles_center_t0_defects, effect_present=dipoles_1stshell_t0_defects,
            effect_future=dipoles_1stshell_tau_defects, weights=alphas, k=k)
        info_imbalances_1st_to_center_defects = d_defects.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0_defects, effect_present=dipoles_center_t0_defects,
            effect_future=dipoles_center_tau_defects, weights=alphas, k=k)

        info_imbalances_center_to_1st_defectless = d_defectless.return_inf_imb_causality(
            cause_present=dipoles_center_t0_defectless, effect_present=dipoles_1stshell_t0_defectless,
            effect_future=dipoles_1stshell_tau_defectless, weights=alphas, k=k)
        info_imbalances_1st_to_center_defectless = d_defectless.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0_defectless, effect_present=dipoles_center_t0_defectless,
            effect_future=dipoles_center_tau_defectless, weights=alphas, k=k)
        
        # center <-> 2nd shell
        info_imbalances_center_to_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_center_t0, effect_present=dipoles_2ndshell_t0,
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nd_to_center = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=dipoles_center_t0,
            effect_future=dipoles_center_tau, weights=alphas, k=k)
        
        info_imbalances_center_to_2nd_defects = d_defects.return_inf_imb_causality(
            cause_present=dipoles_center_t0_defects, effect_present=dipoles_2ndshell_t0_defects,
            effect_future=dipoles_2ndshell_tau_defects, weights=alphas, k=k)
        info_imbalances_2nd_to_center_defects = d_defects.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0_defects, effect_present=dipoles_center_t0_defects,
            effect_future=dipoles_center_tau_defects, weights=alphas, k=k)
        
        info_imbalances_center_to_2nd_defectless = d_defectless.return_inf_imb_causality(
            cause_present=dipoles_center_t0_defectless, effect_present=dipoles_2ndshell_t0_defectless,
            effect_future=dipoles_2ndshell_tau_defectless, weights=alphas, k=k)
        info_imbalances_2nd_to_center_defectless = d_defectless.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0_defectless, effect_present=dipoles_center_t0_defectless,
            effect_future=dipoles_center_tau_defectless, weights=alphas, k=k)

        # 1st shell <-> 2nd shell
        info_imbalances_1st_to_2nd = d.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0, effect_present=dipoles_2ndshell_t0, 
            effect_future=dipoles_2ndshell_tau, weights=alphas, k=k)
        info_imbalances_2nd_to_1st = d.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0, effect_present=dipoles_1stshell_t0, 
            effect_future=dipoles_1stshell_tau, weights=alphas, k=k)
    
        info_imbalances_1st_to_2nd_defects = d_defects.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0_defects, effect_present=dipoles_2ndshell_t0_defects,
            effect_future=dipoles_2ndshell_tau_defects, weights=alphas, k=k)
        info_imbalances_2nd_to_1st_defects = d_defects.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0_defects, effect_present=dipoles_1stshell_t0_defects,
            effect_future=dipoles_1stshell_tau_defects, weights=alphas, k=k)

        info_imbalances_1st_to_2nd_defectless = d_defectless.return_inf_imb_causality(
            cause_present=dipoles_1stshell_t0_defectless, effect_present=dipoles_2ndshell_t0_defectless,
            effect_future=dipoles_2ndshell_tau_defectless, weights=alphas, k=k)
        info_imbalances_2nd_to_1st_defectless = d_defectless.return_inf_imb_causality(
            cause_present=dipoles_2ndshell_t0_defectless, effect_present=dipoles_1stshell_t0_defectless,
            effect_future=dipoles_1stshell_tau_defectless, weights=alphas, k=k)

        # save pickle
        pickle.dump([info_imbalances_center_to_1st, info_imbalances_1st_to_center, 
                     info_imbalances_center_to_1st_defects, info_imbalances_1st_to_center_defects, 
                     info_imbalances_center_to_1st_defectless, info_imbalances_1st_to_center_defectless,
                     info_imbalances_center_to_2nd, info_imbalances_2nd_to_center, 
                     info_imbalances_center_to_2nd_defects, info_imbalances_2nd_to_center_defects, 
                     info_imbalances_center_to_2nd_defectless, info_imbalances_2nd_to_center_defectless,
                     info_imbalances_1st_to_2nd, info_imbalances_2nd_to_1st, 
                     info_imbalances_1st_to_2nd_defects, info_imbalances_2nd_to_1st_defects, 
                     info_imbalances_1st_to_2nd_defectless, info_imbalances_2nd_to_1st_defectless],
                #, info_imbalances_center_to_2nd, info_imbalances_2nd_to_center, info_imbalances_1st_to_2nd, info_imbalances_2nd_to_1st ],
                open(f"./pickles_imb/imb_seed{args.seed}_tau{tau}_k{k}.p", "wb"))

    return

if __name__ == "__main__":
    main()
