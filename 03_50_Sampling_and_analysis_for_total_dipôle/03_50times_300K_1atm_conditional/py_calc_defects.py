import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
import structural_parameters as sp
from MDAnalysis.analysis import distances
import argparse
import sys

parser = argparse.ArgumentParser()
#parser.add_argument("--traj_number", dest="traj_number", default=None, type=int)
parser.add_argument("--seed", dest="seed", default=None, type=int)
#parser.add_argument("--central", dest="central", default=None, type=int)
args = parser.parse_args()
if args.seed is None:
    sys.exit("Error: seed number must be given by user with --seed")
seed = args.seed

u = mda.Universe(f"Run_{seed}/conf.gro", f"Run_{seed}/300w_300K_1atm.xtc")
water = u.select_atoms("type O or type H")
OW = u.select_atoms("type O")
HW = u.select_atoms("type H")
nframes = len(u.trajectory)
selected_frames = np.loadtxt("uniform_starts.dat", dtype=int) 


print(water.n_atoms,OW.n_atoms,HW.n_atoms)

defects_in = []
defects_out = []
for n,ts in enumerate(tqdm(u.trajectory[selected_frames])):
    dims = ts.dimensions 
    pos_o = OW.positions
    pos_h = HW.positions
    oo_dists = np.zeros((OW.n_atoms,OW.n_atoms))
    oh_dists = np.zeros((OW.n_atoms,HW.n_atoms))
    distances.distance_array(pos_o,pos_o,box=dims,result=oo_dists)
    distances.distance_array(pos_o,pos_h,box=dims,result=oh_dists)
    oo_dist_vecs, oo_indices = sp.compute_distance_vectors(pos_o,pos_o,oo_dists,45,dims[:3],same_type=True)
    oh_dist_vecs, _ = sp.compute_distance_vectors(pos_o,pos_h,oh_dists,2,dims[:3],same_type=False)
    defs_in, defs_out, _ = sp.hbond_defects(oo_dist_vecs,oh_dist_vecs,oo_indices)
    defects_in.append(defs_in)
    defects_out.append(defs_out)
defects_in = np.array(defects_in)
defects_out = np.array(defects_out)
np.savetxt(f'defects/defects_in_aw_seed{seed}.txt',defects_in, fmt='%d')
np.savetxt(f'defects/defects_out_aw_seed{seed}.txt',defects_out, fmt='%d')
