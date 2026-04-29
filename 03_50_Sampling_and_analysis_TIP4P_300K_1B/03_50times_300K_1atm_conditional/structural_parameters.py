import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
import MDAnalysis as mda
from itertools import combinations
import networkx as nx

"""
This is a collection of functions used to compute some 
chemically inspired structural parameters for a molecular system (Water). As well as the Smooth Overlap of Atomic Positions (SOAP) descriptor.

All these functions require you to supply all the arguments frame by frame. The easiest way would be to read the trajectory into an MDAnalysis Universe, iterate through the trajectory, and supply the needed arguments of each function.

The example jupyter notebook(s) in this folder will show how this is done.
"""

def compute_distance_vectors(ref_coordinates,neigh_coordinates,distance_matrix,n_neighbours,dimensions,same_type=True):
    """
    Computes the distance vectors between a reference set of atomic coordinates and their n_neighbours.
    Parameters:
        ref_coordinates: 
        (N, 3) Numpy array containing the xyz coordinates of the reference species in a single frame.

        neigh_coordinates:
        (N_neigh, 3) Numpy array containing the xyz coordinates of the neighbouring species in a single frame.
        distance_matrix:
        (N, N) matrix containing the pairwise distances between all pair of points supplied in coordinates.
        
        n_neighbours:
        (int) number of neighbours to consider
        
        dimensions:
        One dimensional Numpy array of length 3, containing the dimensions of the simulation box in a single frame, 
        as well as the angles between the box edges; [lx, ly, lz]
        same_type:
        (bool) Whether or not the ref_coordinates and neigh_coordinates are the same, to take care of the zero distances when selecting neighbours.
    """
    
    #compute neighbour list up to n_neighbours
    if same_type:
        sorted_neighbour_list = np.argsort(distance_matrix,axis=1)[:,1:n_neighbours+1]
    else:
        sorted_neighbour_list = np.argsort(distance_matrix,axis=1)[:,:n_neighbours]
    neighbour_coordinates = neigh_coordinates[sorted_neighbour_list] #neighbour coordinates
    reference_coordinates = ref_coordinates[:,np.newaxis,:] #broadcast reference coordinates along axis 1
    
    #compute distance vectors between reference and neighbour coordinates
    distance_vectors = neighbour_coordinates - reference_coordinates
    distance_vectors -= np.around(distance_vectors/dimensions)*dimensions #apply pbc
    return distance_vectors,sorted_neighbour_list




def compute_tetrahedrality(distance_vectors,n_neighbours=4):
    """
    Compute the tetrahedral order parameter introduced by Debenedetti and Errington - Errington, J. R., & Debenedetti, P. G. (2001). Relationship between structural order and the anomalies of liquid water. Nature, 409(6818), 318-321.
    Parameters:
        distance_vectors:
        (N, max_n_neighbours, 3) Numpy array containing the distance vectors between N reference oxygens and their neighbours up to some max_n_neighbours. Make sure they are at least 4.
        n_neighbours:
        (int) Number of neighbours to consider for computing qtet. The standard definition is with n_neighbours = 4
    """
    N = distance_vectors.shape[0]
    distance_vectors = distance_vectors[:,:n_neighbours,:]
    # Using these 4 distance vectors, filter out the 6 unique possible pairs of angles between them
    pairs = list(combinations(range(n_neighbours), 2))
    qtet_vals = np.zeros(N)

    #iterate through all pairs and compute the angles between them
    for i, j in pairs:
        v1 = distance_vectors[:, i, :]
        v2 = distance_vectors[:, j, :]

        dot = np.einsum('ij,ij->i', v1, v2)
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)

        cos_theta = dot / (norm1 * norm2)
        qtet_vals += (cos_theta + 1./3.)**2  #sum the angles over the 6 pairs and compute second term in qtet formula

    q_tet = 1. - (3./8.) * qtet_vals  #tetrahedrality
    return q_tet


def compute_d5(nn_distances):
    """
    Compute the distance to the fifth oxygen from a reference one
    Parameter:
        nn_distances:
        (N,N) distance matrix between all pairs of atoms we are interested in (usually oxygens)
    """
    sorted_nn_distances = np.sort(nn_distances,axis=1)
    return sorted_nn_distances[:,5]



def compute_LSI(disp_vectors, cutoff=3.7):
    """
    Compute the Local Structural Index (LSI) introduced by Shiratani and Sasai -    Shiratani, E., & Sasai, M. (1996). Growth and collapse of structural patterns in the hydrogen bond network in liquid water. The Journal of chemical physics, 104(19), 7671-7680.
    Parameters:
        disp_vectors:
        (N, n_neighbours, 3) Numpy array containing the OO distance vectors between a reference oxygen and it's n_neighbours

        cutoff: 
        (float) Cut-off distance used in the nearest neighbour search for computing LSI

    """
    natoms = disp_vectors.shape[0]
    max_nn = disp_vectors.shape[1]

    #compute all distances to the max_nn number of neighbours and sort them.
    dists = np.linalg.norm(disp_vectors, axis=2)  #shape is (N,max_nn)
    dists_sorted = np.sort(dists, axis=1)  #shape is (N,max_nn)

    #count how many of these max_nn neighbours are within the cutoff
    mask_within = dists_sorted <= cutoff   #shape is (N,)
    num_within = np.sum(mask_within, axis=1) #total number of neighbours within the cutoff

    # build an empty array of shape (N,max_nn) to store the filtered distances up to 3.7A
    #we pad with nans in order to be able to take the variance over the whole matrix with np.nanvar
    filtered_distances = np.full((natoms, max_nn), np.nan)

    #fill the filtered_distance array with the sorted distances up to 3.7A. and add one more distance beyond 3.7A
    for i in range(natoms):
        n = num_within[i]
        filtered_distances[i, :n] = dists_sorted[i, :n]
        if n < max_nn:   #check if the number of neighbours within a cutoff has gone beyond maxx_nn.
            filtered_distances[i, n] = dists_sorted[i, n]  #append the next distance value beyond 3.7A

    #compute the delta_i which is the differences between sorted consecutive distances up to 3.7A and the next closest distance.
    deltas = np.diff(np.sort(filtered_distances, axis=1), axis=1)

    # compute the variance of the deltas, which is the LSI parameter
    LSI = np.nanvar(deltas, axis=1, keepdims=True)

    #Note: The for loop over the atoms can be vectorized but it is a bit unintuitve in my opinion
    return LSI.flatten()



def switching_function(r,w=3.7,n=30,m=60):
    """
    Switching function to compute the coordination number. It is the same function used in PLUMMED.
    Parameters:
        r: Pairwise distance matrix between atomic species
        w: Cutoff distance defining the coordination shell
        (n,m): Integer numbers determining how fast the switching function goes to zero beyond the cutoff distance 

    """
    numerator = 1. - (r/w)**n
    denominator = 1. - (r/w)**m
    return numerator/denominator


def coordination_number(nn_distances,n,m):
    """
    Computes the coordination number by summing the switching function.
    Parameters:
        nn_distances: 
        (N, N) matrix telling you the distance between the N pairs of atoms
        n and m: Integer numbers determining how fast the switching function goes to zero beyond the cutoff distance 
    """
    nn_dists = nn_distances[~np.eye(nn_distances.shape[0],dtype=bool)].reshape(nn_distances.shape[0],-1) #remove the self distances from the distance matrix

    #apply switching function on distances and sum 
    c = switch(nn_dists)
    coord_number = np.sum(c,axis=1)
    return coord_number


def compute_TTO(distance_vectors,n_neighbours):
    """
    Compute the Translational Tetrahedral Order Parameter introduced by Chau and Hardwick - Chau, P. L., & Hardwick, A. J. (1998). A new order parameter for tetrahedral configurations. Molecular Physics, 93(3), 511-518.
    Parameters:
        distance_vectors:
        (N, max_n_neighbours, 3) Numpy array containing the OO distance vectors beŧween each oxygen and its max_n_neighbour neighbours.
        n_neighbours:
        (int) Maximum number of neighbours to use for the TTO calculation. To use the standard TTO definition, n_neighbours should be 4.
    """
    #find the distance vectors up to the n_neighbours
    distance_vectors = distance_vectors[:,:n_neighbours,:] 
    nn_distances = np.linalg.norm(distance_vectors,axis=2) #find the distances to the n_neighbours
    sorted_dists_to_4 = np.sort(nn_distances,axis=1)
    mean_sorted_dists_to_4 = np.mean(sorted_dists_to_4,axis=1)
    sum_term = (np.sum((sorted_dists_to_4 - mean_sorted_dists_to_4[:,np.newaxis])**2/(4*mean_sorted_dists_to_4[:,np.newaxis]**2),axis=1))*(1/3)
    TTO = 1 - sum_term
    return TTO   



def compute_soap(coordinates,atomic_numbers,atomic_species,dimensions,soap_centers,sigma=1.0,rcut=3.7,nmax=8,lmax=6,average_soap='off'):
    """
    Compute the Smooth Overlap of Atomic Positions (SOAP) descriptor using dscribe and ase
    Parameters:
        coordinates:
        (N, 3) Numpy array of atomic coordinates for all the species in your system in one frame. 
        Must include the coordinates of the atoms you will consider as the soap centers,
        as well as the atoms that contribute to the local density for that center. 
        
        atomic_numbers:
        (N,) numpy array containing the atomic numbers of all the species in the system. eg O = 8 and H = 1
        atomic_species:
        python list containing strings that denote all the species that will be seen by dscribe eg:  ["O","H"] if soap should 
        be computed considering both oxygen and hydrogen contributions to the local density expansion
        
        dimensions:
        One dimensional Numpy array of length 6, containing the dimensions of the simulation box in a single frame, 
        as well as the angles between the box edges; [lx, ly, lz, alpha, beta, gamma]

        sigma:
        Standard deviation of the Gaussians used to compute the local density around a reference species. Default is 1.0 A
        
        rcut:
        Cut-off distance to define the local neighbourhood for the density computation
        
        nmax:
        Number of radial basis functions to use for the density expansion. default is 8

        lmax:
        Number of angular basis functions to use for the density expansion. default is 6

        average_soap:
        Whether you want the average soap spectra in that frame or not. The options are 'off' for no averaging,
        'outer' to average over the powerspectra of different atomic sites and 'inner' to average over atomic sites before summing 
        over the magnetic quantum numbers. 
    """
    species = atomic_species
    rcut = rcut
    nmax = nmax
    lmax = lmax
    sigma = sigma
    
    # Set up the SOAP descriptor
    soap = SOAP(
         species=species,
         periodic=True,
         r_cut=rcut,
         n_max=nmax,
         l_max=lmax,
         sigma=sigma,
         average=average_soap
)

    #create an ase Atoms object defining the simulation box and populate it with the coordinates of the atoms, taking pbc into consideration
    water = Atoms(cell=dimensions,  positions=coordinates, pbc=True) 
    water.set_atomic_numbers(atomic_numbers)   #add atomic numbers to the ase Atoms object - dscribe needs it to define soap
    soap_vectors = soap.create(water,centers=soap_centers) #compute soap from dscribe on the ase Atoms object
    return soap_vectors



def angle(v1,v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    dot = np.dot(v1/norm1,v2/norm2)
    theta = np.arccos(dot)
    theta = theta*(180./np.pi)
    return theta

def hbond_defects(oo_vectors, oh_vectors, oo_indices):
    """
    Compute the hydrogen bond donors/acceptors in a water system

    Parameters:
        oo_vectors: (natoms_o, n_neighbours, 3), sorted PBC-corrected distance vectors from each O to 45 nearest O neighbors
        oh_vectors: (natoms_o, natoms_h, 3), sorted PBC-corrected distance vectors from each O to its 2 nearest H atoms
        oo_indices: (natoms_o, 45), global indices of each O atom's 45 nearest neighbors (same order as oo_vectors)

    """
    natoms_o = oo_vectors.shape[0]
    link_matrix = np.zeros((natoms_o, natoms_o), dtype=int)
    dd_sqrt = np.linalg.norm(oo_vectors, axis=2)
    mask = dd_sqrt < 3.5

    for l in range(natoms_o):
        dr_filtered = oo_vectors[l][mask[l]]
        nb_filtered = oo_indices[l][mask[l]]
        dr_oh = oh_vectors[l]
        for i in range(len(dr_filtered)):
            ang1 = angle(dr_oh[0], dr_filtered[i])
            ang2 = angle(dr_oh[1], dr_filtered[i])
            ang = min(ang1, ang2)
            if ang < 30.0:
                link_matrix[l, nb_filtered[i]] = 1

    defects_in = np.sum(link_matrix, axis=0)   #sum link_matrix over rows to obtain hydrogen bonds received by each molecule
    defects_out = np.sum(link_matrix, axis=1)  #sum link_matrix over columns to obtain hydrogen bonds each molecule gives to another
    return defects_in, defects_out, link_matrix


def compute_psi(adj_matrix,oo_distances):
    """
    Compute the Psi Order parameter introduced by Foffi and Sciortino - Foffi, R., & Sciortino, F. (2022). Correlated fluctuations of structural indicators close to the liquid–liquid transition in supercooled water. The Journal of Physical Chemistry B, 127(1), 378-386.
    Parameters:
        adj_matrix:
        (N, N) Numpy array containing 1s and 0s. If element i,j = 1 then there is a hydrogen bond between oxygen i and j, otherwise 0.
        oo_distances: 
        (N, N) Numpy array containing the OO distances between the reference oxygens
    """
    psi = np.zeros(oo_distances.shape[0])
    #symmetrize adjacency matrix so that it is undirected
    link_matrix = ((adj_matrix + adj_matrix.T) > 0).astype(int) 

    #dump adjacency matrix into a networkx graph
    G=nx.from_numpy_array(link_matrix) 
    for i in range(link_matrix.shape[0]):
        aa = nx.single_source_shortest_path_length(G,i,4) #compute shortest path to each oxygen
        indices_4steps = []
        indices_3steps = []
        for source,steps in aa.items():
            if steps == 4:
                indices_4steps.append(source) #oxygens reachable by 4 Hbonds
            if steps == 3:
                indices_3steps.append(source) #oxygens reachable by 3 Hbonds
        #filter out nodes reachable through 4 Hbonds ONLY
        cc = np.setdiff1d(indices_4steps,indices_3steps)  
        if len(cc) != 0:
        	psi[i] = min(oo_distances[i][cc]) #minimum physical distance between each node and the nodes traversed 4 Hbonds away from reference node
        else:
        	psi[i] = np.nan
    return psi


def compute_eta(link_matrix, oo_distances):
    """
    Compute the eta parameter introduced by Foffi and Sciortino - Foffi, R., & Sciortino, F. (2022). Correlated fluctuations of structural indicators close to the liquid–liquid transition in supercooled water. The Journal of Physical Chemistry B, 127(1), 378-386.
    Parameters:
        link_matrix:
        (N, N) Numpy array containing 1s and 0s. If element i,j = 1 then there is a hydrogen bond between oxygen i and j, otherwise 0.
        oo_distances: 
        (N, N) Numpy array containing the OO distances between the reference oxygens
    """
    eta_parameter = np.zeros(link_matrix.shape[0])
    for l in range(link_matrix.shape[0]):
        sort_dist = np.argsort(oo_distances[l]) #sort the distances to molecule l
        
        #multiply the sorted distances with the link matrix of 1s and 0s. 
        #proj will be zero if there is no Hbond and it will keep the OO distance if there is a hydrogen bond
        proj = np.multiply(oo_distances[l][sort_dist],link_matrix[l][sort_dist]) 
        cond = proj == 0  #mask to filter out the non-hydrogen bonded indices
        counts = np.cumsum(cond)
        idx = np.searchsorted(counts,2) #find the second occurance of 0 to show the closest point NOT hbonded to mol l
        eta_parameter[l] = oo_distances[l][sort_dist][idx] - max(proj) #subtract the distance of the closest mol not hbonded from the distance of the furthest bonded
    return  eta_parameter

def compute_av_descriptor(nn_distances,descriptor,cut_off=3.7):
    av_descriptor = np.zeros((descriptor.shape[0],descriptor.shape[1]))
    for l in range(av_descriptor.shape[0]):
        mask_nn = nn_distances[l] <= cut_off
        av_descriptor[l] = descriptor[mask_nn].mean(axis=0)
    return av_descriptor
