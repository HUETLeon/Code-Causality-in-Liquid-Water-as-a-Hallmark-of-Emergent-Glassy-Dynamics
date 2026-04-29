import numpy as np

def crosscorrelation_1d(signal1, signal2, timestep = 1, cutoff=None):
    """
    Compute 1D autocorrelation of a signal with optional cutoff,
    avoiding full memory expansion.

    Parameters:
    - signal (np.ndarray): Input 2D array (signal vectors).
    - cutoff (int, optional): Maximum lag to compute autocorrelation for.

    Returns:
    - taus (np.ndarray): Array of timelag values.
    - autocorr (np.ndarray): Autocorrelation values at corresponding lags.
    """
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)


    n = len(signal1)

    if (n != len(signal2)):
        print("WARNING: the array give to crosscorrelation_1d are not of the same size")

    # Normalize signal
    signal1 = signal1 - np.mean(signal1, axis=0)
    signal2 = signal2 - np.mean(signal2, axis=0)
    if cutoff is None:
        cutoff = n  # full autocorrelation
    crosscorr = np.empty(cutoff, dtype=np.float64)
    std = np.empty(cutoff, dtype=np.float64)
    cutoff = min(cutoff, n)


    # Compute autocorrelation for each lag manually
    for lag in range(cutoff):
        crosscorr[lag] = np.einsum('ik,ik->', signal2[:n - lag], signal1[lag:]) / (n - lag) / 2 + np.einsum('ik,ik->', signal1[:n - lag], signal2[lag:]) / (n - lag) / 2
        std[lag] =  ((np.einsum('ik,ik->i', signal2[:n - lag], signal1[lag:]) + np.einsum('ik,ik->i', signal1[:n - lag], signal2[lag:]))/2).std()/np.sqrt(n - lag) 

    taus = np.arange(len(crosscorr))*timestep
    return taus, crosscorr, std

