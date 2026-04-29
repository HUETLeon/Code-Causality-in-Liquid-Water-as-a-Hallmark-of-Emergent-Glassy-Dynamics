import numpy as np

def autocorrelation_1d(signal, timestep = 1, cutoff=None):
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
    #print("intputs: signal, timestep, cutoff")
    #print(signal, timestep, cutoff)

    signal = np.asarray(signal)
    n = len(signal)

    #print("shape of signal 1: ", signal.shape)

    # Normalize signal
    signal = signal - np.mean(signal, axis=0)
    
    #print("shape of signal 2: ", signal.shape)
    
    if cutoff is None:
        cutoff = n  # full autocorrelation

    autocorr = np.empty(cutoff, dtype=np.float64)
    std = np.empty(cutoff, dtype=np.float64)
    if (cutoff > n):
        print("Warrning, the cutoff is larger than the signal, therefor cufof is truncated")
    cutoff = min(cutoff, n)

    # Compute autocorrelation for each lag manually
    
    #print("cutoff :" , cutoff)
    #print("n :", n)
    #print("shape of autocorr: ", autocorr.shape)

    for lag in range(cutoff):
        autocorr[lag] = np.einsum('ik,ik->', signal[:n-lag], signal[lag:]) / (n - lag)
        std[lag] = np.einsum('ik,ik-> i', signal[:n-lag], signal[lag:]).std()/np.sqrt(n - lag)

    taus = np.arange(len(autocorr))*timestep
    
    #print("shape of autocorr 2: ", autocorr.shape)
    #print("shape of taus: ", taus.shape)

    return taus, autocorr, std

