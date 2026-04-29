import numpy as np

def autocorrelation_1d(signal, timestep = 1, cutoff=None):
    """
    Compute 1D autocorrelation of a signal with optional cutoff,
    avoiding full memory expansion.

    Parameters:
    - signal (np.ndarray): Input 1D array (signal).
    - cutoff (int, optional): Maximum lag to compute autocorrelation for.

    Returns:
    - taus (np.ndarray): Array of timelag values.
    - autocorr (np.ndarray): Autocorrelation values at corresponding lags.
    """
    signal = np.asarray(signal)
    n = len(signal)

    # Normalize signal
    signal = signal - np.mean(signal)

    if cutoff is None:
        cutoff = n  # full autocorrelation

    cutoff = min(cutoff, n)

    autocorr = np.empty(cutoff, dtype=np.float64)

    # Compute autocorrelation for each lag manually
    for lag in range(cutoff):
        autocorr[lag] = np.dot(signal[:n - lag], signal[lag:]) / (n - lag)

    taus = np.arange(cutoff) * timestep
    return taus, autocorr

