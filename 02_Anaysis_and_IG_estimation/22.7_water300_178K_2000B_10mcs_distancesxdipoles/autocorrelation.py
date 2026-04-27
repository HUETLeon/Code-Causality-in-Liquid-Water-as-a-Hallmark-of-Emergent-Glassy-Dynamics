import numpy as np

def autocorrelation_1d(signal1, timestep=1, cutoff=None, n_blocks=10):
    """
    Compute 1D cross-correlation between two signals with block-averaged error estimation.

    Parameters
    ----------
    signal1, signal2 : np.ndarray
        Input arrays of shape (n, d), where n is number of time steps, d optional vector dimension.
    timestep : float, optional
        Time step between samples (default = 1).
    cutoff : int, optional
        Maximum lag to compute correlation for. If None, uses block size.
    n_blocks : int, optional
        Number of blocks for error estimation (default = 10).

    Returns
    -------
    taus : np.ndarray
        Array of lag times.
    crosscorr_mean : np.ndarray
        Mean cross-correlation as a function of lag.
    crosscorr_std : np.ndarray
        Estimated standard error of the cross-correlation.
    """

    signal1 = np.asarray(signal1, dtype=np.float64)
    signal2 = signal1.copy()
    assert signal1.shape == signal2.shape, "signal1 and signal2 must have the same shape"

    n = len(signal1)
    signal1 -= np.mean(signal1, axis=0)
    signal2 -= np.mean(signal2, axis=0)

    # Divide into blocks
    block_size = n // n_blocks


    # Ensure cutoff doesn't exceed block size
    if cutoff is None:
        cutoff = block_size
    
    crosscorr_blocks = np.zeros((n_blocks, cutoff))
    taus = np.arange(cutoff) * timestep
    
    cutoff = min(cutoff, block_size)


    for b in range(n_blocks):
        block1 = signal1[b * block_size:(b + 1) * block_size]
        block2 = signal2[b * block_size:(b + 1) * block_size]
        nb = len(block1)

        for lag in range(cutoff):
            crosscorr_blocks[b, lag] = np.einsum('ik,ik->', block1[:nb-lag], block2[lag:]) / (nb - lag)

    crosscorr_mean = np.mean(crosscorr_blocks, axis=0)
    crosscorr_std = np.std(crosscorr_blocks, axis=0, ddof=1) / np.sqrt(n_blocks)

    return taus, crosscorr_mean, crosscorr_std

