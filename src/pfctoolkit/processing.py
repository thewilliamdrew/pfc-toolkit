"""
Tools to generate a Precomputed Functional Connectome.

"""

import numpy as np
import time
from numba import jit
import multiprocessing as mp


@jit(nopython=True)
def extract_chunk_signals(connectome_mat, roi_mat):
    """Extract Chunk TC Signal from a connectome subject.

    Parameters
    ----------
    connectome_mat : ndarray
        Connectome subject TC matrix.
    roi_mat : ndarray
        Chunk ROI binary mask

    Returns
    -------
    roi_masked_tc : ndarray
        TC signals extracted from connectome TC according to ROI binary mask
    
    """
    roi_masked_tc = connectome_mat[:, roi_mat > 0]
    return roi_masked_tc

# def make_fz_maps(connectome_files, roi_mat, result_queue):
def make_fz_maps(connectome_files, roi_mat):
    """Make Fz Maps for a chunk ROI and a connctome subject.

    Parameters
    ----------
    connectome_files : (str, str)
        Tuple of paths for connectome subject TC matrix and TC norm vector.
    roi_mat : ndarray
        Brain-masked binary chunk ROI with shape (<size of brain>,)
    result_queue : mp.Queue
        Queue to push result maps to.
        
    """
    connectome_mat = np.load(connectome_files[0]).astype(np.float32)
    connectome_norms_mat = np.load(connectome_files[1]).astype(np.float32)
    chunk_tc = extract_chunk_signals(connectome_mat, roi_mat)
    corr_num = matmul(connectome_mat.T, chunk_tc)
    corr_denom = matmul(connectome_norms_mat.reshape(-1,1),
                        np.linalg.norm(chunk_tc, axis=0).reshape(1,-1))
    np.seterr(invalid='ignore')
    corr = divide(corr_num, corr_denom)
    corr[np.isnan(corr)] = 0
    fz = arctanh(corr)
    # Fix infinite values in the case of single voxel autocorrelations
    finite_max = np.amax(np.ma.masked_invalid(fz), 1).data
    while(np.isinf(np.sum(fz))):
        fz[range(fz.shape[0]), np.argmax(fz, axis=1)] = finite_max
    # result_queue.put(fz)
    return fz

@jit(nopython=True)
def matmul(a, b):
    """Numba wrapped np.matmul.

    Parameters
    ----------
    a : ndarray
        Matrix to multiply.
    b : ndarray
        Matrix to multiply.

    Returns
    -------
    ndarray
        Matrix product.
    """
    return np.dot(a,b)

@jit(nopython=True)
def divide(a, b):
    """Numba wrapped np.divide

    Parameters
    ----------
    a : ndarray
        Matrix as numerator.
    b : ndarray
        Matrix as denominator.

    Returns
    -------
    ndarray
        Element-wise quotient.

    """
    return np.divide(a, b)

@jit(nopython=True)
def arctanh(a):
    """Numba wrapped np.arctanh

    Parameters
    ----------
    a : ndarray
        Matrix as input.

    Returns
    -------
    ndarray
        Element-wise arctanh.

    """
    return np.arctanh(a)

@jit(nopython=True)
def welford_update_map(count, existingAggregateMap, newMap):
    """Update a Welford map with data from a new map. See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Parameters
    ----------
    count : int
        Welford counter.
    existingAggregateMap : ndarray, shape (<brain size>, <ROI size>, 2)
        An existing Welford map.
    newMap : ndarray, shape (<brain size>, <ROI size>)
        New data to incorporate into a Welford map.
        

    Returns
    -------
    updatedAggregateMap : ndarray
        Updated Welford map.

    """
    mean = existingAggregateMap[:,:,0]
    M2 = existingAggregateMap[:,:,1]
    count += 1
    delta = np.subtract(newMap, mean)
    mean += np.divide(delta, count)
    delta2 = np.subtract(newMap, mean)
    M2 += np.multiply(delta, delta2)
    return count, np.stack([mean, M2], axis=2)

@jit(nopython=True)
def welford_finalize_map(count, existingAggregateMap):
    """Convert a Welford map into maps of arrays containing the statistics 
    [mean, variance, sampleVariancce].

    Parameters
    ----------
    count : int
        Welford counter.
    existingAggregateMap : ndarray
        Map of Welford tuples.

    Returns
    -------
    finalWelford : ndarray
        Array of Welford means, variances, and sample variances.

    """
    mean = existingAggregateMap[:,:,0]
    M2 = existingAggregateMap[:,:,1]
    variance = np.divide(M2, count)
    sampleVariance = np.divide(M2, count - 1)
    return np.stack([mean, variance, sampleVariance], axis=2)

def generate_welford_maps_from_queue(result_queue, welford_maps_queue):
    """Creates welford maps from a queue filled with fz maps from make_fz_maps.

    Parameters
    ----------
    result_queue : mp.Queue
        Queue filled with fz maps from make_fz_maps().
    welford_maps_queue : mp.Queue
        Queue onto which welford maps generated are pushed.

    """
    pass

def make_stat_maps(fz_welford_maps, output_dir):
    """Generate statistical maps from welford maps and output to file.

    Parameters
    ----------
    fz_welford_maps : ndarray
        Welford maps from generate_welford_maps_from_queue.
    output_dir : str
        Path to output directory.

    """
    pass

def precomputed_connectome_chunk(chunk_idx_mask,
                                 chunk_idx,
                                 connectome_dir,
                                 output_dir,
                                 workers = 8):
    """Generate a precomputed connectome chunk (AvgR/Fz/T)

    Parameters
    ----------
    chunk_idx_mask : Niimg-like object
        Mask containing voxel-wise chunk labels.
    chunk_idx : int
        Index of chunk to process.
    connectome_dir : str
        Path to individual subject connectome files.
    output_dir : str
        Path to output directory.
    workers : int, optional
        Number of workers, by default 8

    """
    pass
    