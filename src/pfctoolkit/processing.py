"""
Tools to generate a Precomputed Functional Connectome.

"""

import numpy as np
import time
import os
from numba import jit
import multiprocessing as mp
from pathlib import Path

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

def make_fz_maps(connectome_files, roi_mat):
    """Make Fz Maps for a chunk ROI and a connectome subject.

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

    """
    return np.arctanh(a)

@jit(nopython=True)
def tanh(a):
    """Numba wrapped np.tanh

    """
    return np.tanh(a)

def welford_update_map(count, mean, M2, newMap):
    """Update a Welford map with data from a new map. See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Parameters
    ----------
    count : int
        Welford counter.
    mean : ndarray, shape (<brain size>, <ROI size>)
        Mean aggregate.
    M2 : ndarray, shape (<brain size>, <ROI size>)
        M2 aggregate.
    newMap : ndarray, shape (<brain size>, <ROI size>)
        New data to incorporate into a Welford map.
        

    Returns
    -------
    updatedAggregateMap : ndarray
        Updated Welford map.

    """
    count += 1
    # mean, M2 = welford_update(count, newMap, mean, M2)
    start = time.time()
    delta = subtract(newMap, mean)
    print(f"Calculate delta: {time.time()-start} seconds")
    start = time.time()
    mean += divide(delta, count)
    print(f"Calculate mean: {time.time()-start} seconds")
    start = time.time()
    delta2 = subtract(newMap, mean)
    print(f"Calculate delta2: {time.time()-start} seconds")
    start = time.time()
    M2 += multiply(delta, delta2)
    print(f"Calculate M2: {time.time()-start} seconds")
    return count, mean, M2

@jit(nopython=True)
def subtract(a,b):
    return np.subtract(a,b)

@jit(nopython=True)
def multiply(a,b):
    return np.multiply(a,b)

@jit(nopython=True)
def sqrt(a):
    return np.sqrt(a)

@jit(nopython=True)
def welford_update(count, newMap, mean, M2):
    """Numba wrapper for welford_update_map

    """
    delta = np.subtract(newMap, mean)
    mean += np.divide(delta, count)
    delta2 = np.subtract(newMap, mean)
    M2 += np.multiply(delta, delta2)
    return mean, M2

def welford_finalize_map(count, mean, M2):
    """Convert a Welford map into maps of arrays containing the statistics 
    [mean, variance, sampleVariancce].

    Parameters
    ----------
    count : int
        Welford counter.
    mean : ndarray
        Welford Mean array.
    M2 : ndarray
        Welford M2 array.

    Returns
    -------
    finalWelford : ndarray
        Array of Welford means, variances, and sample variances.

    """
    sampleVariance = welford_finalize(count, M2)
    return mean, sampleVariance
    # return np.stack([mean, variance, sampleVariance], axis=2)

@jit(nopython=True)
def welford_finalize(count, M2):
    sampleVariance = np.divide(M2, count-1)
    return sampleVariance

def make_stat_maps(count, mean, M2, output_dir, chunk_idx):
    """Generate statistical maps from welford maps and output to file.

    Parameters
    ----------
    count : int
        Number of connectome subjects.
    mean : ndarray
        Welford Mean array.
    M2 : ndarray
        Welford M2 array.
    output_dir : str
        Path to output directory.
    chunk_idx : int
        Chunk index number being processed.

    """
    mean, sampleVariance = welford_finalize_map(count, mean, M2)
    map_types = ['AvgR_Fz', 'AvgR', 'T']
    output_dir = os.path.abspath(output_dir)
    output_dirs = {}
    for map_type in map_types:
        Path(os.path.join(output_dir, map_type)).mkdir(parents=True,
                                                       exist_ok=True)
        output_dirs[map_type] = os.path.join(output_dir,
                                             map_type,
                                             f"{chunk_idx}_{map_type}.npy")
    # Save AvgR_Fz chunk
    np.save(output_dirs['AvgR_Fz'], mean)
    # Save AvgR chunk
    np.save(output_dirs['AvgR'], tanh(mean))
    # Save T Chunk
    ttest_denom = sqrt(divide(sampleVariance, count))
    np.save(output_dirs['T'], divide(mean, ttest_denom))
    print("Output Chunk files to:")
    print(f"AvgR_Fz: {output_dirs['AvgR_Fz']}")
    print(f"AvgR: {output_dirs['AvgR']}")
    print(f"T: {output_dirs['T']}")

def precomputed_connectome_chunk(chunk_idx_mask,
                                 chunk_idx,
                                 connectome_dir,
                                 output_dir):
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
    