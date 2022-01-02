"""
Tools to generate a Precomputed Functional Connectome.

"""

import numpy as np

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
    """Make Fz Maps for a chunk ROI and a connctome subject.

    Parameters
    ----------
    connectome_files : (str, str)
        Tuple of paths for connectome subject TC matrix and TC norm vector.
    roi_mat : ndarray
        Brain-masked binary chunk ROI with shape (<size of brain>,)

    Returns
    -------
    fz : ndarray
        Fz maps for chunk ROI with shape (<size of brain>, <size of ROI>)
        
    """
    connectome_mat = np.load(connectome_files[0]).astype(np.float32)
    connectome_norms_mat = np.load(connectome_files[1]).astype(np.float32)

    chunk_tc = extract_chunk_signals(connectome_mat, roi_mat)
    corr_num = np.matmul(connectome_mat.T, chunk_tc)
    corr_denom = np.matmul(connectome_norms_mat.reshape(-1,1),np.linalg.norm(chunk_tc, axis=0).reshape(1,-1))
    np.seterr(invalid='ignore')
    corr = np.divide(corr_num, corr_denom)
    corr[np.isnan(corr)] = 0

    fz = np.arctanh(corr)
    # # Fix infinite values in the case of single voxel autocorrelations
    for i in range(fz.shape[1]):
        finite_max = fz[:,i][np.isfinite(fz[:,i])].max()
        fz[:,i][np.isinf(fz[:,i])] = finite_max
    return fz


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
    