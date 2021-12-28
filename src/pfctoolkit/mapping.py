# Tools to perform Lesion Network Mapping with the Precomputed Connectome
# William Drew 2021 (wdrew@bwh.harvard.edu)

import os, shutil
import numpy as np
from nilearn import image
from pfctoolkit import tools, datasets

def process_chunk(chunk, rois, config):
    """Compute chunk contribution to FC maps for a given list of ROIs.

    Parameters
    ----------
    chunk : int
        Index of chunk to be processed.
    rois : list of str
        List of ROI paths to be processed with the chunk.

    Returns
    -------
    contributions : dict of ndarray
        Dictionary containing contributions to network maps.

    """
    brain_masker = tools.NiftiMasker(datasets.get_img(config.get("mask")))
    chunk_masker = tools.NiftiMasker(image.math_img(f"img=={chunk}", 
                                     img=config.get("chunk_idx")))
    brain_weights = [brain_masker.transform(roi) for roi in rois]
    chunk_weights = [chunk_masker.transform(roi) for roi in rois]
    brain_masks = [(weights != 0) for weights in brain_weights]
    chunk_masks = [(weights != 0) for weights in chunk_weights]
    norm_weight = chunk_masker.transform(config.get("norm"))
    std_weight = chunk_masker.transform(config.get("std"))
    norm_weighted_chunk_masks = np.multiply(chunk_weights, norm_weight)
    std_weighted_chunk_masks = np.multiply(chunk_weights, std_weight)
    contributions = {}
    for chunk_type in [("avgr", "AvgR"), 
                       ("fz", "AvgR_Fz"), 
                       ("t", "T"), 
                       ("combo", "Combo"),
                      ]:
        chunk_data = np.load(os.path.join(config.get(chunk_type[0]), 
                                          f"{chunk}_{chunk_type[1]}.npy"))
        if(chunk_data.shape != (config.get('chunk_size'), 
                                config.get('brain_size'))):
            raise TypeError("Chunk expected to have shape {(config.get('chunk_size'), config.get('brain_size'))} but instead has shape {np.shape(chunk_data)}!")
        if(chunk_type[0] == "Combo"):
            numerator = np.sum(norm_weighted_chunk_masks, axis = 1)
            denominator = []
            for i in range(len(rois)):
                chunk_masked = np.multiply(np.reshape(chunk_weights[i][chunk_masks[i]], 
                                           (-1,1)), 
                                           chunk_data[chunk_masks[i],:])
                brain_masked = np.multiply(brain_weights[i][brain_masks[i]], 
                                           chunk_masked[:,brain_masks[i]])
                denominator.append(np.sum(brain_masked))
            contributions[chunk_type[0]] = {
                "numerators": numerator,
                "denominators": denominator
            }
        else:
            network_maps = np.matmul(std_weighted_chunk_masks, chunk_data)
            network_weights = np.sum(std_weighted_chunk_masks, axis = 1)
            contributions[chunk_type[0]] = {
                "maps": network_maps,
                "weights": network_weights
            }