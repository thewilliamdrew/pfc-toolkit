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
    config : pfctoolkit.config.Config
        Config object containing precomputed connectome configuration.

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
        if(chunk_type[0] == "combo"):
            numerator = np.sum(norm_weighted_chunk_masks, axis = 1)
            for i, roi in enumerate(rois):
                chunk_masked = np.multiply(np.reshape(chunk_weights[i][chunk_masks[i]], 
                                           (-1,1)), 
                                           chunk_data[chunk_masks[i],:])
                brain_masked = np.multiply(brain_weights[i][brain_masks[i]], 
                                           chunk_masked[:,brain_masks[i]])
                denominator = np.sum(brain_masked)
                contributions[roi]["numerator"] = numerator[i]
                contributions[roi]["denominator"] = denominator
        else:
            network_maps = np.matmul(std_weighted_chunk_masks, chunk_data)
            for i, roi in enumerate(rois):
                if(chunk_type[0] == "avgr"):
                    contributions[roi] = {
                        chunk_type[0]: network_maps[i,:],
                    }
                else:
                    contributions[roi][chunk_type[0]] = network_maps[i,:]
        network_weights = np.sum(std_weighted_chunk_masks, axis = 1)
        for i, roi in enumerate(rois):
            contributions[roi]["network_weight"] = network_weights[i]
    return contributions

def consolidate(contribution, atlas):
    """Consolidate chunk contributions into running in-progress FC map atlas.

    Parameters
    ----------
    contribution : dict
        dict containing FC and scaling factor contributions from a chunk.
    atlas : dict
        dict containing in-progress FC maps and scaling factor accumulators.

    Returns
    -------
    atlas : dict
        Updated dict containing in-progress FC maps and scaling factor
        accumulators following consolidation of the contribution.

    """
    for roi in contribution.keys():
        if roi in atlas:
            atlas[roi]["avgr"] += contribution[roi]["avgr"]
            atlas[roi]["fz"] += contribution[roi]["fz"]
            atlas[roi]["t"] += contribution[roi]["t"]
            atlas[roi]["network_weight"] += contribution[roi]["network_weight"]
            atlas[roi]["numerator"] += contribution[roi]["numerator"]
            atlas[roi]["denominator"] += contribution[roi]["denominator"]
        else:
            atlas[roi] = {
                "avgr": contribution[roi]["avgr"],
                "fz": contribution[roi]["fz"],
                "t": contribution[roi]["t"],
                "network_weight": contribution[roi]["network_weight"],
                "numerator": contribution[roi]["numerator"],
                "denominator": contribution[roi]["denominator"],
            }
    return atlas