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
    

    """
    brain_masker = tools.NiftiMasker(datasets.get_img(config.get("mask")))
    chunk_masker = tools.NiftiMasker(image.math_img(f"img=={chunk}", 
                                     img=config.get("chunk_idx")))
    roi_brain_masks = [brain_masker.transform(roi) for roi in rois]
    roi_chunk_masks = [chunk_masker.transform(roi) for roi in rois]
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
        
        