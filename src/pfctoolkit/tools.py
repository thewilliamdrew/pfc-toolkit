# Tools, Functions, and Classes for working with the Precomputed Connectome
# Author: William Drew <wdrew@bwh.harvard.edu>

import os
import csv
import numpy as np
from glob import glob
from nilearn import image

def load_roi(roi_path):
    """Load ROIs from path or csv.

    Parameters
    ----------
    roi_path (str):
        Path to CSV containing paths to NIfTI images OR
        Path to directory containing NIfTI images OR
        Path to NIfTI image
    
    Returns
    -------
    roi_paths (list of str):
        List of paths to NIfTI image ROIs
    
    """
    if os.path.isdir(roi_path):
        roi_paths = glob(os.path.join(os.path.abspath(roi_path),"*.nii*"))
        if(len(roi_paths) == 0):
            raise FileNotFoundError("No NIfTI images found!")
    else:
        roi_extension = os.path.basename(roi_path).split('.')[1:]
        if 'csv' in roi_extension:
            with open(roi_path, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
            roi_paths = [path for line in data for path in line]
        elif 'nii' in roi_extension:
            roi_paths = [roi_path]
        else:
            raise ValueError("Input File is not a NIfTI or a CSV containing \
                              paths to a list of NIfTIs")
    print(f"Found {len(roi_paths)} ROIs...")
    return roi_paths

def get_chunks(rois, config):
    """Get list of dicts containing Chunks to load and their associated ROIs

    Parameters
    ----------
    rois : list of str
        List of paths to ROIs to find chunks for.
    config : Config
        Configuration object.

    Returns
    -------
    dict of dicts
        Dict of dicts containing Chunk path and list of associated ROI paths.

    """
    chunk_dict = {}
    chunk_map = image.load_img(config.get('chunk_idx'))
    for roi in rois:
        roi_image = image.load_img(roi)
        roi_chunks = image.math_img("img * mask", img = roi_image, 
                                    mask = chunk_map).get_fdata()
        chunks = np.unique(roi_chunks).astype(int)
        chunks = chunks[chunks != 0]
        for chunk in chunks:
            if chunk in chunk_dict:
                chunk_dict[chunk].append(roi)
            else:
                chunk_dict[chunk] = [roi]
    return chunk_dict


class NiftiMasker:
    """A Faster NiftiMasker.
    Attributes:
        mask_img (nibabel.nifti1.Nifti1Image): Nifti binary mask.
        mask_idx (numpy.ndarray): 1D numpy.ndarray containing indexes from 
            flattened mask image.
        mask_shape (3-tuple of ints): Shape of mask_idx.
        mask_size (int): Number of voxels in entire space, including
            outside the brain.
        
    """
    def __init__(self, mask_img = None):
        """
        Args:
            mask_img (str): File path to brain mask Nifti file.
        """
        self.mask_img = image.load_img(mask_img)
        mask_data = self.mask_img.get_fdata()
        self.mask_idx, = np.where(mask_data.flatten())
        self.mask_shape = mask_data.shape
        self.mask_size = np.prod(self.mask_shape)

    def transform(self, niimg = None):
        """Masks 3D Nifti file into 1D array.
        Args:
            niimg (nibabel.nifti1.Nifti1Image): Nifti to transform. 
        Returns:
            region_signals (1D numpy.ndarray): Masked Nifti file.
        """
        return np.take(image.get_data(niimg).flatten(), self.mask_idx)

    def inverse_transform(self, flat_niimg = None):
        """Unmasks 1D array into 3D Nifti file.
        Args:
            flat_niimg (1D numpy.ndarray): 1D array to unmask. 
        Returns:
            niimg (nibabel.nifti1.Nifti1Image): Unmasked Nifti.
        """
        new_img = np.zeros(self.mask_size)
        new_img[self.mask_idx] = flat_niimg
        return image.new_img_like(self.mask_img, 
                                  new_img.reshape(self.mask_shape))

    def mask(self, niimg = None):
        """Masks 3D Nifti file into Masked 3D Nifti file.
        Args:
            niimg (nibabel.nifti1.Nifti1Image): Nifti to mask. 
        Returns:
            masked_niimg (nibabel.nifti1.Nifti1Image): Masked Nifti file.
        """
        return self.inverse_transform(self.transform(niimg))