# Tools, Functions, and Classes for working with the Precomputed Connectome
# Author: William Drew <wdrew@bwh.harvard.edu>

import os
import numpy as np
from nilearn import image

def binarize(niimg = None, standard = None):
    """Binarize a Nifti.
    Args:
        niimg (nibabel.nifti1.Nifti1Image): Nifti to binarize.
        standard (nibabel.nifti1.Nifti1Image): Standard Nifti template.
    Returns:
        binary_niimg (nibabel.nifti1.Nifti1Image): Binarized Nifti.
    """
    return image.new_img_like(standard, niimg.get_fdata() > 0)

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