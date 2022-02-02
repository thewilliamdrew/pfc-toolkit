"""Tools for Surface Gifti files

"""

import os
import csv
import numpy as np
import nibabel as nib
from pfctoolkit import datasets


class GiftiMasker:
    def __init__(self, mask_img):
        """

        Parameters
        ----------
        mask_img : Giimg-like object
            If str, path to binary gifti mask. If giftiimg, gifti image object. Assumes
            the gifti only contains a single 1-D data array.

        """
        if type(mask_img) == str:
            self.mask_img = nib.load(mask_img)
        else:
            self.mask_img = mask_img
        self.mask_data = self.mask_img.agg_data()
        (self.mask_idx,) = np.where(self.mask_data != 0)
        self.mask_shape = self.mask_data.shape
        self.mask_size = np.prod(self.mask_shape)

    def transform(self, giimg=None, weight=False):
        """Masks Gifti file into 1D array. Retypes to float32.

        Parameters
        ----------
        giimg : Giimg-like object
            If string, consider it as a path to GIfTI image and call `nibabel.load()` on
            it. The '~' symbol is expanded to the user home folder. If it is an object,
            check if affine attribute is present, raise `TypeError` otherwise. If
            ndarray, consider it as a gifti data array.
        weight : bool, default False
            If True, transform the niimg with weighting. If False, transform the niimg
            without weighting.

        Returns
        -------
        region_signals : numpy.ndarray
            Masked Nifti file.

        """
        if type(giimg) == str:
            giimg_data = nib.load(giimg).agg_data().astype(np.float32)
        elif type(giimg) == np.ndarray:
            giimg_data = giimg
        else:
            giimg_data = giimg.agg_data().astype(np.float32)
        if weight:
            img = np.multiply(self.mask_data, giimg_data)
        else:
            img = giimg_data
        return np.take(img, self.mask_idx)

    def inverse_transform(self, flat_giimg=None):
        """Unmasks 1D array into 3D Gifti file. Retypes to float32.

        Parameters
        ----------
        flat_giimg : numpy.ndarray
            1D array to unmask.

        Returns
        -------
        giimg : nibabel.gifti.gifti.GiftiImage
            Unmasked Gifti.

        """
        new_img = np.zeros(self.mask_size, dtype=np.float32)
        new_img[self.mask_idx] = flat_giimg.astype(np.float32)
        return new_gifti_image(data=new_img)

    def mask(self, giimg=None):
        """Masks 3D Gifti file into Masked Gifti file.

        Parameters
        ----------
        giimg : Giimg-like object
            If string, consider it as a path to GIfTI image and call `nibabel.load()` on
            it. The '~' symbol is expanded to the user home folder. If it is an object,
            check if affine attribute is present, raise `TypeError` otherwise.

        Returns
        -------
        masked_giimg : nibabel.gifti.gifti.GiftiImage
            Masked Gifti image.

        """
        return self.inverse_transform(self.transform(giimg))


def new_gifti_image(data, intent=0, datatype=16, metadata=None):
    """NiBabel wrapper to generate a gifti image with data array and metadata.

    Parameters
    ----------
    data : ndarray
        1-D ndarray containing one hemisphere surface data.
    intent : int
        Intent code for Gifti File. Defaults to 0 (Intent = NONE).\n
        Available intent codes:\n
        NIFTI_INTENT_NONE - 0\n
        NIFTI_INTENT_CORREL - 2\n
        NIFTI_INTENT_TTEST - 3\n
        NIFTI_INTENT_ZSCORE - 5\n
        NIFTI_INTENT_PVAL - 22\n
        NIFTI_INTENT_LOGPVAL - 23\n
        NIFTI_INTENT_LOG10PVAL - 24\n
        NIFTI_INTENT_LABEL - 1002\n
        NIFTI_INTENT_POINTSET - 1008\n
        NIFTI_INTENT_TRIANGLE - 1009\n
        NIFTI_INTENT_TIME_SERIES - 2001\n
        NIFTI_INTENT_NODE_INDEX - 2002\n
        NIFTI_INTENT_SHAPE - 2005\n
        More intent codes can be found at: https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/group__NIFTI1__INTENT__CODES.html
    datatype : int
        Datatype for gifti image. Defaults to 16 (dtype = float32)\n
        Available datatypes:\n
        UINT8 - 2\n
        INT32 - 8\n
        FLOAT32 - 16\n
    metadata : dict
        Dictionary of metadata for gifti image.

    Returns
    -------
    nibabel.gifti.gifti.GiftiImage
        Gifti image with specified metadata and data array.

    """
    dtypes = {2: np.uint8, 8: np.int32, 16: np.float32}
    data = data.astype(dtypes[datatype])
    if metadata:
        metadata = nib.gifti.GiftiMetaData.from_dict(metadata)
    gifti_data = nib.gifti.GiftiDataArray(data=data, intent=intent, datatype=datatype)
    gifti_img = nib.gifti.GiftiImage(meta=metadata, darrays=[gifti_data])
    return gifti_img


def concat_hemispheres_to_csv(gifti_paths, output_dir="", mask=""):
    """Concatenate a list of giftis together into a csv to construct a data matrix for
    use with PALM. Assumes that input giftis contain a single 1-D data array with
    functional/statistical data for surface mesh vertices.

    Parameters
    ----------
    gifti_paths : str
        Path to a two-column CSV of paths to giftis where each row corresponds to a
        subject, the first column corresponds to the subject's Left Hemisphere ROI mask,
        and the second column corresponds to the subject's Right Hemisphere ROI mask.\n
        CSV Format:\n
        /path/to/subject1/lh_roi.nii.gz,/path/to/subject1/rh_roi.nii.gz\n
        /path/to/subject2/lh_roi.nii.gz,/path/to/subject2/rh_roi.nii.gz\n
        /path/to/subject3/lh_roi.nii.gz,/path/to/subject3/rh_roi.nii.gz\n
    output_dir : str, optional
        Output directory, by default "". If not specified, generated data matrix csv
        will be output to same directory as input csv.
    mask : str, {'fs5_mask'}, optional
        Mask name (from nimlab.datasets). Defaults to no masking. The provided
        fsaverage5 mask is created from Ryan Darby's ADNI data.

    """
    roi_files = []
    flist = open(gifti_paths)
    reader = csv.reader(flist, delimiter=",")
    for f in reader:
        roi_files.append(f)
    flist.close()
    if mask:
        masker = GiftiMasker(datasets.get_img(mask))
    if output_dir == "":
        output_dir = os.path.dirname(gifti_paths)
    subject_data = []
    for subject in roi_files:
        lh = nib.load(subject[0]).agg_data()
        rh = nib.load(subject[1]).agg_data()
        data = np.concatenate((lh, rh))
        if mask:
            data = masker.transform(data)
        subject_data.append(data)
    fname = os.path.join(output_dir, "data.csv")
    np.savetxt(fname, np.stack(subject_data), delimiter=",")
