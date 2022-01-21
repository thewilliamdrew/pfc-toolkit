"""
Datasets used with the Precomputed Connectome

"""
import nibabel as nib
import importlib.resources as pkg_resources

datasets = {
    "MNI152_T1_2mm_brain": "MNI152_T1_2mm_brain.nii.gz",
    "MNI152_T1_2mm_brain_mask": "MNI152_T1_2mm_brain_mask.nii.gz",
    "MNI152_T1_2mm_brain_mask_dil": "MNI152_T1_2mm_brain_mask_dil.nii.gz",
    "MNI152_T1_1mm_brain": "MNI152_T1_1mm_brain.nii.gz",
    "MNI152_T1_1mm_brain_mask": "MNI152_T1_1mm_brain_mask.nii.gz",
    "MNI152_T1_1mm_brain_mask_dil": "MNI152_T1_1mm_brain_mask_dil.nii.gz",
    "fs5_mask": "fs5_mask.gii",
    "fs5_mask_lh": "fs5_mask_lh.gii",
    "fs5_mask_rh": "fs5_mask_rh.gii",
}


def get_img(ds):
    """Get a standard image file as a Niimg

    Args:
        ds (str) : Name of the image. Options are as follows:

            Volume Masks
            ------------
            "MNI152_T1_2mm_brain"
            "MNI152_T1_2mm_brain_mask"
            "MNI152_T1_2mm_brain_mask_dil"
            "MNI152_T1_1mm_brain"
            "MNI152_T1_1mm_brain_mask"
            "MNI152_T1_1mm_brain_mask_dil"

            Surface Masks
            -------------
            "fs5_mask"
            "fs5_mask_lh"
            "fs5_mask_rh"

    Returns:
        Niimg-like object
    """
    assert ds in datasets.keys(), "Unknown image specified"
    fname = datasets[ds]
    from . import data

    with pkg_resources.path(data, fname) as datafile:
        return nib.load(str(datafile))


def get_img_path(ds):
    """Get a standard image file path

    Args:
        ds (str) : Name of the image. Options are as follows:

            Volume Masks
            ------------
            "MNI152_T1_2mm_brain"
            "MNI152_T1_2mm_brain_mask"
            "MNI152_T1_2mm_brain_mask_dil"
            "MNI152_T1_1mm_brain"
            "MNI152_T1_1mm_brain_mask"
            "MNI152_T1_1mm_brain_mask_dil"

            Surface Masks
            -------------
            "fs5_mask"
            "fs5_mask_lh"
            "fs5_mask_rh"

    Returns:
        str : Path to image
    """
    assert ds in datasets.keys(), "Unknown image specified"
    fname = datasets[ds]
    from . import data

    with pkg_resources.path(data, fname) as datafile:
        return str(datafile)
