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

    Parameters
    ----------
    ds : str
        Name of image to get.\n
        Volume Masks:\n
        "MNI152_T1_2mm_brain"\n
        "MNI152_T1_2mm_brain_mask"\n
        "MNI152_T1_2mm_brain_mask_dil"\n
        "MNI152_T1_1mm_brain"\n
        "MNI152_T1_1mm_brain_mask"\n
        "MNI152_T1_1mm_brain_mask_dil"\n
        Surface Masks:\n
        "fs5_mask"\n
        "fs5_mask_lh"\n
        "fs5_mask_rh"\n

    Returns
    -------
    Niimg-like object

    """
    assert ds in datasets.keys(), "Unknown image specified"
    fname = datasets[ds]
    from . import data

    with pkg_resources.path(data, fname) as datafile:
        return nib.load(str(datafile))


def get_img_path(ds):
    """Get a standard image file path

    Parameters
    ----------
    ds : str
        Name of image to get path of.\n
        Volume Masks\n
        "MNI152_T1_2mm_brain"\n
        "MNI152_T1_2mm_brain_mask"\n
        "MNI152_T1_2mm_brain_mask_dil"\n
        "MNI152_T1_1mm_brain"\n
        "MNI152_T1_1mm_brain_mask"\n
        "MNI152_T1_1mm_brain_mask_dil"\n
        Surface Masks\n
        "fs5_mask"\n
        "fs5_mask_lh"\n
        "fs5_mask_rh"\n

    Returns
    -------
    str
        Path to image.

    """
    assert ds in datasets.keys(), "Unknown image specified"
    fname = datasets[ds]
    from . import data

    with pkg_resources.path(data, fname) as datafile:
        return str(datafile)
