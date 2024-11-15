"""
Useful utilities for working with the Precomputed Connectome

"""

import os
import csv
import numpy as np
from tqdm import tqdm
from glob import glob
from nilearn import image
from nilearn._utils import check_niimg
from pfctoolkit import datasets, surface

import environ
import nibabel as nib
import gzip
from io import BytesIO
import boto3
from botocore.client import Config

env = environ.Env()

ACCESS_KEY_ID = env('S3_ACCESS_KEY_ID', default='default_access_key_id')
SECRET_ACCESS_KEY = env('S3_SECRET_ACCESS_KEY', default='default_secret_access_key')
STORAGE_BUCKET_NAME = env('S3_BUCKET_NAME', default='default_bucket_name')
S3_ENDPOINT_URL = env('S3_ENDPOINT_URL', default='https://default.endpoint.url')
S3_LOCATION = env('S3_LOCATION', default='nyc3')


def load_roi(roi_path):
    """Load ROIs from path or CSV.

    Parameters
    ----------
    roi_path : str
        Path to CSV containing paths to NIfTI images OR Path to directory containing
        NIfTI images OR Path to NIfTI image.

    Returns
    -------
    roi_paths : list of str
        List of paths to NIfTI image ROIs.

    """
    if type(roi_path) == nib.nifti1.Nifti1Image:
        roi_paths = [roi_path]
    elif os.path.isdir(roi_path):
        roi_paths = glob(os.path.join(os.path.abspath(roi_path), "*.nii*"))
        if len(roi_paths) == 0:
            raise FileNotFoundError("No NIfTI images found!")
    else:
        roi_extension = os.path.basename(roi_path).split(".")[1:]
        if "csv" in roi_extension:
            with open(roi_path, newline="") as f:
                reader = csv.reader(f)
                data = list(reader)
            roi_paths = [path for line in data for path in line]
        elif "nii" in roi_extension:
            roi_paths = [roi_path]
        else:
            raise ValueError(
                "Input File is not a NIfTI or a CSV containing paths to NIfTIs"
            )
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
    chunk_map = image.load_img(config.get("chunk_idx")) if not config.get("use_s3", False) else fetch_from_s3(config.get("chunk_idx"))
    for roi in tqdm(rois, desc='Getting chunks'):
        if type(roi) == nib.nifti1.Nifti1Image:
            roi_image = roi
        elif config.get("use_s3", False) or roi.startswith("s3://"):
            roi_image = fetch_from_s3(roi)
        else:
            roi_image = image.load_img(roi)
        bin_roi_image = image.math_img("img != 0", img=roi_image)
        roi_chunks = image.math_img(
            "img * mask", img=bin_roi_image, mask=chunk_map
        ).get_fdata()
        chunks = np.unique(roi_chunks).astype(int)
        chunks = chunks[chunks != 0]
        for chunk in chunks:
            if chunk in chunk_dict:
                chunk_dict[chunk].append(roi)
            else:
                chunk_dict[chunk] = [roi]
    return chunk_dict


def get_voxel_conn_map(x, y, z, map_type, config):
    """Get connectivity map for single voxel

    Parameters
    ----------
    x : int
        X-coordinate for voxel to retrieve connectivity map for.
    y : int
        Y-coordinate for voxel to retrieve connectivity map for.
    z : int
        Z-coordinate for voxel to retrieve connectivity map for.
    map_type : str
        Type of connectivity map to retrieve: "t", "avgr", or "fz".
    config : Config
        Configuration object.

    Returns
    -------
    Nifti1Image
        Nifti image object.

    """
    chunk_paths = {
        "avgr": config.get("avgr"),
        "fz": config.get("fz"),
        "t": config.get("t"),
        "combo": config.get("combo"),
    }
    chunk_type = {
        "avgr": "AvgR",
        "fz": "AvgR_Fz",
        "t": "T",
        "combo": "Combo",
    }
    chunk_map = image.load_img(config.get("chunk_idx"))
    brain_mask = datasets.get_img(config.get("mask"))
    affine = brain_mask.affine
    mni_coordinate = np.array([[x, y, z]])
    voxel_coordinate = (
        np.linalg.inv(affine).dot(np.append(mni_coordinate, 1).T).astype(int)[:3]
    )

    voxel_data = np.zeros(brain_mask.shape)
    voxel_data[voxel_coordinate[0], voxel_coordinate[1], voxel_coordinate[2]] = 1
    voxel_img = image.new_img_like(brain_mask, voxel_data)

    roi_chunks = image.math_img("img * mask", img=voxel_img, mask=chunk_map).get_fdata()
    chunks = np.unique(roi_chunks).astype(int)
    chunk = chunks[chunks != 0][0]

    image_type = config.get("type")
    if image_type == "volume":
        brain_masker = NiftiMasker(datasets.get_img(config.get("mask")))
        chunk_masker = NiftiMasker(
            image.math_img(f"img=={chunk}", img=config.get("chunk_idx"))
        )
    elif image_type == "surface":
        brain_masker = surface.GiftiMasker(datasets.get_img(config.get("mask")))
        chunk_masker = surface.GiftiMasker(
            surface.new_gifti_image(
                datasets.get_img(config.get("chunk_idx")).agg_data() == chunk
            )
        )
    if config.get("use_s3", False):
        chunk_data = np.load(
            fetch_from_s3(
                os.path.join(chunk_paths[map_type], f"{chunk}_{chunk_type[map_type]}.npy")
            )
        )
    else:
        chunk_data = np.load(
            os.path.join(chunk_paths[map_type], f"{chunk}_{chunk_type[map_type]}.npy")
        )
    voxel_connectivity = np.dot(chunk_masker.transform(voxel_img), chunk_data)
    voxel_connectivity_img = brain_masker.inverse_transform(voxel_connectivity)

    return voxel_connectivity_img


class NiftiMasker:
    """A Faster NiftiMasker.

    Attributes
    ----------
    mask_img : nibabel.nifti1.Nifti1Image
        Nifti binary mask.
    mask_idx : numpy.ndarray
        1D numpy.ndarray containing indexes from flattened mask image.
    mask_shape : tuple of ints
        Shape of mask_idx.
    mask_size : int
        Number of voxels in entire space, including outside the brain.

    """

    def __init__(self, mask_img=None):
        """

        Parameters
        ----------
        mask_img : Niimg-like object
            If string, consider it as a path to NIfTI image and call `nibabel.load()` on
            it. The '~' symbol is expanded to the user home folder. If it is an object,
            check if affine attribute is present, raise `TypeError` otherwise.

        """
        self.mask_img = check_niimg(mask_img)
        self.mask_data = self.mask_img.get_fdata().astype(np.float32)
        (self.mask_idx,) = np.where((self.mask_data != 0).flatten())
        self.mask_shape = self.mask_data.shape
        self.mask_size = np.prod(self.mask_shape)

    def transform(self, niimg=None, weight=False):
        """Masks 3D Nifti file into 1D array. Retypes to float32.

        Parameters
        ----------
        niimg : Niimg-like object
            If string, consider it as a path to NIfTI image and call `nibabel.load()` on
            it. The '~' symbol is expanded to the user home folder. If it is an object,
            check if affine attribute is present, raise `TypeError` otherwise.
        weight : bool, default False
            If True, transform the niimg with weighting. If False, transform the niimg
            without weighting.

        Returns
        -------
        region_signals : numpy.ndarray
            Masked Nifti file.

        """
        niimg = np.nan_to_num(check_niimg(niimg).get_fdata()).astype(np.float32)
        if weight:
            img = np.multiply(self.mask_data, niimg)
        else:
            img = niimg
        return np.take(img.flatten(), self.mask_idx)

    def inverse_transform(self, flat_niimg=None):
        """Unmasks 1D array into 3D Nifti file. Retypes to float32.

        Parameters
        ----------
        flat_niimg : numpy.ndarray
            1D array to unmask.

        Returns
        -------
        niimg : nibabel.nifti1.Nifti1Image
            Unmasked Nifti.

        """
        new_img = np.zeros(self.mask_size, dtype=np.float32)
        new_img[self.mask_idx] = flat_niimg.astype(np.float32)
        return image.new_img_like(self.mask_img, new_img.reshape(self.mask_shape))

    def mask(self, niimg=None):
        """Masks 3D Nifti file into Masked 3D Nifti file.

        Parameters
        ----------
        niimg : Niimg-like object
            If string, consider it as a path to NIfTI image and call `nibabel.load()`
            on it. The '~' symbol is expanded to the user home folder. If it is an
            object, check if affine attribute is present, raise `TypeError` otherwise.

        Returns
        -------
        masked_niimg : nibabel.nifti1.Nifti1Image
            Masked Nifti file.

        """
        return self.inverse_transform(self.transform(niimg))


def get_s3_client():
    """Create and return a boto3 client configured for DigitalOcean Spaces"""  
    session = boto3.session.Session()
    client = session.client(
        's3',
        config=Config(s3={'addressing_style': 'virtual'}),
        region_name=S3_LOCATION,
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
    )
    
    return client


def fetch_from_s3(filepath):
    """
    Fetch and load files from DigitalOcean Spaces using boto3
    
    Parameters:
    filepath (str): Path to file within the bucket
    
    Returns:
    Various: Loaded file data depending on the file type
    """

    # First, check if the file is already loaded
    if type(filepath) ==  nib.Nifti1Image or type(filepath) == nib.GiftiImage or type(filepath) == np.ndarray:
        return filepath
    
    # Next, check if the file exists locally
    if os.path.exists(filepath):
        if filepath.endswith(('.nii.gz', '.nii')):
            return nib.load(filepath)
        else:
            return np.load(filepath)

    # Finally, fetch the file from S3 given that we can't find it any other way
    # Get S3 client
    s3_client = get_s3_client()

    # Remove 's3://' prefix if present
    if filepath.startswith('s3://'):
        filepath = filepath.replace('s3://', '', 1)
    
    # Get the file from S3
    try:
        response = s3_client.get_object(Bucket=STORAGE_BUCKET_NAME, Key=filepath)
        file_data = response['Body'].read()
    except s3_client.exceptions.NoSuchKey:
        print(f"Error: The key '{filepath}' does not exist in bucket '{STORAGE_BUCKET_NAME}'.")
        raise
    except Exception as e:
        print(f"Unexpected error fetching file from S3: {str(e)}")
        raise

    # Load the file based on its extension
    try:
        if filepath.endswith('.nii.gz'):
            fh = nib.FileHolder(fileobj=gzip.GzipFile(fileobj=BytesIO(file_data)))
            return nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        
        elif filepath.endswith('.nii'):
            fh = nib.FileHolder(fileobj=BytesIO(file_data))
            return nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        
        elif filepath.endswith(('.npy', '.npz')):
            return np.load(BytesIO(file_data), allow_pickle=True)
        
        elif filepath.endswith('.gii'):
            fh = nib.FileHolder(fileobj=BytesIO(file_data))
            return nib.GiftiImage.from_file_map({'image': fh})
        
        elif filepath.endswith('.gii.gz'):
            fh = nib.FileHolder(fileobj=gzip.GzipFile(fileobj=BytesIO(file_data)))
            return nib.GiftiImage.from_file_map({'image': fh})
        
        else:
            raise ValueError(f"Unsupported file type: {os.path.splitext(filepath)[1]}")
    except Exception as load_error:
        print(f"Error loading file '{filepath}': {str(load_error)}")
        raise