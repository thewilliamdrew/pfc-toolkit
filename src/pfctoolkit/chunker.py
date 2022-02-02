"""
Utilities to generate chunk masks used in the Precomputed Connectome

"""

import os
import numpy as np
from tqdm import trange
from nilearn import image
from scipy.spatial.distance import cityblock


def get_distances(home_coords, locations):
    """Get Manhattan distances of voxels to home coordinates.

    Parameters
    ----------
    home_coords : (int, int, int)
        X/Y/Z coordinates of center of search region.
    locations : ndarray
        2D array of X/Y/Z coordinates of voxels.

    Returns
    -------
    distances : list of double
        Cityblock distances of voxels to home coordinates.

    """
    return [cityblock(home_coords, voxel) for voxel in locations]


def get_locations(volume):
    """Get coords of unassigned voxels in search area.

    Parameters
    ----------
    volume : ndarray
        Volume to search for unassigned voxels. Unassigned voxels have value 1.

    Returns
    -------
    locations : ndarray
        2D array of X/Y/Z coordinates of unassigned voxels.

    """
    return np.transpose(np.stack(np.where(volume)))


def get_extrema(radius_len, home_coords, XMAX, YMAX, ZMAX):
    """Get new boundaries of search area.

    Parameters
    ----------
    radius_len : int
        Radius to expand search area around home coordinates.
    home_coords : tuple of int
        X/Y/Z coordinates of center of search region.
    XMAX : int
        Maximum X-coordinate.
    YMAX : int
        Maximum Y-coordinate.
    ZMAX : int
        Maximum Z-coordinate.

    Returns
    -------
    xMin, xMax, yMin, yMax, zMin, zMax : tuple of int
        X/Y/Z coordinate boundaries of search region.

    """
    xMin = np.max([home_coords[0] - radius_len, 0])
    xMax = np.min([home_coords[0] + radius_len, XMAX - 1])
    yMin = np.max([home_coords[1] - radius_len, 0])
    yMax = np.min([home_coords[1] + radius_len, YMAX - 1])
    zMin = np.max([home_coords[2] - radius_len, 0])
    zMax = np.min([home_coords[2] + radius_len, ZMAX - 1])
    return xMin, xMax, yMin, yMax, zMin, zMax


def label_chunk(
    radius_len, home_coords, voxel_maps, chunk_label, chunk_size, XMAX, YMAX, ZMAX
):
    """Label voxels with chunk index.

    Parameters
    ----------
    radius_len : int
        Radius around home coordinates to search for unassigned voxels.
    home_coords : tuple of int
        X/Y/Z coordinates of home position.
    voxel_maps : ndarray
        Current state of brain volume voxel labels.
    chunk_label : int
        Chunk index to label assigned voxels with.
    chunk_size : int
        Number of voxels in a chunk.
    XMAX : int
        Maximum X-coordinate.
    YMAX : int
        Maximum Y-coordinate.
    ZMAX : int
        Maximum Z-coordinate.

    Returns
    -------
    voxel_maps : ndarray
        State of brain volume voxel labels after adding the latest chunk.

    """
    # Get boundaries of search area
    xMin, xMax, yMin, yMax, zMin, zMax = get_extrema(
        radius_len, home_coords, XMAX, YMAX, ZMAX
    )
    # Make a copy of the voxel_map
    search_area = voxel_maps.copy()
    # zero out everything outside the search area and zero out the home location
    search_area[:xMin, :, :] = 0
    search_area[xMax:, :, :] = 0
    search_area[:, :yMin, :] = 0
    search_area[:, yMax:, :] = 0
    search_area[:, :, :zMin] = 0
    search_area[:, :, zMax:] = 0
    search_area[home_coords[0], home_coords[1], home_coords[2]] = 0
    # Get voxelmap of unassigned voxels inside search area
    search_area = search_area == -1
    # Check if there are enough unassigned voxels to fill a chunk
    if np.sum(search_area) < (chunk_size - 1):
        label_chunk(
            radius_len + 1,
            home_coords,
            voxel_maps,
            chunk_label,
            chunk_size,
            XMAX,
            YMAX,
            ZMAX,
        )
    else:
        # Get coords of unassigned voxels in search area
        locations = get_locations(search_area)
        # Get Manhattan distances of unassigned voxels to home location
        distances = get_distances(home_coords, locations)
        for i in range(chunk_size - 1):
            minidx = np.argmin(distances)
            x, y, z = locations[minidx]
            distances[minidx] = 99999
            voxel_maps[x, y, z] = chunk_label
        voxel_maps[home_coords[0], home_coords[1], home_coords[2]] = chunk_label
    return voxel_maps


def get_next_home(voxel_maps, XMAX, YMAX, ZMAX):
    """Get the next home coordinates from a current voxel-mapping.

    Parameters
    ----------
    voxel_maps : ndarray
        Volume containing voxel-index mapping to chunks.
    XMAX : int
        Maximum X-coordinate.
    YMAX : int
        Maximum Y-coordinate.
    ZMAX : int
        Maximum Z-coordinate.

    Returns
    -------
    x, y, z : tuple of int
        X/Y/Z coordinates of next home location.

    """
    search_area = voxel_maps.copy()
    search_area = search_area == -1
    x, y, z = 0, 0, 0
    for i in range(ZMAX):
        if np.sum(search_area[:, :, i]) > 0:
            z += i
            break
    for j in range(YMAX):
        if np.sum(search_area[:, j, z]) > 0:
            y += j
            break
    for k in range(XMAX):
        if np.sum(search_area[k, y, z]) > 0:
            x += k
            break
    return x, y, z


def generate_chunk_mask(mask, num_chunks, chunk_size, out_dir, radius=20):
    """Generate chunk index mask.

    Parameters
    ----------
    mask : niimg-like object
        Binary brain volume niimg to divide into chunks
    num_chunks : int
        Number of chunks to divide brain volume into.
    chunk_size : int
        Number of voxels in each chunk.
    out_dir : str
        Directory to save chunk index mask to.
    radius : int, optional
        Radius to search for unassigned voxels, by default 20.

    """
    XMAX, YMAX, ZMAX = mask.shape
    voxel_map = mask.get_fdata().astype(int).copy() * -1
    for i in trange(num_chunks):
        home = get_next_home(voxel_map, XMAX, YMAX, ZMAX)
        voxel_map = label_chunk(
            radius, home, voxel_map, i + 1, chunk_size, XMAX, YMAX, ZMAX
        )
    out_fname = os.path.join(os.path.abspath(out_dir), "chunk_idx.nii.gz")
    image.new_img_like(mask, voxel_map).to_filename(out_fname)
