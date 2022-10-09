"""
Tools to perform Lesion Network Mapping with the Precomputed Connectome

"""
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from numba import jit
from nilearn import image
from pfctoolkit import tools, datasets, surface


def process_chunk(chunk, rois, config):
    """Compute chunk contribution to FC maps for a given list of ROIs.

    Parameters
    ----------
    chunk : int
        Index of chunk to be processed.
    rois : list of str
        List of ROI paths to be processed.
    config : pfctoolkit.config.Config
        Configuration of the precomputed connectome.

    Returns
    -------
    contributions : dict of ndarray
        Dictionary containing contributions to network maps.

    """
    chunk_paths = {
        "avgr": config.get("avgr"),
        "fz": config.get("fz"),
        "t": config.get("t"),
        "combo": config.get("combo"),
    }
    image_type = config.get("type")
    if image_type == "volume":
        brain_masker = tools.NiftiMasker(datasets.get_img(config.get("mask")))
        chunk_masker = tools.NiftiMasker(
            image.math_img(f"img=={chunk}", img=config.get("chunk_idx"))
        )
    elif image_type == "surface":
        brain_masker = surface.GiftiMasker(datasets.get_img(config.get("mask")))
        chunk_masker = surface.GiftiMasker(
            surface.new_gifti_image(
                datasets.get_img(config.get("chunk_idx")).agg_data() == chunk
            )
        )
    brain_weights = np.array([brain_masker.transform(roi) for roi in rois])
    chunk_weights = np.array([chunk_masker.transform(roi) for roi in rois])
    brain_masks = np.array([(weights != 0) for weights in brain_weights])
    chunk_masks = np.array([(weights != 0) for weights in chunk_weights])
    norm_weight = chunk_masker.transform(config.get("norm"))
    std_weight = chunk_masker.transform(config.get("std"))

    norm_chunk_masks, std_chunk_masks = compute_chunk_masks(
        chunk_weights, norm_weight, std_weight
    )
    contributions = {}
    for chunk_type in [
        ("avgr", "AvgR"),
        ("fz", "AvgR_Fz"),
        ("t", "T"),
        ("combo", "Combo"),
    ]:
        chunk_data = np.load(
            os.path.join(chunk_paths[chunk_type[0]], f"{chunk}_{chunk_type[1]}.npy")
        )
        expected_shape = (config.get("chunk_size"), config.get("brain_size"))
        if chunk_data.shape != expected_shape:
            raise TypeError(
                f"Chunk expected to have shape {expected_shape} but"
                f" instead has shape {chunk_data.shape}!"
            )
        if chunk_type[0] == "combo":
            numerator = compute_numerator(norm_chunk_masks)
            for i, roi in enumerate(rois):
                denominator = compute_denominator(
                    brain_weights,
                    chunk_weights,
                    brain_masks,
                    chunk_masks,
                    chunk_data,
                    i,
                )
                contributions[roi]["numerator"] = numerator[i]
                contributions[roi]["denominator"] = denominator
        else:
            network_maps = compute_network_maps(std_chunk_masks, chunk_data)
            for i, roi in enumerate(rois):
                if chunk_type[0] == "avgr":
                    contributions[roi] = {
                        chunk_type[0]: network_maps[i, :],
                    }
                else:
                    contributions[roi][chunk_type[0]] = network_maps[i, :]
    network_weights = compute_network_weights(std_chunk_masks)
    for i, roi in enumerate(rois):
        contributions[roi]["network_weight"] = network_weights[i]
    return contributions


def update_atlas(contribution, atlas):
    """Update atlas with chunk contributions.

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


def publish_atlas(atlas, output_dir, config):
    """Runs final computation on the atlas and outputs network maps to file.

    Parameters
    ----------
    atlas : dict
        dict containing in-progress FC maps and scaling factor accumulators.
    output_dir : str
        Output directory.
    config : pfctoolkit.config.Config
        Configuration of the precomputed connectome.

    """
    output_dir = os.path.abspath(output_dir)
    image_type = config.get("type")
    if image_type == "volume":
        brain_masker = tools.NiftiMasker(datasets.get_img(config.get("mask")))
        extension = ".nii.gz"
    elif image_type == "surface":
        brain_masker = surface.GiftiMasker(datasets.get_img(config.get("mask")))
        extension = ".gii"
    for roi in atlas:
        atlas[roi]["denominator"] = final_denominator(atlas[roi]["denominator"])
        atlas[roi]["avgr"] = atlas[roi]["avgr"] / atlas[roi]["network_weight"]
        atlas[roi]["fz"] = atlas[roi]["fz"] / atlas[roi]["network_weight"]
        atlas[roi]["t"] = atlas[roi]["t"] / atlas[roi]["network_weight"]
        scaling_factor = atlas[roi]["numerator"] / atlas[roi]["denominator"]
        subject_name = os.path.basename(roi).split(".nii")[0]
        for map_type in [("avgr", "AvgR"), ("fz", "AvgR_Fz"), ("t", "T")]:
            output_fname = f"{subject_name}_Precom_{map_type[1]}{extension}"
            output_path = os.path.join(output_dir, output_fname)
            atlas[roi][map_type[0]] = atlas[roi][map_type[0]] * scaling_factor
            out_img = brain_masker.inverse_transform(atlas[roi][map_type[0]])
            out_img.to_filename(output_path)
    print(f"Network maps output to {output_dir}")
    print("Done!")


def get_roi_voxel_maps(chunks, roi, config, map_type="t", output_dir=""):
    """Get array of voxel maps for a given ROI. If output_dir is specified. Save array as .npy file.

    Parameters
    ----------
    chunks : int
        Index of chunk to be processed.
    roi :  str
        ROI path to obtain voxel maps for.
    config : pfctoolkit.config.Config
        Configuration of the precomputed connectome.
    map_type : str
        Type of voxel map to return. Defaults to "t".
    output_dir : str
        Path to save roi voxel map array to.

    Returns
    -------
    voxel_maps : ndarray or None
        ndarray containing voxel connectivity maps. None if output_dir is specified.

    """
    chunk_types = {
        "avgr": "AvgR",
        "fz": "AvgR_Fz",
        "t": "T",
        "combo": "Combo",
    }
    chunk_path = config.get(map_type)
    image_type = config.get("type")

    roi_name = os.path.basename(roi).split(".nii")[0].split(".gii")[0]

    if image_type == "volume":
        brain_masker = tools.NiftiMasker(datasets.get_img(config.get("mask")))
        roi_masker = tools.NiftiMasker(brain_masker.mask(roi))
    elif image_type == "surface":
        brain_masker = surface.GiftiMasker(datasets.get_img(config.get("mask")))
        roi_masker = surface.GiftiMasker(brain_masker.mask(roi))

    brain_weight = brain_masker.transform(roi)
    brain_mask = brain_weight != 0

    roi_size = np.sum(brain_mask)
    voxel_map = np.zeros((roi_size, config.get("brain_size")), dtype=np.float32)

    index_map = roi_masker.inverse_transform(np.arange(1, roi_size + 1))

    for chunk in tqdm(chunks):
        if image_type == "volume":
            chunk_masker = tools.NiftiMasker(
                image.math_img(f"img=={chunk}", img=config.get("chunk_idx"))
            )
        elif image_type == "surface":
            chunk_masker = surface.GiftiMasker(
                surface.new_gifti_image(
                    datasets.get_img(config.get("chunk_idx")).agg_data() == chunk
                )
            )

        chunk_weight = chunk_masker.transform(roi)
        chunk_mask = chunk_weight != 0

        chunk_data = np.load(
            os.path.join(chunk_path, f"{chunk}_{chunk_types[map_type]}.npy")
        )
        expected_shape = (config.get("chunk_size"), config.get("brain_size"))
        if chunk_data.shape != expected_shape:
            raise TypeError(
                f"Chunk expected to have shape {expected_shape} but"
                f" instead has shape {chunk_data.shape}!"
            )

        chunk_index_map = chunk_masker.transform(index_map)
        chunk_index_locations = np.where(chunk_index_map)[0]
        trimmed_chunk_index_map = (
            chunk_index_map[chunk_index_locations].astype(np.int32) - 1
        )
        voxel_map[[trimmed_chunk_index_map], :] = chunk_data[[chunk_index_locations], :]

    if output_dir:
        out_fname = os.path.join(
            os.path.abspath(output_dir),
            f"{roi_name}_voxel_maps_{chunk_types[map_type]}.npy",
        )
        np.save(out_fname, voxel_map)
        print(f"Saved to: {out_fname} !")
        return None
    else:
        return voxel_map


@jit(nopython=True)
def final_denominator(denominator):
    return np.sqrt(denominator)


@jit(nopython=True)
def compute_network_weights(std_chunk_masks):
    """Compute network weights.

    Parameters
    ----------
    std_chunk_masks : ndarray
        Chunk-masked ROIs weighted by BOLD standard deviation.

    Returns
    -------
    network_weights : ndarray
        Contribution to total network map weights.

    """
    network_weights = np.sum(std_chunk_masks, axis=1)
    return network_weights


@jit(nopython=True)
def compute_network_maps(std_chunk_masks, chunk_data):
    """Compute network maps.

    Parameters
    ----------
    std_chunk_masks : ndarray
        Chunk-masked ROIs weighted by BOLD standard deviation.
    chunk_data : ndarray
        Chunk data.

    Returns
    -------
    network maps : ndarray
        Network map contributions from chunk.

    """
    network_maps = np.dot(std_chunk_masks, chunk_data)
    return network_maps


@jit(nopython=True)
def compute_denominator(
    brain_weights, chunk_weights, brain_masks, chunk_masks, chunk_data, i
):
    """Compute denominator contribution.

    Parameters
    ----------
    brain_weights : ndarray
        Brain-masked weighted ROIs.
    chunk_weights : ndarray
        Chunk-masked weighted ROIs.
    brain_masks : ndarray
        Brain-masked unweighted ROIs.
    chunk_masks : ndarray
        Chunk-masked unweighted ROIs.
    chunk_data : ndarray
        Chunk data.
    i : int
        Index of processed ROI.

    Returns
    -------
    denominator : float
        Contribution to denominator.

    """
    chunk_masked = np.multiply(
        np.reshape(chunk_weights[i][chunk_masks[i]], (-1, 1)),
        chunk_data[chunk_masks[i], :],
    )
    brain_masked = np.multiply(
        brain_weights[i][brain_masks[i]], chunk_masked[:, brain_masks[i]]
    )
    denominator = np.sum(brain_masked)
    return denominator


@jit(nopython=True)
def compute_numerator(norm_chunk_masks):
    """Compute numerator contribution.

    Parameters
    ----------
    norm_chunk_masks : ndarray
        ROI chunk masks weighted with BOLD norms.

    Returns
    -------
    numerator : float
        Numerator contribution

    """
    numerator = np.sum(norm_chunk_masks, axis=1)
    return numerator


@jit(nopython=True)
def compute_chunk_masks(chunk_weights, norm_weight, std_weight):
    """Compute weighted chunk masks.

    Parameters
    ----------
    chunk_weights : ndarray
        Chunk-masked weighted ROIs.
    norm_weight : ndarray
        Chunk-masked voxel BOLD norms.
    std_weight : ndarray
        Chunk-masked voxel BOLD standard deviations.

    Returns
    -------
    norm_weighted_chunk_masks : ndarray
        ROI chunk masks weighted with BOLD norms.
    std_weighted_chunk_masks : ndarray
        ROI chunk masks weighted with BOLD standard deviations.

    """
    norm_weighted_chunk_masks = np.multiply(chunk_weights, norm_weight)
    std_weighted_chunk_masks = np.multiply(chunk_weights, std_weight)
    return norm_weighted_chunk_masks, std_weighted_chunk_masks
