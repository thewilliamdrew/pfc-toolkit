"""
Tools to perform brain segmentation and network clustering with the Precomputed Connectome

"""
import os
import numpy as np

from pfctoolkit import tools, datasets, surface
from nilearn import image
from tqdm import tqdm


def generate_weighted_network_matrix(roi, chunks, pcc_config):
    image_type = pcc_config.get("type")
    if image_type == "volume":
        brain_masker = tools.NiftiMasker(datasets.get_img(pcc_config.get("mask")))
    elif image_type == "surface":
        brain_masker = surface.GiftiMasker(datasets.get_img(pcc_config.get("mask")))

    brain_weight = brain_masker.transform(roi[0])
    brain_mask = brain_weight != 0
    roi_masker = tools.NiftiMasker(brain_masker.mask(roi[0]))
    roi_size = np.sum(brain_mask)
    print(f"ROI Size: {roi_size}")

    roi_weight_matrix = np.zeros((roi_size, roi_size), dtype=np.float32)
    for chunk in tqdm(chunks):
        chunk_path = pcc_config.get("avgr")
        if image_type == "volume":
            chunk_masker = tools.NiftiMasker(
                image.math_img(f"img=={chunk}", img=pcc_config.get("chunk_idx"))
            )
        elif image_type == "surface":
            chunk_masker = surface.GiftiMasker(
                surface.new_gifti_image(
                    datasets.get_img(pcc_config.get("chunk_idx")).agg_data() == chunk
                )
            )
        chunk_index_map = chunk_masker.inverse_transform(
            np.arange(1, pcc_config.get("chunk_size") + 1, dtype=np.int32)
        )

        # Load chunk data
        chunk_data = np.load(os.path.join(chunk_path, f"{chunk}_AvgR.npy"))
        expected_shape = (pcc_config.get("chunk_size"), pcc_config.get("brain_size"))
        if chunk_data.shape != expected_shape:
            raise TypeError(
                f"Chunk expected to have shape {expected_shape} but"
                f" instead has shape {chunk_data.shape}!"
            )

        # Mask chunk data on whole brain for roi
        chunk_data = chunk_data[:, brain_mask]

        # The order of this is in roi index ordering
        chunk_index_in_roi = roi_masker.transform(chunk_index_map).astype(np.int32)

        # Get mapping of chunk idx to roi idx
        roi_index_in_chunk = np.where(chunk_index_in_roi > 0)[0]

        roi_weight_matrix[roi_index_in_chunk, :] = (
            chunk_data[chunk_index_in_roi[roi_index_in_chunk] - 1, :] + 1
        )
    return roi_weight_matrix, roi_size
