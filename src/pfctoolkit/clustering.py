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

        roi_weight_matrix[roi_index_in_chunk, :] = chunk_data[
            chunk_index_in_roi[roi_index_in_chunk] - 1, :
        ]
    return roi_weight_matrix, roi_size


#########################################
#########################################
# The functions below were received from Steve Petersen's lab with minor
# tweaks by Shan. This python implementation is by William. The original
# scripts are also in the Midnight Scan Club's github.
#########################################
#########################################
def matrix_thresholder(matrix, threshold, threshold_type):
    #
    # Name:matrix_thresholder.m
    # $Revision: 1.2 $
    # $Date: 2011/03/08 20:30:16 $
    #
    # jdp 10/10/10
    #
    # This script takes a 2D matrix and thresholds it at a given value or edge
    # density, and returns a thresholded matrix. It also returns the precise
    # edge density and minimum r value of the thresholded matrix (which will
    # differ from the pre-specified kden or r value used to threshold the
    # matrix!)
    #
    # USAGE: [matrix r kden] = matrix_thresholder(matrix,threshold,thresholdtype)
    # USAGE: [matrix r kden] = matrix_thresholder(mat,0.8,'kden')
    # USAGE: [matrix r kden] = matrix_thresholder(rmat,0.25,'r')
    #
    # Thresholdtype is 'r' or 'kden'
    #
    # NOTE:
    # This script presumes an undirected (symmetric) matrix !!!!
    #
    # It also ignores values on the diagonal (should be zero anyways) !!!!

    d = np.shape(matrix)
    num_possible_edges = d[0] * (d[1] - 1) / 2
    if threshold_type == "r":
        matrix[matrix < threshold] = 0
        kden = np.count_nonzero(np.triu(matrix, 1)) / num_possible_edges
        notzero = np.triu(matrix, 1) != 0
        r = np.min(matrix[notzero])
        if len(notzero) == 0:
            print(f"No edges passing threshold {threshold}")
            r = threshold
    elif threshold_type == "kden":
        if (threshold < 0) or (threshold > 1):
            raise ValueError("threshold needs to be between 0 and 1")
        if d[0] < 20000:  # if it's a smaller matrix...
            edges_left = np.ceil(threshold * num_possible_edges)
            matrix = np.triu(matrix, 1)
            v, i = np.sort(matrix)[::-1]  # CHECK THIS
            kept_edges = np.zeros(d[0])
            kept_edges[0:edges_left] = 1
            kept_edges = kept_edges.astype(bool)
            v[~kept_edges] = 0
            matrix[i] = v
            matrix = np.reshape(matrix, d)
            matrix = np.max(matrix, matrix.transpose())
            r = v[edges_left]
            kden = edges_left / num_possible_edges
        else:
            print(
                f"matrix_thresholder: Using alternate kden thresholding for large matrices to spare RAM\n"
            )
            matrix, kden, r = alternate_threshold(matrix, kden)
    elif threshold_type == "abs":
        matrix[np.abs(matrix) < threshold] = 0
        kden = np.count_nonzero(np.triu(matrix, 1)) / num_possible_edges
        r = threshold
    else:
        raise ValueError("matrix_thresholder: must use 'r', 'kden', or 'abs'")
    return matrix, r, kden


def alternate_threshold(matrix, kden):
    d = matrix.shape
    num_edges = d[0] ** 2
    repsneeded = np.ceil(
        np.log2(num_edges)
    )  # This is the maximum useful number of cycles when bisecting differences of numedges

    # Check that matrix can meet the desired kden
    begin_kden = np.count_nonzero(matrix) / num_edges
    if begin_kden < kden:
        print(
            f"matrix_thresholder: matrix starts with kden {begin_kden}, cannot achieve kden of {kden}.\n"
        )
        print("Supply a different kden or a different matrix")

    # Obtain upper and lower bounds of the matrix
    amin = np.min(np.min(np.nonzero(matrix)))
    amax = np.max(np.max(matrix))

    for i in range(repsneeded):
        # Perform the next obvious threshold
        temp_thresh = (amax + amin) / 2  # Set the next threshold
        print(f"\tCycle {i}/{repsneeded}, threshold is {temp_thresh}\t")
        matrix2 = matrix >= temp_thresh  # Apply the threshold, create logical array
        kden2 = np.count_nonzero(matrix2) / (num_edges)  # Calculate the edge density
        kdiff = kden2 - kden  # How far from our target kden are we?

        print(f"kden is {kden2}, offtarget is {kdiff}\n")

        if kdiff > 0:  # if the matrix is too dense still we bump the lower threshold up
            amin = temp_thresh
        elif kdiff < 0:  # if the matrix is too sparse we drop the upper threshold down
            amax = temp_thresh
        elif kdiff == 0:
            break

    threshold = temp_thresh
    matrix[matrix < threshold] = 0
    kden = np.count_nonzero(matrix) / num_edges
    print(
        f"Final result is r={threshold}\tkden={kden}\tedges={np.count_nonzero(matrix)}"
    )

    return matrix, kden, threshold
