"""
Generate AvgR, AvgR_Fz, and T functional connectivity chunks for a single chunk.

Usage
-----
    Run this script once for every chunk in your precomputed connectome.
    Use -i to specify which chunk is being generated.

"""
import os
import numpy as np
import argparse
from pfctoolkit import processing, datasets
from nilearn import image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate precomputed connectome functional connectivity chunks.")

    parser.add_argument("-b","--bmask", metavar='brain_mask', help="Name of binary brain mask found in pfc.toolkits.datasets. Must be the same mask used to generate the chunk index map. Defaults to 'MNI152_T1_2mm_brain_mask_dil'", type=str, default='MNI152_T1_2mm_brain_mask_dil')

    parser.add_argument("-c", "--cmask", metavar='chunk_idx_mask', help="Path to mask containing voxel-wise chunk labels.", type=str, required=True)

    parser.add_argument("-i", "--idx", metavar='chunk_idx', help="Index of chunk to process.", type=int, required=True)

    parser.add_argument("-cs","--conn", metavar='connectome_dir', help="Path to directory containing individual subject connectome files.", type=str, required=True)

    parser.add_argument("-o", "--out", metavar='output_dir', help="Path to output directory. */AvgR, */AvgR_Fz, and */T will be created in this directory if they do not already exist.", type=str, required=True)

    args = parser.parse_args()

    mask = datasets.get_img_path(args.bmask)
    chunk_idx_mask = os.path.abspath(args.cmask)
    chunk_idx = args.idx
    connectome_dir = os.path.abspath(args.conn)
    output_dir = os.path.abspath(args.out)

    assert os.path.exists(chunk_idx_mask), f"Chunk mask not found: {chunk_idx_mask}"
    assert os.path.exists(connectome_dir), f"Connectome directory not found: {connectome_dir}"
    assert os.path.exists(output_dir), f"Output directory not found: {output_dir}"
    max_idx = np.max(image.get_data(chunk_idx_mask))
    assert (chunk_idx > 0) & (chunk_idx < max_idx), f"Chunk index out of range. Choose an index from 1-{max_idx}."

    print(f"mask: {mask}")
    print(f"chunk_idx_mask: {chunk_idx_mask}")
    print(f"chunk_idx: {chunk_idx}")
    print(f"connectome_dir: {connectome_dir}")
    print(f"output_dir: {output_dir}")

    processing.precomputed_connectome_chunk(mask,
                                            chunk_idx_mask,
                                            chunk_idx,
                                            connectome_dir,
                                            output_dir)
