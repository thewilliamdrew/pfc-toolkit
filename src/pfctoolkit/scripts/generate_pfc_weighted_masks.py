"""usage: generate_pfc_combo_chunks.py [-h] [-b] -c -i -cs -o 

Generate AvgR, AvgR_Fz, and T functional connectivity chunks for a single chunk.

Arguments:
  -h, --help        show this help message and exit

  -b, --brain-mask  Name of binary brain mask found in pfc.toolkits.datasets.
                    Must be the same mask used to generate the chunk index map.
                    Defaults to 'MNI152_T1_2mm_brain_mask_dil'

  -c, --chunk-mask  Path to mask containing voxel-wise chunk labels.

  -i, --chunk-idx   Index of chunk to process.

  -cs, --conn-dir   Path to directory containing individual subject connectome
                    files.

  -o, --output-dir  Path to output directory. */AvgR, */AvgR_Fz, and */T will be
                    created in this directory if they do not already exist.

"""
import os
import numpy as np
import argparse
from pfctoolkit import processing, datasets
from nilearn import image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AvgR, AvgR_Fz, and T"
    " functional connectivity chunks for a single chunk.")

    parser.add_argument("-b","--brain-mask", metavar='\b', help="Name of binary"
    " brain mask found in pfc.toolkits.datasets. Must be the same mask used to"
    " generate the chunk index map. Defaults to 'MNI152_T1_2mm_brain_mask_dil'",
    type=str, default='MNI152_T1_2mm_brain_mask_dil')

    parser.add_argument("-cs","--conn-dir", metavar='\b', help="Path to "
    "directory containing individual subject connectome files.", type=str,
    required=True)

    parser.add_argument("-n","--conn-name", metavar='\b', help="Name of connectome to use for naming mask files.", type=str, default="")

    parser.add_argument("-o", "--output-dir", metavar='\b', help="Path to "
    "output directory. */AvgR, */AvgR_Fz, and */T will be created in this "
    "directory if they do not already exist.", type=str, required=True)

    args = parser.parse_args()

    mask = datasets.get_img_path(args.b)
    connectome_dir = os.path.abspath(args.cs)
    connectome_name = args.n
    output_dir = os.path.abspath(args.o)

    assert os.path.exists(connectome_dir), \
           f"Connectome directory not found: {connectome_dir}"
    assert os.path.exists(output_dir), \
           f"Output directory not found: {output_dir}"

    processing.precomputed_connectome_weighted_masks(mask,
                                                     connectome_dir,
                                                     output_dir,
                                                     connectome_name)
