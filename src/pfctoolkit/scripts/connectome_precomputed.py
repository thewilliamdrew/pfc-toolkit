"""usage: connectome_precomputed.py [-h] -r -c -o

Generate AvgR, AvgR_Fz, and T functional connectivity chunks for a single chunk.

Arguments:
  -h, --help        show this help message and exit

  -r, --roi-dir     Path to directory containing binary Nifti ROIs, or path to a single 
                    binary Nifti ROI, or path to CSV containing binary Nifti ROI paths.

  -c, --config      Name of precomputed connectome config file to use.

  -o, --output-dir  Path to output directory.

"""
import os
import argparse
from tqdm import tqdm
from pfctoolkit import tools
from pfctoolkit import config
from pfctoolkit import mapping
from pfctoolkit import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate functional lesion network maps using the precomputed connectome."
    )

    parser.add_argument(
        "-r",
        "--roi-dir",
        metavar="\b",
        help="Path to directory containing binary Nifti ROIs, or path to a single binary Nifti ROI, or path to CSV containing binary Nifti ROI paths.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="\b",
        help="Name of precomputed connectome config file to use.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="\b",
        help="Path to output directory.",
        type=str,
        required=True,
    )

    # Parse arguments
    args = parser.parse_args()

    # Load ROI list
    roi_paths = tools.load_roi(os.path.abspath(args.roi_dir))

    # Load and check PCC configuration
    pcc_config = config.Config(args.config)

    # Set output directory
    output_dir = os.path.abspath(args.output_dir)

    # Load brain mask
    brain_mask = datasets.get_img(pcc_config.get("mask"))

    # Get chunks
    chunks = tools.get_chunks(roi_paths, pcc_config)

    # Process Chunks
    atlas = {}
    for chunk in tqdm(chunks):
        contribution = mapping.process_chunk(chunk, chunks[chunk], pcc_config)
        atlas = mapping.update_atlas(contribution, atlas)

    # Consolidate outputs
    mapping.publish_atlas(atlas, output_dir, pcc_config)
