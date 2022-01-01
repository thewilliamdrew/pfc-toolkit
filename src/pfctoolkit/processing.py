"""
Tools to generate a Precomputed Functional Connectome.

"""

def precomputed_connectome_chunk(chunk_idx_mask,
                                 chunk_idx,
                                 connectome_dir,
                                 output_dir,
                                 workers = 8):
    """Generate a precomputed connectome chunk (AvgR/Fz/T)

    Parameters
    ----------
    chunk_idx_mask : Niimg-like object
        Mask containing voxel-wise chunk labels.
    chunk_idx : int
        Index of chunk to process.
    connectome_dir : str
        Path to individual subject connectome files.
    output_dir : str
        Path to output directory.
    workers : int, optional
        Number of workers, by default 8

    """
    