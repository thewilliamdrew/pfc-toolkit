[![PyPI version shields.io](https://img.shields.io/pypi/v/pfc-toolkit.svg)](https://pypi.python.org/pypi/pfc-toolkit/)

# pfc-toolkit
The Precomputed Functional Connectome Toolkit (**pfc-toolkit**) is a Python module for functional lesion network mapping using the Precomputed Functional Connectome and is distributed under the 3-Cause BSD license.

The project was started in 2019 by William Drew for his Harvard undergraduate thesis project ('21).

The project was supervised by Dr. Michael D. Fox, MD, PhD first at the Berenson-Allen Center for Noninvasive Brain Stimulation at Beth Israel Deaconess Medical Center and later at the Center for Brain Circuit Therapeutics at Brigham and Women's Hospital.

This project references work from [@clin045](https://github.com/clin045)'s and [@alexlicohen](https://github.com/alexlicohen)'s `connectome_quick.py` from [nimlab](https://github.com/nimlab) and [@andreashorn](https://github.com/andreashorn)'s `connectome.sh` from [LeadDBS](https://github.com/netstim/leaddbs).

The Precomputed Functional Connectome Toolkit is for research only; please do not use results from PFC Toolkit for clinical decisions.

## Installation
### Dependencies
pfc-toolkit requires:
- Python (>=3.6)
- NumPy
- SciPy
- Numba
- Nibabel
- Nilearn
- tqdm
- natsort
- importlib

### User Installation
Install using `pip`
```
pip install pfc-toolkit
```

## Usage

### Generate Functional Lesion Network Maps

1. Create a folder named `pfctoolkit_config` in your home directory. 
```
mkdir ~/pfctoolkit_config
```
2. Set up your precomputed connectome config file. An example config file is located [here](https://github.com/thewilliamdrew/pfc-toolkit/blob/master/src/pfctoolkit/configs/discovery-yeo1000_dil.json). Make a copy of the config file to the `pfctoolkit_config` folder. Edit the config file and swap out the precomputed connectome paths to where you have downloaded the connectome files on your machine.

3. Run the precomputed connectome script. If your precomputed connectome config file is named `yeo1000_dil.json`, the name of your precomputed connectome config is `yeo1000_dil`.

```bash
connectome_precomputed -r <path to directory containing rois> -c <name of precomputed connectome config> -o <output directory>
```
***
### Generate a Precomputed Connectome (Instructions WIP)

If instead of using the provided precomputed connectome, you would like to generate your own, first you need preprocessed BOLD fMRI timecourse data from a set of subjects. Each subject's preprocessed data must contain the same number of timepoints, must be registered to the same MNI152 space, and must be masked with the same `MASK`. If you are not using the default `MNI152_T1_2mm_brain_mask_dil.nii.gz` mask file included with FSL, please use `pfctoolkit.chunker.generate_chunk_mask` to generate a chunk mask file `chunk_idx.nii.gz`.

Once you have such data, for each subject please create two `.npy` files. 

For a subject with ID `SUB001`, the first `.npy` file should be named `SUB001.npy` and contain the preprocessed BOLD fMRI timecourse data in a `numpy` array with shape `(n_timepoints, n_voxels)`. The order of the voxels should be determined with `nilearn.maskers.NiftiMasker`. 

The second `.npy` file should be named `SUB001_norms.npy` and contain the norms of each voxel's BOLD fMRI timecourse signal in a `numpy` array with shape `(n_voxels,)` in the same order as in the first `.npy` file. 

After you have done this for all subjects in your dataset, please place all `.npy` files in a single "connectome" folder. 

Next, to generate functional connectivity chunks for the precomputed connectome, run the following script for each chunk index `i`:
```bash
generate_pfc_fc_chunks -b <name of brain mask> -c <path to chunk_idx.nii.gz> -i <index of chunk to process, i> -cs <path to connectome directory containing .npy files> -o <path to output directory>
```

Please see additional usage instructions with `generate_pfc_fc_chunks -h`.

Next, to generate combo chunks for the precomputed connectome, run the following script for each chunk index `i`:
```bash
generate_pfc_combo_chunks -b <name of brain mask> -c <path to chunk_idx.nii.gz> -i <index of chunk to process, i> -cs <path to connectome directory containing .npy files> -o <path to output directory>
```

Please see additional usage instructions with `generate_pfc_combo_chunks -h`.

Lastly, to generate BOLD timecourse norm and standard-deviation weighted masks for the precomputed connectome, run the following script for each chunk index `i`:
```bash
generate_pfc_weighted_masks -b <name of brain mask> -cs <path to connectome directory containing .npy files> -n <name of precomputed connectome> -o <path to output directory>
```

Please see additional usage instructions with `generate_pfc_weighted_masks -h`.

## Development
### Source code
You can check the latest sources with the command:
```
git clone https://github.com/thewilliamdrew/pfc-toolkit.git
```

## Help and Support
### Documentation
Documentation is located [here](https://thewilliamdrew.github.io/pfc-toolkit). (WIP)
