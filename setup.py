import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

LICENSE = "BSD-3-Clause"

setuptools.setup(
    name="pfc-toolkit",
    version="2026.9.18.0",
    author="William Drew",
    author_email="william.drew100@gmail.com",
    description="The Precomputed Functional Connectome Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thewilliamdrew/pfc-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/thewilliamdrew/pfc-toolkit/issues",
    },
    license=LICENSE,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={
        "pfctoolkit.chunks": ["*.nii.gz", "*.gii"],
        "pfctoolkit.data": ["*.nii.gz", "*.gii"],
        "pfctoolkit.configs": ["*.json"],
    },
    scripts=[
        "src/pfctoolkit/scripts/connectome_precomputed",
        "src/pfctoolkit/scripts/generate_pfc_combo_chunks",
        "src/pfctoolkit/scripts/generate_pfc_fc_chunks",
        "src/pfctoolkit/scripts/generate_pfc_weighted_masks",
    ],
    install_requires=[
        "tqdm",
        "numba",
        "numpy",
        "scipy",
        "nibabel",
        "nilearn",
        "natsort",
        "python-environ",
        "boto3"
    ],
    python_requires=">=3.6",
)
