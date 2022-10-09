import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

LICENSE = "BSD License"

setuptools.setup(
    name="pfc-toolkit",
    version="0.1.5.0",
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={
        "pfctoolkit.chunks": ["*.nii.gz", "*.gii"],
        "pfctoolkit.data": ["*.nii.gz", "*.gii"],
        "pfctoolkit.configs": ["*.json"],
        "pfctoolkit.scripts": ["*.py"],
    },
    scripts=[
        "src/pfctoolkit/scripts/connectome_precomputed",
        "src/pfctoolkit/scripts/generate_pfc_combo_chunks",
        "src/pfctoolkit/scripts/generate_pfc_fc_chunks",
        "src/pfctoolkit/scripts/generate_pfc_weighted_masks",
        "src/pfctoolkit/scripts/generate_weighted_network_linklist",
    ],
    install_requires=[
        "tqdm",
        "numba",
        "numpy",
        "scipy",
        "nibabel",
        "nilearn",
        "natsort",
        "importlib",
    ],
    python_requires=">=3.6",
)
