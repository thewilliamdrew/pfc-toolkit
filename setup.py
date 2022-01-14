import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

LICENSE = "BSD License"

setuptools.setup(
    name="pfc-toolkit",
    version="0.0.3",
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
    package_data={"pfctoolkit.data": ["*.nii.gz"], "pfctoolkit.configs": ["*.json"]},
    install_requires=[
        "tqdm",
        "numba",
        "numpy",
        "scipy",
        "nilearn",
        "natsort",
        "importlib",
    ],
    python_requires=">=3.6",
)
