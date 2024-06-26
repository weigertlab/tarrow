from setuptools import setup, find_packages

setup(
    name="tarrow",
    version="0.2.0",
    description="Time Arrow Pretraining",
    long_description_content_type="text/markdown",
    author="Benjamin Gallusser, Max Stieber, Martin Weigert",
    author_email="benjamin.gallusser@epfl.ch",
    license="BSD 3-Clause License",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "scikit-image",
        "torch",
        "torchvision>=0.13",
        "pyyaml",
        "matplotlib",
        "configargparse",
        "pyyaml",
        "tensorboard",
        "pre-commit",
        "black",
        "imagecodecs",
        "tifffile",
        "imageio>=2.19",
        "pandas",
        "dill",
        "scikit-learn",
        "gitpython",
        "cython",
        "numba",
        "requests",
    ],
)
