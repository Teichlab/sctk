# sctk

**S**ingle **C**ell analysis **T**ool **K**it: A large collection of functions built around the scanpy ecosystem that faciliates preprocessing, clustering, annotation, integration and some down-stream analysis of scRNA-seq data.

## Installation

```bash
pip install git+https://github.com/Teichlab/sctk.git
```

It is recommended to install sctk into a fresh python 3.9 virtual environment due to its large amount of dependencies. If you encounter a dependency conflict when installing into a fresh environment, try updating pip:

```bash
pip install --upgrade pip
```

## Usage

An example of applying the automatic QC workflow can be found in the [demo notebook](notebooks/automatic_qc.ipynb).