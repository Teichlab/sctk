![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)(https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
<!-- [![PyPI version](https://badge.fury.io/py/sctk.svg)](https://badge.fury.io/py/sctk) -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) 
![CI](https://github.com/slobentanzer/sctk/actions/workflows/ci-cd.yaml/badge.svg)
![Coverage](coverage.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

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
