[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
<!-- [![PyPI version](https://badge.fury.io/py/sctk.svg)](https://badge.fury.io/py/sctk) -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) 
[![CI](https://github.com/Teichlab/sctk/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/Teichlab/sctk/actions/workflows/ci-cd.yaml)
[![Tests](https://github.com/Teichlab/sctk/actions/workflows/sphinx-autodoc.yaml/badge.svg)](https://github.com/Teichlab/sctk/actions/workflows/sphinx-autodoc.yaml)
![Coverage](coverage.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

# sctk

**S**ingle **C**ell analysis **T**ool **K**it: A large collection of functions built around the scanpy ecosystem that faciliates preprocessing, clustering, annotation, integration and some down-stream analysis of scRNA-seq data.

## Installation

```bash
pip install sctk
```

If you encounter dependency clashes, create a fresh python 3.9 virtual environment (e.g. via conda) and install SCTK there. It may also help to update pip:

```bash
pip install --upgrade pip
```

## Usage and Documentation

SCTK's documentation is available [here](https://teichlab.github.io/sctk/), and features a tutorial and API reference for the automated QC workflow.
