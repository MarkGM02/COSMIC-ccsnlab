"""
ccsnlab
=======

A lightweight toolkit for analyzing and visualizing
core-collapse supernova (CCSN) COSMIC output data.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("ccsnlab")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Mark Martinez"

__all__ = ["sn_subtypes", "data_loading", "cee_rerun", "plotting"]
