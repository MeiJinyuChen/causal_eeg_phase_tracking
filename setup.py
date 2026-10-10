"""Compatibility shim for legacy tooling.

All project metadata lives in ``pyproject.toml``; this file exists only so that
installers that still expect a ``setup.py`` can install the project.
"""

from setuptools import setup

setup()
