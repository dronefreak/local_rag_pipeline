"""Setup script to create a Python package for the project."""

from setuptools import find_packages, setup

setup(
    name="local_rag_pipeline",
    version="1.0.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
)
