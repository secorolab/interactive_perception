"""
Setup script for simulate_algorithm package.

This makes simulate_algorithm installable and ensures core_algorithm is available.

Installation:
  pip install -e ../core_algorithm   # Install core_algorithm first
  pip install -e .                   # Then install simulation
  
Or install together:
  pip install -e ../core_algorithm ../simulate_algorithm
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simulate-algorithm",
    version="1.0.0",
    author="Interactive Perception Lab",
    description="Standalone simulation for polygon exploration algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/felix-thesis/interactive_perception",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "core-algorithm>=1.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "sympy>=1.9",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [],
    },
    include_package_data=True,
    zip_safe=False,
)
