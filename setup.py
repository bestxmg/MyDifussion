#!/usr/bin/env python3
"""
Setup script for MyDifussion - Stable Diffusion GPU Learning Project
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("docs/requirements_source.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mydifussion",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive learning project for Stable Diffusion on GPU with user-friendly GUI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MyDifussion",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/MyDifussion/issues",
        "Source": "https://github.com/yourusername/MyDifussion",
        "Documentation": "https://github.com/yourusername/MyDifussion#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pre-commit",
            "black",
            "flake8",
            "pytest",
        ],
        "gui": [
            "tkinter",
            "Pillow",
        ],
    },
    entry_points={
        "console_scripts": [
            "mydifussion-gui=core.stable_diffusion_gui:main",
            "mydifussion-gpu=core.final_working_gpu_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    keywords="stable-diffusion, ai, image-generation, gpu, machine-learning, deep-learning",
    license="MIT",
    zip_safe=False,
)
