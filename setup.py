# -*- coding: utf-8 -*-

from __future__ import print_function

from pathlib import Path
from setuptools import setup
import sys, re, os, pathlib

this_directory = Path(__file__).parent

with open(this_directory / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="diffshadow",
    version="0.1.0",
    author="Markus Worchel",
    author_email="m.worchel@tu-berlin.de",
    description="Differentiable shadow mapping",
    url="https://github.com/mworchel/differentiable-shadow-mapping",
    license="MIT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['diffshadow']
)