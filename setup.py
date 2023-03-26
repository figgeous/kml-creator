# -*- coding: utf-8 -*-
# Learn more: https://github.com/kennethreitz/setup.py
from setuptools import find_packages
from setuptools import setup


with open("README.md") as f:
    readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name="sample",
    version="0.1.0",
    description="",
    long_description=readme,
    author="",
    author_email="",
    url="",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
)
