import os
from os import path

from setuptools import find_packages, setup

NAME = "pfvwmi"
DESCRIPTION = "Benchmarking WMI-based probabilistic formal verification"
URL = "http://github.com/paolomorettin/pfvwmi"
EMAIL = "paolo.morettin@unitn.it"
AUTHOR = "Paolo Morettin"
VERSION = "0.9"

# What packages are required for this module to be executed?
REQUIRED = [
    "gymnasium",
    "matplotlib",
    "numpy",
    "pysmt @ git+https://git@github.com/pysmt/pysmt@optimization#egg=pysmt",
    "torch",
    "wmipa @ git+https://git@github.com/unitn-sml/wmi-pa@interface-refactoring#egg=wmi-pa",
]

here = os.path.abspath(os.path.dirname(__file__))

with open(path.join(here, "README.md")) as ref:
    long_description = ref.read()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=("experiments")),
    zip_safe=False,
    install_requires=REQUIRED,
)
