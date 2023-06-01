from pathlib import Path
from setuptools import setup, find_packages

NAME = "soundmonitor"
DESCRIPTION = "Long-term sound level measurements analysis in Python"
URL = "https://github.com/valerianF/soundmonitor"
EMAIL = "valerian.fraisse@mail.mcgill.ca"
AUTHOR = "Valérian Fraisse"
REQUIRES_PYTHON = ">=3.7"
VERSION = "0.0.6"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=["soundmonitor"],
    install_requires=["matplotlib>=3.7.0", "numpy>=1.21.6", "pandas>=1.3.5", "xlrd>=2.0.1"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Acoustics",
        "Topic :: Scientific/Engineering",
    ],
)
