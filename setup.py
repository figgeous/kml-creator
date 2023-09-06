from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Seavis: Mussel Data",
    version="1.0.0",
    author="Thomas O'Neill",
    author_email="thomas.oneill@gmail.com",
    long_description=readme,
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "your-command=your_package.module:main_function",
        ],
    },
)
