from setuptools import setup, find_packages

setup(
    name='mint',
    version='0.0.1',
    packages=find_packages(
        where='mint',
    ),
    package_dir={"": "mint"}
)
