from setuptools import find_packages, setup

setup(
    name="soul",
    version="0.0.1",
    packages=find_packages(include=["soul", "soul.*"]),
)
