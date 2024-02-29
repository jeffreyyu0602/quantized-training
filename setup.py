from setuptools import setup, find_packages

setup(
    name="quantized-training",
    version="0.1.dev0",
    author="Jeffrey Yu",
    author_email="jeffreyy@stanford.edu",
    description="Quantization on PyTorch models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9.0",
)