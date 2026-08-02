from setuptools import find_packages, setup


setup(
    name="crystal_gnn",
    version="0.1.0",
    description="Structure-aware multi-scale crystal GNN with calibrated uncertainty",
    packages=find_packages(include=["crystal_gnn*"]),
    include_package_data=True,
    python_requires=">=3.10",
)
