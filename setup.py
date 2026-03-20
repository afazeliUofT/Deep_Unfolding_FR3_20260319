from setuptools import find_packages, setup

setup(
    name="deep-unfolding-fr3-twc",
    version="0.1.2",
    description="FR3 deep-unfolding TWC extension",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["fr3_sim", "fr3_sim.*", "fr3_twc", "fr3_twc.*"],
    ),
    include_package_data=True,
)
