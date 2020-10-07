import os
from setuptools import find_packages, setup


package_dir = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(package_dir, "requirements.txt")
with open(requirements_file, "r") as rf:
    requirements = [
        req.strip() for req in rf.readlines() if req.strip() and not req.startswith("#")
    ]

setup(
    author="Maximilian Welsch",
    author_email="maxwelsch93@gmail.com",
    python_requires=">=3.6",
    name="inclusivekinematicfit",
    packages=find_packages(),
    description="Inclusive Kinematic Fit provides code to kinematically fit the four momenta of the tag-side B meson, the signal lepton and the inclusive X system in inclusive semi-leptonic B decays ast e+e- B factories like Belle II.",
    install_requires=requirements,
    test_suite="tests",
    tests_require=["pytest>=3"],
    version="0.1.0",
)
