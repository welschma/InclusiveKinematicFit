from setuptools import find_packages, setup

requirements = ["numba", "numpy", "scipy", "pandas"]

setup(
    author="Maximilian Welsch",
    author_email="maxwelsch93@gmail.com",
    python_requires=">=3.6",
    name="kinfit",
    packages=find_packages(),
    description="Inclusive Kinematic Fit provides code to kinematically fit the four momenta of the tag-side B meson, the signal lepton and the inclusive X system in inclusive semi-leptonic B decays ast e+e- B factories like Belle II.",
    install_requires=requirements,
    test_suite="tests",
    tests_require=["pytest>=3"],
    version="0.1.0",
)
