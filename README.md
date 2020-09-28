# Inclusive B→Xℓν Analysis at Belle II

Inclusive Kinematic Fit provides code to kinematically fit the four momenta of the tag-side B meson, the signal lepton and the inclusive X system in inclusive semi-leptonic B decays ast e+e- B factories like Belle II.

## Installation

To install the dependencies of the package simply run

```bash
# setup your basf2 environment release in which you want to install this package
pip3 install --upgrade pip
pip3 install --upgrade --editable .
```

An additional `--user` argument is required if you don't have writing rights to
your basf2 externals, e.g. if you use a basf2 release from `cvmfs`, so pip will
then install the package into your `~/.local` directory.


## Development

If you just want to import functions from it, you don't need anything else, but if you want to work on the code, and submit PR's, you should run:

```bash
pip3 install -r requirements-dev.txt
pre-commit install
```

The pip command will install some code linters and formatters that check if the style of your code conforms to the project rule, e.g.[pylint](https://www.pylint.org/) and [pre-commit](https://pre-commit.com/).
The second line install the git hooks to check your code before creating a commit.

To check your code without trying to commit you can just run `pre-commit run
--all` or just run `black <filename.py>`.

[pytest](https://docs.pytest.org/en/stable/index.html) is used to create and run unit tests.
Please ensure new code has a corresponding test suite that runs succesfully.
You can run the tests via
```bash
pytest
```
in the package root directory.


## Contributors
- Maximilian Welsch ([mwelsch@uni-bonn.de](mailto:mwelsch@uni-bonn.de))
