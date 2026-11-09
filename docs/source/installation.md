# Installation

The package requires Python 3.12 or newer, and it installs *motulator* and *motulator-multiphase* as its dependencies. It supports *motulator* 0.23. A new release of *motulator* is supported after it has been tested.

## Option 1: Use pip

Install the package in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install motulator-bearingless
```

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install motulator-bearingless
```

If PowerShell does not allow running the activation script, allow it with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

The plots need a GUI library for *matplotlib*. On some Linux systems, it must be installed separately; see the [installation of *motulator*](https://aalto-electric-drives.github.io/motulator/installation.html).

## Option 2: Clone the Repository

Clone the repository to run the examples or to develop the package. On Linux and macOS:

```bash
git clone https://github.com/Aalto-Electric-Drives/motulator-bearingless
cd motulator-bearingless
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,doc]"
```

On Windows, create and activate the virtual environment as in Option 1. The optional dependencies are installed with the following extras:

| Extra | Purpose                                                                         |
| ----- | ------------------------------------------------------------------------------- |
| `dev` | Tests, linting, and type checking                                               |
| `doc` | Building this documentation                                                     |
| `fem` | Generating the finite-element dataset and training the networks, see {doc}`fem` |

The trained networks are evaluated with NumPy, so that PyTorch is needed only in the training. The examples with the gradient networks are run in the clone, since they load the trained networks and the dataset from `examples/`. The scripts downloaded from the example pages do not include these data files. Run an example, e.g.:

```bash
python examples/plot_3kw_bsyrm_dfc.py
```

The tests are run with `pytest`, and the documentation is built with the command:

```bash
sphinx-build -b html docs/source docs/build/html
```
