# Notebooks

A series of [Jupyter Notebooks](https://www.jupyter.org) are provided that demonstrate how to use the TopoStats package
in a more interactive manner, calling individual steps. This is useful as it allows the user to explore interactively
and with rapid feedback the parameters that may need adjusting in order to process a batch of scans. The notebooks can
be found in the `notebook/` directory after cloning the GitHub repository.


| Notebook                                | Description                                                                                                             |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `00-Walkthrough-minicircle.ipynb`       | Step-by-step walkthrough of processing `minicircle.spm` from the `tests/resources/` directory.                          |
| `01-Walthrhgouh-interactive.ipynb`      | **Work in Progress** As above but uploading a single scan. Will be deployed in Google Colab/Binder for interactive use. |
| `02-Summary-statistics-and-plots.ipynb` | Plotting statistics interactively.                                                                                      |


## Installation

To be able to run the Notebooks you need some additional Python packages installed. You will have to clone the
repository from GitHub (see [installation](installation#cloning-from-github)) and then install the Notebook dependencies
with the following commands under your Virtual Environment (e.g. Conda)...

``` bash
cd TopoStats
pip install .[notebooks]
```

## Running Notebooks

Start a Jupyter server under the Virtual Environment you have installed the dependencies from.

``` bash
cd TopoStats/notebooks
jupyter notebook
```

For more on Jupyter Notebooks please refer to the [official documentation](https://docs.jupyter.org/en/latest/).