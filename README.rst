Mowjsub
=======

.. image:: https://img.shields.io/pypi/v/mowjsub.svg
    :target: https://pypi.org/project/mowjsub/
    :alt: PyPI version

.. image:: https://readthedocs.org/projects/contsub/badge/?version=latest
    :target: https://contsub.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


Image and uv-plane continuum subtraction tools for spectral line data.

Installation
------------

PyPI (stable; recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install mowjsub

Latest (in development)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install git+https://github.com/laduma-dev/mowjsub.git


Using uv (Recommended development install)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The package requires Python >=3.11, so uv is the recommended way to install
from a local clone. If you do not have `uv <https://docs.astral.sh/uv/>`_ installed, install it first:

.. code-block:: bash

    pip install uv

Then clone the repository, navigate into it, activate your existing virtual
environment, and sync dependencies:

.. code-block:: bash

    git clone https://github.com/laduma-dev/mowjsub.git
    cd mowjsub

    source /path/to/your/.venv/bin/activate

    uv sync --python 3.12 --active #NOTE: --active ensures the package is installed in your activated virtual environment

The full documentation is available on `readthedocs <https://contsub.readthedocs.io/>`_.


