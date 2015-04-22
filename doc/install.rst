Get and install PECas
=====================

Within the next sections, you will get to know how to obtain and install PECas,
and what prerequisites have to be met in order to get PECas to work correctly.

Prerequisites
-------------

In order to use PECas, please make sure that
`Python <https://www.python.org/>`_ (currently supported version is Python 2.7) as well as
`Python Numpy <http://www.numpy.org/>`_ and
`PyLab <http://wiki.scipy.org/PyLab>`_ are installed on your system. E. g. on Ubuntu 14.04, this can easily be ensured by running

.. code:: bash

    sudo apt-get update
    sudo apt-get install python python-numpy python-scipy python-matplotlib

Note that you need root privileges to do this. Also, a recent version of `CasADi <http://casadi.org>`_ needs to be installed on your system. You can obtain a recent version from the `CasADi web page <http://casadi.org>`_, and follow the installation instructions there.

PECas has only been run on Ubuntu Linux systems so far, and has not been tested for Windows. However, if the prerequisites are met, usage on Windows should also be possible.

Get PECas
---------

The preferred way to obtain PECas is `directly from its
git repository <https://github.com/adbuerger/PECas>`_. You can then either clone the repository, or download the current files within a zip-archive.

To obtain the zip-file you do not need to have `git <http://git-scm.com/>`_ installed, but cloning the repository provides an easy way to receive updates on PECas by pulling from the repository in regular intervals.

On Ubuntu 14.04 again, you can install git and obatin PECas using the following commands

.. code:: bash

    sudo apt-get update
    sudo apt-get install git
    git clone git@github.com:adbuerger/PECas.git

Install PECas
-------------

You can install PECas on Ubuntu by running the command

.. code:: bash
    
    sudo python setup.py install

from within the PECas directory. Note that you need root privileges to be able to do this. If you are planning to install PECas on systems different from Ubuntu, this command might need to be adapted.

Update PECas
------------

If you recieved PECas by cloning the git repository, you can update the contents of your local copy by running

.. code:: bash
    
    git pull

from within the PECas directory. Afterwards, you need to install the recent version by running

.. code:: bash
    
    sudo python setup.py install

again. Note that due to the fact that PECas is still in a very early state, an update might lead to changes e. g. in syntax and functionalities, causing that some code that was written for older versions of PECas might not work anymore with future version, and will then need to be adjusted accordingly.
