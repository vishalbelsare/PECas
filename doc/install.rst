.. Copyright 2014-2015 Adrian Bürger
..
.. This file is part of PECas.
..
.. PECas is free software: you can redistribute it and/or modify
.. it under the terms of the GNU Lesser General Public License as published by
.. the Free Software Foundation, either version 3 of the License, or
.. (at your option) any later version.
..
.. PECas is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
.. GNU Lesser General Public License for more details.
..
.. You should have received a copy of the GNU Lesser General Public License
.. along with PECas. If not, see <http://www.gnu.org/licenses/>.

Get and install PECas
=====================

Within the next sections, you will get to know how to obtain and install PECas, and what prerequisites have to be met to get PECas working correctly.

Installation on Ubuntu 14.04
----------------------------

The following instructions show the installation on Ubuntu 14.04. If you are planning to install PECas on Linux systems different from Ubuntu 14.04, this commands need to be adapted accordingly.

Prerequesites
~~~~~~~~~~~~~

Python
^^^^^^

In order to use PECas, please make sure that
`Python <https://www.python.org/>`_ (currently supported version is Python 2.7) as well as
`Python Numpy <http://www.numpy.org/>`_ (>= 1.8), 
`PyLab <http://wiki.scipy.org/PyLab>`_ and `Python Setuptools <http://wiki.ubuntuusers.de/Python_setuptools>`_ are installed on your system. This can easily be ensured by running

.. code:: bash

    sudo apt-get update
    sudo apt-get install python python-numpy python-scipy python-matplotlib python-setuptools --install-recommends

If you want to install PECas using `pip <https://wiki.ubuntuusers.de/pip>`_, which is the recommended and easiest way, you also need to install pip by running

.. code:: bash

    sudo apt-get install python-pip

Also, you might want to install the `Spyder IDE <https://pythonhosted.org/spyder/>`_ for working with Python. You can install it by running

.. code:: bash

    sudo apt-get install spyder

.. note:: These commands require root privileges. In case you do not have root privileges on your system, consider using `Miniconda <http://conda.pydata.org/docs/install/quick.html>`_ to install Python and the necessary modules into a user-writeable directory.

CasADi
^^^^^^

For PECas to work correctly, a recent version of `CasADi <http://casadi.org>`_ (>= 2.4.0-rc2) needs to be installed on your system. Installation instructions for CasADi can be found  `here <https://github.com/casadi/casadi/wiki/InstallationInstructions>`_. You can simpy use `"Option 1: Download a binary installation" <https://github.com/casadi/casadi/wiki/InstallationInstructions#option-1-download-a-binary-installation-recommended>`_ and download and unpack e. g. `this archive <http://sourceforge.net/projects/casadi/files/CasADi/2.4.1/linux/casadi-py27-np1.9.1-v2.4.1.tar.gz/download>`_.

.. note:: Some plugins for CasADi require extra prerequisites to work on Linux. For a list of the required libraries and installation instructions, see `the corresponding section in the CasADi installation guide <https://github.com/casadi/casadi/wiki/linuxplugins>`_. If something goes wrong with executing CasADi and/or PECas, missing one or more of these libraries might be the reason.

In addition to unpacking the archive you just obtained, please make sure that the unpacked folder that contains CasADi can be found by Python permanently. As mentioned in the CasADi installation instructions, this can e. g. be ensured by adding the CasADi directory to the :code:`PYTHONPATH` variable permanently. Just open the file :code:`~/.bashrc` on your system with your favorite text editor, and add the line

.. code:: bash

    export PYTHONPATH=$PYTHONPATH:/<path>/<to>/<casadi>/<folder>

while :code:`/<path>/<to>/<casadi>/<folder>` needs to be adapted to the path of your unpacked archive. Afterwards, save these changes and close all open terminals. Now open a new terminal, and have a look at the value of :code:`PYTHONPATH` by typing

.. code:: bash

    echo $PYTHONPATH

It should now contain at least the path your just inserted. If everything went well, you should be able to open a Python console, and execute the following commands

.. code::

    >>> from casadi import *
    >>> x = MX.sym("x")
    >>> print jacobian(sin(x),x)

without recieving error messages.

.. _option1:

Option 1: Get PECas using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

PECas is listed on the `Python Package Index <https://pypi.python.org/pypi?name=pecas&version=0.5&:action=display>`_. You can obtain it from there by simply running

.. code:: bash

    sudo pip install pecas

.. note:: These commands require root privileges. In case you do not have root privileges ony your system, consider :ref:`Option 2: Get PECas from GitHub <option2>`.

Upgrades
^^^^^^^^

Upgrades to new releases of PECas can simply be obtained by running

.. code:: bash

    sudo pip install pecas --upgrade

.. note:: You need root privileges to do this!

.. _option2:

Option 2: Get PECas from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

You can also obtain the PECas module `directly from its
GitHub repository <https://github.com/adbuerger/PECas>`_. You can either clone the repository, or download the contained files within a compressed archive. To just obtain an archive, you do not need to have `git <http://git-scm.com/>`_ installed, but cloning the repository provides an easy way to receive updates on PECas by pulling from the repository.

You can install git by running

.. code:: bash

    sudo apt-get update
    sudo apt-get install git

.. note:: These commands require root privileges. In case you do not have root priviliges and git ist not installed on you system, consider downloading the archive from the `GitHub page <https://github.com/adbuerger/PECas>`_ using your favorite web browser instead of cloning the git repository.

Afterwards, you can clone the repository using the following commands

.. code:: bash

    git clone git@github.com:adbuerger/PECas.git

and install PECas by running

.. code:: bash
    
    sudo python setup.py install

from within the PECas directory.

.. note:: This command requires root privileges. In case you do not have root priviliges, consider adding the PECas directory to :code:`PYTHONPATH`, as described above for CasADi.

Upgrades
^^^^^^^^

If you recieved PECas by cloning the git repository, you can update the contents of your local copy by running

.. code:: bash
    
    git pull

from within the PECas directory. In case you did not clone the repository, you would again need to download a compressed archive.

Afterwards, you need to install the recent version again by running

.. code:: bash
    
    sudo python setup.py install

.. note:: This command requires root privileges.

.. warning:: If you installed PECas by adding the directory to :code:`PYTHONPATH`, just place the newly obtained files in the previously defined path to upgrade to a new version of PECas. You do not not need to add the directory again to :code:`PYTHONPATH` then. Also, make sure not to add multiple versions of PECas to :code:`PYTHONPATH`, since this might lead to conflicts.


Installation on Windows
-----------------------

The following instructions have been tested on Windows 7 64 bit.

.. note:: You need to have administrator rights on your system to be able to follow the instructions below.

Prerequesites
~~~~~~~~~~~~~

Python
^^^^^^

The easiest way to meet the prerequesites for PECas and CasADi on a Windows system might be to install a recent version of `Python(x,y) <http://python-xy.github.io/>`_, which is also the procedure recommended by the CasADi developers. It is recommended to do a "Full" installation. In the following, the instructions also assume that you are installing Python(x,y) and all components with their default paths.

CasADi
^^^^^^

After installing Python(x,y), a recent version of `CasADi <http://casadi.org>`_ (>= 2.4.0-rc2) needs to be installed on your system. Installation instructions for CasADi can be found  `here <https://github.com/casadi/casadi/wiki/InstallationInstructions>`_. You can simpy use `"Option 1: Download a binary installation" <https://github.com/casadi/casadi/wiki/InstallationInstructions#option-1-download-a-binary-installation-recommended>`_ and download and unpack e. g. `this archive <http://sourceforge.net/projects/casadi/files/CasADi/2.4.1/windows/casadi-py27-np1.9.1-v2.4.1.zip/download>`_, which is compatible with the previous Python(x,y) installation.

After unpacking the archive, go to :code:`My Computer > Properties > Advanced System Settings > Environment Variables`. If a variable :code:`PYTHONPATH` already exists, apply the full path to the CasADi folder to the end of the variable value, and separate this new path from the ones already contained by :code:`;`. If :code:`PYTHONPATH` does not yet exist on the system, create a new environmental variable with this name, and fill in the path to the unpacked CasADi folder.

.. _option1win:

Option 1: Get PECas using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

PECas is listed on the `Python Package Index <https://pypi.python.org/pypi?name=pecas&version=0.5&:action=display>`_. Since you installed `pip <https://wiki.ubuntuusers.de/pip>`_ with Python(x,y), you can obtain PECas by opening a command line and running

.. code:: bash

    pip install pecas

.. note:: If you have problems obtaining PECas with pip (which can e. g. be caused by a company's proxy server) consider :ref:`Option 2: Get PECas from GitHub <option2win>`.

Upgrades
^^^^^^^^

Upgrades to new releases of PECas can simply be obtained by running

.. code:: bash

    pip install pecas --upgrade


.. _option2win:

Option 2: Get PECas from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

You can also obtain the PECas module `directly from its
GitHub repository <https://github.com/adbuerger/PECas>`_. Since installing git is more time-consuming on Windows then it is on most Linux systems, it is recommended (at least for less experienced users) to just download the contained files for PECas within a compressed archive.

Afterwards, unpack the archive, and install PECas by running

.. code:: bash
    
    python setup.py install

from the command line, within the unzipped folder.

.. note:: If this procedure is for some reason not applicable for you, you can consider adding the PECas directory to :code:`PYTHONPATH` instead, as described above for CasADi.

Upgrades
^^^^^^^^

For upgrading PECas, you would again need to download a compressed archive.

Afterwards, you need to install the recent version by again running

.. code:: bash
    
    python setup.py install

.. warning:: If you installed PECas by adding the directory to :code:`PYTHONPATH`, just place the newly obtained files in the previously defined path to upgrade to a new version of PECas. You do not not need to add the directory again to :code:`PYTHONPATH` then. Also, make sure not to add multiple versions of PECas to :code:`PYTHONPATH`, since this might lead to conflicts.

Recommendations
---------------

To speed up computations in PECas, it is recommended to install `HSL for IPOPT <http://www.hsl.rl.ac.uk/ipopt/>`_. On how to install the solvers and for further information, see the page `Obtaining HSL <https://github.com/casadi/casadi/wiki/Obtaining-HSL>`_ in the CasADi wiki.
