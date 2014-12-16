Get and install PECas
=====================

Within the next sections, you will get to know how to obtain and install PECas,
and what prerequesites have to be met in order to get PECas to work correctly.

Prerequesites
-------------

In order to use PECas, please make sure that
`Python <https://www.python.org/>`_ as well as
`Python Numpy <http://www.numpy.org/>`_
and a recent version of `CasADi <https://github.com/casadi/casadi/wiki>`_ are
installed on your system.

PECas has only been run on Ubuntu Linux systems so far, and has not been tested for Windows. However, if the prerequesites are met, usage on Windows should also be possible.

Get PECas
---------

The preferred way to obtain PECas is `directly from its
git repository <https://github.com/adbuerger/PECas>`_. You can then either clone the repository, or download the current files within a zip-archive.

To obtain the zip-file you do not need to have `git <http://git-scm.com/>`_ installed, but cloning the repository provides an easy way to recieve updates on PECas by pulling from the repository in regular intervals.

Install PECas
-------------

If you meet the requirements in from the above sections, you can run PECas directly from within the 
directory in which you obtained or unpacked it,
or copy ``pecas.py`` to another destination.
If you
want to use PECas from multiple locations and not only from the containing
directory, consider adding the PECas directory to your PYTHONPATH by running

.. code:: bash
    
    export PYTHONPATH=$PYTHONPATH:/path/to/PECas

inside your shell. If you also want to permanently add PECas to your
PYTHONPATH, you can e. g. on Ubuntu add the above line at the end of your
``.bashrc`` file, which is usually located in your home directory.
