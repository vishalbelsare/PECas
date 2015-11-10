#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2015 Adrian Bürger
#
# This file is part of PECas.
#
# PECas is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PECas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

from codecs import open
from os import path, environ

from subprocess import call
from setuptools.command.install import install

# todo: check dependencies

# Custom install class to be able to check for CasADi installtion as well, see:
# http://stackoverflow.com/questions/21915469/python-setuptools-install-
# requires-is-ignored-when-overriding-cmdclass

class CustomInstall(install):

    def run(self):

        print("Checking for compatible CasADi installation ...")

        try:

            import casadi

        except ImportError:

            errmsg = '''
It seems that you are missing CasADi or it's Python interface, or the interface
cannot be found.

Please visit www.casadi.org for information on how to obtain CasADi and it's
Python interface for your system.

CasADi is distributed under the LGPL license, meaning the code can be used
royalty-free even in commercial applications.
'''
            raise RuntimeError(errmsg)

        casadi_version = casadi.CasadiMeta.getVersion()

        if not float(casadi_version[:3]) >= 2.4:

            errmsg = '''
The version of CasADi found on your system is {0} <= 2.4.0-rc2, and therefor
not suitable for use with PECas.

If you think that you have already installed a newer version of CasADi on your
system, it might not be found due to an older version that still remained
on the system.

Please visit www.casadi.org for information on how to update CasADi correctly
and how to handle these problems.
'''.format(casadi_version)

            raise RuntimeError(errmsg)

        print("--> Compatible CasADi installation found (version {0})."\
            .format(casadi_version))

        install.run(self)


# Get the long description from the README file, see:
# https://github.com/pypa/sampleproject/blob/master/setup.py

with open(path.join(path.abspath(path.dirname(__file__)), 'README.rst'), \
    encoding='utf-8') as f:
        long_description = f.read()


on_rtd = environ.get('READTHEDOCS', None) == 'True'


if on_rtd:

    setup(

        name='pecas',
        version='0.5',

        author='Adrian Buerger',
        author_email='adrian.buerger@hs-karlsruhe.de',

        packages=['pecas'],
        package_dir={'pecas': 'pecas'},
        url='http://github.com/adbuerger/PECas/',

        license='LGPL',
        zip_safe=False,

        description='Parameter estimation using CasADi',
        long_description = long_description,

        platforms = ["Linux", "Windows"], 
        use_2to3=True,

    )


else:

    setup(
        name='pecas',
        version='0.5',

        author='Adrian Buerger',
        author_email='adrian.buerger@hs-karlsruhe.de',

        packages=['pecas'],
        package_dir={'pecas': 'pecas'},
        url='http://github.com/adbuerger/PECas/',

        license='LGPL',
        zip_safe=False,

        description='Parameter estimation using CasADi',
        long_description = long_description,

        install_requires=[ 

                "numpy>=1.8.2",
                "scipy", 
                "matplotlib",

            ],

        platforms = ["Linux", "Windows"], 
        use_2to3=True,

        cmdclass={"install" : CustomInstall},

    )
