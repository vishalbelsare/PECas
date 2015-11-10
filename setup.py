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

# todo: check dependencies

setup(
    name='pecas',
    version='0.4',
    author='Adrian Buerger',
    author_email='adrian.buerger@hs-karlsruhe.de',
    packages=['pecas'],
    package_dir={'pecas': 'pecas'},
        url='http://github.com/adbuerger/PECas/',
    license='LGPL',
    zip_safe=False,
    description='Parameter estimation using CasADi',

    install_requires=[ 

            "numpy>=1.8.2",
            "scipy", 
            "matplotlib",
            # "casadi?",

        ],

    use_2to3=True,
)
