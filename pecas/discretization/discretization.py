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

from abc import ABCMeta, abstractmethod

from ..interfaces import casadi_interface as ci
from .. import inputchecks

class Discretization(object):

    __metaclass__ = ABCMeta

    @property
    def number_of_intervals(self):

        return self.time_points.size - 1

    @abstractmethod
    def __init__(self, system, tu):

        self.system = inputchecks.set_system(system)
        self.time_points = inputchecks.check_and_set_time_points_input(tu)
