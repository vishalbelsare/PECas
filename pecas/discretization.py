from abc import ABCMeta, abstractmethod

Class Discretization(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):

        pass


class NoDiscretization(Discretization):

    def __init__(self):

        pass


class ODEDiscretization(Discretization):

    def __init__(self):

        pass
