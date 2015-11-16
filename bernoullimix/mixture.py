from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

class BernoulliMixture(object):

    _number_of_components = None
    _number_of_dimensions = None

    _mixing_coefficients = None
    _emission_probabilities = None

    def __init__(self, number_of_components,
                 number_of_dimensions,
                 mixing_coefficients,
                 emission_probabilities):
        """
        Initialises a Bernoulli Mixture Model

        :param number_of_components: number of components in the model (i.e. K)
        :param number_of_dimensions: number of independent Bernoullis in the model (i.e. D)
        :param mixing_coefficients: K-dimensional array of the mixture components for the data
        :param emission_probabilities: (K, D)-dimensional matrix of the probabilities of emitting
                                       `True` in each bernoulli, given the component.
        """

        self._number_of_components = int(number_of_components)
        self._number_of_dimensions = int(number_of_dimensions)

        self._mixing_coefficients = np.asarray(mixing_coefficients, dtype=float)
        self._emission_probabilities = np.asarray(emission_probabilities, dtype=float)

        self._validate()

    def _bounded_between_zero_and_one(self, array_):
        return np.all((array_ >= 0) & (array_ <= 1))

    def _validate(self):
        K = self.number_of_components
        D = self.number_of_dimensions

        if not self.mixing_coefficients.shape == (K, ):
            raise ValueError('Wrong shape of mixing coefficients provided. '
                             'Expected {}, got {}'.format((K,), self.mixing_coefficients))

        if not np.sum(self.mixing_coefficients) == 1:
            raise ValueError('Mixing coefficient probabilities do not sum to one.')

        if not self._bounded_between_zero_and_one(self.mixing_coefficients):
            raise ValueError('Mixing coefficients not bounded between 0 and 1')

        if not self.emission_probabilities.shape == (K, D):
            raise ValueError('Wrong shape of emission probabilities matrix. '
                             'Expected {}, got {}'.format((K,D), self.emission_probabilities.shape))

        if not self._bounded_between_zero_and_one(self.emission_probabilities):
            raise ValueError('Emission probabilities not bounded between 0 and 1')

    @property
    def number_of_components(self):
        return self._number_of_components

    @property
    def number_of_dimensions(self):
        return self._number_of_dimensions

    @property
    def emission_probabilities(self):
        return self._emission_probabilities

    @property
    def mixing_coefficients(self):
        return self._mixing_coefficients

