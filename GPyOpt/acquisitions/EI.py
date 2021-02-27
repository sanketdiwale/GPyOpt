# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles
import numpy as np
from IPython import embed
class AcquisitionEI(AcquisitionBase):
    """
    Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.0):
        self.optimizer = optimizer
        super(AcquisitionEI, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.name = "EI"

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return AcquisitionEI(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x,with_noise=False)
        s = np.sqrt(s)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * phi + (fmin - m - self.jitter)*Phi 
        # embed()
        # if isinstance(s, np.ndarray): # correcting the internal heuristic used in get_quantiles
        #     f_acqu[s<np.sqrt(0.01+self.model.noise_var)] = max(0,fmin - m[s<np.sqrt(self.model.noise_var)] - self.jitter)
        #     embed()
        # elif s< 0.1+np.sqrt(self.model.noise_var):
        #     # embed()
        #     f_acqu = max(0,fmin - m - self.jitter)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        s = np.sqrt(s)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        # f_acqu = s * (u * Phi + phi)
        f_acqu = s * phi + (fmin - m - self.jitter)*Phi 
        # if isinstance(s, np.ndarray): # correcting the internal heuristic used in get_quantiles
        #     f_acqu[s<1e-10] = max(0,fmin - m[s<1e-10] - self.jitter)
        # elif s< 1e-10:
        #     f_acqu = max(0,fmin - m - self.jitter)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu
