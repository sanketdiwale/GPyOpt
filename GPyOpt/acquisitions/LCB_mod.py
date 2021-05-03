# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles
import numpy as np
class AcquisitionLCB(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function with constant exploration weight.
    See:
    
    Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
    Srinivas et al., Proc. International Conference on Machine Learning (ICML), 2010

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(AcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight
        self.name = "LCB"

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        """
        m, v = self.model.predict(x, full_cov=False, include_likelihood=False)
        f_acqu = m - self.exploration_weight * np.sqrt(v)
        return -f_acqu

    def _compute_acq_withGradients(self, X):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        # m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X,include_likelihood=False)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))

        f_acqu = m - self.exploration_weight * np.sqrt(v)       
        df_acqu = dmdx - self.exploration_weight * dsdx
        return -f_acqu, -df_acqu

