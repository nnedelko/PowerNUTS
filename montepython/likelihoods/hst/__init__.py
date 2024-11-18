import os
from jax import value_and_grad,jit
from montepython.likelihood_class import Likelihood_prior


class hst(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl_call(pdict):
        return -0.5 * (pdict['h0'] - pdict['h']) ** 2 / (pdict['sigma'] ** 2)
    def loglkl_and_grad(self, cosmo, data):
        loglkl = jit(value_and_grad(loglkl_call))
        return loglkl({'h0':cosmo.h,'h':self.h,'sigma':self.sigma})
    def loglkl_jit(self, cosmo, data):
        h = cosmo.h()
        loglkl = jit(loglkl_call)
        return loglkl({'h0':cosmo.h,'h':self.h,'sigma':self.sigma})
