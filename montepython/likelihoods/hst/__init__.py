import os
from jax import value_and_grad,jit
from montepython.likelihood_class import Likelihood_prior


class hst(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl_call(h0,h,sigma):
        return -0.5 * (h0 - h) ** 2 / (sigma ** 2)
    def loglkl_and_grad(self, cosmo, data):
        h = cosmo.h()
        loglkl = value_and_grad(loglkl_call,[0,1,2])
        return loglkl(cosmo.h,self.h,self.sigma)
    def loglkl_jit(self, cosmo, data):
        h = cosmo.h()
        loglkl = jit(loglkl_call)
        return loglkl(cosmo.h,self.h,self.sigma)
