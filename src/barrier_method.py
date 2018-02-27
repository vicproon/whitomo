# /usr/bin/python
# coding=utf-8

import numpy as np


class BarrierFunction:
    """Class represents a barrier penalty to constraint in
    the interrior point method:
    $$
    \varphi(x) = - \log (-f(x))
    $$
    """

    def __init__(self, func, grad):
        """accepts functions for evaluating f(x) and $\nabla f(x)$ at a point.
        """
        self.f = func
        self.g = grad
        self.num_feval = 0
        self.num_geval = 0

    def eval(self, x, do_grad=True):
        """evals $f(x)$ and $\nabla f(x)$ at point x.
        :param x point of evaluation
        :param do_grad turn of gradient evaluation

        returns tuple of f(x) and its grad
        """
        fx = self.f(x)
        phi = -np.log(-self.f(x))
        self.last_f = fx
        self.last_phi = phi
        self.num_feval += 1
        grad_phi = None

        if do_grad:
            gx = self.g(x)
            grad_phi = - gx / fx
            self.last_g = gx
            self.last_grad = grad_phi
            self.num_geval += 1
        return (phi, grad_phi)

    def func(self, x):
        """Shortcut to eval only function value
        """
        return self.eval(x, do_grad=False)[0]

    def grad(self, x):
        """shortcut to eval only function gradient
        """
        return self.eval(x)[1]


def barrier_method():
    pass