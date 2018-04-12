# /usr/bin/python
# coding=utf-8

import sys
sys.path.append('../src')

import barrier_method
import numpy as np

class HoughBarrier(barrier_method.BarrierFunction):
    """Class represents special type of barrier function, 
    for which the constraint is formulated onto hough transform
    """
    def __init__(self, FP, BP, m, b, r_like):
        self.FP = FP
        self.BP = BP
        self.m = m
        self.b = b
        self.r = r_like

        def bc_func(x):
            f = FP(x)
            return b - f[m]

        def bc_grad(x):
            grad = np.zeros_like(r).flatten()
            grad[m] = -1
            return BP(grad)

        def bc_project(x, n_iter=100, alpha=1e-2):
            '''Projects x onto feasible point set
            '''
            f = FP(x)
            infeas_i = m & (f < b)
            x_new = x.copy()
            for i in range(n_iter):
                infeas_i = m & (f < b)
                if not np.any(infeas_i):
                    break
                f[~infeas_i] = 0
                f[infeas_i] -= b+ 1e-3 #..?
                x_new = x_new - alpha * BP(f)
                f = FP(x_new)
            # import ipdb; ipdb.set_trace()
            return x_new
        
        super(HoughBarrier, self).__init__(bc_func, bc_grad, bc_project)


    def eval(self, x, do_grad=True):
        fp = self.FP(x)
        fx = self.b - fp[self.m]
        phi = -np.log(-fx)
        phi[fx >= 0] = np.inf

        self.last_f = fx
        self.last_phi = phi
        self.num_feval += 1
        grad_phi = None

        if do_grad:
             grad = np.zeros_like(self.r).flatten()
             grad[self.m] = -1 / fx
             grad_phi = -self.BP(grad)
             self.last_g = grad
             self.last_grad = grad_phi
             self.num_geval += 1

        return (phi, grad_phi)