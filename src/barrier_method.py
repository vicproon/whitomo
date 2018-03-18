# /usr/bin/python
# coding=utf-8

from __future__ import print_function, division
import numpy as np
import collections
import helpers

# a named tuple of x values, function eval at x, and function gradient at x
# to store optimisation statistics
StepStat = collections.namedtuple('StepStat', ['x', 'func', 'grad'])


class BarrierFunction(object):
    """Class represents a barrier penalty to constraint in
    the interrior point method:
    $$
    \varphi(x) = - \log (-f(x))
    $$
    """

    def __init__(self, func, grad, proj=None):
        """accepts functions for evaluating f(x) and $\nabla f(x)$ at a point.
        """
        self.f = func
        self.g = grad
        self.num_feval = 0
        self.num_geval = 0
        self.proj = proj

    def eval(self, x, do_grad=True):
        """evals $f(x)$ and $\\nabla f(x)$ at point x.
        :param x point of evaluation
        :param do_grad turn of gradient evaluation

        returns tuple of f(x) and its grad
        """
        fx = self.f(x)
        phi = -np.log(-fx)
        # for infeasible point replace nans with infs
        phi[fx >= 0] = np.inf 
        self.last_f = fx
        self.last_phi = phi
        self.num_feval += 1
        grad_phi = None

        if do_grad:
            gx = self.g(x)
            n = fx.flatten().shape[0]
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

    def is_feasible(self, x):
        """ checks if x is a feasible point
        """
        return np.all(self.f(x) <= 0)

    def project(self, x):
        """projects point x to feasible set
        """
        if self.proj is None:
            return x

        return self.proj(x)

def barrier_method(x0, goal_function, reg_dict={}, ineq_dict={}, n_iter=200, t0=1.0, n_biter=10, alpha=0.1,
                   beta_reg = 1.0, t_step = 0.1,
                   add_stat_cb=None):
    '''performs minimization of goal_function with regularization functions from
    reg_dict subject to inequality constraints from ineq_dict (in form of g(x) <= 0)
    using gradient descent with barrier functions.
    t0 is start of barrier method step. n_biter is number of barrier iterations
    t = t / 10 each barrier iteration.
    n_iter is number if gradient descent iterations done for each barrier iteration.

    :param x0 initial FEASIBLE solution
    :param goal_function tuple of $(f(x), \\nabla f(x))$ function objects
    :param reg_dict a dict of {'reg_name': (r(x), \\nabla r(x))}. r(x) can be None.
    :param ineq_dict a dict of {'ineq_name': (g(x), \\nabla g(x)}, tuple is used as an arg for BarrierFunction
    :param n_iter number of gradient steps for each barreir methor t val
    :param t0 starting t parameter
    :param n_biter number of differet t steps in barrier method. 
    :param aplha gradient step size
    :param beta_reg common relaxation parameter for regularization terms (both reg and ineq)
    :param t_step exponential step of t decay
    :param add_stat_cb callback on iteration add. takes a 3-tuple of goal, reg and ineq stepstat vals
    '''
    bf_objs={}
    bf = {}

    for k,v in ineq_dict.iteritems():
        if type(v) is tuple:
            f = BarrierFunction(*v)
            bf_objs[k] = f
            bf[k] = (f.func, f.grad)
        elif issubclass(type(v), BarrierFunction):
            bf_objs[k] = v
            bf[k] = (v.func, v.grad)

    # check that initial guess is FEASIBLE:
    feasible = True
    for _, v in bf_objs.iteritems():
        if not v.is_feasible(x0):
            feasible = False
            break

    if not feasible:
        print('WARNING! the initial guess is not feasible.')
        
        print('trying to project in onto fieasible set..')
        for _, v in bf_objs.iteritems():
            x0 = v.proj(x0)

        feasible = True
        infeas_list = []
        for k, v in bf_objs.iteritems():
            if not v.is_feasible(x0):
                feasible = False
                infeas_list.append(k)

        if not feasible:
            print('ERROR! feasibility projection unsuccesfull')
            print('non-satisfied constraints: ', ' '.join(infeas_list))
            return x0, None


    t = t0
    x = x0
    opt_stats = []
    for i_biter in range(n_biter):

        print('starting ', i_biter, 'barrier iteration.')
        for i_iter in range(n_iter):
            goal_grad = goal_function[1](x)
            reg_grads = {k:v[1](x) for k,v in reg_dict.iteritems()}
            bf_grads = {k:v[1](x) for k,v in bf.iteritems()}

            grad = goal_grad
            for _, g in reg_grads.iteritems():
                grad += beta_reg * g

            for _, g in bf_grads.iteritems():
                grad += beta_reg * g * t

            step = -alpha * grad
            # collect stats for the step

            helpers.printProgressBar(i_iter, n_iter - 1, 
                prefix='barrier iter %d/%d' % (i_biter + 1, n_biter),
                suffix='done (%03d/%03d)' % (i_iter + 1, n_iter), length=50)

            if i_iter % 10 == 0:
                x_stat = x.copy()
                goal_stat = StepStat(x_stat, goal_function[0](x), goal_grad.copy())
                reg_stats = {}
                for k,v in reg_dict.iteritems():
                    reg_stats[k] = StepStat(x_stat, v[0](x), reg_grads[k].copy())

                bf_stats = {}
                for k,v in bf_objs.iteritems():
                    bf_stats[k] = StepStat(x_stat, v.last_phi.copy(), v.last_grad.copy())

                # print('\noptimisation progress: ', i_biter, 'barrier iteration', 
                #       'step: ', i_iter, 'out of', n_iter)

#                opt_stats.append((goal_stat, reg_stats, bf_stats))
                add_stat_cb((goal_stat, reg_stats, bf_stats))

            x = x + step

        t = t * t_step
        alpha = alpha * t_step
        n_iter = int(n_iter / t_step)
        if n_iter > 1000:
            n_iter = 1000
        print('')

    return x , opt_stats
