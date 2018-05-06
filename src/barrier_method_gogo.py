# /usr/bin/python
# coding=utf-8

from __future__ import print_function, division
import white_projection
import white_rec
import barrier_method
import phantoms
import scipy, scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable

import collections


input_data_dict = phantoms.get_input('button_4_synth')

energy_grid = input_data_dict['grid']
gt_concentrations = input_data_dict['gt_concentrations']
# tmp = gt_concentrations.copy()
# gt_concentrations[0] = tmp[1]
# gt_concentrations[1] = tmp[0]
# gt_concentrations = tmp
source = input_data_dict['source']
pixel_size = input_data_dict['pixel_size']
element_numbers = input_data_dict['element_numbers']
element_absorptions = input_data_dict['element_absorptions']
ph_size = gt_concentrations.shape[1:]

# parameters of reconstruction
n_angles = 360   # projection angles
alpha = 0.7    # gradient step (aka relaxation aka learning rate)
beta_reg = 1e-2   # regularization coefficiten [update = alpha * (BP + beta * reg)]

# set up tomography operator
wp = white_projection.WhiteProjection(source=source,
    pixel_size=pixel_size, element_numbers=element_numbers,
    element_absorptions=element_absorptions, ph_size=ph_size,
    n_angles=n_angles, energy_grid=energy_grid)

wp.calc_sinogram_with_gt(gt_concentrations)

goal = (lambda x: wp.func(x), lambda x: wp.grad(x))

# init arrays for the concentrations:

conc_shape = (len(element_numbers), ph_size[0], ph_size[1])
# concentrations = np.zeros(shape=conc_shape, dtype=np.float64)
# init as truncated normal
concentrations = scipy.stats.truncnorm(
    a=-0.05, b=0.05, scale=0.05).rvs(size=conc_shape) 
# concentrations += np.array(gt_concentrations)
concentrations += 0.15

# test if goal works
def test_goal():
    global concentrations
    f = goal[0](concentrations)
    g = goal[1](concentrations)

    print('f size is', f.shape, 'g size is', g.shape)

    print('making some iters...')
    for i in range(100):
        concentrations -= goal[1](concentrations)

    plt.subplot(121)
    plt.imshow(concentrations[0])
    plt.subplot(122)
    plt.imshow(concentrations[1])
    plt.show()

# test_goal()

# ==============
# initialize c1*c2 regularisation
ir_beta = 0.05
ineq_reg = (lambda x: ir_beta * np.abs(x[0] * x[1]), 
            lambda x: ir_beta * white_rec.calc_uneq_reg_grad(x))

# ==============
# initialize C constraints

# c >= 0, i.e. -c <= 0
conc_non_negative = (lambda x: -x,
                     lambda x: -np.ones_like(x),
                     lambda x: np.clip(x, 0, x.max()))

# c <= 1, i.e. c - 1 <= 0
conc_less_than_one = (lambda x: x - 1,
                      lambda x: np.ones_like(x),
                      lambda x: np.clip(x, x.min(), 1))

# ==============
# Setup stat callback

# Result plotting function
def plot_x(x, iter, num, show=True, save=False, out='../../exp_results/movie'):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    im0 = axes[0].imshow(x[0], vmin=0, vmax=1)
    axes[0].set_title('$c_0$')
    im1 = axes[1].imshow(x[1], vmin=0, vmax=1)
    axes[1].set_title('$c_1$')

    plt.suptitle('Iteration %d' % iter)

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    plt.tight_layout()
    if save:
        try:
            os.makedirs(out)
        except os.error:
            pass

        plt.savefig(os.path.join(out, '%04d.png' % num),
            dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


opt_stats = collections.deque([], maxlen=5000)
stat_iter = 0



def plot_x_2(x, iter, num, show=True, save=False, out='../../exp_results/movie'):
    x1 = x[0]
    x2 = x[1]

    # need smth like x1 * (255, 0, 0) + x2 * (0, 255, 0)

    img = np.expand_dims(x1, 2) * np.array([[[255, 0, 0]]], dtype=x1.dtype) + \
          np.expand_dims(x2, 2) * np.array([[[0, 255, 0]]], dtype=x1.dtype)
    img = np.clip(img, 0, 255).astype(np.uint8)
    fig = plt.figure()
    plt.imshow(img)
    plt.title('Iteration %d' % iter)
    if save:
        try:
            os.makedirs(out)
        except os.error:
            pass

        plt.savefig(os.path.join(out, '%04d_comb.png' % num),
            dpi=150)
        np.savetxt(os.path.join(out, 'x0_%04d.nptxt' % num), x[0])
        np.savetxt(os.path.join(out, 'x1_%04d.nptxt' % num), x[1])
    if show:
        plt.show()
    else:
        plt.close(fig)


def stat_cb(statrecord):
    global opt_stats
    global stat_iter
    out = '../../exp_results/movie4'
    opt_stats.append(statrecord)
    x = statrecord[0].x
    plot_x(x, stat_iter * 10, stat_iter, show=False, save=True, out=out)
    plot_x_2(x, stat_iter * 10, stat_iter, show=False, save=True, out=out)
    stat_iter += 1

plot_x(gt_concentrations, 0, 0, show=False, save=True, out='../../exp_results/movie4/gt')
plot_x_2(gt_concentrations, 0, 0, show=False, save=True, out='../../exp_results/movie4/gt')

# ==============
# run barrier method with C constraints only
ans, opt_stats1 = barrier_method.barrier_method(concentrations,
    goal,
    reg_dict={'ineq_reg': ineq_reg},
    ineq_dict={'conc_non_negative': conc_non_negative,
               'conc_less_than_one': conc_less_than_one},
    n_iter=50,
    n_biter=20,
    t0=0.1,
    t_step=0.5,
    beta_reg=1.0,
    alpha=1.0,
    add_stat_cb=stat_cb,
    max_iter=5000,
    n_steps_without_progress=15
    )

plt.subplot(121)
plt.imshow(ans[0])
plt.subplot(122)
plt.imshow(ans[1])
plt.show()


# ==============
# stat plots
x = np.arange(len(opt_stats))
loss = np.array([stat[0].func for stat in opt_stats])
grad_norm = np.array([np.linalg.norm(stat[0].grad) for stat in opt_stats])

max_non_negative = np.array([(-np.exp(-stat[2]['conc_non_negative'].func)).max() for stat in opt_stats])
grad_norm_non_neg = np.array([np.linalg.norm(stat[2]['conc_non_negative'].grad) for stat in opt_stats])

max_less_than_one = np.array([(-np.exp(-stat[2]['conc_less_than_one'].func)).max() for stat in opt_stats])
grad_norm_less_one = np.array([np.linalg.norm(stat[2]['conc_less_than_one'].grad) for stat in opt_stats])

plt.subplot(221)
plt.plot(x, loss)
plt.title('goal loss')

plt.subplot(222)
plt.plot(x, grad_norm)
plt.title('goal grad norm')

plt.subplot(223)
plt.plot(x, max_non_negative, x , max_less_than_one)
plt.legend(['$c \geq 0$', '$c \leq 1$'])
plt.title('max constraints values')

plt.subplot(224)
plt.plot(x, grad_norm_non_neg, x, grad_norm_less_one)
plt.legend(['$c \geq 0$', '$c \leq 1$'])
plt.title('grad norms for barrier functions')

plt.show()

# plot_x(ans, 1000, 100, True, True)

def save_iteration_movie(opt_stats, out):
    for i, stat in enumerate(opt_stats):
        x = stat[0].x
        plot_x(x, i * 10, i, show=False, save=True, out=out)

# save_iteration_movie(opt_stats, '../../exp_results/movie3')
