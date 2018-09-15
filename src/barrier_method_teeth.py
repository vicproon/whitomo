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

exp_number = 105
experiment_desctiption = u'''
Эксперимент 105

Нумерация экспериментов по восстановлению зуба с помощью white_rec
начинается со 101.

Данные: измерения зуба со свинцовой вставкой на молибденовом аноде, 
по высоте 820 слайс.
Даунскейл до размера 256х256, углов проекции - ???
Спектральные кривые поглощения в соответствии с xraylib и физикой.
Спектр анода соответстует присланному.


В следующих экспериментах планируется уменьшить ширину фантома, 
сделать фиктивные спектры.
'''

inp_name = 'teeth_data_256'
input_data_dict = phantoms.get_input(inp_name)

energy_grid = input_data_dict['grid']
angles = input_data_dict['angles']
proj_data = input_data_dict['proj_data']
source = input_data_dict['source']
pixel_size = input_data_dict['pixel_size']
element_numbers = input_data_dict['element_numbers']
element_absorptions = input_data_dict['element_absorptions']
ph_size = (proj_data.shape[1], proj_data.shape[1])
n_angles = angles.shape[0]

plt.figure()
plt.subplot(211)
plt.plot(energy_grid[:,0], element_absorptions[0])
plt.plot(energy_grid[:,0], element_absorptions[1])
plt.legend(['Ca', 'Pb'])
plt.xlabel(u'энергия, кЭв')
plt.ylabel(u'$\\kappa$, отн.ед.')
plt.grid()
plt.title(u'Спектральные коэффициенты поглощения')

plt.subplot(212)
plt.plot(energy_grid[:,0], source)
plt.xlabel(u'энергия, кЭв')
plt.ylabel(u'интенсивность, отн.ед.')
plt.grid()
plt.title(u'Спектр источника')

plt.tight_layout()
plt.savefig('zub_spectre.png', dpi=350)
plt.show()
sys.exit(0)

# parameters of reconstruction
alpha = 1         # gradient step (aka relaxation aka learning rate)
beta_reg = 1e-2   # regularization coefficiten [update = alpha * (BP + beta * reg)]

# set up tomography operator
wp = white_projection.WhiteProjection(source=source,
    pixel_size=pixel_size, element_numbers=element_numbers,
    element_absorptions=element_absorptions, ph_size=ph_size,
    energy_grid=energy_grid, sinogram=proj_data,
    # n_angles=400),
    angles=angles)
ph_size = wp.ph_size

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
ir_beta = 1.0
ineq_reg = (lambda x: ir_beta * np.abs(x[0] * x[1]), 
            lambda x: ir_beta * white_rec.calc_uneq_reg_grad(x))

# ==============
# initialize C constraints

# consider clipping with some small epsilon remaining (as log(0) is still infty)

# c >= 0, i.e. -c <= 0
conc_non_negative = (lambda x: -x,
                     lambda x: -np.ones_like(x),
                     lambda x: np.clip(x, 1e-4, x.max()))

# c <= 1, i.e. c - 1 <= 0
conc_less_than_one = (lambda x: x - 1,
                      lambda x: np.ones_like(x),
                      lambda x: np.clip(x, x.min(), 1 - 1e-4))

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


opt_stats = collections.deque([], maxlen=500)
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

def analyse_grads(x):
    plt.subplot(221)
    plt.imshow(x[0])
    plt.subplot(222)
    plt.imshow(x[1])

    wfp = wp.FP_white(x)[0].reshape(proj_data.shape)
    plt.subplot(223)
    plt.imshow(wfp)
    plt.subplot(224)
    plt.imshow(wp.sinogram.reshape(proj_data.shape))

    plt.show()

def stat_cb(statrecord):
    global opt_stats
    global stat_iter
    out = '../../exp_results/movie_teeth_%03d' % exp_number
    opt_stats.append(statrecord)
    x = statrecord[0].x
    plot_x(x, stat_iter * 10, stat_iter, show=False, save=True, out=out)
    plot_x_2(x, stat_iter * 10, stat_iter, show=False, save=True, out=out + '/merged')
    stat_iter += 1

n_iter = 50
n_biter = 20
t0 = 1e-4
t_step = 0.5
max_iter = 5000
n_steps_without_progress=15

# ==============
# output of experiment description
def output_exp_stats():
    out = '../../exp_results/movie_teeth_%03d' % exp_number
    try:
        os.makedirs(out)
    except OSError:
        pass

    outf = os.path.join(out, 'description_%03d.txt' % exp_number)
    with open(outf, 'wb') as f:
        f.write(experiment_desctiption.encode('UTF-8'))
        f.write('ph_size = [%d, %d]\n' % ph_size)
        f.write('inp_name = %s\n' % inp_name)
        f.write('pixel_size = %f\n' % pixel_size)
        f.write('element_numbers = [' + 
                ', '.join(['%d' % en for en in element_numbers]) + 
                ']\n')

        f.write('alpha = %f\n' % alpha)
        f.write('beta_reg = %f\n' % beta_reg)
        f.write('n_iter = %d\n' % n_iter)
        f.write('n_biter = %d\n' % n_biter)
        f.write('t0 = %f\n' % t0)
        f.write('t_step = %f\n' % t_step)
        f.write('max_iter = %d\n' % max_iter)
        f.write('n_steps_without_progress = %d\n' % n_steps_without_progress)

output_exp_stats()
# ==============
# run barrier method with C constraints
ans, opt_stats1 = barrier_method.barrier_method(concentrations,
    goal,
    reg_dict={'ineq_reg': ineq_reg},
    ineq_dict={'conc_non_negative': conc_non_negative,
               'conc_less_than_one': conc_less_than_one},
    n_iter=n_iter,
    n_biter=n_biter,
    t0=t0,
    t_step=t_step,
    beta_reg=beta_reg,
    alpha=alpha,
    add_stat_cb=stat_cb,
    max_iter=max_iter,
    n_steps_without_progress=n_steps_without_progress
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
