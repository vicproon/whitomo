# /usr/bin/python
# coding=utf-8

# todo рандомные углы при проекции 
# todo рандомные пиксели изображения для регуляризации произведением 
# todo все буфферы вынести в класс
from __future__ import print_function

import numpy as np

import os
import os.path

import matplotlib.pyplot as plt
import astra

import scipy.stats
import sys

# local imports
import batch_filter
import phantoms
from astra_proxy import *

# get input data and load proxy objects
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

# output params
exp_root = '../../exp_output'
experiment_name = 'exp38'
exp_dir = os.path.join(exp_root, experiment_name)
try:
    os.mkdir(exp_dir)
except:
    pass

with open(exp_dir + '/readme.txt', 'w') as f:
    notes = ['''Эксперимент40: оптимизация с помощью барьерного метода''', 
              'фантом: button_4_synth',
              'без батч-фльтров',
              'n_angles: %d' % n_angles,
              'alpha: %.3f' % alpha,
              'beta_reg: %.3f' % beta_reg,
              '']
    f.writelines('\n'.join(notes + ['']))


# setup astra geometry
proj_geom = astra.create_proj_geom(
    'parallel',
    1.0,
    # This should be enough to register full object.
    int(np.sqrt(2) * ph_size[0] + 1),
    np.linspace(0, np.pi, n_angles)
)
vol_geom = astra.create_vol_geom(*ph_size)
projector = astra.create_projector('linear', proj_geom, vol_geom)
proj_shape = (n_angles, proj_geom['DetectorCount'])

# setup placeholders for results

conc_shape = (len(element_numbers), ph_size[0], ph_size[1])
# concentrations = np.zeros(shape=conc_shape, dtype=np.float64)
# init as truncated normal
concentrations = scipy.stats.truncnorm(
    a=-0.1, b=0.1, scale=0.1).rvs(size=conc_shape) 
# concentrations += np.array(gt_concentrations)
concentrations += 0.1


# # Check if GT concentrations give loss < 0.35
# concentrations = gt_concentrations.copy

# Check if starting near GT leads to GT optimum with zero loss
# concentrations += gt_concentrations

# What if we not only noise concentrations, but mix them a little bit?
# c = concentrations.copy()
# concentrations[0] = 0.8 * c[0] + 0.2 * c[1]
# concentrations[1] = 0.2 * c[0] + 0.8 * c[1]


def integrate(ar, energy_grid):
    res_shape = ar.shape[:-1]
    num_en = energy_grid.shape[0]
    return np.sum(ar.reshape(-1, num_en) * energy_grid[:, 1]
                  .reshape(1, -1), axis=1).reshape(res_shape)


sum_energy = integrate(source, energy_grid)
source /= sum_energy


def FP_white(energy, source, pixel_size, concentrations):
    """ white forward projection
    """
    global proj_geom
    global vol_geom
    global projector
    K = len(concentrations)

    def FP(x):
        return gpu_fp(proj_geom, vol_geom, x, projector)
    conc_fp = np.array([FP(c) for c in concentrations])
    flat_fp = conc_fp.reshape(K, -1, 1)
    ea = element_absorptions.reshape(K, 1, -1)
    # print('flat_fp ', flat_fp.shape, ' * ea ',
    #       ea.shape, ' = ', (flat_fp * ea).shape)
    exp_arg = -pixel_size * np.sum(flat_fp * ea, axis=0)

    # plt.imshow(exp_arg.reshape(proj_shape[0], proj_shape[1], -1)[:,:, 10])
    # plt.show()
    exp = np.exp(exp_arg)
    # print('exp.shape is :', exp.shape)
    Integral = integrate(source.reshape(1, -1) * exp, energy)
    return Integral, exp_arg, exp, flat_fp

# calculate sinogram - input for reconstruction
sinogram, exp_arg, exp, flat_fp, = FP_white(
    energy_grid, source, pixel_size, gt_concentrations)


def calc_mu(energy, source, exp, element_absorptions):
    """ wighted residuals
    """
    mu = np.array([-integrate(
        source.reshape(1, -1) * pixel_size * exp * ea.reshape(1, -1),
        energy)
        for ea in element_absorptions])

    return mu




bp_conc_buffer = np.zeros_like(concentrations)


def BP_white(energy, source, exp, concentrations, Q):
    """white bp step
    """
    mu = calc_mu(energy, source, exp, element_absorptions)
    global bp_conc_buffer
    conc = bp_conc_buffer

    def BP(x):
        return gpu_bp(proj_geom, vol_geom, x, projector)

    bp_arg = Q * mu

    for k, cc in enumerate(conc):
        conc[k] = - BP(bp_arg[k].reshape(proj_shape))
    return conc, mu


c = np.array(concentrations)

fp = None

uneq_reg_buffer = None# np.zeros_like(concentrations)


def calc_uneq_reg_grad(c):
    global uneq_reg_buffer
    if uneq_reg_buffer is None:
        uneq_reg_buffer = np.zeros_like(c)
    for i, cc in enumerate(uneq_reg_buffer):
        uneq_reg_buffer[i] = (c[np.arange(len(c)) != i] * np.sign(c[0] * c[1])).sum(axis=0)
    return uneq_reg_buffer

def calc_triv_conc_grad(c):
    '''||c(c - 1)||^2 '''
    # 2 * c * (c - 1) * (2 * c - 1) = 2 * (2 * c ** 3 - c ** 2 - 2 * c ** 2 + c) =
    # 4 * c ** 3 - 3 * c ** 2 + c
    res = 4 * c.copy()
    res = c * (res - 3)
    return c * (res + 1)

#bf = batch_filter.BatchFilter(concentrations.shape[1:], (6, 6))
bf = batch_filter.GaussBatchGen(concentrations.shape[1:], 16, 100)

def Iteration(c,
              batch_filtering=False,
              clip_concentrations=False, 
              dissimilarity_constraint=False,
              triv_conc_constraint=False,
              iter_num=None):    
    global fp
    fp, exp_arg, exp, flat_fp = FP_white(energy_grid, source, pixel_size, c)
    q = (sinogram - fp)
    bp_grad, mu = BP_white(energy_grid, source, exp, c, q)
    grads = {'bp_grad': bp_grad}
    reg_grad = 0;
    reg_loss = 0;
    if dissimilarity_constraint:
        reg_grad = calc_uneq_reg_grad(c)
        grads['reg_grad'] = reg_grad
        # gradient update with regularisation
        reg_loss += np.linalg.norm(beta_reg * c[0] * c[1])

    tc_grad = 0
    if triv_conc_constraint:
        tc_grad = calc_triv_conc_grad(c)
        grads['triv_conc'] = tc_grad

    step = alpha * (bp_grad + beta_reg * (reg_grad + tc_grad))
    
    if batch_filtering:
        masks = np.array([bf.new_mask(), bf.new_mask()])
        c = c - step * masks # batch_filtering is ON
    else:
        if iter_num is not None:
            if iter_num % 4 == 0:
                step[1] = 0
            else:
                step[0] = 0
        c = c - step

    hough_loss = np.linalg.norm(q)
    full_loss = hough_loss + reg_loss

    if (clip_concentrations):
        c_new = np.clip(c, 0, 1)
        grads['clip_grad'] = c_new - c
        c = c_new

    return c, mu, q, full_loss, reg_grad, reg_loss, grads


def showres(res1, res2, iter_num=None, suffix='iteration'):
    if iter_num % 30 == 0 or iter_num is None:
        f = plt.figure(1)
        plt.subplot(231)
        plt.imshow(res1[0], interpolation='none')

        plt.subplot(232)
        plt.imshow(res1[1], interpolation='none')

        plt.subplot(233)
        plt.imshow(np.clip(np.around(res1[0] / res1[1], 4), -0.5, 2.0), interpolation='none')

        plt.subplot(234)
        plt.imshow(res2[0], interpolation='none')

        plt.subplot(235)
        plt.imshow(res2[1], interpolation='none')

        plt.subplot(236)
        plt.imshow(np.around(res2[0] / res2[1], 4), interpolation='none')
        # plt.pause(0.1)
        # plt.hold(hold)
        plt.savefig(exp_dir + '/%s_%02d.png' % (suffix, iter_num))
      
    # elif iter_num is None:
        if iter_num is None:
            plt.show()
        plt.close(f)

def showres2(c, grads, iter_num=None, suffix='iteration'):
    """ Plots concentrations and gradients of all additive grad pairs
    """
    if iter_num is not None and iter_num % 30 != 0:
        return;

    # Calc multiplot shape
    n_elements = len(c)
    n_grads = len(grads)
    plot_rows = n_grads + 1
    plot_cols = n_elements

    # Create matplotlib objects
    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols,
                             #sharex=True, sharey='row',
                             figsize=(2 * plot_cols + 0.5, 2 * plot_rows), num=1)
    
    # Set correct limits for all axes
    for ax in axes.ravel():
        ax.set_xticks(np.linspace(0, c.shape[2], 5))
        ax.set_yticks(np.linspace(0, c.shape[1], 5))

    # Intensity limits for concentrations
    v_min = -0.1
    v_max = 1.1

    # Plot images of concentrations
    for i in range(n_elements):
        ax = axes[0, i]
        ax.imshow(c[i], interpolation='none', vmin=v_min, vmax=v_max)
        ax.set_title('Element %d' % i)

        # Use ylabel as row title for multiplot.
    axes[0, 0].set_ylabel('Concentrations')

    # Plot their BP grads
    bp_grad = grads['bp_grad'].reshape(c.shape) # do we need to reshape here?
    for i in range(n_elements):
        ax = axes[1, i]
        ax.imshow(bp_grad[i], interpolation='none', vmin=v_min, vmax=v_max)
    axes[1, 0].set_ylabel('bp_grad')

    # Plot other grads.
    other_grads = [k for k in grads.keys() if k != 'bp_grad']
    for r, k in enumerate(other_grads):
        grad = grads[k]
        
        for i in range(n_elements):
            ax = axes[2 + r, i]
            ax.imshow(grad[i], interpolation='None', vmin=v_min, vmax=v_max)

        # Use ylabel as row title for multiplot
        axes[2 + r, 0].set_ylabel(k)

    if iter_num is not None:
        fig.savefig(exp_dir + '/%s_%02d.png' % (suffix, iter_num))
    else:
        fig.show()

    plt.close(fig)


if __name__ == '__main__':
    # итерационная минимизация.
    iters = 1000
    stat = np.zeros(shape=(iters, 2 + len(concentrations)), dtype=np.float64)
    do_batch_filtering = False
    do_clipping = True # включаем клиппинг
    do_dissimilarity_constraint = True
    do_triv_conc = True
    for i in range(iters):
        #if i % 30 == 0 and i >= 750:
        #    do_batch_filtering = False
        #else:
        #    do_batch_filtering = True

        c, mu, q, loss, reg_loss, reg_grad, grads = Iteration(c, 
                                            batch_filtering=do_batch_filtering,
                                            clip_concentrations=do_clipping,
                                            dissimilarity_constraint = do_dissimilarity_constraint,
                                            triv_conc_constraint=do_triv_conc,
                                            iter_num=i)
        # showres(mu.reshape(2, *proj_shape), i, 'mu')
        # showres(c, mu.reshape(2, *proj_shape), i)
        showres2(c, grads, i)
        # print('reg loss is ', reg_loss)

        c_acc = np.linalg.norm(c - gt_concentrations, axis=(1, 2))
        sum_acc = np.linalg.norm(c - gt_concentrations)
        stat[i, :] = np.array([loss, sum_acc] + [ac for ac in c_acc])

        print('iter: %03d, min(c) = %.2f, max(c) = %.2f; loss = %.2f' %
              (i, c.min(), c.max(), loss))

    # for i in range(iters, 5 * iters):
    #     c, mu, q, loss, reg_loss, reg_grad = Iteration(c, batch_filtering=False)
    #     # showres(mu.reshape(2, *proj_shape), i, 'mu')
    #     showres(c, mu.reshape(2, *proj_shape), i)
    #     # print('reg loss is ', reg_loss)
    # 
    #     c_acc = np.linalg.norm(c - gt_concentrations, axis=(1, 2))
    #     sum_acc = np.linalg.norm(c - gt_concentrations)
    #     stat[i, :] = np.array([loss, sum_acc] + [ac for ac in c_acc])
    # 
    #     print('iter: %03d, min(c) = %.2f, max(c) = %.2f; loss = %.2f' %
    #           (i, c.min(), c.max(), loss))
    # 
    # iters = 5 * iters

    plt.figure()

    plt.subplot(221)
    plt.plot(stat[:, 0])
    plt.xlabel('Iterations')
    plt.title('loss')

    plt.subplot(222)
    plt.plot(stat[:, 1])
    plt.xlabel('Iterations')
    plt.title('sum accuracy')

    plt.subplot(223)
    plt.plot(stat[:, 2])
    plt.xlabel('Iterations')
    plt.title('c1 accuracy')

    plt.subplot(224)
    plt.plot(stat[:, 3])
    plt.xlabel('Iterations')
    plt.title('c2 accuracy')

    plt.savefig(exp_dir  + '/error_plots.png')
    plt.show()

    for k, cc in enumerate(c):
        np.savetxt(exp_dir + '/c_%d.txt' % k, cc)

    # sys.exit(0)

    # regular SIRT and FBP reconstruction
    i0 = np.dot(source, energy_grid[:, 1])
    mono_sino = (np.log(i0) - np.log(sinogram)).reshape(proj_shape)

    sirt_res = cpu_sirt(proj_geom, vol_geom, projector, mono_sino)
    fbp_res = cpu_fbp(proj_geom, vol_geom, projector, mono_sino)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(mono_sino)
    plt.title('non_log fp')

    plt.subplot(2, 2, 2)
    plt.imshow(mono_sino)
    plt.title('log fp mono_sino')

    plt.subplot(2, 2, 3)
    plt.imshow(sirt_res)
    plt.title('SIRT 100 iters')

    plt.subplot(2, 2, 4)
    plt.imshow(fbp_res)
    plt.title('FBP')

    plt.savefig(os.path.join(exp_dir, 'mono_recon.png'))
    plt.show()

    np.savetxt(exp_dir + '/sirt.txt', sirt_res)
    np.savetxt(exp_dir + '/fbp.txt', fbp_res)



    gt = gt_concentrations
    plt.tight_layout()
    plt.subplot(221)
    plt.imshow(gt[1], vmin=0, vmax=1, interpolation=None)
    plt.xticks(np.linspace(0, 256, 5), [])
    plt.yticks(np.linspace(0, 256, 5), [])
    plt.ylabel("ground\ntruth")
    plt.title("$c_0$")

    plt.subplot(222)
    plt.imshow(gt[0], vmin=0, vmax=1, interpolation=None)
    plt.xticks(np.linspace(0, 256, 5), [])
    plt.yticks(np.linspace(0, 256, 5))
    plt.title("$c_1$")

    plt.subplot(223)
    plt.imshow(c[0], vmin=0, vmax=1, interpolation=None)
    plt.xticks(np.linspace(0, 256, 5))
    plt.yticks(np.linspace(0, 256, 5), [])
    plt.ylabel("recon")

    plt.subplot(224)
    plt.imshow(c[1], vmin=0, vmax=1, interpolation=None)
    plt.xticks(np.linspace(0, 256, 5))
    plt.yticks(np.linspace(0, 256, 5))

    plt.savefig(os.path.join(exp_dir, 'recon.png'), dpi=300)
    plt.show()