# /usr/bin/python
# coding=utf-8

# todo спектр из двух линий и два элемента
# todo модуль штрафа произведения (+)
# todo рандомные углы при проекции
# todo рандомные пиксели изображения для регуляризации произведением
from __future__ import print_function

import numpy as np
import xraylib_np as xraylib

import os
import os.path

import matplotlib.pyplot as plt
import astra
import scipy.stats
import sys

# local imports
import batch_filter

xraylib.XRayInit()

input_dir = '../testdata/whitereconstruct/input'
energy_grid = np.loadtxt(os.path.join(input_dir, 'grid.txt'))
source = np.loadtxt(os.path.join(input_dir, 'source.txt'))


pixel_size = 3e-5  # 0.05 in icmv experiments
ph_size = (256, 256)
n_angles = 180
element_numbers = np.array([22, 28])

gt_dir = '../testdata/whitereconstruct/correct'
gt_concentrations = [np.loadtxt(os.path.join(gt_dir, f))
                     for f in os.listdir(gt_dir)]
f = [np.loadtxt(os.path.join(input_dir, 'f_%02d.txt' % en))
     for en in element_numbers]


def absorption(energy, element):
    element = np.array(element)
    dens = xraylib.ElementDensity(element)
    cross = xraylib.CS_Total(element, energy)
    return dens.reshape(-1, 1) * cross


element_absorptions = absorption(energy_grid[:, 0], element_numbers)

sinogram = np.loadtxt(os.path.join(input_dir, 'white_ht.txt'))


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


class AstraProxy:
    def __init__(self, dims):
        if dims == 2:
            self.data_create = astra.data2d.create
            self.data_delete = astra.data2d.delete
            self.data_get = astra.data2d.get
            self.fp_algo_name = 'FP'
            self.bp_algo_name = 'BP'
            self.sirt_algo_name = 'SIRT'
            self.fbp_algo_name = 'FBP'
            self.dart_mask_name = 'DARTMASK'
            self.dart_smoothing_name = 'DARTSMOOTHING'
        elif dims == 3:
            self.data_create = astra.data3d.create
            self.data_delete = astra.data3d.delete
            self.data_get = astra.data3d.get
            self.fp_algo_name = 'FP3D'
            self.bp_algo_name = 'BP3D'
            self.sirt_algo_name = 'SIRT3D'
            self.fbp_algo_name = 'FBP3D'
            self.dart_mask_name = 'DARTMASK3D'
            self.dart_smoothing_name = 'DARTSMOOTHING3D'
        else:
            raise NotImplementedError


def gpu_fp(pg, vg, v, proj_id):
    ap = AstraProxy(len(v.shape))
    v_id = ap.data_create('-vol', vg, v)
    rt_id = ap.data_create('-sino', pg)
    fp_cfg = astra.astra_dict(ap.fp_algo_name)
    fp_cfg['VolumeDataId'] = v_id
    fp_cfg['ProjectionDataId'] = rt_id
    fp_cfg['ProjectorId'] = proj_id
    fp_id = astra.algorithm.create(fp_cfg)
    astra.algorithm.run(fp_id)
    out = ap.data_get(rt_id)
    astra.algorithm.delete(fp_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out


def gpu_bp(pg, vg, rt, proj_id, supersampling=1):
    ap = AstraProxy(len(rt.shape))
    v_id = ap.data_create('-vol', vg)
    rt_id = ap.data_create('-sino', pg, rt)
    bp_cfg = astra.astra_dict(ap.bp_algo_name)
    bp_cfg['ReconstructionDataId'] = v_id
    bp_cfg['ProjectionDataId'] = rt_id
    bp_cfg['ProjectorId'] = proj_id
    bp_id = astra.algorithm.create(bp_cfg)
    astra.algorithm.run(bp_id)
    out = ap.data_get(v_id)
    astra.algorithm.delete(bp_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out


def cpu_sirt(pg, vg, proj_id, sm, n_iters=100):
    ap = AstraProxy(len(sm.shape))
    rt_id = ap.data_create('-sino', pg, data=sm)
    v_id = ap.data_create('-vol', vg)
    sirt_cfg = astra.astra_dict(ap.sirt_algo_name)
    sirt_cfg['ReconstructionDataId'] = v_id
    sirt_cfg['ProjectionDataId'] = rt_id
    sirt_cfg['ProjectorId'] = proj_id
    sirt_id = astra.algorithm.create(sirt_cfg)
    astra.algorithm.run(sirt_id, n_iters)
    out = ap.data_get(v_id)

    astra.algorithm.delete(sirt_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out


def cpu_fbp(pg, vg, proj_id, sm, n_iters=100):
    ap = AstraProxy(len(sm.shape))
    rt_id = ap.data_create('-sino', pg, data=sm)
    v_id = ap.data_create('-vol', vg)
    fbp_cfg = astra.astra_dict(ap.fbp_algo_name)
    fbp_cfg['ReconstructionDataId'] = v_id
    fbp_cfg['ProjectionDataId'] = rt_id
    fbp_cfg['ProjectorId'] = proj_id
    fbp_id = astra.algorithm.create(fbp_cfg)
    astra.algorithm.run(fbp_id, n_iters)
    out = ap.data_get(v_id)

    astra.algorithm.delete(fbp_id)
    ap.data_delete(rt_id)
    ap.data_delete(v_id)
    return out
# ---------------


conc_shape = (len(element_numbers), ph_size[0], ph_size[1])
concentrations = np.zeros(shape=conc_shape, dtype=np.float64)

concentrations = scipy.stats.truncnorm(
    a=-0.1, b=0.1, scale=0.1).rvs(size=conc_shape) + 0.09


def integrate(ar, energy_grid):
    res_shape = ar.shape[:-1]
    num_en = energy_grid.shape[0]
    return np.sum(ar.reshape(-1, num_en) * energy_grid[:, 1]
                  .reshape(1, -1), axis=1).reshape(res_shape)


sum_energy = integrate(source, energy_grid)
source /= sum_energy


def FP_white(energy, source, pixel_size, concentrations):
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


sinogram, exp_arg, exp, flat_fp, = FP_white(
    energy_grid, source, pixel_size, gt_concentrations)


def calc_mu(energy, source, exp, element_absorptions):

    mu = np.array([-integrate(
        source.reshape(1, -1) * pixel_size * exp * ea.reshape(1, -1),
        energy)
        for ea in element_absorptions])

    return mu


alpha = 0.05
beta_reg = 0.1


bp_conc_buffer = np.zeros_like(concentrations)


def BP_white(energy, source, exp, concentrations, Q):
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

uneq_reg_buffer = np.zeros_like(concentrations)


def calc_uneq_reg_grad(c):
    for i, cc in enumerate(uneq_reg_buffer):
        uneq_reg_buffer[i] = c[np.arange(len(c)) != i].sum(axis=0)
    return uneq_reg_buffer

#bf = batch_filter.BatchFilter(concentrations.shape[1:], (6, 6))
bf = batch_filter.GaussBatchGen(concentrations.shape[1:], 16, 100)

def Iteration(c, batch_filtering=False):
    global fp
    fp, exp_arg, exp, flat_fp = FP_white(energy_grid, source, pixel_size, c)
    q = (sinogram - fp)
    bp_grad, mu = BP_white(energy_grid, source, exp, c, q)
    reg_grad = calc_uneq_reg_grad(c)

    # gradient update with regularisation
    reg_loss = np.linalg.norm(beta_reg * c[0] * c[1])
    step = alpha * (bp_grad + beta_reg * reg_grad)
    
    if batch_filtering:
        masks = np.array([bf.new_mask(), bf.new_mask()])
        c = c - step * masks # batch_filtering is ON
    else:
        c = c - step

    hough_loss = np.linalg.norm(q)

    np.linalg.norm(beta_reg * reg_grad)
    full_loss = hough_loss + reg_loss
    return c, mu, q, full_loss, reg_grad, reg_loss

experiment_name = 'exp8'
try:
    os.mkdir(experiment_name)
except:
    pass

with open(experiment_name + '/readme.txt', 'w') as f:
    f.writelines('\n'.join(['Эксперимент8: гауссово сглаживание батч генератора']))


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
        plt.savefig(experiment_name + '/%s_%02d.png' % (suffix, iter_num))
      
    # elif iter_num is None:
        if iter_num is None:
            plt.show()
        plt.close(f)


# итерационная минимизация.
iters = 5000
stat = np.zeros(shape=(iters + 100, 2 + len(concentrations)), dtype=np.float64)

for i in xrange(iters):
    if i % 30 == 0 and i >= 4250:
        do_batch_filtering = False
    else:
        do_batch_filtering = True

    c, mu, q, loss, reg_loss, reg_grad = Iteration(c, 
                                        batch_filtering=do_batch_filtering)
    # showres(mu.reshape(2, *proj_shape), i, 'mu')
    showres(c, mu.reshape(2, *proj_shape), i)
    # print('reg loss is ', reg_loss)

    c_acc = np.linalg.norm(c - gt_concentrations, axis=(1, 2))
    sum_acc = np.linalg.norm(c - gt_concentrations)
    stat[i, :] = np.array([loss, sum_acc] + [ac for ac in c_acc])

    print('iter: %03d, min(c) = %.2f, max(c) = %.2f; loss = %.2f' %
          (i, c.min(), c.max(), loss))

# for i in xrange(iters, 5 * iters):
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

plt.figure(2)

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

plt.savefig(experiment_name + '/error_plots.png')
plt.show()

sys.exit(0)

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

plt.savefig('mono_recon.png')
plt.show()

# todo проверить размерности и переходы нормировочных коэффициентов
# при дифференциировании. Установить причину больших чисел.

# лол - фантом норм восстанавливается и без "белого пучка". Нужен фантом на
# котором проявляются артефакты.
# 1) спектр источника сделать с более выреженными пиками
# 2) фантом взять а-ля квадрат с включениями
# 3) сделать разбиение на случайные подзоны и апдейт текущего состояния по
# подзонам.
