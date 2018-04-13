# /usr/bin/python
# coding=utf-8
from __future__ import print_function

import astra
import numpy as np
import xraylib_np as xrl_np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import os
import pickle
import json
import scipy.optimize as opt
import datetime

from cvxopt import matrix
from cvxopt import solvers

import sys
sys.path.append('../src')
import barrier_method
from barrier import HoughBarrier

import logging as log
log.basicConfig(level=log.INFO)
import multiprocessing

import collections

cmap = 'pink'
# cmap = 'viridis'
plt.rcParams['image.cmap'] = cmap

def create_phantom_tooth(size_x, energy, elem_tooth, elem_implant, pixel_size, isimplant):
    """
    Create phantom, share is two flowers (with 3 and 6 petals)

    :param sx: size of phantom
    :param energy: energy, set in keV
    :param elem1: Number of the chemical element
    :param elem2: Number of the chemical element
    :param pixel_size: size of one pixel, set in microns
    :return: 2d array of phantom
    """
    xrl_np.XRayInit()
    phantom = np.zeros((size_x, size_x))
    sx_half = size_x / 2
    sq = size_x / 14

    #calculate mu
    density_tooth = xrl_np.ElementDensity(np.array([elem_tooth]))
    cross_section_tooth = xrl_np.CS_Total(np.array([elem_tooth]), np.array([energy]))
    mu_tooth = density_tooth * cross_section_tooth

    density_implant = xrl_np.ElementDensity(np.array([elem_implant]))
    cross_section_implant = xrl_np.CS_Total(np.array([elem_implant]), np.array([energy]))
    mu_implant = density_implant * cross_section_implant

    #buld mesh
    y, x = np.meshgrid(range(size_x), range(size_x))
    xx = (x - sx_half).astype('float32')
    yy = (y - sx_half).astype('float32')
    r = np.sqrt(xx*xx + yy*yy)
    tetta = np.arctan2(yy, xx)

    #make teeth
    mask_tooth = r <= sq*(1 + np.cos(2*tetta) + np.sin(2*tetta)**2)
    mask_tooth += (xx*xx + yy*yy) <=(0.09*size_x)**2
    mask_tooth += np.roll(mask_tooth, size_x//3, axis=0) + np.roll(mask_tooth, -size_x//3, axis=0) #  make 3 teeth
    phantom[mask_tooth] = mu_tooth[0]

    #make implant
    mask_implant = (xx / (0.11*size_x))**2 + (yy / (0.07*size_x))**2 < 1
    mask_implant *= y <= sx_half
    mask_implant *= ((xx / (0.11*size_x))**2 + (((yy - 0.025*size_x) / (0.07*size_x)))**2) > 1

    if(isimplant):
        phantom[mask_implant] = mu_implant[0]

    phantom *= pixel_size
    print("for Ca:")
    print(mu_tooth)
    print("for Au")
    print(mu_implant)
    return phantom

def create_phantom(size):
    """

    :param size: shape of the phantom
    :return: np array of the phantom
    """
    energy = 45.0
    elem1 = 20  # Ca
    elem2 = 79  # Au
    pixel_size = 0.05
    return create_phantom_tooth(size, energy, elem1, elem2, pixel_size, True)

# -----------------------------------------------------------------------------


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

def cpu_sirt(pg, vg, proj_id,sm, n_iters=100):
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
# ------------------------------------------------------------------------------


def create_projection_with_poisson_noise(i0, pg, vg, v, pj):
    r = gpu_fp(pg, vg, v, pj)
    p = np.exp(np.log(i0) - r)
    p = np.random.poisson(lam=p.flatten()).astype('float32').reshape(r.shape)
    return p

def create_data_sample(i0, size, n_angles, n_samples):
    if (size == 8):
        phantom = np.zeros((size,size))
        phantom[3,6] = 160
        phantom[5,6] = 160
        phantom[4,5] = 160
        phantom[4,7] = 160
        phantom[1,3] = 250
        phantom[1,4] = 250
        phantom[2,2] = 250
        phantom[2,1] = 250
        phantom[3,1] = 250
        phantom[4,1] = 250
        phantom[5,1] = 250
        phantom[6,2] = 250
        phantom[7,3] = 250
        phantom[7,4] = 250

        original = phantom
    else:
        original = create_phantom(size)

    np.savetxt('phantom.txt', original)
    proj_geom = astra.create_proj_geom(
        'parallel',
        1.0,
        int(np.sqrt(2)*size + 1),  # This should be enough to register full object.
        np.linspace(0, np.pi, n_angles)
    )
    vol_geom = astra.create_vol_geom(*original.shape)
    projector = astra.create_projector('linear', proj_geom, vol_geom)
    projections = []
    for _ in range(n_samples):
        projections.append(
            create_projection_with_poisson_noise(
                i0, proj_geom, vol_geom, original, projector)
        )

    sino = projections[0]
    np.savetxt('projjections.txt', sino.T)
    return {
        'i0': i0,
        'n_angles': n_angles,
        'n_samples': n_samples,
        'original': original,
        'proj_geom': proj_geom,
        'vol_geom': vol_geom,
        'projections': projections,
        'projector': projector
    }

def all_i0s():
    mult = [1, 4, 8]
    powers = [2, 3, 4]

    i0s = []
    for m, p in itertools.product(mult, powers):
        i0s.append(m*(10**p))
    return sorted(i0s)

# ----------------------------------------------------------------------------


def e1_prepare_data():
    SEED = 12345
    np.random.seed(SEED)
    log.info('Running Experiment #1: Prepare data')
    i0s = all_i0s()
    size = 256
    n_angles = 512
    n_samples = 20

    os.mkdir('e1-data')
    with open('e1-data/description.json', 'w') as f:
        f.write(json.dumps(
            {'i0s': i0s, 'size': size, 'n_angles': n_angles, 'n_samples': n_samples},
            indent=2, sort_keys=True))
    for idx, i0 in enumerate(i0s):
        d = create_data_sample(i0, size, n_angles, n_samples)
        with open('e1-data/item-{idx:03d}.pkl'.format(idx=idx), 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------


def ray_transform_from_projection(p, bound, i0):
    m = (p <= bound)
    p[m] = bound
    return np.log(i0) - np.log(p), m

def vg_to_shape(vg):
    return (vg['GridRowCount'], vg['GridColCount'])

def mse(x0, x1):
    size = x0.size
    return np.sum((x0 - x1)**2)/size

def process_item(item, func):
    i0 = item['i0']
    projections = item['projections']
    pg = item['proj_geom']
    vg = item['vol_geom']
    x_orig = item['original']
    proj_id = item['projector']

    errors = []
    results = []
    ineq = []
    for pr in projections:
        x = func(i0, pg, vg, pr, proj_id)
        results.append(x)
        errors.append(mse(x_orig, x))

    return {
        'errors': errors,
        'results': results,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors)
    }

def dump_item_res(res, dest):
    for idx, x in enumerate(res['results']):
        np.save(os.path.join(dest, 'result-{:02}.npy'.format(idx)), x)
    v = {}
    for k in ('errors', 'mean_error', 'std_error'):
        v[k] = res[k]
    with open(os.path.join(dest, 'info.json'), 'w') as f:
        f.write(json.dumps(v, indent=2, sort_keys=True))

def run_experiment(out_dir, func):
    log.info('Running for: {:s}'.format(out_dir))
    i0s = all_i0s()

    os.makedirs(out_dir)

    for idx, i0 in enumerate(i0s):
        with open('e1-data/item-{idx:03d}.pkl'.format(idx=idx), 'rb') as f:
            d = pickle.load(f)

        res = process_item(d, func)
        dest = os.path.join(out_dir, 'item-{idx:03d}'.format(idx=idx))
        os.makedirs(dest)
        dump_item_res(res, dest)

# -----------------------------------------------------------------------------


def sirt(i0, pg, vg, pr, proj_id, bound, orig=None):
    r, m = ray_transform_from_projection(pr, bound, i0)
    r[m] = np.log(i0) - np.log(bound)

    return cpu_sirt(pg, vg, proj_id, r, n_iters=200)

def fbp(i0, pg, vg, pr, proj_id, bound, orig=None):
    r, m = ray_transform_from_projection(pr, bound, i0)
    r[m] = np.log(i0) - np.log(bound)

    return cpu_fbp(pg, vg, proj_id, r, n_iters=200)
# -----------------------------------------------------------------------------


def mask_linear_least_squares(i0, pg, vg, pr, proj_id, bound, orig=None):
    r, m = ray_transform_from_projection(pr, bound, i0)
    v_shape = vg_to_shape(vg)
    b = r.flatten()

    FP = lambda x: gpu_fp(pg, vg, x.reshape(v_shape), proj_id).flatten()
    BP = lambda x: gpu_bp(pg, vg, x.reshape(r.shape), proj_id).flatten()

    def cost(x):
        d = FP(x) - b
        d[m.flatten()] = 0.0
        return np.sum(d**2)

    def grad(x):
        d = FP(x) - b
        d[m.flatten()] = 0.0
        return 2.0*BP(d)

    plot_cost = []
    plot_mse = []

    def cb(x):
        return None
        plot_cost.append(cost(x))
        plot_mse.append(mse(x, orig.flatten()))

    x0 = np.zeros(v_shape, dtype='float32').flatten()
    x_opt = opt.fmin_cg(cost, x0, grad, maxiter=200, callback=cb)

    # n = len(plot_cost)
    # plt.figure()
    # plt.plot(np.arange(n), plot_cost)
    # ax = plt.gca().twinx()
    # ax.plot(np.arange(n), plot_mse, '--')

    return x_opt.reshape(v_shape)

# -----------------------------------------------------------------------------


def solve_poisson_maximum_likelihood(i0, pg, vg, pr, proj_id):
    pr /= np.max(pr)

    x_shape = (vg['GridRowCount'], vg['GridColCount'])

    def _func(_x):
        x = _x.reshape(x_shape)
        fp = gpu_fp(pg, vg, x, proj_id)
        return np.sum(np.exp(-fp) + pr*fp)

    def _grad(_x):
        x = _x.reshape(x_shape)
        fp = gpu_fp(pg, vg, x, proj_id)
        gr = gpu_bp(pg, vg, pr - np.exp(-fp), proj_id)
        return gr.flatten()

    x0 = np.zeros(x_shape).flatten()
    x_opt = opt.fmin_cg(_func, x0, _grad, maxiter=100)
    x_opt = x_opt.reshape(x_shape)
    return x_opt

# --------------------------------------------------------------------

def save_im(image1, title1, name):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(image1, cmap=plt.cm.pink, interpolation='none')
    ax.set_title(title1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #ax.set_xlabel("Variable")
    #ax.set_ylabel("Variable")
    f.savefig(name)
    return

def prepare_weight(pg, vg, v, pj, n_angles):
    sx = v.shape[0]
    vol = np.zeros((sx, sx))
    vol[0,0] = 1
    s_test = gpu_fp(pg, vg, vol, pj)
    houghsize = s_test.shape
    vol[0,0] = 0
    H = np.zeros((houghsize[0] * houghsize[1], sx * sx))
    z = 0
    for i in range(0,sx):
        for j in range(0, sx):
            vol[i, j] = 1
            s_test = gpu_fp(pg, vg, vol, pj)
            H[:, z] = s_test.flatten()
            z += 1
            #np.savetxt(str(i) + '_' + str(j) + '_hh.txt', H)
            vol[i,j] = 0
    return H

def quadratic_programming(pg, vg, v, pj, n_angles, P, bound):
    solvers.options['maxiters'] = 200

    H = prepare_weight(pg, vg, v, pj, n_angles)
    #Task (0.5 x'Qx + f'x -> min(x))
    p = P.flatten()
    Q = np.dot(2 * H.T, H)
    f = np.dot(- 2 * p.T, H)
    f = f.T

    Q_s = matrix(Q, tc='d')
    f_s = matrix(f, tc='d')
    #without inequalities and without boundares
    #x3 = solvers.qp(Q_s,f_s)

    #inequalities Ax <= bp
    indices = np.array([])
    A = np.empty((0, H.shape[1]))
    b = np.array([])
    new_p = p.copy()
    for i in range(0, H.shape[0]):
        if (p[i] >= bound):
            new_p[i] = bound
            indices = np.hstack((indices,i))
            A = np.vstack((A, -H[i,:]))
            b = np.hstack((b, -bound))
    b = b.T
    f_new = np.dot(-2 * new_p.T, H)
    f_new = f_new.T
    A_diag = np.zeros((H.shape[1], H.shape[1]))
    np.fill_diagonal(A_diag, -1)
    A = np.concatenate((A, A_diag), axis=0)
    b_zero = np.zeros((H.shape[1], 1))
    b_zero = b_zero.flatten()
    b = np.concatenate((b, b_zero), axis=0)

    f_new_s = matrix(f_new, tc='d')
    G_s = matrix(A, tc='d')
    h_s = matrix(b, tc='d')
    x2 = solvers.qp(Q_s,f_new_s, G_s, h_s)
    #print (x2['z'])
    print ('the end')
    #without inequalities
    x1 = solvers.qp(Q_s,f_new_s)
    print ('the end')
    return {
        'without_inequalities':x1['x'],
        'inequalities': x2['x']
    }

# --------------------------------------------------------------------
counter = 0

def ineq_linear_least_squares(i0, pg, vg, pr, proj_id, bound, alpha, orig=None):
    r, m = ray_transform_from_projection(pr, bound, i0)
    # после этого в r[m] лежит log(i0) - log(bound)
    v_shape = vg_to_shape(vg)
    m = m.flatten()
    b = r.flatten()
    b[m] = np.log(i0) - np.log(bound)

    n_bad = np.sum(m)
    n_good = m.size - n_bad

    FP = lambda x: gpu_fp(pg, vg, x.reshape(v_shape), proj_id).flatten()
    BP = lambda x: gpu_bp(pg, vg, x.reshape(r.shape), proj_id).flatten()

    def cost(x, a):
        d = FP(x) - b

        p_mask = (d > 0)*m
        d[p_mask] = 0.0
        d[m] *= a

        d[1-m] /= np.sqrt(n_good)
        d[m] /= np.sqrt(n_bad)

        return np.sum(d**2)

    def grad(x, a):
        d = FP(x) - b

        p_mask = (d > 0)*m
        d[p_mask] = 0.0
        d[m] *= a

        d[1-m] /= n_good
        d[m] /= n_bad

        return 2.0*BP(d)

    plot_cost = []
    plot_mse = []
    def cb(x):
        global counter
        counter += 1
        #print "iteration %d, cost: %.2f" % (counter, cost(x, alpha))
        #return None
        plot_cost.append(cost(x, alpha))
        plot_mse.append(mse(x, orig.flatten()))

    x_cur = np.zeros(v_shape, dtype='float32').flatten()
    # for i in range(100):
    #     x_cur = x_cur - 1e-4 * grad(x_cur, alpha)
    #     cb(x_cur)

    x_cur = opt.fmin_cg(lambda x: cost(x, alpha),
                        x_cur,
                        lambda x: grad(x, alpha),
                        maxiter=2000,
                        callback=cb)

    # print(opt.check_grad(lambda x: cost(x, alpha),
    #                      lambda x: grad(x, alpha),
    #                      x_cur))

    n = len(plot_cost)
    plt.figure()
    plt.plot(np.arange(n), plot_cost)
    ax = plt.gca().twinx()
    ax.plot(np.arange(n), plot_mse, '--')

    return x_cur.reshape(v_shape)


# Result plotting function for barrier method
def plot_x(x, iter, num, show=True, save=False, out='movie'):
    fig = plt.figure()
    im0 = plt.imshow(x, vmin=-0.1, vmax=7)
    plt.title('barrier method')
    plt.suptitle('Iteration %d' % iter)
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

def stat_cb(statrecord):
    global opt_stats
    global stat_iter
    out = 'movie_qp_64'
    opt_stats.append(statrecord)
    x = statrecord[0].x.reshape((64, 64))
    plot_x(x, stat_iter * 10, stat_iter, show=False, save=True, out=out)
    stat_iter += 1

def plot_stats(opt_stats):
    # ==============
    # stat plots
    x = np.arange(len(opt_stats))
    loss = np.array([stat[0].func for stat in opt_stats])
    grad_norm = np.array([np.linalg.norm(stat[0].grad) for stat in opt_stats])

    bound = np.array([(-np.exp(-stat[2]['bound'].func)).max() for stat in opt_stats])
    grad_norm_bound = np.array([np.linalg.norm(stat[2]['bound'].grad) for stat in opt_stats])

    morezero = np.array([(-np.exp(-stat[2]['morezero'].func)).max() for stat in opt_stats])
    grad_norm_morezero = np.array([np.linalg.norm(stat[2]['morezero'].grad) for stat in opt_stats])

    plt.subplot(221)
    plt.plot(x, loss)
    plt.title('goal loss')

    plt.subplot(222)
    plt.plot(x, grad_norm)
    plt.title('goal grad norm')

    plt.subplot(223)
    plt.plot(x, bound, x , morezero)
    plt.legend(['$Wf \geq \delta$', '$f \geq 0$'])
    plt.title('max constraints values')

    plt.subplot(224)
    plt.plot(x, grad_norm_bound, x, grad_norm_morezero)
    plt.legend(['$Wf \geq \delta$', '$f \geq 0$'])
    plt.title('grad norms for barrier functions')
    plt.tight_layout()
    plt.savefig('barrier_method_plots.png', dpi=400)
    plt.show()


def barrier_least_squares(i0, pg, vg, pr, proj_id, bound, alpha, orig=None):
    r, m = ray_transform_from_projection(pr, bound, i0)
    # после этого в r[m] лежит log(i0) - log(bound)
    v_shape = vg_to_shape(vg)
    print ('v_shape:', v_shape)
    print('m shape:', m.shape)
    m = m.flatten()
    b = r.flatten()
    bound_log = np.log(i0) - np.log(bound)
    b[m] = bound_log

    FP = lambda x: gpu_fp(pg, vg, x.reshape(v_shape), proj_id).flatten()
    BP = lambda x: gpu_bp(pg, vg, x.reshape(r.shape), proj_id).flatten()

    # FP(x)[m] >= bound <-> bound - FP(x)[m] <= 0
    bound_log = 0.95 * bound_log
    def bc_func(x):
        f = FP(x)
        return bound_log - f[m]

    def bc_grad(x):
        grad = np.zeros_like(r).flatten()
        grad[m] = -1
        return BP(grad)

    def bc_project(x, n_iter=100, alpha=1e-2):
        '''Projects x onto feasible point set
        '''
        f = FP(x)
        infeas_i = m & (f < bound_log)
        x_new = x.copy()
        for i in range(n_iter):
            infeas_i = m & (f < bound_log)
            if not np.any(infeas_i):
                break
            f[~infeas_i] = 0
            f[infeas_i] -= bound_log + 1e-3 #..?
            x_new = x_new - alpha * BP(f)
            f = FP(x_new)
        # import ipdb; ipdb.set_trace()
        return x_new

    # import pdb; pdb.set_trace()

    bound_constraints = (bc_func, bc_grad, bc_project)

    hb = HoughBarrier(FP, BP, m, bound_log, r)

    # x >= 0 constraints
    def nonzero_project(x):
        return np.clip(x, 0, x.max())

    morezero_constraints= (lambda x: -x,                     # func
                           lambda x: -np.ones_like(x),       # grad
                           lambda x: np.clip(x, 0, x.max())) # project

    n_bad = np.sum(m)
    n_good = m.size - n_bad



    def cost(x):
        d = FP(x) - b
        d[m] = 0
        d /= n_good
        return np.sum(d**2)

    def grad(x):
        d = FP(x) - b
        d[m] = 0
        d /= n_good
        return 2.0*BP(d)

    plot_cost = []
    plot_mse = []
    def cb(x):
        global counter
        counter += 1
        print("iteration %d, cost: %.2f" % (counter, cost(x, alpha)))
        return None
        # plot_cost.append(cost(x, alpha))
        # plot_mse.append(mse(x, orig.flatten()))


    x_cur = 1e-2 + np.zeros(v_shape, dtype='float32').flatten()
    
    ans, stats = barrier_method.barrier_method(x_cur,
        (cost, grad),
        ineq_dict={'bound': hb,
                   'morezero': morezero_constraints},
        n_iter=200,
        n_biter=10,
        t0=0.2,
        t_step=0.2,
        beta_reg=1.0,
        add_stat_cb=stat_cb,
        alpha=0.1,
        max_iter=400,
        do_alpha_decay=False)
    plot_stats(opt_stats)

    x_cur_new = 1e-2 + np.zeros(v_shape, dtype='float32').flatten()
    ans_no_ineq, stats_no_ineq = barrier_method.barrier_method(x_cur_new,
        (cost, grad),
        ineq_dict={'morezero': morezero_constraints},
        n_iter=100,
        n_biter=10,
        t0=0.1,
        t_step=0.2,
        beta_reg=1.0)

    #x_cur = opt.fmin_cg(lambda x: cost(x, alpha),
    #                    x_cur,
    #                    lambda x: grad(x, alpha),
    #                    maxiter=2000,
    #                    callback=cb)

    # print(opt.check_grad(lambda x: cost(x, alpha),
    #                      lambda x: grad(x, alpha),
    #                      x_cur))

    # n = len(plot_cost)
    # plt.figure()
    # plt.plot(np.arange(n), plot_cost)
    # ax = plt.gca().twinx()
    # ax.plot(np.arange(n), plot_mse, '--')

    return ans.reshape(v_shape), stats, ans_no_ineq.reshape(v_shape), stats_no_ineq


def run_all_experiments():
    # run_experiment(
    #     'pois', lambda i0, pg, vg, pr: solve_poisson_maximum_likelihood(i0, pg, vg, pr))
    run_experiment(
        'mask', lambda i0, pg, vg, pr, proj_id: mask_linear_least_squares(i0, pg, vg, pr, proj_id, 1))
    run_experiment(
        'ineq', lambda i0, pg, vg, pr, proj_id: ineq_linear_least_squares(i0, pg, vg, pr, proj_id, 1, 30))

def save_image(image1, image2, title1, title2, name, bounds1, bounds2):
    f = plt.figure(figsize=(9, 4))

    ax = [
        f.add_axes([0.05, 0.05, 0.44, 0.9]),
        f.add_axes([0.51, 0.05, 0.44, 0.9])
    ]

    im1 = ax[0].imshow(image1, cmap=cmap, interpolation='none',
                       vmin=bounds2[0], vmax=bounds2[-1])
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title(title1)
    f.colorbar(im1, ax=ax[0], shrink=0.9,  boundaries=bounds1)

    im2 = ax[1].imshow(image2, cmap=cmap, interpolation='none', 
                       vmin=bounds2[0], vmax=bounds2[-1])
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title(title2)
    f.colorbar(im2, ax=ax[1], shrink=0.9,  boundaries=bounds2)

    plt.savefig(name)
    return


def main():
    global x3
    global x4
    global x3_stats
    global x4_stats
    # e1_prepare_data()
    # run_all_experiments()
    i0 = 1e9
    n_angles = 90
    size_x = 64
    bound = 8

    # i0 = 1e3
    # n_angles = 512
    # size_x = 256
    # bound = 2

    d = datetime.datetime.now()
    np.random.seed(d.microsecond + d.hour + d.minute + d.second)
    item = create_data_sample(i0, size_x, n_angles, 1)
    pg = item['proj_geom']
    vg = item['vol_geom']
    pr = item['projections'][0]
    proj_id = item['projector']
    v = item['original']

    # plt.figure();
    # 
    # plt.subplot(121)
    # plt.imshow(v)
    # 
    # plt.subplot(122)
    # #plt.imshow((np.log(i0) - np.log(pr)).T)
    # plt.imshow(pr.T)
    # 
    # plt.show()
    # sys.exit(0)

    print('min value of sinogram:')
    print(pr.min())
    print('\nmax value of sinogram:')
    print(pr.max())

    r, m = ray_transform_from_projection(pr, bound, i0)
    r[m] = np.log(i0 / bound) # prun : этого не нужно, в ray_transform это уже учтено
    print(r.shape)
    np.savetxt('sinogram.txt', r)
    v_max = v.max()
    bounds1 =  np.arange(0, v_max, v_max/5)
    v_max = r.max()
    bounds2 =  np.arange(0, v_max, v_max/5)
    bounds1 = np.linspace(0, 4.23, 9)
    save_image(v, r.T, 'Phantom', 'Sinogram', 'sample.png', bounds1, bounds2)
    sys.exit(0)
    x1 = sirt(i0, pg, vg, pr, proj_id, bound, orig=item['original'])
    x2 = fbp(i0, pg, vg, pr, proj_id, bound, orig=item['original'])
    x1[x1 < 0] = 0
    x2[x2 < 0] = 0
    np.savetxt('x1.nptxt', x1)
    np.savetxt('x2.nptxt', x2)
    # solves = quadratic_programming(pg, vg, v, proj_id, n_angles, r, bound)
    # x3 = np.array(solves['inequalities'])
    # x3 = x3.reshape(x2.shape)
    # x4 = np.array(solves['without_inequalities'])
    # x4 = x4.reshape(x1.shape)
    # x3[x3 < 0] = 0
    # x4[x4 < 0] = 0

    x3, x3_stats, x4, x4_stats = barrier_least_squares(i0, pg, vg, pr, proj_id, bound, 100, orig=item['original'])
    np.savetxt('x3.nptxt', x3)
    np.savetxt('x4.nptxt', x4)

    x5 = ineq_linear_least_squares(i0, pg, vg, pr, proj_id, bound, 300, orig=item['original'])
    x6 = mask_linear_least_squares(i0, pg, vg, pr, proj_id, bound, orig=item['original'])
    x5[x5 < 0] = 0
    x6[x6 < 0] = 0
    np.savetxt('x5.nptxt', x5)
    np.savetxt('x6.nptxt', x6)


    v_max = np.amax([x1.max(), x2.max(), x5.max(), x6.max()])
    bounds = np.arange(0.0, v_max, v_max/5)
    save_image(x1, x2, 'SIRT', 'FBP', 'sample1.png', bounds, bounds)
    save_image(x3, x4, 'QP with inequalities', 'QP without inequalities', 'sample2.png', bounds, bounds)
    save_image(x5, x6, 'Soft Inequalities method', 'Missing Data method', 'sample3.png', bounds, bounds)
    return x1, x2, x3, x4, x5, x6, x3_stats, x4_stats

if __name__ == '__main__':
    x1, x2, x3, x4, x5, x6, x3_stats, x4_stats = main()


def save_3_images(image1, image2, image3,
                  title1, title2, title3,
                  bounds, name):
    global cmap
    f, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    im1 = ax[0].imshow(image1, cmap=cmap, interpolation='none',
                       vmin=bounds[0], vmax=bounds[-1])
    ax[0].set_xlim((0, image1.shape[1]))
    ax[0].set_ylim((0, image1.shape[0]))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title(title1)
   # f.colorbar(im1, ax=ax[0], shrink=0.9,  boundaries=bounds)

    im2 = ax[1].imshow(image2, cmap=cmap, interpolation='none', 
                       vmin=bounds[0], vmax=bounds[-1])
    ax[1].set_xlim((0, image1.shape[1]))
    ax[1].set_ylim((0, image1.shape[0]))
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title(title2)
   # f.colorbar(im2, ax=ax[1], shrink=0.9,  boundaries=bounds)

    im3 = ax[2].imshow(image3, cmap=cmap, interpolation='none', 
                       vmin=bounds[0], vmax=bounds[-1])
    ax[2].set_xlim((0, image1.shape[1]))
    ax[2].set_ylim((0, image1.shape[0]))
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_title(title3)
    f.colorbar(im3, ax=ax[2], shrink=0.9,  boundaries=bounds)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    return

save_3_images(v, x2, x3, u'Фантом', 'FBP', 'QP (barrier method)', bounds, 'qp_threesome_pink.png')
cmap = 'viridis'
save_3_images(v, x2, x3, u'Фантом', 'FBP', 'QP (barrier method)', bounds, 'qp_threesome.png')
cmap = 'hot'
save_3_images(v, x2, x3, u'Фантом', 'FBP', 'QP (barrier method)', bounds, 'qp_threesome_hot.png')

def save_4_images(image1, image2, image3, image4,
                  title1, title2, title3, title4,
                  bounds, name):
    global cmap
    f, ax = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    ax = ax.flatten()
    im1 = ax[0].imshow(image1, cmap=cmap, interpolation='none',
                       vmin=bounds[0], vmax=bounds[-1])
    ax[0].set_xlim((0, image1.shape[1]))
    ax[0].set_ylim((0, image1.shape[0]))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title(title1)
   # f.colorbar(im1, ax=ax[0], shrink=0.9,  boundaries=bounds)

    im2 = ax[1].imshow(image2, cmap=cmap, interpolation='none', 
                       vmin=bounds[0], vmax=bounds[-1])
    ax[1].set_xlim((0, image1.shape[1]))
    ax[1].set_ylim((0, image1.shape[0]))
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title(title2)
    f.colorbar(im2, ax=ax[1], shrink=0.9,  boundaries=bounds)

    im3 = ax[2].imshow(image3, cmap=cmap, interpolation='none', 
                       vmin=bounds[0], vmax=bounds[-1])
    ax[2].set_xlim((0, image1.shape[1]))
    ax[2].set_ylim((0, image1.shape[0]))
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_title(title3)
    # f.colorbar(im3, ax=ax[2], shrink=0.9,  boundaries=bounds)

    im4 = ax[3].imshow(image4, cmap=cmap, interpolation='none', 
                       vmin=bounds[0], vmax=bounds[-1])
    ax[3].set_xlim((0, image1.shape[1]))
    ax[3].set_ylim((0, image1.shape[0]))
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)
    ax[3].set_title(title4)
    f.colorbar(im4, ax=ax[3], shrink=0.9,  boundaries=bounds)

    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    return

cmap = 'pink'
save_4_images(v, x2, x3, x5, u'Фантом', 'FBP', 'QP (barrier method)', u'Soft inequalities', bounds, 'qp_foursome_pink.png')
cmap = 'viridis'
save_4_images(v, x2, x3, x5, u'Фантом', 'FBP', 'QP (barrier method)', u'Soft inequalities', bounds, 'qp_foursome.png')