# /usr/bin/python
# coding=utf-8
from __future__ import absolute_import, unicode_literals, division, print_function
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import astra
from experiments import AstraProxy, sirt, fbp, ineq_linear_least_squares, \
    mask_linear_least_squares, cpu_sirt, cpu_fbp

import cPickle as pickle
import preproc_data
import teeth_io
import os

def read_oldstyle():
    if os.path.exists('tomodata.pickle'):
        print('loading from pickle...')
        with open('tomodata.pickle', 'rb') as f:
            tomodata = pickle.load(f)
    else:
        print('reading data from storage')
        tomodata = load_data_from_images()
        with open('tomodata.pickle', 'wb') as f:
            print('pickling tomodata...')
            pickle.dump(tomodata, f)

    data = tomodata['data']
    angles = tomodata['angles']
    dark_imgs = tomodata['dark_imgs']
    empty_imgs = tomodata['empty_imgs']

    pixel_size = 1.0# 9 * 1e-6 # 9 mkm
    print('normalizing data...')

    norm_data, i0 = normalize_data(data, dark_imgs, empty_imgs, pixel_size)
    with open('norm_data.pickle', 'wb') as f:
        pickle.dump({'norm_data':norm_data, 'angles': angles}, f)

def read_newstyle():
    slice_line = 820
    clean_data = teeth_io.read_clear_data(slice_line=slice_line)
    pb_data = teeth_io.read_pb_data(slice_line=slice_line)
    f, ax = plt.subplots(2, 1)
    plt.ylabel('angle')
    im = ax[0].imshow(clean_data['data'])
    ax[0].set_title('clean sinogram')
    # f.colorbar(im)
    # ax[0].xlabel('pos')
    im = ax[1].imshow(pb_data['data'])
    ax[1].set_title('Pb sinogram')
    # ax[1].xlabel('pos')
    # f.colorbar(im)

    plt.savefig('sinograms.png')
    # plt.show(block=False)
    return clean_data, pb_data

def check_bounds(r, maxbound, minbound):
    M = r > maxbound
    m = r < minbound
    
    r[M] = maxbound
    r[m] = minbound

    return r, M, m

# def main():
plt.figure()
cl, pb = read_newstyle()
plt.figure()
plt.hist(pb['data'].ravel(), bins=50)
plt.title('Pb Sinogram histogram')
plt.savefig('pb_hist.png')
# plt.show(block=False)

bound = 2.9
cl_pr, cl_M, cl_m = check_bounds(cl['data'], bound, 0)
pb_pr, pb_M, pb_m = check_bounds(pb['data'], bound, 0)

def run_soft_ineq(data, name, alpha, bound):
    d_pr, M, m = check_bounds(data['data'], bound, 0)
    pixel_size = 1.0
    proj_geom = astra.create_proj_geom(
        'parallel',
        pixel_size,
        d_pr.shape[1],
        data['angles']
    )

    rec_pix = d_pr.shape[1]

    vol_geom = astra.create_vol_geom(rec_pix, rec_pix)
    projector = astra.create_projector('linear', proj_geom, vol_geom)

    res = ineq_linear_least_squares(1.0, 
        proj_geom, vol_geom,
        np.exp(-d_pr), projector, np.exp(-bound),
        alpha)

    plt.figure()
    plt.imshow(res, cmap=plt.cm.viridis)
    plt.colorbar(orientation='horizontal')
    name_suffix = '%s_a%.1f' % (name, alpha)
    plt.title("%s, alpha %.1f" % (name, alpha))
    plt.savefig('soft_ineq_' + name_suffix + '.png')
    # plt.show(block=False)
    np.savetxt(name_suffix, res)

#run_soft_ineq(pb, name='res', alpha=10)
def shrink(data_dict, shrink_ratio):
    angle_ind = np.arange(data_dict['angles'].shape[0], step=shrink_ratio)
    data_ind = np.arange(data_dict['data'].shape[1], step=shrink_ratio)
    new_data = data_dict['data'][angle_ind, :]
    new_data = new_data[:, data_ind]
    new_angles = data_dict['angles'][angle_ind]
    return {'data': new_data, 'angles': new_angles}

small_sinodata = shrink(pb, 4)

#alphas = np.exp(np.arange(np.log(0.1), np.log(300), 0.7))
#print('running reconstruction for following alphas:', alphas)
#for i, a in enumerate(alphas):
#    print('alpha =', a, '(%d / %d)' % (i, len(alphas)))
#    run_soft_ineq(small_sinodata, name='pb_small_exp3', alpha=a, bound=bound)
run_soft_ineq(small_sinodata, name='pb_small_exp7_nocg', alpha=6.67, bound=bound)

# if __name__ == '__main__':
#     main()
# slice_line = 0
# flat_normalized = norm_data[:, slice_line, :]
# plt.figure()
# plt.imshow(flat_normalized)
# plt.title('sinogram')
# plt.savefig('sinogram.png')
# # now prepare for test astra reconstruction
# 
# 
# proj_geom = astra.create_proj_geom(
#     'parallel',
#     pixel_size,
#     flat_normalized.shape[1],
#     angles
# )
# 
# rec_pix = flat_normalized.shape[1]
# 
# vol_geom = astra.create_vol_geom(rec_pix, rec_pix)
# projector = astra.create_projector('linear', proj_geom, vol_geom)
# 
# print('good')
# x1 = cpu_sirt(proj_geom, vol_geom, projector, flat_normalized, n_iters=200)
# plt.imshow(x1)
# plt.show()
