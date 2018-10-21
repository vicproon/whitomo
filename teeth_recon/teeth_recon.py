# /usr/bin/python
# coding=utf-8
from __future__ import absolute_import, unicode_literals, division, print_function
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

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

def run_fbp(data, bound):
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

    res = fbp(1.0, proj_geom, vol_geom, np.exp(-d_pr), projector, np.exp(-bound))

    plt.figure()
    plt.imshow(res, cmap=plt.cm.viridis)
    plt.colorbar(orientation='vertical')
    name_suffix = 'FBP'
    plt.title("FBP")
    plt.savefig(name_suffix + '.png')
    np.savetxt(name_suffix, res)
    return res    

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
# run_soft_ineq(small_sinodata, name='pb_small_exp7_nocg', alpha=6.67, bound=bound)

fbp_res = run_fbp(pb, bound=bound)

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
soft_res = np.loadtxt('/home/vic/teeth_recon/pb_big_exp4_a6.7')
fbp_res = np.flip(fbp_res, 0)
soft_res = np.flip(soft_res, 0)

def save_2_images(image1, image2,
                  title1, title2,
                  bounds, name, cmap='viridis', show=False):
    f, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    im1 = ax[0].imshow(image1, cmap=cmap, interpolation='none',
                       vmin=bounds[0], vmax=bounds[-1])
    ax[0].set_xlim((0, image1.shape[1]))
    ax[0].set_ylim((image1.shape[0], 0))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title(title1)
    f.colorbar(im1, ax=ax[0], shrink=0.9,  boundaries=bounds)

    im2 = ax[1].imshow(image2, cmap=cmap, interpolation='none', 
                       vmin=bounds[0], vmax=bounds[-1])
    ax[1].set_xlim((0, image1.shape[1]))
    ax[1].set_ylim((image1.shape[0], 0))
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title(title2)
    f.colorbar(im2, ax=ax[1], shrink=0.9,  boundaries=bounds)

    plt.tight_layout()
    plt.savefig(name, dpi=300)
    if show:
        plt.show()
    plt.close(f)


    y_level = 500
    x_roi = (450, 800)

    return

def save_2_images_cs(image1, image2,
                  title1, title2,
                  bounds, name, 
                  y_level, x_roi,
                  cmap='viridis', show=False):
    f = plt.figure()
    gs = gridspec.GridSpec(7,2) # f.add_gridspec(2, 2)
    # ax = [f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1]),
    #                   f.add_subplot(gs[1,:])              ];

    ax = [plt.subplot(gs[0:4, 0:1]), plt.subplot(gs[0:4, 1:2]),
                  plt.subplot(gs[4:,:])              ];

    def cs_patch():
        return patches.Polygon([[x_roi[0], y_level], [x_roi[1], y_level]],
                              closed=False, color='r', lw=1)

    # f, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    im1 = ax[0].imshow(image1, cmap=cmap, interpolation='none',
                       vmin=bounds[0], vmax=bounds[-1])

    ax[0].add_patch(cs_patch())

    ax[0].set_xlim((0, image1.shape[1]))
    ax[0].set_ylim((image1.shape[0], 0))
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title(title1)
    f.colorbar(im1, ax=ax[0], shrink=0.9,  boundaries=bounds)

    im2 = ax[1].imshow(image2, cmap=cmap, interpolation='none', 
                       vmin=bounds[0], vmax=bounds[-1])
    ax[1].add_patch(cs_patch())
    ax[1].set_xlim((0, image1.shape[1]))
    ax[1].set_ylim((image1.shape[0], 0))
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title(title2)
    f.colorbar(im2, ax=ax[1], shrink=0.9,  boundaries=bounds)

    x = np.arange(x_roi[0], x_roi[1])
    ax[2].plot(x, image1[y_level, x_roi[0]: x_roi[1]], 'b:')
    ax[2].plot(x, image2[y_level, x_roi[0]: x_roi[1]], 'g--')
    ax[2].set_xlim(x_roi)
    ax[2].set_title(u'Кросс-секция y = %d' % y_level)
    
    min_tick = min(image1[y_level, x_roi[0]:x_roi[1]].min(),
                   image2[y_level, x_roi[0]:x_roi[1]].min())

    max_tick = max(image1[y_level, x_roi[0]:x_roi[1]].max(),
                   image2[y_level, x_roi[0]:x_roi[1]].max())
    ticks = np.linspace(min_tick, max_tick, 10)
    ax[2].set_yticks(ticks)
    ax[2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    ax[2].legend([title1, title2])
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    if show:
        plt.show()
    plt.close(f)
    return



vmin = -0.005
vmax = 0.025
bounds = np.linspace(vmin, vmax, 6)

y_level = 630
x_roi = (470, 800)

tmp_fbp = 100 * fbp_res
tmp_sft = 100 * soft_res
bounds = 100 * bounds

save_2_images_cs(tmp_fbp, tmp_sft, 
              'FBP', u'ММО', 
              bounds, 'pb_big__fbp_vs_soft__cs__viridis.png', cmap='viridis',
              y_level=y_level, x_roi=x_roi, show=True)

save_2_images_cs(tmp_fbp, tmp_sft, 
              'FBP', u'ММО', 
              bounds, 'pb_big__fbp_vs_soft__cs__pink.png', cmap='pink',
              y_level=y_level, x_roi=x_roi, show=False)

save_2_images_cs(tmp_fbp, tmp_sft, 
              'FBP', u'ММО', 
              bounds, 'pb_big__fbp_vs_soft__cs__hot.png', cmap='hot',
              y_level=y_level, x_roi=x_roi, show=False)

save_2_images_cs(tmp_fbp, tmp_sft, 
              'FBP', u'ММО', 
              bounds, 'pb_big__fbp_vs_soft__cs__gray.png', cmap='gray',
              y_level=y_level, x_roi=x_roi, show=False)   