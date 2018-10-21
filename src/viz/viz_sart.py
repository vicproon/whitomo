# /usr/bin/python
# coding=utf-8

# нужно переполучить картинку с различными сечениями восстановлений 
# SART и FHT-SIRT, получить ошибки и точности.


from __future__ import print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path

from skimage.transform import radon
from skimage.morphology.selem import disk
from skimage.filters.rank import gradient

import scipy.ndimage

# зачитаем картинки
data_dir = '../../testdata/fht_sirt_results'

phantom = np.loadtxt(os.path.join(data_dir, 'phantom_line_256.txt'))
phantom = phantom.reshape((256, 256))

sart = np.loadtxt(os.path.join(data_dir, 'buz_line_256.txt'))
sart = sart.reshape((256, 256))

fht_sirt = np.loadtxt(os.path.join(data_dir, 'vp_line_256.txt'))
fht_sirt = fht_sirt.reshape((256, 256))

# рассчитаем синограммы
theta = np.linspace(0, 180, 180)

phantom_sino = radon(phantom, theta, circle=True)
sart_sino = radon(sart, theta, circle=True)
fht_sirt_sino = radon(fht_sirt, theta, circle=True)

# пронаблюдаем промежуточный результат
def show_obj_and_sino(obj, sino, title, bounds, name):
    #f = plt.figure(figsize=(9, 5))
    #axes = [
    #    f.add_axes([0.05, 0.05, 0.44, 0.9]),
    #    f.add_axes([0.51, 0.05, 0.44, 0.9])
    #]

    f, axes = plt.subplots(1,2, figsize=(11, 4), sharey=True)
    ax = axes[0]
    im = ax.imshow(obj, vmin=bounds[0][0], vmax=bounds[0][-1], 
                   interpolation='none', aspect=1)
    ax.set_title(title[0])
    ax.set_xlim((0, obj.shape[1]))
    ax.set_ylim((0, obj.shape[0]))
    f.colorbar(im, ax=ax, shrink=0.9,  boundaries=bounds[0])
    
    ax = axes[1]
    im = ax.imshow(sino, vmin=bounds[1][0], vmax=bounds[1][-1],
                   interpolation='none')
    ax.set_title(title[1])
    ax.set_ylim((0, sino.shape[0]))
    ax.set_xlim((0, sino.shape[1]))
    f.colorbar(im, ax=ax, shrink=0.9,  boundaries=bounds[1])

    f.savefig(name)

ph_max = max([phantom.max(), sart.max(), fht_sirt.max()])
bounds_ph = np.linspace(0, ph_max, 9)

si_max = max([phantom_sino.max(), sart_sino.max(), fht_sirt_sino.max()])
bounds_si = np.linspace(0, si_max, 9)


show_obj_and_sino(phantom, phantom_sino, 
                 ('Фантом', 'Синограмма'), 
                 (bounds_ph, bounds_si),
                 name='sl.png')

show_obj_and_sino(sart, sart_sino, 
                 ('SART', 'FP(SART)'), 
                 (bounds_ph, bounds_si),
                 name='sart.png')

show_obj_and_sino(fht_sirt, fht_sirt_sino, 
                 ('FHT-SIRT', 'FP(FHT-SIRT)'), 
                 (bounds_ph, bounds_si),
                 name='fht_sirt.png')


show_obj_and_sino(sart, fht_sirt, 
                  ('SART', 'FHT-SIRT'),
                  (bounds_ph, bounds_ph),
                  'sart__fht_sirt.png')


# plt.show()



def plot_cross_sections(lines, titles, colors, lw, name='cs.png', title=None, startx=0):
    plt.figure(figsize=(6, 4))

    x = np.arange(startx, startx + lines[0].shape[0])
    plt.plot(x, lines[0], colors[0], lw=lw[0])
    plt.plot(x, lines[1], colors[1], lw=lw[1])
    plt.plot(x, lines[2], colors[2], lw=lw[2])
    plt.legend(titles)
    plt.xlim((startx, startx + x.shape[0]))
    plt.xlabel('пиксели')
    plt.ylabel('интенсивность')

    if title:
        plt.title(title)

    plt.savefig(name, dpi=350)

lines_80 = [phantom[80, :], sart[80, :], fht_sirt[80, :]]
titles = ['Фантом', 'SART 1 итерация','FHT-SIRT 40 итераций']
colors = ['b:', 'c--', 'r']
lw = [1.5, 1, 1.2]

plot_cross_sections(lines_80[-1::-1], titles[-1::-1],
                    colors[-1::-1], lw[-1::-1],
                    name='cs_80.png',
                    title='кросс-секция $y = 80$')

lines_127 = [phantom[127, :], sart[127, :], fht_sirt[127, :]]
plot_cross_sections(lines_127[-1::-1], titles[-1::-1],
                    colors[-1::-1], lw[-1::-1],
                    name='cs_127.png',
                    title='кросс-секция $y = 127$')

lines_v_50 = [phantom[100:150, 50], sart[100:150, 50], fht_sirt[100:150, 50]]
plot_cross_sections(lines_v_50[-1::-1], titles[-1::-1],
                    colors[-1::-1], lw[-1::-1],
                    startx=100,
                    name='cs_v_50.png',
                    title='кросс-секция $x = 50$')


def accuracy(ph, rec):
    acc = np.linalg.norm(ph - rec) / np.linalg.norm(ph)
    acc = np.clip(np.abs(acc), 0, 1)
    return acc


sart_loss = np.mean((phantom_sino - sart_sino)**2)
sart_acc = accuracy(phantom, sart)

fht_sirt_loss = np.mean((phantom_sino - fht_sirt_sino)**2)
fht_sirt_acc = accuracy(phantom, fht_sirt)

print('sart loss: ', sart_loss)
print('sart_accuracy:', sart_acc)

print('fht_sirt loss: ', fht_sirt_loss)
print('fht_sirt_accuracy: ', fht_sirt_acc)

with open('acc_loss.txt', 'wb') as f:
    f.write('name\tloss\tacc\tn_iter\ttime\n')
    f.write('sart\t%.4f\t%.4f\t%d\t%.4f\n' % (sart_loss, sart_acc, 1, 0.93099999427))
    f.write('fht_sirt\t%.4f\t%.4f\t%d\t%.4f\n' % (fht_sirt_loss, fht_sirt_acc, 40, 0.935111))


def draw_patches():
    f = plt.figure()
    ax = plt.gca()
    ax.imshow(phantom)
    ax.invert_yaxis()
    cs_v_50_patch = patches.Polygon([[50, 100], [50, 150]], closed=False, color='r', lw=2)
    ax.add_patch(cs_v_50_patch)

    cs_80_patch = patches.Polygon([[0, 80], [255, 80]], closed=False, color='b', lw=2)
    ax.add_patch(cs_80_patch)

    cs_127_patch = patches.Polygon([[0, 127], [255, 127]], closed=False, color='m', lw=2)
    ax.add_patch(cs_127_patch)
    
    ax.set_xlim((0, 256))
    ax.set_ylim((256, 0))

    ax.set_title('визуализация кросс-секций')
    plt.savefig('cs_viz.png')


    f = plt.figure()
    ax = plt.gca()
    ax.imshow(phantom)
    ax.invert_yaxis()
    cs_v_50_patch = patches.Polygon([[50, 100], [50, 150]], closed=False, color='r', lw=2)
    ax.add_patch(cs_v_50_patch)
    ax.set_xlim((0, 256))
    ax.set_ylim((256, 0))

    ax.set_title('визуализация кросс-секций')
    plt.savefig('cs_v_50_viz.png')

    f = plt.figure()
    ax = plt.gca()
    ax.imshow(phantom)
    ax.invert_yaxis()
    cs_80_patch = patches.Polygon([[0, 80], [255, 80]], closed=False, color='b', lw=2)
    ax.add_patch(cs_80_patch)
    ax.set_xlim((0, 256))
    ax.set_ylim((256, 0))

    ax.set_title('визуализация кросс-секций')
    plt.savefig('cs_80_viz.png')

    f = plt.figure()
    ax = plt.gca()
    ax.imshow(phantom)
    ax.invert_yaxis()
    cs_127_patch = patches.Polygon([[0, 127], [255, 127]], closed=False, color='m', lw=2)
    ax.add_patch(cs_127_patch)
    ax.set_xlim((0, 256))
    ax.set_ylim((256, 0))

    ax.set_title('визуализация кросс-секций')
    plt.savefig('cs_127_viz.png')




draw_patches()

# phantom - sart

def grad(img):
    return scipy.ndimage.gaussian_gradient_magnitude(img, 2)

def grad_dist(phantom, recon):
    diff = grad(phantom) - grad(recon)
    return np.linalg.norm(diff)

def morph_grad(img):
    gr = gradient(img, disk(1))
    return gr.astype(np.float32) / 256

def show_grad_metrics():
    f, axes = plt.subplots(2, 5, figsize=(12, 4), sharex=True, sharey=True)
    axes = axes.T
    for ax in axes.ravel():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    grad_ph = grad(phantom)
    grad_fht_sirt = grad(fht_sirt)
    grad_sart = grad(sart)

    g_max = grad_ph.max()

    dist_fht_sirt = grad_dist(phantom, fht_sirt)
    dist_sart = grad_dist(phantom, sart)

    ax = axes[0, 0]
    ax.imshow(fht_sirt, vmin=0, vmax=1)
    ax.set_title('FHT-SIRT')

    
    ax = axes[1, 0]
    ax.set_title('GRAD')
    ax.imshow(grad_fht_sirt, vmin=0, vmax=g_max)

    ax = axes[2, 0]
    ax.imshow(grad_fht_sirt - grad_ph, vmin=-g_max, vmax=g_max)
    ax.set_title('dist: %.2f' % dist_fht_sirt)



    ax = axes[0, 1]
    ax.imshow(sart, vmin=0, vmax=1)
    ax.set_title('SART')

    ax = axes[1, 1]
    ax.set_title('GRAD')
    ax.imshow(grad_sart, vmin=0, vmax=g_max)

    ax = axes[2, 1]
    ax.imshow(grad_sart - grad_ph, vmin=-g_max, vmax=g_max)
    ax.set_title('dist: %.2f' % dist_sart)


    morph_ph = morph_grad(phantom)
    morph_fht_sirt = morph_grad(fht_sirt)
    morph_sart = morph_grad(sart)
    dist_morph_fht_sirt = np.linalg.norm(morph_fht_sirt - morph_ph)
    dist_morph_sart = np.linalg.norm(morph_sart - morph_ph)
    g_max = morph_ph.max()


    ax = axes[3, 0]
    ax.imshow(morph_fht_sirt, vmin=0,vmax=g_max)
    ax.set_title('MORPH_GRAD')

    ax = axes[4, 0]
    ax.imshow(morph_fht_sirt - morph_ph, vmin=-g_max, vmax = g_max)
    ax.set_title('dist: %.2f' % dist_morph_fht_sirt)


    ax = axes[3, 1]
    ax.imshow(morph_sart, vmin=0,vmax=g_max)
    ax.set_title('MORPH_GRAD')

    ax = axes[4, 1]
    ax.imshow(morph_sart - morph_ph, vmin=-g_max, vmax = g_max)
    ax.set_title('dist: %.2f' % dist_morph_sart)



    f.show()

show_grad_metrics()

plt.show()