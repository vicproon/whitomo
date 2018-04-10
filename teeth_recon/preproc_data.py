# !/usr/bin.python
# coding=utf-8
# from __future__ import absolute_import, unicode_literals, division, print_function

import numpy as np
from scipy.misc import imresize


def resize_data(data, dst_size, angle_sparsify=1):
    """Resizes tomography data imitating increase in pixel size
    and sparsity in angles.
      param: data - angles,x,y 3d numpy array tomodata
      param: dst_size - float resize ratio
      param: angle_sparsify - int take only each k-th angle

      Returns: (resized_data, angle_indices)
    """
    assert(len(data.shape) == 3)
    res = list()
    angle_indices = list()
    for i, im in enumerate(data):
        if i % angle_sparsify == 0:
            angle_indices.append(i)
            res.append(imresize(im, dst_size)) #todo area interpolation
    
    res = np.array(res)
    return res, angle_indices

def normalize_data(data, dark, empty, pix_size):
    # тут нужно прологарифмировать и скорректировать на интенсивность источника
    avg_dark = np.mean(dark)
    I0 = np.mean(empty - avg_dark)
    del(dark)
    del(empty)
    data[data < avg_dark] = avg_dark
    return (np.log(I0 + 1e-3) - np.log(data - avg_dark + 1e-3)) / pix_size, I0
