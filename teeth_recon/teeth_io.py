# !/usr/bin/python
# coding=utf-8
from __future__ import absolute_import, unicode_literals, division, print_function
import os
import numpy as np
import scipy.misc
from os.path import join, splitext
import h5py

def load_data_from_images(slice_line=500):    
    # assuming projection data located in /testdata/tomodata/
    clear_data_path = '/testdata/tomodata/zub_sonya_clear_mo_40_25'
    pb_data_path = '/testdata/tomodata/zub_sonya_pb_mo_40_25'

    # read dark and empty data
    dark_imgs = np.array([scipy.misc.imread(join(clear_data_path, f)) 
        for f in os.listdir(clear_data_path) if f.startswith('dark')])

    empty_imgs = np.array([scipy.misc.imread(join(clear_data_path, f)) 
        for f in os.listdir(clear_data_path) if f.startswith('empty')])

    # read projection data with angles from filenames
    data, angles = zip(*[(scipy.misc.imread(join(clear_data_path, f))[slice_line : slice_line + 1, 500:1500], 
        float(splitext(f)[0].split('_')[1]))
        for f in os.listdir(clear_data_path) if f.startswith('data')])

    print('dark, empty and data read')

    # data = np.array(data)
    angles = np.array(angles)

    # sort angles array
    print('sorting data angles...')

    sort_indices = np.argsort(angles)
    angles = angles[sort_indices] * np.pi / 180.0
    data = np.array([data[i] for i in sort_indices])

    print('parallel projection resolution is ', data.shape)
    print('proj angles number is ', angles.shape, 'avg delta is', np.mean(angles[1:] - angles[:-1]))

    return {'data': data,
            'angles': angles,
            'dark_imgs': dark_imgs,
            'empty_imgs': empty_imgs}

def read_data_from_hdf5(in_filename, slice_line):
    with h5py.File(in_filename) as h5f:
        data = h5f['sinogram'][:, :, slice_line]
    angles = (np.arange(data.shape[0]) * 0.5) * np.pi / 180.0
    return {'data' : data, 'angles' : angles}

def read_clear_data(slice_line):
    return read_data_from_hdf5('/home/vic/datasets/tomodata/zub_clear.h5', slice_line)

def read_pb_data(slice_line):
    return read_data_from_hdf5('/home/vic/datasets/tomodata/zub_pb.h5', slice_line)
    

def save_data_to_hdf5(tomodata, out_filename):
    pass