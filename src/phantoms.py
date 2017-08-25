# /usr/bin/python
# coding=utf-8

#todo: make it possible to pass args and kwargs to get_input

"""This module is intent to provide an easy proxy for different
phantoms and input data for experiments
"""

import numpy as np
import os.path
import xraylib_np as xraylib

xraylib.XRayInit()


def absorption(energy, element):
    """Returns total element absorption for given energy.
    Both energy and element could be 1d-arrays
    """
    element = np.array(element)
    dens = xraylib.ElementDensity(element)
    cross = xraylib.CS_Total(element, energy)
    return dens.reshape(-1, 1) * cross


def get_eggs_data():
    input_dir = '../testdata/whitereconstruct/input'
    gt_dir = '../testdata/whitereconstruct/correct'

    energy_grid = np.loadtxt(os.path.join(input_dir, 'grid.txt'))
    source = np.loadtxt(os.path.join(input_dir, 'source.txt'))


    pixel_size = 3e-5  # 0.05 in icmv experiments
    ph_size = (256, 256)
    n_angles = 180
    element_numbers = np.array([22, 28])

    gt_concentrations = np.array([np.loadtxt(os.path.join(gt_dir, f))
                         for f in os.listdir(gt_dir)])
    element_absorptions = absorption(energy_grid[:, 0], element_numbers)

    return {'grid': energy_grid, 
            'source': source,
            'pixel_size': pixel_size,
            'element_numbers': element_numbers,
            'gt_concentrations': gt_concentrations,
            'element_absorptions': element_absorptions}

__proxy_dict={'eggs': get_eggs_data}

def get_input(ph_name='eggs'):
    """main proxy function, retruns input dict by the name.\
    throws KeyError if not found
    """
    return __proxy_dict[ph_name]()