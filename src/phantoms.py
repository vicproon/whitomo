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

import scipy.constants

def __angs_kev_coeff():
    c = scipy.constants.c
    hh = scipy.constants.physical_constants['Planck constant over 2 pi in eV s'][0]
    pi = scipy.constants.pi
    return 1e-3 * 2 * pi * hh * c * 1e10


an_kev = __angs_kev_coeff()


def angstrom_to_kev(wavelength_angstrom):
    return an_kev / wavelength_angstrom

def kev_to_angstrom(energy_kev):
    return an_kev / energy_kev

def absorption(energy, element):
    """Returns total element absorption for given energy.
    Both energy and element could be 1d-arrays
    """
    element = np.array(element)
    dens = xraylib.ElementDensity(element)
    cross = xraylib.CS_Total(element, energy)
    return dens.reshape(-1, 1) * cross


def load_phantom(gt_dir):
    input_dir = '../testdata/whitereconstruct/input'

    energy_grid = np.loadtxt(os.path.join(input_dir, 'grid.txt'))
    source = np.loadtxt(os.path.join(input_dir, 'source.txt'))

    pixel_size = 3e-5  # 0.05 in icmv experiments
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



def get_eggs_data():
    return load_phantom('../testdata/whitereconstruct/correct')
    

def delta_spectrum():
    source = np.array([5.25770000e+03,   5.25770000e+03,   5.25770000e+03,
        4.62190000e+08,   5.25770000e+03,   5.25770000e+03,
        5.25770000e+03,   5.25770000e+03,   5.25770000e+03,
        5.25770000e+03,   2.31095000e+08,   5.25770000e+03,
        5.25770000e+03,   5.25770000e+03,   5.25770000e+03,
        5.25770000e+03,   5.25770000e+03,   5.25770000e+03,
        5.25770000e+03,   5.25770000e+03,   5.25770000e+03,
        5.25770000e+03,   5.25770000e+03,   5.25770000e+03,
        5.25770000e+03])
    return source


def get_phantom_with_delta_spectrum(ph):
    ph['source'] = delta_spectrum()
    return ph


def get_eggs_with_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_eggs_data())


def get_button():
    return load_phantom('../testdata/whitereconstruct/correct_button')


def get_button_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button)


def get_button_1():
    return load_phantom('../testdata/whitereconstruct/correct_button1')


def get_button_2():
    return load_phantom('../testdata/whitereconstruct/correct_button2')


def get_button_3():
    return load_phantom('../testdata/whitereconstruct/correct_button3')


def get_button_1_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button_1())


def get_button_2_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button_2())


def get_button_3_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button_3())


def calc_biology_grid():
    anchors_angstrom = np.array([22,40])
    anchors_kev = angstrom_to_kev(anchors_angstrom)
    inner_grid_kev = np.linspace(anchors_kev[1], anchors_kev[0], 10)
    grid_step = inner_grid_kev[1] - inner_grid_kev[0]
    grid_pts = np.arange(anchors_kev[1] - 3 * grid_step, 
        anchors_kev[0] + 4 * grid_step, grid_step)
    grid = np.array([grid_pts, np.repeat(grid_step, len(grid_pts))])
    return grid.T

def get_button_2_biology():
    ph_base = get_button_2()
    grid = calc_biology_grid()
    ph_base['grid'] = grid

    biology_source = np.zeros_like(grid[:, 0])
    biology_source[3] = 0.5
    biology_source[-5] = 1.0
    ph_base['source'] = biology_source

    element_numbers = np.array([8, 6])
    biology_absorptions = absorption(grid[:, 0], element_numbers)
    ph_base['element_absorptions'] = biology_absorptions
    ph_base['element_numbers'] = element_numbers

    ph_base['pixel_size'] = 1e-9
    return ph_base


def get_imaginary_absorptions():
    spectrum = delta_spectrum()
    

__proxy_dict={'eggs': get_eggs_data,
              'eggs_delta_spectrum': get_eggs_with_delta_spectrum,
              'button': get_button,
              'button_delta_spectrum': get_button_delta_spectrum,
              'button_1': get_button_1,
              'button_1_delta_spectrum': get_button_1_delta_spectrum,
              'button_2': get_button_2,
              'button_2_delta_spectrum': get_button_2_delta_spectrum,
              'button_3': get_button_3,
              'button_3_delta_spectrum': get_button_3_delta_spectrum,
              'button_2_biology': get_button_2_biology}


def get_input(ph_name='eggs'):
    """main proxy function, retruns input dict by the name.\
    throws KeyError if not found
    """
    return __proxy_dict[ph_name]()