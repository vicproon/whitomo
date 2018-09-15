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

def get_button_4():
    return load_phantom('../testdata/whitereconstruct/correct_button4')

def get_button_1_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button_1())


def get_button_2_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button_2())


def get_button_3_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button_3())

def get_button_4_delta_spectrum():
    return get_phantom_with_delta_spectrum(get_button_4())



def calc_biology_grid(ang_from=22, ang_to=40, num_pts=10):
    anchors_angstrom = np.array([ang_from,ang_to])
    anchors_kev = angstrom_to_kev(anchors_angstrom)
    inner_grid_kev = np.linspace(anchors_kev[1], anchors_kev[0], num_pts)
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


def get_synth_data_small():
    # Grid with respect to only 2.5 and 10
    grid = np.array([[2.5, 10],
                     [0.005, 0.005]]).T

    # Source with peaks at 2.5 and 10
    source = np.array([2.0, 1.0])

    # Taken from synth_spec.py
    abs_1 = np.array([4000, 302.04159662])
    # уменьшили абсорпцию второго в максимуме первого
    # abs_2 = np.array([1470.60170339, 2000])
    abs_2 = np.array([300, 2000])

    # Emphasize that elements are imaginary.
    element_numbers = np.array([301, 302]) 

    return {'grid': grid, 
            'source': source,
            'pixel_size': 1e-6, # adjust for numerical stability
            'element_numbers': element_numbers,
            'gt_concentrations': get_button_3()['gt_concentrations'],
            'element_absorptions': np.stack([abs_1, abs_2])}


def get_synth_data_b2_small():
    data = get_synth_data_small()
    data['gt_concentrations'] = get_button_2()['gt_concentrations']
    return data

def get_synth_data_b4_small():
    data = get_synth_data_small()
    data['gt_concentrations'] = get_button_4()['gt_concentrations']
    return data


import white_projection
import sys
sys.path.append('../teeth_recon')
import teeth_io

def read_teeth_data():
    # First we can get energy grid in keV, source, and element absorptions using spectre file.
    spectre = np.loadtxt('../testdata/whitereconstruct/zub/spectre.txt', skiprows=1)
    grid_vals = spectre[:,0]
    source = spectre[:, 1]

    # Grid is given with equal distances of 0.40363636, which can be calculated
    # from grid_vals, but for simplicity and cleaness of the code i will just use the
    # hard-coded constant
    grid_widths = 0.40363636 * np.ones_like(grid_vals)
    
    # Finally, the grid is..
    grid = np.vstack([grid_vals, grid_widths]).T


    # Now lets get the absorptions: Ca and Pb
    element_numbers = np.array([20, 82])
    element_absorptions = absorption(grid[:, 0], element_numbers)

    # The only thing left is reading the sinogram itself
    pb_data = teeth_io.read_pb_data(820)
    # Problem is that pb_data is normed and logarithmed. need to undo this.
    sum_intensity = white_projection.integrate(source, grid)
    # p = log(i0 / i)
    # i = I0 exp (-p)
    proj_data = sum_intensity * np.exp(-pb_data['data'])

    return {'grid': grid, 
            'source': source,
            'pixel_size': 1e-8, # adjust for numerical stability
            'element_numbers': element_numbers,
            'element_absorptions': element_absorptions,
            'proj_data': proj_data,
            'angles': pb_data['angles']}

def shrink(angles, proj, shrink_ratio):
    angle_ind = np.arange(angles.shape[0], step=shrink_ratio)
    data_ind = np.arange(proj.shape[1], step=shrink_ratio)
    new_data = proj[angle_ind, :]
    new_data = new_data[:, data_ind]
    new_angles = angles[angle_ind]
    return new_data, new_angles

def read_teeth_data_small(ratio=4):
    data = read_teeth_data()
    small_proj, small_angles = shrink(data['angles'], data['proj_data'], ratio)
    data['angles'] = small_angles
    data['proj_data'] = small_proj
    #data['pixel_size'] = 6.4e-5
    data['pixel_size'] = 1.3e-4
    return data


__proxy_dict={'eggs': get_eggs_data,
              'eggs_delta_spectrum': get_eggs_with_delta_spectrum,
              'button': get_button,
              'button_delta_spectrum': get_button_delta_spectrum,
              'button_1': get_button_1,
              'button_1_delta_spectrum': get_button_1_delta_spectrum,
              'button_2': get_button_2,
              'button_2_delta_spectrum': get_button_2_delta_spectrum,
              'button_2_biology': get_button_2_biology,
              'button_2_synth': get_synth_data_b2_small,
              'button_3': get_button_3,
              'button_3_delta_spectrum': get_button_3_delta_spectrum,
              'button_3_synth': get_synth_data_small,
              'button_4': get_button_4,
              'button_4_delta_spectrum': get_button_4_delta_spectrum,
              'button_4_synth': get_synth_data_b4_small,
              'teeth_data': read_teeth_data,
              'teeth_data_256': lambda : read_teeth_data_small()}


def get_input(ph_name='eggs'):
    """main proxy function, retruns input dict by the name.\
    throws KeyError if not found
    """
    return __proxy_dict[ph_name]()