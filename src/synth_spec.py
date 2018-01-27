'''A script that generates synthetic spectrum for issue9 of
https://github.com/vicproon/whitomo
and does a pretty drawing of it.
'''
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy
import scipy.interpolate

import  phantoms

# Get biological energy grid, source spectrum and element absorptions.
bio_data = phantoms.get_button_2_biology()
grid = bio_data['grid']
spec = bio_data['source']
abs8, abs6 = bio_data['element_absorptions']

# A helper function for plotting.
def plot_spectrums(grid, spec, absorptions, elem_names=[]):
    """ Plot element absorptions and energy spectrum using energy grid
    """
    assert grid.shape[0] == spec.shape[0] and grid.shape[0] == absorptions[0].shape[0], \
        'inconsistent array shapes'

    # All plot share same X axis -- energy grid
    X = grid[:, 0]
    
    # TODO: consider setting minor ticks to middles between grid elements
    # X ticks should correspond to grid widths.
    minor_xticks = [(x - 0.5 * w, x + 0.5 * w) for (x,w) in grid]
    
    # In case if we change step width the middle minor tick should lie in the middle.
    leftest_xtick = minor_xticks[0][0]
    rightest_xtick = minor_xticks[-1][1]
    middle_xticks = []
    for i in range(len(minor_xticks) - 1):
        r = minor_xticks[i][1]
        l = minor_xticks[i+1][0]
        middle_xticks.append(0.5 * (l + r))
    minor_xticks = [leftest_xtick] + middle_xticks + [rightest_xtick]
    
    # Begin plotting. First create a figure for plot.
    plt.figure()

    # Top -- element absorptions.
    ax = plt.subplot(211)
    
    # If element names were not provided we should provide them ourselves 
    if len(elem_names) != len(absorptions):
        elem_names = ['elem_%d' % i for i in range(len(absorptions))]

    # Create title for subplot.
    plt.title('Element absorptions')
    plt.xlabel('E, kev')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticks([2.5, 10])
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks([2000, 4000])
    plt.ylabel('$\mu$', rotation=0)
    plt.grid(True, linestyle='--')
    
    # Plot graphs.
    for Y in absorptions:
        plt.plot(X, Y)

    # Create legend.
    plt.legend(elem_names)

    # Bottom -- source spectrum
    ax = plt.subplot(212)
    plt.title('Energy spectrum')
    plt.xlabel('E, kev')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticks([2.5, 10])
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks([1, 2])
    plt.grid(True, linestyle='--')
    plt.plot(X, spec)


    # Shot plots
    plt.show()


# Lets plot the data.
# plot_spectrums(grid, spec, [abs8, abs6 / 10000], ['8', '6'])



def before_after_plot(x_before, f_before, x_after, f_after):
    plt.figure()
    ax = plt.subplot(211)
    plt.title('before')
    plt.plot(x_before, f_before)
    ax.set_xticks(x_before)
    plt.grid(True)

    ax = plt.subplot(212)
    plt.title('after')
    plt.plot(x_after, f_after)
    ax.set_xticks(x_after)
    plt.grid(True)
    plt.show()


def transform_function(f, x, new_x, pivots):
    """Given a scalar function with values f at points x transform it's shape 
    to the new grid new_x, preserving values at pivots.
    pivots are pairs of sorted indices (i, i_new) such that
    f(x[i]) == f_new(new_x[i_new]). 
    pivots (0, 0) and (len(x) - 1, len(new_x) - 1) are always implicitly included
    """
    pivots.append((len(x) - 1, len(new_x) - 1))
    lp = (0, 0)
    nf = np.zeros_like(new_x)
    for i, rp in enumerate(pivots):
        # Select parts of functions that correspond to current pivot.
        f_ = f[lp[0]: rp[0] + 1]
        x_ = x[lp[0]: rp[0] + 1]
        nx_ = new_x[lp[1]: rp[1] + 1]
        nf_ = nf[lp[1]: rp[1] + 1]
        
        # Transform x_ to corresponding value rates as nx_.
        scale = (nx_[-1] - nx_[0]) / (x_[-1] - x_[0])
        x_ = (nx_[0] + (x_ - x_[0]) * scale).copy()

        # Interpolate source function.
        inter = scipy.interpolate.interp1d(x_, f_, kind='linear')

        # Get new values for current pivot.
        nf_ += inter(nx_)

        #before_after_plot(x_, f_, nx_, nf_)
        # Set next left pivot as prev right pivot.
        lp = (rp[0] + 1, rp[1] + 1)

    #before_after_plot(x, f, new_x, nf)

    return nf


new_grid = phantoms.calc_biology_grid(10, 45, 25)
xx = new_grid[:,0]
absorb = phantoms.absorption(new_grid[:, 0], [8, 6])
maxvals = np.max(absorb[:, 2:], axis=1)
imaxvals = 2 + np.argmax(absorb[:, 2:], axis=1)
absorb = [absorb[0] / maxvals[0] * 2000, absorb[1] / maxvals[1] * 4000]
src = np.zeros_like(new_grid[:, 0])
# plot_spectrums(new_grid, src, [absorb[0], absorb[1]], ['scaled O', 'scaled C'])

# new_x = np.unique(np.hstack([np.linspace(1.5, 2.5, 5),
#                              np.linspace(2.5, 3.5, 5),
#                              np.linspace(3.5, 9, 8),
#                              np.linspace(9, 10, 5),
#                              np.linspace(10., 11, 5),
#                              np.linspace(11, 16.5, 8)]))


new_x = np.arange(0.5, 17, step=0.125)

peak_for_2 = np.where(new_x == 2.5)[0][0]
peak_for_1 = np.where(new_x == 10.0)[0][0]
peak_for = [peak_for_1, peak_for_2]

new_abs = [transform_function(absorb[0], new_grid[:, 0], new_x, [(imaxvals[0], peak_for[0])]),
           transform_function(absorb[1], new_grid[:, 0], new_x, [(imaxvals[1], peak_for[1])])]



new_spec = np.zeros_like(new_x)
new_spec[peak_for[0]] = 1.0
new_spec[peak_for[1]] = 2.0

new_widths = new_x[1:] - new_x[:-1]
new_grid = np.vstack([new_x, np.array([new_widths[0]] + list(new_widths))]).T
plot_spectrums(new_grid, new_spec, new_abs)