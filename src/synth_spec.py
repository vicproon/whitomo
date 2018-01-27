'''A script that generates synthetic spectrum for issue9 of
https://github.com/vicproon/whitomo
'''
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy

import  phantoms

# Get biological energy grid, source spectrum and element absorptions.
bio_data = phantoms.get_button_2_biology()
grid = bio_data['grid']
spec = bio_data['source']
abs8, abs6 = bio_data['element_absorptions']

# A helper function for plotting.
def plot_spectrums(grid, spec, absorptions, elem_names=None):
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
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks(X)
    ax.set_xticks(minor_xticks, minor=True)
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
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks(X)
    ax.set_xticks(minor_xticks, minor=True)
    plt.grid(True, linestyle='--')
    plt.plot(X, spec)


    # Shot plots
    plt.show()


# Lets plot the data.
plot_spectrums(grid, spec, [abs8, abs6 / 10000], ['8', '6'])