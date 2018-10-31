from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

x1 = np.loadtxt('x1.nptxt')
x2 = np.loadtxt('x2.nptxt')
x3 = np.loadtxt('x3.nptxt')
x4 = np.loadtxt('x4.nptxt')
x5 = np.loadtxt('x5.nptxt')
x6 = np.loadtxt('x6.nptxt')

x0 = np.loadtxt('../phantom.txt')

gt_intensity_vals = np.unique(x0) # array([ 0.        ,  0.10358164,  9.21754054])

bgrnd_mask = x0 == gt_intensity_vals[0]
teeth_mask = x0 == gt_intensity_vals[1]
metal_mask = x0 == gt_intensity_vals[2]

def calc_stats_by_mask(x, m):
    '''calculates stats like mean, std, min, max, median of array x by mask m'''
    xx = x[m]
    return np.array([xx.mean(), xx.std(), xx.min(), xx.max(), np.median(xx)])

def calc_lebesgue_stats_by_mask(x, m, val, eps):
    xx = x[m]
    mm = (xx >= val - eps) & (xx <= val + eps)
    return sum(mm) / sum(m.ravel())

np.set_printoptions(precision=3, suppress=True)

print('phantom')
print('bgrnd: ', calc_stats_by_mask(x0, bgrnd_mask), calc_lebesgue_stats_by_mask(x0, bgrnd_mask, gt_intensity_vals[0], 0.05))
print('teeth: ', calc_stats_by_mask(x0, teeth_mask), calc_lebesgue_stats_by_mask(x0, teeth_mask, gt_intensity_vals[1], 0.95))
print('metal: ', calc_stats_by_mask(x0, metal_mask), calc_lebesgue_stats_by_mask(x0, metal_mask, gt_intensity_vals[2], 7.5))

print('FBP')
print('bgrnd: ', calc_stats_by_mask(x2, bgrnd_mask), calc_lebesgue_stats_by_mask(x2, bgrnd_mask, gt_intensity_vals[0], 0.05))
print('teeth: ', calc_stats_by_mask(x2, teeth_mask), calc_lebesgue_stats_by_mask(x2, teeth_mask, gt_intensity_vals[1], 0.95))
print('metal: ', calc_stats_by_mask(x2, metal_mask), calc_lebesgue_stats_by_mask(x2, metal_mask, gt_intensity_vals[2], 7.5))

x3[x3 < 0] = 0
print('Barrier Method')
print('bgrnd: ', calc_stats_by_mask(x3, bgrnd_mask), calc_lebesgue_stats_by_mask(x3, bgrnd_mask, gt_intensity_vals[0], 0.05))
print('teeth: ', calc_stats_by_mask(x3, teeth_mask), calc_lebesgue_stats_by_mask(x3, teeth_mask, gt_intensity_vals[1], 0.95))
print('metal: ', calc_stats_by_mask(x3, metal_mask), calc_lebesgue_stats_by_mask(x3, metal_mask, gt_intensity_vals[2], 7.5))

print('Soft Inequalities')
print('bgrnd: ', calc_stats_by_mask(x5, bgrnd_mask), calc_lebesgue_stats_by_mask(x5, bgrnd_mask, gt_intensity_vals[0], 0.05))
print('teeth: ', calc_stats_by_mask(x5, teeth_mask), calc_lebesgue_stats_by_mask(x5, teeth_mask, gt_intensity_vals[1], 0.95))
print('metal: ', calc_stats_by_mask(x5, metal_mask), calc_lebesgue_stats_by_mask(x5, metal_mask, gt_intensity_vals[2], 7.5))