# /usr/bin/python
# coding=utf-8

from __future__ import print_function, division

# from decorator import staticmethod
import numpy as np
from astra_proxy import *


def integrate(ar, energy_grid):
    '''Integrates function (possibly multidimentional, eg image) given in array on energy_grid'''
    res_shape = ar.shape[:-1]
    num_en = energy_grid.shape[0]
    return np.sum(ar.reshape(-1, num_en) * energy_grid[:, 1]
              .reshape(1, -1), axis=1).reshape(res_shape)


class WhiteProjection:
    ''' Class for white projection and backprojection in spectrum
    '''

    def __init__(self, source, pixel_size, element_numbers, element_absorptions,
                 energy_grid, ph_size, sinogram=None, n_angles=360, angles=None):
        self.dt = np.float64
        self.source_init = source
        self.pixel_size = pixel_size
        self.element_numbers = element_numbers
        self.element_absorptions = element_absorptions
        self.ph_size = ph_size
        self.n_angles = n_angles
        self.energy_grid = energy_grid
        self.sinogram = sinogram.ravel()
        self.angles = angles

        if sinogram is not None:
            self.ph_size = (sinogram.shape[1], sinogram.shape[1])

        # setup astra geometry
        if self.angles is None:
            self.proj_geom = astra.create_proj_geom(
                'parallel',
                1.0,
                # This should be enough to register full object.
                int(np.sqrt(2) * self.ph_size[0] + 1),
                np.linspace(0, np.pi, n_angles)
            )
        else:
            self.n_angles = angles.shape[0]
            self.proj_geom = astra.create_proj_geom(
                'parallel',
                1.0,
                # This should be enough to register full object.
                self.ph_size[0],
                self.angles
            )

        self.vol_geom = astra.create_vol_geom(*self.ph_size)
        self.projector = astra.create_projector('linear',
                                                self.proj_geom,
                                                self.vol_geom)
        self.proj_shape = (self.n_angles, self.proj_geom['DetectorCount'])
        self.conc_shape = (len(self.element_numbers), 
                           self.ph_size[0],
                           self.ph_size[1])

        # norm source intensity
        sum_energy = self.integrate(self.source_init)
        self.source = self.source_init / sum_energy

    def integrate(self, ar):
        return integrate(ar, self.energy_grid)

    def FP(self, x):
        ''' forward projetion proxy for single sinogram '''
        return gpu_fp(self.proj_geom, self.vol_geom, x, self.projector)

    def BP(self, x):
        return gpu_bp(self.proj_geom, self.vol_geom, x, self.projector)


    def FP_white(self, c):
        """ white forward projection
        """
        K = len(c)

        conc_fp = np.array([self.FP(cc) for cc in c])
        flat_fp = conc_fp.reshape(K, -1, 1)
        ea = self.element_absorptions.reshape(K, 1, -1)
        exp_arg = -self.pixel_size * np.sum(flat_fp * ea, axis=0)

        exp = np.exp(exp_arg)
        Integral = self.integrate(self.source.reshape(1, -1) * exp)
        return Integral, exp_arg, exp, flat_fp

    def calc_sinogram_with_gt(self, gt):
        self.gt = gt
        self.sinogram = self.FP_white(gt)[0]

    def BP_white(self, Q, exp):
        """white backprojection"""
        mu = self.calc_mu(exp)
        bp_arg = Q * mu

        conc = np.zeros(shape=self.conc_shape, dtype=self.dt)

        for k, cc in enumerate(conc):
            conc[k] = - self.BP(bp_arg[k].reshape(self.proj_shape))
        return conc, mu


    def func(self, concentrations):
        """returns half squared l2 loss of reconstruction"""
        fp, exp_arg, exp, flat_fp = self.FP_white(concentrations)
        q = (self.sinogram - fp)
        return 0.5 * np.linalg.norm(q)** 2

    def grad(self, concentrations):
        """returns grad of halfed l2 loss."""
        fp, exp_arg, exp, flat_fp = self.FP_white(concentrations)
        q = (self.sinogram - fp)
        bp_grad, mu = self.BP_white(q, exp)
        return bp_grad


    def calc_mu(self,exp):
        """ wighted residuals
        """
        pix = self.pixel_size
        source = self.source.reshape(1, -1)
        mu = np.array([-self.integrate(
            source * pix * exp * ea.reshape(1, -1))
            for ea in self.element_absorptions])

        return mu


