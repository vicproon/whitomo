# /usr/bin/python
# coding=utf-8

from __future__ import print_function
import random
import numpy as np


class BatchFilter(object):
    def __init__(self, shape, batches_shape):
        assert len(shape) == 2, 'BatchFilter can be applied only to 2d images'
        assert len(batches_shape) == 2, 'BatchFilter.batches_shape must be 2d'
        self.shape = shape
        self.batches_shape = batches_shape
        self.num_batches = batches_shape[0] * batches_shape[1]
        self.batch_shape = tuple(int(s / bs) + 1 for s, bs in
                                 zip(self.shape, self.batches_shape))
        import datetime
        now = datetime.datetime.now()
        random.seed(now.microsecond + now.second * 1e6 + 101991048)
        self.epoch_counter = 1
        self.reset_iterations()

    def __gen_new_batches(self):
        # план действий такой. берем batches_shape и генерим пермутацию из
        # индексов от 1 до batches_shape[0] * batches_shape[1]. по индексам
        # позже генерим соответствующие маски.
        self.batch_order = np.random.permutation(self.num_batches)
        self.batch_index = 0

    def __get_batch(self, arr, ibatch):
        batch_indices = np.unravel_index(ibatch, self.batches_shape)
        ind_start = tuple(
            i * b for i, b in zip(batch_indices, self.batch_shape))
        ind_end = tuple(np.clip(i * b + b, 0, s)
                        for i, b, s in zip(batch_indices, self.batch_shape, self.shape))
        return arr[ind_start[0]: ind_end[0], ind_start[1]: ind_end[1]]

    def new_mask(self):
        '''Основная фунция. Генерирует булевскую маску для очередного шага,
           в которой указано, какую часть изображения обновлять сейчас'''
        mask = np.zeros(shape=self.shape, dtype=np.bool)
        b = self.__get_batch(mask, self.batch_order[self.batch_index])
        b.fill(True)
        self.batch_index += 1
        if (self.batch_index >= self.num_batches):
            self.reset_iterations()
        return mask

    def reset_iterations(self):
        print('generating new batch order (', self.epoch_counter, ')')
        self.epoch_counter += 1
        self.__gen_new_batches()

#    def apply_update(grad):
#        ''''''


def test_batch_generator():
    import matplotlib.pyplot as plt
    import os
    try:
        os.mkdir('test_batch')
    except:
        pass
    shape = (256, 256)
    batches_shape = (8, 6)
    bg = GaussBatchGen(shape, 10, 100)
    for i in range(100):
        plt.figure(1)
        plt.imshow(bg.new_mask(), interpolation='none')
        plt.title('batch %02d' % i)
        plt.savefig('test_batch/batch_%02d.png' % i)


class GaussBatchGen:
    def __init__(self, shape, sigma, batch_size):
        self.sigma = sigma
        self.kernel = self.gen_gauss_kernel()
        self.batch_size = batch_size
        self.shape = shape
        self.epoch_counter = 0
        self.reset_iterations()
        self.mask_buffer = np.zeros(shape=self.shape, dtype=np.float32)

    def gen_gauss_kernel(self):
        kernel_shape = 8 * self.sigma
        self.kernel_shape = kernel_shape
        sigma = self.sigma
        x,y = np.meshgrid(np.arange(start=-kernel_shape / 2, stop=kernel_shape / 2),
                          np.arange(start=-kernel_shape / 2, stop=kernel_shape / 2))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        s2 = sigma ** 2
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * s2))
        return np.clip(1.2 * kernel / kernel.max(), 0.0, 1.0)

    def new_mask(self):
        mask = self.mask_buffer
        mask.fill(0.0)

        i = self.ind[self.batch_index]
        i_center, j_center = np.unravel_index(i, self.shape)
        i_low, i_high = 0, self.kernel.shape[0]
        j_low, j_high = 0, self.kernel.shape[1]

        def trim_high_low(center, low, high, dim, k_dim):
            kd2 = k_dim / 2
            if center + (low - kd2) < 0:
                low = kd2 - center
            if center + (high - kd2) > dim:
                high = dim - center + kd2

            return low, high, center + (low - kd2), center + (high - kd2)

        i_low, i_high, mi_low, mi_high = trim_high_low(i_center, i_low, i_high, 
                                      self.shape[0], self.kernel.shape[0])
        j_low, j_high, mj_low, mj_high  = trim_high_low(j_center, j_low, j_high, 
                                      self.shape[1], self.kernel.shape[1])
        # print(i_center, j_center)
        # print(i_low, i_high, mi_low, mi_high)
        # print(j_low, j_high, mj_low, mj_high)
        mask[mi_low : mi_high, mj_low : mj_high] = \
            self.kernel[i_low : i_high, j_low : j_high]
 
        self.batch_index += 1
        if (self.batch_index >= self.batch_size):
            self.reset_iterations()
        #print (mask.max(), mask.min(), mask.sum())
        #print (self.kernel.max(), self.kernel.sum() )
        return mask

    def reset_iterations(self):
        print('generating new batch order (', self.epoch_counter, ')')
        self.epoch_counter += 1
        self.ind = np.random.randint(0, self.shape[0] * self.shape[1], self.batch_size)
        self.batch_index = 0

if __name__ == '__main__':
    test_batch_generator()
