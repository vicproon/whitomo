# /usr/bin/python
# coding=utf-8

from __future__ import print_function
import random


class BatchFilter:
    def __init__(self, shape, batches_shape):
        assert(len(shape) == 2, 'BatchFilter can be applied only to 2d images')
        assert(len(batches_shape) == 2, 'BatchFilter.batches_shape must be 2d')
        self.shape = shape
        self.batches_shape = batches_shape
        self.num_batches = batches_shape[0] * batches_shape[1]
        import datetime
        now = datetime.datetime.now()
        random.seed(now.microsecond + now.second * 1e6 + 101991048)
        self.__gen_new_batches()

    def __gen_new_batches(self):
        pass

    def new_mask(self):
        pass
