# /usr/bin/python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def circle_base(x0, y0, r, x_grid, y_grid):
    return (x_grid - x0) ** 2 + (y_grid - y0) ** 2 <= r ** 2


def draw_2comp_concentrations(c1, c2):
    color_1 = np.array((1.0, 0, 0))
    color_2 = np.array((0, 0, 1.0))
    img_shape = (c1.shape[0], c1.shape[1], 3)
    colored_c1 = np.stack((c1,) * 3, -1)
    colored_c2 = np.stack((c2,) * 3, -1)
    img_todraw = np.broadcast_to(color_1, img_shape) * colored_c1 + \
        np.broadcast_to(color_2, img_shape) * colored_c2

    return (img_todraw * 255).astype('uint8')


if __name__ == '__main__':

    c1 = np.zeros(shape=(256, 256), dtype=np.float32)
    c2 = np.zeros(shape=(256, 256), dtype=np.float32)

    x_grid, y_grid = np.meshgrid(np.arange(0, 256), np.arange(0, 256))

    def circle(x0, y0, r):
        return circle_base(x0, y0, r, x_grid, y_grid)

    c1[circle(80, 80, 50)] = 1
    c2[circle(200, 200, 20)] = 1

    plt.imshow(draw_2comp_concentrations(c1, c2))
    plt.show()

    np.savetxt('../testdata/whitereconstruct/correct_button4/c1.txt', c1)
    np.savetxt('../testdata/whitereconstruct/correct_button4/c2.txt', c2)
