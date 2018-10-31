#tmp_plot
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(-2, 0, 5000)
t = np.array([2, 4, 16])
ls = ['--', '-.', ':']

plt.axes().set_aspect('equal')
plt.xlim([-1.5, 1.5])
plt.ylim([-0.25, 2.5])
plt.grid(True)
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')

line_handles = []
for tt,l in zip(t, ls):
    ll = -np.log(-x[:-1]) / tt
    lh = plt.plot(x[:-1], ll, linestyle=l, color='black', linewidth=1.5)
    line_handles.append(lh)

line_handles.append(plt.plot(x2, l2, color='black', linewidth=1.5))
plt.legend([lh[0] for lh in line_handles],
            ['$-\\frac{1}{t} log(-x),\\ t = 2$', 
            '$-\\frac{1}{t} log(-x),\\ t = 4$',
            '$-\\frac{1}{t} log(-x),\\ t = 16$',
            '$(x_+)^2$'])

plt.savefig('log_vs_l2.png', dpi=450)
plt.show()
