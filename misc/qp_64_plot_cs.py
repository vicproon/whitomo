import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x1 = np.loadtxt('x1.nptxt')
x2 = np.loadtxt('x2.nptxt')
x3 = np.loadtxt('x3.nptxt')
x4 = np.loadtxt('x4.nptxt')
x4 = np.loadtxt('x5.nptxt')
x4 = np.loadtxt('x4.nptxt')
x5 = np.loadtxt('x5.nptxt')
x6 = np.loadtxt('x6.nptxt')
x0 = np.loadtxt('../phantom.txt')

x3 = np.clip(x3, 0.0, 1.1 * x0.max())

def gamma_correct(im_, gamma):
    return gamma_correct_like(im_, im_, gamma)

def gamma_correct_like(im_, like, gamma):
    im = im_.copy()
    m = like.min()
    M = like.max()
    if M == m:
        return im
    im = (im.astype(np.float) - m) / (M - m)
    negative = im <= 0
    im[~negative] = np.power(im[~negative], gamma)
    im[negative] = - np.power(-im[negative], gamma)
    im = m + im * (M - m)
    return im

imgs = [x0, x2, x3, x5]
the_x3 = x0.copy()
imgs = [gamma_correct_like(x, the_x3, 0.5) for x in imgs]
labels = [u'Фантом', 'FBP', 'QP(barrier method)', 'Soft Inequalities']
colors = ['k', 'c', 'm', 'g']
lss = ['-', '-.', ':', '--']
lws = [2, 2, 3, 2]
import matplotlib.pyplot as plt
x = 28
lines = [img[:, x] for img in imgs]

f, ax = plt.subplots(1, 2, figsize=(11, 5))

def cs_patch():
    return patches.Polygon([[x + 0.5, 0], [x + 0.5, x0.shape[0]]],
                          closed=False, color='r', lw=4)
# f, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
im1 = ax[0].imshow(gamma_correct_like(x0, the_x3, 0.5), cmap='viridis', interpolation='none')

ax[0].add_patch(cs_patch())

for i in range(4):
    ax[1].plot(lines[i], 
             color=colors[i],
             linestyle=lss[i],
             label=labels[i],
             linewidth=lws[i])
ax[1].legend(labels)
ax[1].set_xlabel(u'пиксели')
ax[1].set_ylabel(u'интенсивность')

plt.tight_layout()
plt.savefig('cs_x%d.png' % x, dpi=300)
plt.show()
