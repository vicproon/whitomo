import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import teeth_io

with h5py.File('zub_pb.h5') as h5f:
    full_zub = np.array(h5f['sinogram'])

sino = full_zub[:, :, 820]

for i, angle in enumerate(range(0, 360, 10)):
    f = plt.figure(figsize=(6,3))
    plt.subplot(121)
    ax = plt.gca()
    line_820 = patches.Polygon([[0, 820], [full_zub.shape[1], 820]], closed=False, color='r', lw=2)
    ax.add_patch(line_820)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('x, пикс')
    plt.ylabel('y, пикс')
    plt.title('объект')

    plt.imshow(full_zub[angle].T, vmin=0, vmax=4)
    
    plt.subplot(122)
    this_sino = sino.copy()
    this_sino[angle + 10:, :] = 0
    plt.yticks(np.arange(0, 180, 30))
    plt.ylabel('$\\varphi$, град.')
    plt.xlabel('$\\xi$, пикс')
    plt.xticks([])
    plt.title('измерения')
    plt.imshow(this_sino, vmin=0, vmax=4, aspect='auto', extent=(0, full_zub.shape[1], 180, 0))
    
    plt.suptitle('$\\varphi = %d$ град' % int((angle / 2)), fontsize=16)
    plt.tight_layout()
    plt.savefig('anim/sino_%03d.png' % i, dpi=200)
    plt.close(f)
    # break
