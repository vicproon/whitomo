import matplotlib.patches as patches

sino = full_zub[:, :, 820]

for i, angle in enumerate(range(0, 360, 10)):
    f = plt.figure(figsize=(8,4))
    plt.title('$\\varphi = %d$ град' % angle)
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
    
    plt.suptitle('$\\varphi = %d$ град' % angle, fontsize=16)
    plt.tight_layout()
    plt.savefig('anim/sino_%03d.png' % i, dpi=300)
    plt.close(f)
    # break
