from __future__ import division, print_function
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
to_bw = np.array([0.299, 0.587, 0.114])

def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if isinstance(ims[0], np.ndarray):
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()
