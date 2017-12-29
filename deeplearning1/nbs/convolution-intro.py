import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from scipy.ndimage.filters import correlate, convolve
np.set_printoptions(precision=4, linewidth=100)


'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
images, labels = mnist.train.images, mnist.train.labels
images = images.reshape((55000,28,28))
np.savez_compressed("MNIST_data/train", images=images, labels=labels)
1
'''


def plots(ims, interp=False, titles=None):
    ims = np.array(ims)
    mn, mx = ims.min(), ims.max()
    f = plt.figure(figsize=(12, 24))
    for i in range(len(ims)):
        sp = f.add_subplot(1, len(ims), i + 1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(
            ims[i],
            interpolation=None if interp else 'none',
            vmin=mn,
            vmax=mx)
    plt.show()


def plot(im, interp=False):
    plt.figure(figsize=(3, 6), frameon=True)
    plt.imshow(im, interpolation=None if interp else 'none')
    plt.show()


plt.gray()
plt.close()


data = np.load("MNIST_data/train.npz")
images = data['images']
labels = data['labels']
n = len(images)
images.shape


plot(images[0])


labels[0]


plots(images[:5], titles=labels[:5])


top = [[-1, -1, -1],
       [1, 1, 1],
       [0, 0, 0]]

plot(top)
corrtop = correlate(images[0], top)
plot(corrtop)
np.rot90(top, 1)


convtop = convolve(images[0], np.rot90(top, 2))
plot(convtop)
np.allclose(convtop, corrtop)


straights = [np.rot90(top, i) for i in range(4)]
plots(straights)


br = [[0, 0, 1],
      [0, 1, -1.5],
      [1, -1.5, 0]]

diags = [np.rot90(br, i) for i in range(4)]
plots(diags)


rots = straights + diags
corrs = [correlate(images[0], rot) for rot in rots]
plots(corrs)


def pool(im): return block_reduce(im, (7, 7), np.max)


plots([pool(im) for im in corrs])


eights = [images[i] for i in xrange(n) if labels[i] == 8]
ones = [images[i] for i in xrange(n) if labels[i] == 1]


plots(eights[:5])
plots(ones[:5])


pool8 = [np.array([pool(correlate(im, rot)) for im in eights]) for rot in rots]


len(pool8), pool8[0].shape


plots(pool8[0][0:5])


def normalize(arr): return (arr - arr.mean()) / arr.std()


filts8 = np.array([ims.mean(axis=0) for ims in pool8])
filts8 = normalize(filts8)


plots(filts8)


pool1 = [np.array([pool(correlate(im, rot)) for im in ones]) for rot in rots]
filts1 = np.array([ims.mean(axis=0) for ims in pool1])
filts1 = normalize(filts1)


plots(filts1)


def pool_corr(im): return np.array([pool(correlate(im, rot)) for rot in rots])


plots(pool_corr(eights[0]))


def sse(a, b): return ((a - b)**2).sum()


def is8_n2(im): return 1 if sse(
    pool_corr(im),
    filts1) > sse(
        pool_corr(im),
    filts8) else 0


sse(pool_corr(eights[0]), filts8), sse(pool_corr(eights[0]), filts1)


[np.array([is8_n2(im) for im in ims]).sum() for ims in [eights, ones]]


[np.array([(1 - is8_n2(im)) for im in ims]).sum() for ims in [eights, ones]]


def n1(a, b): return (np.fabs(a - b)).sum()


def is8_n1(im): return 1 if n1(
    pool_corr(im),
    filts1) > n1(
        pool_corr(im),
    filts8) else 0


[np.array([is8_n1(im) for im in ims]).sum() for ims in [eights, ones]]


[np.array([(1 - is8_n1(im)) for im in ims]).sum() for ims in [eights, ones]]
