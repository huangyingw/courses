
# coding: utf-8

# This notebook contains an implementation of the One Hundred Layers
# Tiramisu as described in Simon Jegou et al.'s paper [The One Hundred
# Layers Tiramisu: Fully Convolutional DenseNets for Semantic
# Segmentation](https://arxiv.org/abs/1611.09326).

import importlib
import utils2
importlib.reload(utils2)
from utils2 import *


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


limit_mem()


# # Tiramisu / Camvid

# Tiramisu is a fully-convolutional neural network based on DenseNet architecture. It was designed as a state-of-the-art approach to semantic image segmentation.
#
# We're going to use the same dataset they did, CamVid.
#
# CamVid is a dataset of images from a video. It has ~ 600 images, so it's quite small, and given that it is from a video the information content of the dataset is small.
#
# We're going to train this Tiramisu network from scratch to segment the
# CamVid dataset. This seems extremely ambitious!

# ## Setup

# Modify the following to point to the appropriate paths on your machine

PATH = '/data/datasets/SegNet-Tutorial/CamVid/'


frames_path = PATH + 'all/'


labels_path = PATH + 'allannot/'


PATH = '/data/datasets/camvid/'


# The images in CamVid come with labels defining the segments of the input
# image. We're going to load both the images and the labels.

frames_path = PATH + '701_StillsRaw_full/'


labels_path = PATH + 'LabeledApproved_full/'


fnames = glob.glob(frames_path + '*.png')


lnames = [labels_path + os.path.basename(fn)[:-4] + '_L.png' for fn in fnames]


img_sz = (480, 360)


# Helper function to resize images.

def open_image(fn): return np.array(
    Image.open(fn).resize(
        img_sz, Image.NEAREST))


img = Image.open(fnames[0]).resize(img_sz, Image.NEAREST)


img


imgs = np.stack([open_image(fn) for fn in fnames])


labels = np.stack([open_image(fn) for fn in lnames])


imgs.shape, labels.shape


# Normalize pixel values.

# Save array for easier use.

save_array(PATH + 'results/imgs.bc', imgs)
save_array(PATH + 'results/labels.bc', labels)


imgs = load_array(PATH + 'results/imgs.bc')
labels = load_array(PATH + 'results/labels.bc')


# Standardize

imgs -= 0.4
imgs /= 0.3


n, r, c, ch = imgs.shape


# ## Preprocessing

# ### Generator

# This implementation employs data augmentation on CamVid.
#
# Augmentation includes random cropping / horizontal flipping, as done by
# `segm_generator()`. `BatchIndices()` lets us randomly sample batches
# from input array.

class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n:
                self.reset()
            ni = min(self.bs, self.n - self.curr)
            res = self.idxs[self.curr:self.curr + ni]
            self.curr += ni
            return res


bi = BatchIndices(10, 3)
[next(bi) for o in range(5)]


bi = BatchIndices(10, 3, True)
[next(bi) for o in range(5)]


class segm_generator(object):
    def __init__(self, x, y, bs=64, out_sz=(224, 224), train=True):
        self.x, self.y, self.bs, self.train = x, y, bs, train
        self.n, self.ri, self.ci, _ = x.shape
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape) == 4 else 1

    def get_slice(self, i, o):
        start = random.randint(0, i - o) if self.train else (i - o)
        return slice(start, start + o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]
        if self.train and (random.random() > 0.5):
            y = y[:, ::-1]
            x = x[:, ::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs, ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.ych)


# As an example, here's a crop of the first image.

sg = segm_generator(imgs, labels, 4, train=False)
b_img, b_label = next(sg)
plt.imshow(b_img[0] * 0.3 + 0.4)


plt.imshow(imgs[0] * 0.3 + 0.4)


sg = segm_generator(imgs, labels, 4, train=True)
b_img, b_label = next(sg)
plt.imshow(b_img[0] * 0.3 + 0.4)


# ### Convert labels

# The following loads, parses, and converts the segment labels we need for targets.
#
# In particular we're looking to make the segmented targets into integers
# for classification purposes.

def parse_code(l):
    a, b = l.strip().split("\t")
    return tuple(int(o) for o in a.split(' ')), b


label_codes, label_names = zip(*[
    parse_code(l) for l in open(PATH + "label_colors.txt")])


label_codes, label_names = list(label_codes), list(label_names)


# Each segment / category is indicated by a particular color. The
# following maps each unique pixel to it's category.

list(zip(label_codes, label_names))[:5]


# We're going to map each unique pixel color to an integer so we can
# classify w/ our NN. (Think how a fill-in-the color image looks)

code2id = {v: k for k, v in enumerate(label_codes)}


# We'll include an integer for erroneous pixel values.

failed_code = len(label_codes) + 1


label_codes.append((0, 0, 0))
label_names.append('unk')


def conv_one_label(i):
    res = np.zeros((r, c), 'uint8')
    for j in range(r):
        for k in range(c):
            try:
                res[j, k] = code2id[tuple(labels[i, j, k])]
            except BaseException:
                res[j, k] = failed_code
    return res


from concurrent.futures import ProcessPoolExecutor


def conv_all_labels():
    ex = ProcessPoolExecutor(8)
    return np.stack(ex.map(conv_one_label, range(n)))


# Now we'll create integer-mapped labels for all our colored images.

get_ipython().magic(u'time labels_int =conv_all_labels()')


np.count_nonzero(labels_int == failed_code)


# Set erroneous pixels to zero.

labels_int[labels_int == failed_code] = 0


save_array(PATH + 'results/labels_int.bc', labels_int)


labels_int = load_array(PATH + 'results/labels_int.bc')


sg = segm_generator(imgs, labels, 4, train=True)
b_img, b_label = next(sg)
plt.imshow(b_img[0] * 0.3 + 0.4)


# Here is an example of how the segmented image looks.

plt.imshow(b_label[0].reshape(224, 224, 3))


# ### Test set

# Next we load test set, set training/test images and labels.

fn_test = set(o.strip() for o in open(PATH + 'test.txt', 'r'))


is_test = np.array([o.split('/')[-1] in fn_test for o in fnames])


trn = imgs[is_test == False]
trn_labels = labels_int[is_test == False]
test = imgs[is_test]
test_labels = labels_int[is_test]
trn.shape, test_labels.shape


rnd_trn = len(trn_labels)
rnd_test = len(test_labels)


# ## The Tiramisu

# Now that we've prepared our data, we're ready to introduce the Tiramisu.
#
# Conventional CNN's for image segmentation are very similar to the kind we looked at for style transfer. Recall that it involved convolutions with downsampling (stride 2, pooling) to increase the receptive field, followed by upsampling with deconvolutions until reaching the original side.
#
# Tiramisu uses a similar down / up architecture, but with some key caveats.
#
# As opposed to normal convolutional layers, Tiramisu uses the DenseNet method of concatenating inputs to outputs. Tiramisu also uses *skip connections* from the downsampling branch to the upsampling branch.
#
# Specifically, the *skip connection* functions by concatenating the output of a Dense block in the down-sampling branch **onto** the input of the corresponding Dense block in the upsampling branch. By "corresponding", we mean the down-sample/up-sample Dense blocks that are equidistant from the input / output respectively.
#
# One way of interpreting this architecture is that by re-introducing
# earlier stages of the network to later stages, we're forcing the network
# to "remember" the finer details of the input image.

# ### The pieces

# This should all be familiar.

def relu(x): return Activation('relu')(x)


def dropout(x, p): return Dropout(p)(x) if p else x


def bn(x): return BatchNormalization(mode=2, axis=-1)(x)


def relu_bn(x): return relu(bn(x))


def concat(xs): return merge(xs, mode='concat', concat_axis=-1)


def conv(x, nf, sz, wd, p, stride=1):
    x = Convolution2D(nf, sz, sz, init='he_uniform', border_mode='same',
                      subsample=(stride, stride), W_regularizer=l2(wd))(x)
    return dropout(x, p)


def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)


# Recall the dense block from DenseNet.

def dense_block(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x, added


# This is the downsampling transition.
#
# In the original paper, downsampling consists of 1x1 convolution followed
# by max pooling. However we've found a stride 2 1x1 convolution to give
# better results.

def transition_dn(x, p, wd):
    #     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
    #     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()
                        [-1], sz=1, p=p, wd=wd, stride=2)


# Next we build the entire downward path, keeping track of Dense block
# outputs in a list called `skip`.

def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = dense_block(n, x, growth_rate, p, wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added


# This is the upsampling transition. We use a deconvolution layer.

def transition_up(added, wd=0):
    x = concat(added)
    _, r, c, ch = x.get_shape().as_list()
    return Deconvolution2D(ch, 3, 3, (None, r * 2, c * 2, ch), init='he_uniform',
                           border_mode='same', subsample=(2, 2), W_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)


# This builds our upward path, concatenating the skip connections from
# `skip` to the Dense block inputs as mentioned.

def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i, n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x, skips[i]])
        x, added = dense_block(n, x, growth_rate, p, wd)
    return x


# ### Build the tiramisu model

# - nb_classes: number of classes
# - img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
# - depth: number or layers
# - nb_dense_block: number of dense blocks to add to end (generally = 3)
# - growth_rate: number of filters to add per dense block
# - nb_filter:  initial number of filters
# - nb_layers_per_block: number of layers in each dense block.
#   - If positive integer, a set number of layers per dense block.
#   - If list, nb_layer is used as provided
# - p: dropout rate
# - wd: weight decay

def reverse(a): return list(reversed(a))


# Finally we put together the entire network.

def create_tiramisu(nb_classes, img_input, nb_dense_block=6,
                    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):

    if isinstance(nb_layers_per_block, list) or isinstance(
            nb_layers_per_block, tuple):
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added,
                reverse(skips[:-1]),
                reverse(nb_layers[:-1]),
                growth_rate,
                p,
                wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)


# ## Train

# Now we can train.
#
# These architectures can take quite some time to train.

limit_mem()


input_shape = (224, 224, 3)


img_input = Input(shape=input_shape)


x = create_tiramisu(
    12,
    img_input,
    nb_layers_per_block=[
        4,
        5,
        7,
        10,
        12,
        15],
    p=0.2,
    wd=1e-4)


model = Model(img_input, x)


gen = segm_generator(trn, trn_labels, 3, train=True)


gen_test = segm_generator(test, test_labels, 3, train=False)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"])


model.optimizer = keras.optimizers.RMSprop(1e-3, decay=1 - 0.99995)


model.optimizer = keras.optimizers.RMSprop(1e-3)


K.set_value(model.optimizer.lr, 1e-3)


model.fit_generator(gen, rnd_trn, 100, verbose=2,
                    validation_data=gen_test, nb_val_samples=rnd_test)


model.optimizer = keras.optimizers.RMSprop(3e-4, decay=1 - 0.9995)


model.fit_generator(gen, rnd_trn, 500, verbose=2,
                    validation_data=gen_test, nb_val_samples=rnd_test)


model.optimizer = keras.optimizers.RMSprop(2e-4, decay=1 - 0.9995)


model.fit_generator(gen, rnd_trn, 500, verbose=2,
                    validation_data=gen_test, nb_val_samples=rnd_test)


model.optimizer = keras.optimizers.RMSprop(1e-5, decay=1 - 0.9995)


model.fit_generator(gen, rnd_trn, 500, verbose=2,
                    validation_data=gen_test, nb_val_samples=rnd_test)


lrg_sz = (352, 480)
gen = segm_generator(trn, trn_labels, 2, out_sz=lrg_sz, train=True)
gen_test = segm_generator(test, test_labels, 2, out_sz=lrg_sz, train=False)


lrg_shape = lrg_sz + (3,)
lrg_input = Input(shape=lrg_shape)


x = create_tiramisu(
    12,
    lrg_input,
    nb_layers_per_block=[
        4,
        5,
        7,
        10,
        12,
        15],
    p=0.2,
    wd=1e-4)


lrg_model = Model(lrg_input, x)


lrg_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(1e-4), metrics=["accuracy"])


lrg_model.fit_generator(gen, rnd_trn, 100, verbose=2,
                        validation_data=gen_test, nb_val_samples=rnd_test)


lrg_model.fit_generator(gen, rnd_trn, 100, verbose=2,
                        validation_data=gen_test, nb_val_samples=rnd_test)


lrg_model.optimizer = keras.optimizers.RMSprop(1e-5)


lrg_model.fit_generator(gen, rnd_trn, 2, verbose=2,
                        validation_data=gen_test, nb_val_samples=rnd_test)


lrg_model.save_weights(PATH + 'results/8758.h5')


# ## View results

# Let's take a look at some of the results we achieved.

colors = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192),
          (128, 128, 0), (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0),
          (0, 128, 192), (0, 0, 0)]
names = ['sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree',
                'sign', 'fence', 'car', 'pedestrian', 'bicyclist', 'void']


gen_test = segm_generator(test, test_labels, 2, out_sz=lrg_sz, train=False)


preds = lrg_model.predict_generator(gen_test, rnd_test)
preds = np.argmax(preds, axis=-1)
preds = preds.reshape((-1, 352, 480))


target = test_labels.reshape((233, 360, 480))[:, 8:]


(target == preds).mean()


non_void = target != 11
(target[non_void] == preds[non_void]).mean()


idx = 1


p = lrg_model.predict(np.expand_dims(test[idx, 8:], 0))
p = np.argmax(p[0], -1).reshape(352, 480)
pred = color_label(p)


# This is pretty good! We can see it is having some difficulty with the
# street between the light posts, but we would expect that a model that
# was pre-trained on a much larger dataset would perform better.

plt.imshow(pred)


plt.figure(figsize=(9, 9))
plt.imshow(test[idx] * 0.3 + 0.4)


# ## End
