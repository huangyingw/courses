
# coding: utf-8

# # Fisheries competition

# In this notebook we're going to investigate a range of different
# architectures for the [Kaggle fisheries
# competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/).
# The video states that vgg.py and ``vgg_ft()`` from utils.py have been
# updated to include VGG with batch normalization, but this is not the
# case.  We've instead created a new file
# [vgg_bn.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16bn.py)
# and an additional method ``vgg_ft_bn()`` (which is already in utils.py)
# which we use in this notebook.


import utils
reload(utils)
from utils import *


#path = "data/fish/sample/"
path = "data/fish/"
batch_size = 64


batches = get_batches(path + 'train', batch_size=batch_size)
val_batches = get_batches(
    path + 'valid',
    batch_size=batch_size * 2,
    shuffle=False)

(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)


# Sometimes it's helpful to have just the filenames, without the path.

raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]


# ## Setup dirs

# We create the validation and sample sets in the usual way.

get_ipython().magic(u'cd data/fish')
get_ipython().magic(u'cd train')
get_ipython().magic(u'mkdir ../valid')


g = glob('*')
for d in g:
    os.mkdir('../valid/' + d)

g = glob('*/*.jpg')
shuf = np.random.permutation(g)
for i in range(500):
    os.rename(shuf[i], '../valid/' + shuf[i])


get_ipython().magic(u'mkdir ../sample')
get_ipython().magic(u'mkdir ../sample/train')
get_ipython().magic(u'mkdir ../sample/valid')


from shutil import copyfile

g = glob('*')
for d in g:
    os.mkdir('../sample/train/' + d)
    os.mkdir('../sample/valid/' + d)


g = glob('*/*.jpg')
shuf = np.random.permutation(g)
for i in range(400):
    copyfile(shuf[i], '../sample/train/' + shuf[i])

get_ipython().magic(u'cd ../valid')

g = glob('*/*.jpg')
shuf = np.random.permutation(g)
for i in range(200):
    copyfile(shuf[i], '../sample/valid/' + shuf[i])

get_ipython().magic(u'cd ..')


get_ipython().magic(u'mkdir results')
get_ipython().magic(u'mkdir sample/results')
get_ipython().magic(u'cd ../..')


# ## Basic VGG

# We start with our usual VGG approach.  We will be using VGG with batch
# normalization.  We explained how to add batch normalization to VGG in
# the [imagenet_batchnorm
# notebook](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/imagenet_batchnorm.ipynb).
# VGG with batch normalization is implemented in
# [vgg_bn.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16bn.py),
# and there is a version of ``vgg_ft`` (our fine tuning function) with
# batch norm called ``vgg_ft_bn`` in
# [utils.py](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py).

# ### Initial model

# First we create a simple fine-tuned VGG model to be our starting point.

from vgg16bn import Vgg16BN
model = vgg_ft_bn(8)


trn = get_data(path + 'train')
val = get_data(path + 'valid')


test = get_data(path + 'test')


save_array(path + 'results/trn.dat', trn)
save_array(path + 'results/val.dat', val)


save_array(path + 'results/test.dat', test)


trn = load_array(path + 'results/trn.dat')
val = load_array(path + 'results/val.dat')


test = load_array(path + 'results/test.dat')


gen = image.ImageDataGenerator()


model.compile(optimizer=Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    trn,
    trn_labels,
    batch_size=batch_size,
    nb_epoch=3,
    validation_data=(
        val,
        val_labels))


model.save_weights(path + 'results/ft1.h5')


# ### Precompute convolutional output

# We pre-compute the output of the last convolution layer of VGG, since
# we're unlikely to need to fine-tune those layers. (All following
# analysis will be done on just the pre-computed convolutional features.)

model.load_weights(path + 'results/ft1.h5')


conv_layers, fc_layers = split_at(model, Convolution2D)


conv_model = Sequential(conv_layers)


conv_feat = conv_model.predict(trn)
conv_val_feat = conv_model.predict(val)


conv_test_feat = conv_model.predict(test)


save_array(path + 'results/conv_val_feat.dat', conv_val_feat)
save_array(path + 'results/conv_feat.dat', conv_feat)


save_array(path + 'results/conv_test_feat.dat', conv_test_feat)


conv_feat = load_array(path + 'results/conv_feat.dat')
conv_val_feat = load_array(path + 'results/conv_val_feat.dat')


conv_test_feat = load_array(path + 'results/conv_test_feat.dat')


conv_val_feat.shape


# ### Train model

# We can now create our first baseline model - a simple 3-layer FC net.

def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        BatchNormalization(axis=1),
        Dropout(p / 4),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p / 2),
        Dense(8, activation='softmax')
    ]


p = 0.6


bn_model = Sequential(get_bn_layers(p))
bn_model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=3,
             validation_data=(conv_val_feat, val_labels))


bn_model.optimizer.lr = 1e-4


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=7,
             validation_data=(conv_val_feat, val_labels))


bn_model.save_weights(path + 'models/conv_512_6.h5')


bn_model.evaluate(conv_val_feat, val_labels)


bn_model.load_weights(path + 'models/conv_512_6.h5')


# ## Multi-input

# The images are of different sizes, which are likely to represent the
# boat they came from (since different boats will use different cameras).
# Perhaps this creates some data leakage that we can take advantage of to
# get a better Kaggle leaderboard position? To find out, first we create
# arrays of the file sizes for each image:

sizes = [PIL.Image.open(path + 'train/' + f).size for f in filenames]
id2size = list(set(sizes))
size2id = {o: i for i, o in enumerate(id2size)}


import collections
collections.Counter(sizes)


# Then we one-hot encode them (since we want to treat them as categorical)
# and normalize the data.

trn_sizes_orig = to_categorical([size2id[o] for o in sizes], len(id2size))


raw_val_sizes = [
    PIL.Image.open(
        path +
        'valid/' +
        f).size for f in val_filenames]
val_sizes = to_categorical([size2id[o] for o in raw_val_sizes], len(id2size))


trn_sizes = trn_sizes_orig - \
    trn_sizes_orig.mean(axis=0) / trn_sizes_orig.std(axis=0)
val_sizes = val_sizes - \
    trn_sizes_orig.mean(axis=0) / trn_sizes_orig.std(axis=0)


# To use this additional "meta-data", we create a model with multiple
# input layers - `sz_inp` will be our input for the size information.

p = 0.6


inp = Input(conv_layers[-1].output_shape[1:])
sz_inp = Input((len(id2size),))
bn_inp = BatchNormalization()(sz_inp)

x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p / 4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p / 2)(x)
x = merge([x, bn_inp], 'concat')
x = Dense(8, activation='softmax')(x)


# When we compile the model, we have to specify all the input layers in an
# array.

model = Model([inp, sz_inp], x)
model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# And when we train the model, we have to provide all the input layers'
# data in an array.

model.fit([conv_feat, trn_sizes], trn_labels, batch_size=batch_size, nb_epoch=3,
          validation_data=([conv_val_feat, val_sizes], val_labels))


bn_model.optimizer.lr = 1e-4


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=8,
             validation_data=(conv_val_feat, val_labels))


# The model did not show an improvement by using the leakage, other than
# in the early epochs. This is most likely because the information about
# what boat the picture came from is readily identified from the image
# itself, so the meta-data turned out not to add any additional
# information.

# ## Bounding boxes & multi output

# ### Import / view bounding boxes

# A kaggle user has created bounding box annotations for each fish in each
# training set image. You can download them [from
# here](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25902/complete-bounding-box-annotation).
# We will see if we can utilize this additional information. First, we'll
# load in the data, and keep just the largest bounding box for each image.

import ujson as json


anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']


def get_annotations():
    annot_urls = {
        '5458/bet_labels.json': 'bd20591439b650f44b36b72a98d3ce27',
        '5459/shark_labels.json': '94b1b3110ca58ff4788fb659eda7da90',
        '5460/dol_labels.json': '91a25d29a29b7e8b8d7a8770355993de',
        '5461/yft_labels.json': '9ef63caad8f076457d48a21986d81ddc',
        '5462/alb_labels.json': '731c74d347748b5272042f0661dad37c',
        '5463/lag_labels.json': '92d75d9218c3333ac31d74125f2b380a'
    }
    cache_subdir = os.path.abspath(os.path.join(path, 'annos'))
    url_prefix = 'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/'

    makedirs(cache_subdir)

    for url_suffix, md5_hash in annot_urls.iteritems():
        fname = url_suffix.rsplit('/', 1)[-1]
        get_file(
            fname,
            url_prefix +
            url_suffix,
            cache_subdir=cache_subdir,
            md5_hash=md5_hash)


get_annotations()


bb_json = {}
for c in anno_classes:
    if c == 'other':
        continue  # no annotation file for "other" class
    j = json.load(open('{}annos/{}_labels.json'.format(path, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations']) > 0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height'] * x['width'])[-1]


bb_json['img_04908.jpg']


file2idx = {o: i for i, o in enumerate(raw_filenames)}
val_file2idx = {o: i for i, o in enumerate(raw_val_filenames)}


# For any images that have no annotations, we'll create an empty bounding box.

empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}


for f in raw_filenames:
    if not f in bb_json.keys():
        bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys():
        bb_json[f] = empty_bbox


# Finally, we convert the dictionary into an array, and convert the
# coordinates to our resized 224x224 images.

bb_params = ['height', 'width', 'x', 'y']


def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0] * conv_y
    bb[1] = bb[1] * conv_x
    bb[2] = max(bb[2] * conv_x, 0)
    bb[3] = max(bb[3] * conv_y, 0)
    return bb


trn_bbox = np.stack([convert_bb(bb_json[f], s) for f, s in zip(raw_filenames, sizes)],
                    ).astype(np.float32)
val_bbox = np.stack([convert_bb(bb_json[f], s)
                     for f, s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)


# Now we can check our work by drawing one of the annotations.

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0],
                         color=color, fill=False, lw=3)


def show_bb(i):
    bb = val_bbox[i]
    plot(val[i])
    plt.gca().add_patch(create_rect(bb))


show_bb(0)


# ### Create & train model

# Since we're not allowed (by the kaggle rules) to manually annotate the
# test set, we'll need to create a model that predicts the locations of
# the bounding box on each image. To do so, we create a model with
# multiple outputs: it will predict both the type of fish (the 'class'),
# and the 4 bounding box coordinates. We prefer this approach to only
# predicting the bounding box coordinates, since we hope that giving the
# model more context about what it's looking for will help it with both
# tasks.

p = 0.6


inp = Input(conv_layers[-1].output_shape[1:])
x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p / 4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p / 2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)


# Since we have multiple outputs, we need to provide them to the model
# constructor in an array, and we also need to say what loss function to
# use for each. We also weight the bounding box loss function down by
# 1000x since the scale of the cross-entropy loss and the MSE is very
# different.

model = Model([inp], [x_bb, x_class])
model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'],
              loss_weights=[.001, 1.])


model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=3,
          validation_data=(conv_val_feat, [val_bbox, val_labels]))


model.optimizer.lr = 1e-5


model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=batch_size, nb_epoch=10,
          validation_data=(conv_val_feat, [val_bbox, val_labels]))


# Excitingly, it turned out that the classification model is much improved
# by giving it this additional task. Let's see how well the bounding box
# model did by taking a look at its output.

pred = model.predict(conv_val_feat[0:10])


def show_bb_pred(i):
    bb = val_bbox[i]
    bb_pred = pred[0][i]
    plt.figure(figsize=(6, 6))
    plot(val[i])
    ax = plt.gca()
    ax.add_patch(create_rect(bb_pred, 'yellow'))
    ax.add_patch(create_rect(bb))


# The image shows that it can find fish that are tricky for us to see!

show_bb_pred(6)


model.evaluate(conv_val_feat, [val_bbox, val_labels])


model.save_weights(path + 'models/bn_anno.h5')


model.load_weights(path + 'models/bn_anno.h5')


# ## Larger size

# ### Set up data

# Let's see if we get better results if we use larger images. We'll use
# 640x360, since it's the same shape as the most common size we saw
# earlier (1280x720), without being too big.

trn = get_data(path + 'train', (360, 640))
val = get_data(path + 'valid', (360, 640))


# The image shows that things are much clearer at this size.

plot(trn[0])


test = get_data(path + 'test', (360, 640))


save_array(path + 'results/trn_640.dat', trn)
save_array(path + 'results/val_640.dat', val)


save_array(path + 'results/test_640.dat', test)


trn = load_array(path + 'results/trn_640.dat')
val = load_array(path + 'results/val_640.dat')


# We can now create our VGG model - we'll need to tell it we're not using
# the normal 224x224 images, which also means it won't include the fully
# connected layers (since they don't make sense for non-default sizes). We
# will also remove the last max pooling layer, since we don't want to
# throw away information yet.

vgg640 = Vgg16BN((360, 640)).model
vgg640.pop()
vgg640.input_shape, vgg640.output_shape
vgg640.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])


# We can now pre-compute the output of the convolutional part of VGG.

conv_val_feat = vgg640.predict(val, batch_size=32, verbose=1)
conv_trn_feat = vgg640.predict(trn, batch_size=32, verbose=1)


save_array(path + 'results/conv_val_640.dat', conv_val_feat)
save_array(path + 'results/conv_trn_640.dat', conv_trn_feat)


conv_test_feat = vgg640.predict(test, batch_size=32, verbose=1)


save_array(path + 'results/conv_test_640.dat', conv_test_feat)


conv_val_feat = load_array(path + 'results/conv_val_640.dat')
conv_trn_feat = load_array(path + 'results/conv_trn_640.dat')


conv_test_feat = load_array(path + 'results/conv_test_640.dat')


# ### Fully convolutional net (FCN)

# Since we're using a larger input, the output of the final convolutional
# layer is also larger. So we probably don't want to put a dense layer
# there - that would be a *lot* of parameters! Instead, let's use a fully
# convolutional net (FCN); this also has the benefit that they tend to
# generalize well, and also seems like a good fit for our problem (since
# the fish are a small part of the image).

conv_layers, _ = split_at(vgg640, Convolution2D)


# I'm not using any dropout, since I found I got better results without it.

nf = 128
p = 0.


def get_lrg_layers():
    return [
        BatchNormalization(axis=1,
                           input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(nf, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D((1, 2)),
        Convolution2D(8, 3, 3, border_mode='same'),
        Dropout(p),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]


lrg_model = Sequential(get_lrg_layers())


lrg_model.summary()


lrg_model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2,
              validation_data=(conv_val_feat, val_labels))


lrg_model.optimizer.lr = 1e-5


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6,
              validation_data=(conv_val_feat, val_labels))


# When I submitted the results of this model to Kaggle, I got the best
# single model results of any shown here (ranked 22nd on the leaderboard
# as at Dec-6-2016.)

lrg_model.save_weights(path + 'models/lrg_nmp.h5')


lrg_model.load_weights(path + 'models/lrg_nmp.h5')


lrg_model.evaluate(conv_val_feat, val_labels)


# Another benefit of this kind of model is that the last convolutional
# layer has to learn to classify each part of the image (since there's
# only an average pooling layer after). Let's create a function that grabs
# the output of this layer (which is the 4th-last layer of our model).

l = lrg_model.layers
conv_fn = K.function([l[0].input, K.learning_phase()], l[-4].output)


def get_cm(inp, label):
    conv = conv_fn([inp, 0])[0, label]
    return scipy.misc.imresize(conv, (360, 640), interp='nearest')


# We have to add an extra dimension to our input since the CNN expects a
# 'batch' (even if it's just a batch of one).

inp = np.expand_dims(conv_val_feat[0], 0)
np.round(lrg_model.predict(inp)[0], 2)


plt.imshow(to_plot(val[0]))


cm = get_cm(inp, 0)


# The heatmap shows that (at very low resolution) the model is finding the
# fish!

plt.imshow(cm, cmap="cool")


# ### All convolutional net heatmap

# To create a higher resolution heatmap, we'll remove all the max pooling
# layers, and repeat the previous steps.

def get_lrg_layers():
    return [
        BatchNormalization(axis=1,
                           input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(nf, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(nf, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(nf, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(8, 3, 3, border_mode='same'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]


lrg_model = Sequential(get_lrg_layers())


lrg_model.summary()


lrg_model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2,
              validation_data=(conv_val_feat, val_labels))


lrg_model.optimizer.lr = 1e-5


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6,
              validation_data=(conv_val_feat, val_labels))


lrg_model.save_weights(path + 'models/lrg_0mp.h5')


lrg_model.load_weights(path + 'models/lrg_0mp.h5')


# #### Create heatmap

l = lrg_model.layers
conv_fn = K.function([l[0].input, K.learning_phase()], l[-3].output)


def get_cm2(inp, label):
    conv = conv_fn([inp, 0])[0, label]
    return scipy.misc.imresize(conv, (360, 640))


inp = np.expand_dims(conv_val_feat[0], 0)


plt.imshow(to_plot(val[0]))


cm = get_cm2(inp, 0)


cm = get_cm2(inp, 4)


plt.imshow(cm, cmap="cool")


plt.figure(figsize=(10, 10))
plot(val[0])
plt.imshow(cm, cmap="cool", alpha=0.5)


# ### Inception mini-net

# Here's an example of how to create and use "inception blocks" - as you
# see, they use multiple different convolution filter sizes and
# concatenate the results together. We'll talk more about these next year.

def conv2d_bn(x, nb_filter, nb_row, nb_col, subsample=(1, 1)):
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample, activation='relu', border_mode='same')(x)
    return BatchNormalization(axis=1)(x)


def incep_block(x):
    branch1x1 = conv2d_bn(x, 32, 1, 1, subsample=(2, 2))
    branch5x5 = conv2d_bn(x, 24, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, subsample=(2, 2))

    branch3x3dbl = conv2d_bn(x, 32, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, subsample=(2, 2))

    branch_pool = AveragePooling2D(
        (3, 3), strides=(2, 2), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1)
    return merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                 mode='concat', concat_axis=1)


inp = Input(vgg640.layers[-1].output_shape[1:])
x = BatchNormalization(axis=1)(inp)
x = incep_block(x)
x = incep_block(x)
x = incep_block(x)
x = Dropout(0.75)(x)
x = Convolution2D(8, 3, 3, border_mode='same')(x)
x = GlobalAveragePooling2D()(x)
outp = Activation('softmax')(x)


lrg_model = Model([inp], outp)


lrg_model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2,
              validation_data=(conv_val_feat, val_labels))


lrg_model.optimizer.lr = 1e-5


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=6,
              validation_data=(conv_val_feat, val_labels))


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=10,
              validation_data=(conv_val_feat, val_labels))


lrg_model.save_weights(path + 'models/lrg_nmp.h5')


lrg_model.load_weights(path + 'models/lrg_nmp.h5')


# ## Pseudo-labeling

preds = model.predict([conv_test_feat, test_sizes], batch_size=batch_size * 2)


gen = image.ImageDataGenerator()


test_batches = gen.flow(conv_test_feat, preds, batch_size=16)


val_batches = gen.flow(conv_val_feat, val_labels, batch_size=4)


batches = gen.flow(conv_feat, trn_labels, batch_size=44)


mi = MixIterator([batches, test_batches, val_batches])


bn_model.fit_generator(
    mi,
    mi.N,
    nb_epoch=8,
    validation_data=(
        conv_val_feat,
        val_labels))


# ## Submit

def do_clip(arr, mx): return np.clip(arr, (1 - mx) / 7, mx)


lrg_model.evaluate(conv_val_feat, val_labels, batch_size * 2)


preds = model.predict(conv_test_feat, batch_size=batch_size)


preds = preds[1]


test = load_array(path + 'results/test_640.dat')


test = load_array(path + 'results/test.dat')


preds = conv_model.predict(test, batch_size=32)


subm = do_clip(preds, 0.82)


subm_name = path + 'results/subm_bb.gz'


# classes = sorted(batches.class_indices, key=batches.class_indices.get)
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', raw_test_filenames)
submission.head()


submission.to_csv(subm_name, index=False, compression='gzip')


FileLink(subm_name)
