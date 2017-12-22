
# coding: utf-8

# # Enter State Farm

from theano.sandbox import cuda
from keras.models import load_model
cuda.use('gpu0')


from __future__ import print_function, division
path = "data/state/"
#path = "data/state/sample/"
import utils
reload(utils)
from utils import *
from IPython.display import FileLink


model_path = path + 'models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
file_name = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.model'


batch_size = 64


# ## Setup batches

batches = get_batches(path + 'train', batch_size=batch_size)
val_batches = get_batches(
    path + 'valid',
    batch_size=batch_size * 2,
    shuffle=False)


(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)


# Rather than using batches, we could just import all the data into an
# array to save some processing time. (In most examples I'm using the
# batches, however - just because that's how I happened to start out.)

trn = get_data(path + 'train')
val = get_data(path + 'valid')


save_array(path + 'results/val.dat', val)
save_array(path + 'results/trn.dat', trn)


val = load_array(path + 'results/val.dat')
trn = load_array(path + 'results/trn.dat')


# ## Re-run sample experiments on full dataset

# We should find that everything that worked on the sample (see
# statefarm-sample.ipynb), works on the full dataset too. Only better!
# Because now we have more data. So let's see how they go - the models in
# this section are exact copies of the sample notebook models.

# ### Single conv layer

def conv1(batches):
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])

    model.compile(
        Adam(
            lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches,
                        nb_val_samples=val_batches.nb_sample)
    model.optimizer.lr = 0.001
    model.fit_generator(batches, batches.nb_sample, nb_epoch=4, validation_data=val_batches,
                        nb_val_samples=val_batches.nb_sample)
    return model


model = conv1(batches)


model.save_weights(model_path + 'statefarm1.h5')


model.load_weights(model_path + 'statefarm1.h5')


# Interestingly, with no regularization or augmentation we're getting some
# reasonable results from our simple convolutional model. So with
# augmentation, we hopefully will see some very good results.

# ### Data augmentation

gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                                 shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


model = conv1(batches)


model.save(model_path + 'statefarm2.h5')


model = load_model(model_path + 'statefarm2.h5')


model.optimizer.lr = 0.0001
model.fit_generator(batches, batches.nb_sample, nb_epoch=15, validation_data=val_batches,
                    nb_val_samples=val_batches.nb_sample)


# I'm shocked by *how* good these results are! We're regularly seeing
# 75-80% accuracy on the validation set, which puts us into the top third
# or better of the competition. With such a simple model and no dropout or
# semi-supervised learning, this really speaks to the power of this
# approach to data augmentation.

# ### Four conv/pooling pairs + dropout

# Unfortunately, the results are still very unstable - the validation
# accuracy jumps from epoch to epoch. Perhaps a deeper model with some
# dropout would help.

gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                                 shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


model = Sequential([
    BatchNormalization(axis=1, input_shape=(3, 224, 224)),
    Convolution2D(32, 3, 3, activation='relu'),
    BatchNormalization(axis=1),
    MaxPooling2D(),
    Convolution2D(64, 3, 3, activation='relu'),
    BatchNormalization(axis=1),
    MaxPooling2D(),
    Convolution2D(128, 3, 3, activation='relu'),
    BatchNormalization(axis=1),
    MaxPooling2D(),
    Flatten(),
    Dense(200, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(200, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


model.compile(
    Adam(
        lr=10e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


model.fit_generator(batches, batches.nb_sample, nb_epoch=2, validation_data=val_batches,
                    nb_val_samples=val_batches.nb_sample)


model.optimizer.lr = 0.001


model.fit_generator(batches, batches.nb_sample, nb_epoch=10, validation_data=val_batches,
                    nb_val_samples=val_batches.nb_sample)


model.optimizer.lr = 0.00001


model.fit_generator(batches, batches.nb_sample, nb_epoch=10, validation_data=val_batches,
                    nb_val_samples=val_batches.nb_sample)


# This is looking quite a bit better - the accuracy is similar, but the
# stability is higher. There's still some way to go however...

# ### Imagenet conv features

# Since we have so little data, and it is similar to imagenet images (full
# color photos), using pre-trained VGG weights is likely to be helpful -
# in fact it seems likely that we won't need to fine-tune the
# convolutional layer weights much, if at all. So we can pre-compute the
# output of the last convolutional layer, as we did in lesson 3 when we
# experimented with dropout. (However this means that we can't use full
# data augmentation, since we can't pre-compute something that changes
# every image.)

vgg = Vgg16()
model = vgg.model
last_conv_idx = [i for i, l in enumerate(
    model.layers) if isinstance(l, Convolution2D)][-1]
conv_layers = model.layers[:last_conv_idx + 1]


conv_model = Sequential(conv_layers)


# batches shuffle must be set to False when pre-computing features
batches = get_batches(path + 'train', batch_size=batch_size, shuffle=False)


(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)


conv_feat = conv_model.predict_generator(batches, batches.nb_sample)
conv_val_feat = conv_model.predict_generator(
    val_batches, val_batches.nb_sample)
conv_test_feat = conv_model.predict_generator(
    test_batches, test_batches.nb_sample)


save_array(path + 'results/conv_val_feat.dat', conv_val_feat)
save_array(path + 'results/conv_test_feat.dat', conv_test_feat)
save_array(path + 'results/conv_feat.dat', conv_feat)


conv_feat = load_array(path + 'results/conv_feat.dat')
conv_val_feat = load_array(path + 'results/conv_val_feat.dat')
conv_val_feat.shape


# ### Batchnorm dense layers on pretrained conv layers

# Since we've pre-computed the output of the last convolutional layer, we
# need to create a network that takes that as input, and predicts our 10
# classes. Let's try using a simplified version of VGG's dense layers.

def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p / 2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p / 2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]


p = 0.8


bn_model = Sequential(get_bn_layers(p))
bn_model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=1,
             validation_data=(conv_val_feat, val_labels))


bn_model.optimizer.lr = 0.01


bn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=2,
             validation_data=(conv_val_feat, val_labels))


bn_model.save(path + 'models/conv8.h5')


# Looking good! Let's try pre-computing 5 epochs worth of augmented data,
# so we can experiment with combining dropout and augmentation on the
# pre-trained model.

# ### Pre-computed data augmentation + dropout

# We'll use our usual data augmentation parameters:

gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                                 shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
da_batches = get_batches(
    path + 'train',
    gen_t,
    batch_size=batch_size,
    shuffle=False)


# We use those to create a dataset of convolutional features 5x bigger
# than the training set.

da_conv_feat = conv_model.predict_generator(
    da_batches, da_batches.nb_sample * 5)


save_array(path + 'results/da_conv_feat2.dat', da_conv_feat)


da_conv_feat = load_array(path + 'results/da_conv_feat2.dat')


# Let's include the real training data as well in its non-augmented form.

da_conv_feat = np.concatenate([da_conv_feat, conv_feat])


# Since we've now got a dataset 6x bigger than before, we'll need to copy
# our labels 6 times too.

da_trn_labels = np.concatenate([trn_labels] * 6)


# Based on some experiments the previous model works well, with bigger
# dense layers.

def get_bn_da_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]


p = 0.8


bn_model = Sequential(get_bn_da_layers(p))
bn_model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# Now we can train the model as usual, with pre-computed augmented data.

bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=1,
             validation_data=(conv_val_feat, val_labels))


bn_model.optimizer.lr = 0.01


bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=4,
             validation_data=(conv_val_feat, val_labels))


bn_model.optimizer.lr = 0.0001


bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=4,
             validation_data=(conv_val_feat, val_labels))


# Looks good - let's save those weights.

bn_model.save_weights(path + 'models/da_conv8_1.h5')


# ### Pseudo labeling

# We're going to try using a combination of [pseudo
# labeling](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf)
# and [knowledge distillation](https://arxiv.org/abs/1503.02531) to allow
# us to use unlabeled data (i.e. do semi-supervised learning). For our
# initial experiment we'll use the validation set as the unlabeled data,
# so that we can see that it is working without using the test set. At a
# later date we'll try using the test set.

# To do this, we simply calculate the predictions of our model...

val_pseudo = bn_model.predict(conv_val_feat, batch_size=batch_size)


# ...concatenate them with our training labels...

comb_pseudo = np.concatenate([da_trn_labels, val_pseudo])


comb_feat = np.concatenate([da_conv_feat, conv_val_feat])


# ...and fine-tune our model using that data.

bn_model.load_weights(path + 'models/da_conv8_1.h5')


bn_model.fit(comb_feat, comb_pseudo, batch_size=batch_size, nb_epoch=1,
             validation_data=(conv_val_feat, val_labels))


bn_model.fit(comb_feat, comb_pseudo, batch_size=batch_size, nb_epoch=4,
             validation_data=(conv_val_feat, val_labels))


bn_model.optimizer.lr = 0.00001


bn_model.fit(comb_feat, comb_pseudo, batch_size=batch_size, nb_epoch=4,
             validation_data=(conv_val_feat, val_labels))


# That's a distinct improvement - even although the validation set isn't
# very big. This looks encouraging for when we try this on the test set.

bn_model.save_weights(path + 'models/bn-ps8.h5')


# ### Submit

# We'll find a good clipping amount using the validation set, prior to
# submitting.

def do_clip(arr, mx): return np.clip(arr, (1 - mx) / 9, mx)


keras.metrics.categorical_crossentropy(
    val_labels, do_clip(val_preds, 0.93)).eval()


conv_test_feat = load_array(path + 'results/conv_test_feat.dat')


preds = bn_model.predict(conv_test_feat, batch_size=batch_size * 2)


subm = do_clip(preds, 0.93)


subm_name = path + 'results/subm.gz'


classes = sorted(batches.class_indices, key=batches.class_indices.get)


submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'img', [a[4:] for a in test_filenames])
submission.head()


submission.to_csv(subm_name, index=False, compression='gzip')


FileLink(subm_name)


# This gets 0.534 on the leaderboard.

# ## The "things that didn't really work" section

# You can safely ignore everything from here on, because they didn't
# really help.

# ### Finetune some conv layers too

for l in get_bn_layers(p):
    conv_model.add(l)


for l1, l2 in zip(bn_model.layers, conv_model.layers[last_conv_idx + 1:]):
    l2.set_weights(l1.get_weights())


for l in conv_model.layers:
    l.trainable = False


for l in conv_model.layers[last_conv_idx + 1:]:
    l.trainable = True


comb = np.concatenate([trn, val])


gen_t = image.ImageDataGenerator(rotation_range=8, height_shift_range=0.04,
                                 shear_range=0.03, channel_shift_range=10, width_shift_range=0.08)


batches = gen_t.flow(comb, comb_pseudo, batch_size=batch_size)


val_batches = get_batches(
    path + 'valid',
    batch_size=batch_size * 2,
    shuffle=False)


conv_model.compile(
    Adam(
        lr=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


conv_model.fit_generator(batches, batches.N, nb_epoch=1, validation_data=val_batches,
                         nb_val_samples=val_batches.N)


conv_model.optimizer.lr = 0.0001


conv_model.fit_generator(batches, batches.N, nb_epoch=3, validation_data=val_batches,
                         nb_val_samples=val_batches.N)


for l in conv_model.layers[16:]:
    l.trainable = True


conv_model.optimizer.lr = 0.00001


conv_model.fit_generator(batches, batches.N, nb_epoch=8, validation_data=val_batches,
                         nb_val_samples=val_batches.N)


conv_model.save_weights(path + 'models/conv8_ps.h5')


conv_model.load_weights(path + 'models/conv8_da.h5')


val_pseudo = conv_model.predict(val, batch_size=batch_size * 2)


save_array(path + 'models/pseudo8_da.dat', val_pseudo)


# ### Ensembling

drivers_ds = pd.read_csv(path + 'driver_imgs_list.csv')
drivers_ds.head()


img2driver = drivers_ds.set_index('img')['subject'].to_dict()


driver2imgs = {k: g["img"].tolist()
               for k, g in drivers_ds[['subject', 'img']].groupby("subject")}


def get_idx(driver_list):
    return [i for i, f in enumerate(
        filenames) if img2driver[f[3:]] in driver_list]


drivers = driver2imgs.keys()


rnd_drivers = np.random.permutation(drivers)


ds1 = rnd_drivers[:len(rnd_drivers) // 2]
ds2 = rnd_drivers[len(rnd_drivers) // 2:]


models = [fit_conv([d]) for d in drivers]
models = [m for m in models if m is not None]


all_preds = np.stack([m.predict(conv_test_feat, batch_size=128)
                      for m in models])
avg_preds = all_preds.mean(axis=0)
avg_preds = avg_preds / np.expand_dims(avg_preds.sum(axis=1), 1)


keras.metrics.categorical_crossentropy(
    val_labels, np.clip(
        avg_val_preds, 0.01, 0.99)).eval()


keras.metrics.categorical_accuracy(
    val_labels, np.clip(
        avg_val_preds, 0.01, 0.99)).eval()
