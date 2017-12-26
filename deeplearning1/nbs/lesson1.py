from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from vgg16 import Vgg16
import json
import numpy as np


# Define path to data: (It's a good idea to put it in a subdirectory of
# your notebooks folder, and then exclude that directory from git control
# by adding it to .gitignore.)

path = "data/dogscats/"
#path = "data/dogscats/sample/"


# A few basic libraries that we'll need for the initial exercises:


np.set_printoptions(precision=4, linewidth=100)

# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will
# have to decrease this.
batch_size = 64


# Import our class, and instantiate

vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size * 2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)


# Vgg16 is built on top of *Keras* (which we will be learning much more about shortly!), a flexible, easy to use deep learning library that sits on top of Theano or Tensorflow. Keras reads groups of images and labels in *batches*, using a fixed directory structure, where images from each category for training must be placed in a separate folder.
#
# Let's grab batches of data from our training folder:

#batches = vgg.get_batches(path + 'train', batch_size=4)


# (BTW, when Keras refers to 'classes', it doesn't mean python classes - but rather it refers to the categories of the labels, such as 'pug', or 'tabby'.)
#
# *Batches* is just a regular python iterator. Each iteration returns both the images themselves, as well as the labels.

#imgs, labels = next(batches)


# As you can see, the labels for each image are an array, containing a 1 in the first position if it's a cat, and in the second position if it's a dog. This approach to encoding categorical variables, where an array containing just a single 1 in the position corresponding to the category, is very common in deep learning. It is called *one hot encoding*.
#
# The arrays contain two elements, because we have two categories (cat,
# and dog). If we had three categories (e.g. cats, dogs, and kangaroos),
# then the arrays would each contain two 0's, and one 1.

#plots(imgs, titles=labels)


# We can now pass the images to Vgg16's predict() function to get back
# probabilities, category indexes, and category names for each image's VGG
# prediction.

#vgg.predict(imgs, True)


# The category indexes are based on the ordering of categories used in the
# VGG model - e.g here are the first four:

#vgg.classes[:4]


# (Note that, other than creating the Vgg16 object, none of these steps are necessary to build a model; they are just showing how to use the class to view imagenet predictions.)

# ## Use our Vgg16 class to finetune a Dogs vs Cats model
#
# To change our model so that it outputs "cat" vs "dog", instead of one of 1,000 very specific categories, we need to use a process called "finetuning". Finetuning looks from the outside to be identical to normal machine learning training - we provide a training set with data and labels to learn from, and a validation set to test against. The model learns a set of parameters based on the data provided.
#
# However, the difference is that we start with a model that is already trained to solve a similar problem. The idea is that many of the parameters should be very similar, or the same, between the existing model, and the model we wish to create. Therefore, we only select a subset of parameters to train, and leave the rest untouched. This happens automatically when we call *fit()* after calling *finetune()*.
#
# We create our batches just like before, and making the validation set
# available as well. A 'batch' (or *mini-batch* as it is commonly known)
# is simply a subset of the training data - we use a subset at a time when
# training or predicting, in order to speed up training, and to avoid
# running out of memory.

batch_size = 64


batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size)


# Calling *finetune()* modifies the model such that it will be trained
# based on the data in the batches provided - in this case, to predict
# either 'dog' or 'cat'.

vgg.finetune(batches)


# Finally, we *fit()* the parameters of the model using the training data,
# reporting the accuracy on the validation set after every epoch. (An
# *epoch* is one full pass through the training data.)

vgg.fit(batches, val_batches, nb_epoch=1)


# That shows all of the steps involved in using the Vgg16 class to create an image recognition model using whatever labels you are interested in. For instance, this process could classify paintings by style, or leaves by type of disease, or satellite photos by type of crop, and so forth.
#
# Next up, we'll dig one level deeper to see what's going on in the Vgg16
# class.

# # Create a VGG model from scratch in Keras
#
# For the rest of this tutorial, we will not be using the Vgg16 class at
# all. Instead, we will recreate from scratch the functionality we just
# used. This is not necessary if all you want to do is use the existing
# model - but if you want to create your own models, you'll need to
# understand these details. It will also help you in the future when you
# debug any problems with your models, since you'll understand what's
# going on behind the scenes.


# Let's import the mappings from VGG ids to imagenet category ids and
# descriptions, for display purposes later.

FILES_PATH = 'http://files.fast.ai/models/'
CLASS_FILE = 'imagenet_class_index.json'
# Keras' get_file() is a handy function that downloads files, and caches
# them for re-use later
fpath = get_file(CLASS_FILE, FILES_PATH + CLASS_FILE, cache_subdir='models')
with open(fpath) as f:
    class_dict = json.load(f)
# Convert dictionary with string indexes into an array
classes = [class_dict[str(i)][1] for i in range(len(class_dict))]


# Here's a few examples of the categories we just imported:

classes[:5]


# ## Model creation
#
# Creating the model involves creating the model architecture, and then loading the model weights into that architecture. We will start by defining the basic pieces of the VGG architecture.
#
# VGG has just one type of convolutional block, and one type of fully
# connected ('dense') block. Here's the convolutional block definition:

def ConvBlock(layers, model, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


# ...and here's the fully-connected definition.

def FCBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))


# When the VGG model was trained in 2014, the creators subtracted the
# average of each of the three (R,G,B) channels first, so that the data
# for each channel had a mean of zero. Furthermore, their software that
# expected the channels to be in B,G,R order, whereas Python by default
# uses R,G,B. We need to preprocess our data to make these two changes, so
# that it is compatible with the VGG model:

# Mean of each channel as provided by VGG researchers
vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))


def vgg_preprocess(x):
    x = x - vgg_mean     # subtract mean
    return x[:, ::-1]    # reverse axis bgr->rgb


# Now we're ready to define the VGG model architecture - look at how
# simple it is, now that we have the basic blocks defined!

def VGG_16():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224)))

    ConvBlock(2, model, 64)
    ConvBlock(2, model, 128)
    ConvBlock(3, model, 256)
    ConvBlock(3, model, 512)
    ConvBlock(3, model, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)
    model.add(Dense(1000, activation='softmax'))
    return model


# We'll learn about what these different blocks do later in the course. For now, it's enough to know that:
#
# - Convolution layers are for finding patterns in images
# - Dense (fully connected) layers are for combining patterns across an image
#
# Now that we've defined the architecture, we can create the model like
# any python object:

model = VGG_16()


# As well as the architecture, we need the weights that the VGG creators trained. The weights are the part of the model that is learnt from the data, whereas the architecture is pre-defined based on the nature of the problem.
#
# Downloading pre-trained weights is much preferred to training the model
# ourselves, since otherwise we would have to download the entire Imagenet
# archive, and train the model for many days! It's very helpful when
# researchers release their weights, as they did here.

fpath = get_file('vgg16.h5', FILES_PATH + 'vgg16.h5', cache_subdir='models')
model.load_weights(fpath)


# ## Getting imagenet predictions
#
# The setup of the imagenet model is now complete, so all we have to do is
# grab a batch of images and call *predict()* on them.

batch_size = 4


# Keras provides functionality to create batches of data from directories
# containing images; all we have to do is to define the size to resize the
# images to, what type of labels to create, whether to randomly shuffle
# the images, and how many images to include in each batch. We use this
# little wrapper to define some helpful defaults appropriate for imagenet
# data:

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True,
                batch_size=batch_size, class_mode='categorical'):
    return gen.flow_from_directory(path + dirname, target_size=(224, 224),
                                   class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


# From here we can use exactly the same steps as before to look at
# predictions from the model.

batches = get_batches('train', batch_size=batch_size)
val_batches = get_batches('valid', batch_size=batch_size)
imgs, labels = next(batches)

# This shows the 'ground truth'
#plots(imgs, titles=labels)


# The VGG model returns 1,000 probabilities for each image, representing
# the probability that the model assigns to each possible imagenet
# category for each image. By finding the index with the largest
# probability (with *np.argmax()*) we can find the predicted label.

def pred_batch(imgs):
    preds = model.predict(imgs)
    idxs = np.argmax(preds, axis=1)

    print('Shape: {}'.format(preds.shape))
    print('First 5 classes: {}'.format(classes[:5]))
    print('First 5 probabilities: {}\n'.format(preds[0, :5]))
    print('Predictions prob/class: ')

    for i in range(len(idxs)):
        idx = idxs[i]
        print ('  {:.4f}/{}'.format(preds[i, idx], classes[idx]))


pred_batch(imgs)
