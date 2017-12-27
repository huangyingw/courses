from IPython.display import FileLink
from IPython.lib.display import FileLink
from PIL import Image
from glob import glob
from keras.layers.convolutional import *
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from utils import *
from vgg16 import *
from vgg16 import Vgg16
from vgg16bn import *
import numpy as np
import os
import shutil

# # Dogs vs Cat Redux

# In this tutorial, you will learn how generate and submit predictions to a Kaggle competiton
#
# [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
#
#

# To start you will need to download and unzip the competition data from Kaggle and ensure your directory structure looks like this
# ```
# utils/
#     vgg16.py
#     utils.py
# lesson1/
#     redux.ipynb
#     data/
#         redux/
#             train/
#                 cat.437.jpg
#                 dog.9924.jpg
#                 cat.1029.jpg
#                 dog.4374.jpg
#             test/
#                 231.jpg
#                 325.jpg
#                 1235.jpg
#                 9923.jpg
# ```
#
# You can download the data files from the competition page [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) or you can download them from the command line using the [Kaggle CLI](https://github.com/floydwch/kaggle-cli).
#
# You should launch your notebook inside the lesson1 directory
# ```
# cd lesson1
# jupyter notebook
# ```

# Create references to important directories we will use over and over
current_dir = os.getcwd()
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir + '/data/redux'


# ## Create validation set and sample

# Create directories
os.chdir(DATA_HOME_DIR)
makedirs('valid')
makedirs('results')
makedirs('sample/train')
makedirs('sample/test')
makedirs('sample/valid')
makedirs('sample/results')
makedirs('test/unknown')

os.chdir(DATA_HOME_DIR + '/train')

'''
g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(2000):
    os.rename(shuf[i], DATA_HOME_DIR + '/valid/' + shuf[i])

g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(200):
    copyfile(shuf[i], DATA_HOME_DIR + '/sample/train/' + shuf[i])

os.chdir(DATA_HOME_DIR + '/valid')

g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(50):
    copyfile(shuf[i], DATA_HOME_DIR + '/sample/valid/' + shuf[i])
'''

# ## Rearrange image files into their respective directories

# Divide cat/dog images into separate directories

def makedirs_dogs_cat(path):
    os.chdir(path)
    makedirs('cats')
    makedirs('dogs')
    for file in glob('cat.*.jpg'):
        shutil.move(file, 'cats/')
    for file in glob('dog.*.jpg'):
        shutil.move(file, 'dogs/')

makedirs_dogs_cat(DATA_HOME_DIR + '/sample/train')
makedirs_dogs_cat(DATA_HOME_DIR + '/sample/valid')
makedirs_dogs_cat(DATA_HOME_DIR + '/valid')
makedirs_dogs_cat(DATA_HOME_DIR + '/train')
makedirs_dogs_cat(DATA_HOME_DIR + '/sample/train')
makedirs_dogs_cat(DATA_HOME_DIR + '/sample/train')

# Create single 'unknown' class for test set
os.chdir(DATA_HOME_DIR + '/test')
for file in glob('*.jpg'):
    shutil.move(file, 'unknown/')


# ## Finetuning and Training

os.chdir(DATA_HOME_DIR)

# Set path to sample/ path if desired
path = DATA_HOME_DIR + '/'  # '/sample/'
test_path = DATA_HOME_DIR + '/test/'  # We use all the test data
results_path = DATA_HOME_DIR + '/results/'
train_path = path + '/train/'
valid_path = path + '/valid/'


vgg = Vgg16()


# Set constants. You can experiment with no_of_epochs to improve the model
batch_size = 64
no_of_epochs = 3


# Finetune the model
batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size * 2)
'''
vgg.finetune(batches)

# Not sure if we set this for all fits
vgg.model.optimizer.lr = 0.01


# Notice we are passing in the validation dataset to the fit() method
# For each epoch we test our model against the validation set
latest_weights_filename = None
for epoch in range(no_of_epochs):
    print "Running epoch: %d" % epoch
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = 'ft%d.h5' % epoch
    vgg.model.save_weights(results_path + latest_weights_filename)
print "Completed %s fit operations" % no_of_epochs
'''
latest_weights_filename = 'ft%d.h5' % (no_of_epochs - 1)
vgg.model.load_weights(results_path + latest_weights_filename)


# ## Generate Predictions

# Let's use our new model to make predictions on the test dataset

batches, preds = vgg.test(test_path, batch_size=batch_size * 2)


# For every image, vgg.test() generates two probabilities
# based on how we've ordered the cats/dogs directories.
# It looks like column one is cats and column two is dogs
print preds[:5]

filenames = batches.filenames
print filenames[:5]


# You can verify the column ordering by viewing some images
Image.open(test_path + filenames[2])


# Save our test results arrays so we can use them again later
save_array(results_path + 'test_preds.dat', preds)
save_array(results_path + 'filenames.dat', filenames)


# ## Validate Predictions

# Keras' *fit()* function conveniently shows us the value of the loss function, and the accuracy, after every epoch ("*epoch*" refers to one full run through all training examples). The most important metrics for us to look at are for the validation set, since we want to check for over-fitting.
#
# - **Tip**: with our first model we should try to overfit before we start worrying about how to reduce over-fitting - there's no point even thinking about regularization, data augmentation, etc if you're still under-fitting! (We'll be looking at these techniques shortly).
#
# As well as looking at the overall metrics, it's also a good idea to look at examples of each of:
# 1. A few correct labels at random
# 2. A few incorrect labels at random
# 3. The most correct labels of each class (ie those with highest probability that are correct)
# 4. The most incorrect labels of each class (ie those with highest probability that are incorrect)
# 5. The most uncertain labels (ie those with probability closest to 0.5).

# Let's see what we can learn from these examples. (In general, this is a particularly useful technique for debugging problems in the model. However, since this model is so simple, there may not be too much to learn at this stage.)
#
# Calculate predictions on validation set, so we can find correct and
# incorrect examples:


val_batches, probs = vgg.test(valid_path, batch_size=batch_size)


filenames = val_batches.filenames
expected_labels = val_batches.classes  # 0 or 1

# Round our predictions to 0/1 to generate labels
our_predictions = probs[:, 0]
our_labels = np.round(1 - our_predictions)


# Helper function to plot images by index in the validation set
# Plots is a helper function in utils.py


def plots_idx(idx, titles=None):
    plots([image.load_img(valid_path + filenames[i])
           for i in idx], titles=titles)


# Number of images to view for each visualization task
n_view = 4


'''
# 2. A few incorrect labels at random
incorrect = np.where(our_labels != expected_labels)[0]
print "Found %d incorrect labels" % len(incorrect)
idx = permutation(incorrect)[:n_view]
plots_idx(idx, our_predictions[idx])
'''

# 4a. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where(
    (our_labels == 0) & (
        our_labels != expected_labels))[0]
print "Found %d incorrect cats" % len(incorrect_cats)
if len(incorrect_cats):
    most_incorrect_cats = np.argsort(
        our_predictions[incorrect_cats])[::-1][:n_view]
    plots_idx(
        incorrect_cats[most_incorrect_cats],
        our_predictions[incorrect_cats][most_incorrect_cats])


# 4b. The images we were most confident were dogs, but are actually cats
incorrect_dogs = np.where(
    (our_labels == 1) & (
        our_labels != expected_labels))[0]
print "Found %d incorrect dogs" % len(incorrect_dogs)
if len(incorrect_dogs):
    most_incorrect_dogs = np.argsort(our_predictions[incorrect_dogs])[:n_view]
    plots_idx(
        incorrect_dogs[most_incorrect_dogs],
        our_predictions[incorrect_dogs][most_incorrect_dogs])


# 5. The most uncertain labels (ie those with probability closest to 0.5).
most_uncertain = np.argsort(np.abs(our_predictions - 0.5))
plots_idx(most_uncertain[:n_view], our_predictions[most_uncertain])

# Perhaps the most common way to analyze the result of a classification
# model is to use a [confusion
# matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).
# Scikit-learn has a convenient function we can use for this purpose:

cm = confusion_matrix(expected_labels, our_labels)


# We can just print out the confusion matrix, or we can show a graphical
# view (which is mainly useful for dependents with a larger number of
# categories).

plot_confusion_matrix(cm, val_batches.class_indices)


# ## Submit Predictions to Kaggle!

# Here's the format Kaggle requires for new submissions:
# ```
# imageId,isDog
# 1242, .3984
# 3947, .1000
# 4539, .9082
# 2345, .0000
# ```
#
# Kaggle wants the imageId followed by the probability of the image being
# a dog. Kaggle uses a metric called [Log
# Loss](http://wiki.fast.ai/index.php/Log_Loss) to evaluate your
# submission.

# Load our test predictions from file
preds = load_array(results_path + 'test_preds.dat')
filenames = load_array(results_path + 'filenames.dat')


# Grab the dog prediction column
isdog = preds[:, 1]
print "Raw Predictions: " + str(isdog[:5])
print "Mid Predictions: " + str(isdog[(isdog < .6) & (isdog > .4)])
print "Edge Predictions: " + str(isdog[(isdog == 1) | (isdog == 0)])


# So to play it safe, we use a sneaky trick to round down our edge predictions
# Swap all ones with .95 and all zeros with .05
isdog = isdog.clip(min=0.05, max=0.95)


# Extract imageIds from the filenames in our test/unknown directory
filenames = batches.filenames
ids = np.array([int(f[8:f.find('.')]) for f in filenames])


# Here we join the two columns into an array of [imageId, isDog]

subm = np.stack([ids, isdog], axis=1)
subm[:5]


os.chdir(DATA_HOME_DIR)
submission_file_name = 'submission1.csv'
np.savetxt(
    submission_file_name,
    subm,
    fmt='%d,%.5f',
    header='id,label',
    comments='')


os.chdir(LESSON_HOME_DIR)
FileLink('data/redux/' + submission_file_name)


# You can download this file and submit on the Kaggle website or use the
# Kaggle command line tool's "submit" method.
