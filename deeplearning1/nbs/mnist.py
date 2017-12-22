
# coding: utf-8

from theano.sandbox import cuda
cuda.use('gpu2')


import utils
reload(utils)
from utils import *


# ## Setup

batch_size = 64


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


X_test = np.expand_dims(X_test, 1)
X_train = np.expand_dims(X_train, 1)


X_train.shape


y_train[:5]


y_train = onehot(y_train)
y_test = onehot(y_test)


y_train[:5]


mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)


def norm_input(x): return (x - mean_px) / std_px


# ## Linear model

def get_lin_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


lm = get_lin_model()


gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=64)
test_batches = gen.flow(X_test, y_test, batch_size=64)


lm.fit_generator(batches, batches.N, nb_epoch=1,
                 validation_data=test_batches, nb_val_samples=test_batches.N)


lm.optimizer.lr = 0.1


lm.fit_generator(batches, batches.N, nb_epoch=1,
                 validation_data=test_batches, nb_val_samples=test_batches.N)


lm.optimizer.lr = 0.01


lm.fit_generator(batches, batches.N, nb_epoch=4,
                 validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Single dense layer

def get_fc_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Flatten(),
        Dense(512, activation='softmax'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


fc = get_fc_model()


fc.fit_generator(batches, batches.N, nb_epoch=1,
                 validation_data=test_batches, nb_val_samples=test_batches.N)


fc.optimizer.lr = 0.1


fc.fit_generator(batches, batches.N, nb_epoch=4,
                 validation_data=test_batches, nb_val_samples=test_batches.N)


fc.optimizer.lr = 0.01


fc.fit_generator(batches, batches.N, nb_epoch=4,
                 validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Basic 'VGG-style' CNN

def get_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Convolution2D(32, 3, 3, activation='relu'),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),
        Convolution2D(64, 3, 3, activation='relu'),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


model = get_model()


model.fit_generator(batches, batches.N, nb_epoch=1,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.1


model.fit_generator(batches, batches.N, nb_epoch=1,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.01


model.fit_generator(batches, batches.N, nb_epoch=8,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Data augmentation

model = get_model()


gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=64)
test_batches = gen.flow(X_test, y_test, batch_size=64)


model.fit_generator(batches, batches.N, nb_epoch=1,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.1


model.fit_generator(batches, batches.N, nb_epoch=4,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.01


model.fit_generator(batches, batches.N, nb_epoch=8,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.001


model.fit_generator(batches, batches.N, nb_epoch=14,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.0001


model.fit_generator(batches, batches.N, nb_epoch=10,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Batchnorm + data augmentation

def get_model_bn():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


model = get_model_bn()


model.fit_generator(batches, batches.N, nb_epoch=1,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.1


model.fit_generator(batches, batches.N, nb_epoch=4,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.01


model.fit_generator(batches, batches.N, nb_epoch=12,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.001


model.fit_generator(batches, batches.N, nb_epoch=12,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Batchnorm + dropout + data augmentation

def get_model_bn_do():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(
        Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


model = get_model_bn_do()


model.fit_generator(batches, batches.N, nb_epoch=1,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.1


model.fit_generator(batches, batches.N, nb_epoch=4,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.01


model.fit_generator(batches, batches.N, nb_epoch=12,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


model.optimizer.lr = 0.001


model.fit_generator(batches, batches.N, nb_epoch=1,
                    validation_data=test_batches, nb_val_samples=test_batches.N)


# ## Ensembling

def fit_model():
    model = get_model_bn_do()
    model.fit_generator(batches, batches.N, nb_epoch=1, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr = 0.1
    model.fit_generator(batches, batches.N, nb_epoch=4, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr = 0.01
    model.fit_generator(batches, batches.N, nb_epoch=12, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr = 0.001
    model.fit_generator(batches, batches.N, nb_epoch=18, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    return model


models = [fit_model() for i in range(6)]


path = "data/mnist/"
model_path = path + 'models/'


for i, m in enumerate(models):
    m.save_weights(model_path + 'cnn-mnist23-' + str(i) + '.pkl')


evals = np.array([m.evaluate(X_test, y_test, batch_size=256) for m in models])


evals.mean(axis=0)


all_preds = np.stack([m.predict(X_test, batch_size=256) for m in models])


all_preds.shape


avg_preds = all_preds.mean(axis=0)


keras.metrics.categorical_accuracy(y_test, avg_preds).eval()
