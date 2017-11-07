
# coding: utf-8

import ast

import pandas as pd

import datetime

from keras.layers import Input, Dense, Embedding, merge, Flatten, Merge, BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import SGD
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth

import utils

import data

from sklearn.model_selection import train_test_split

from bcolz_array_iterator import BcolzArrayIterator

import bcolz

from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import ModelCheckpoint


# Below path is a shared directory, swap to own

data_path = "/data/datasets/taxi/"


# ## Replication of 'csv_to_hdf5.py'

# Original repo used some bizarre tuple method of reading in data to save
# in a hdf5 file using fuel. The following does the same approach in that
# module, only using pandas and saving in a bcolz format (w/ training data
# as example)

meta = pd.read_csv(
    data_path +
    'metaData_taxistandsID_name_GPSlocation.csv',
    header=0)


meta.head()


train = pd.read_csv(data_path + 'train/train.csv', header=0)


train.head()


train['ORIGIN_CALL'] = pd.Series(pd.factorize(train['ORIGIN_CALL'])[0]) + 1


train['ORIGIN_STAND'] = pd.Series(
    [0 if pd.isnull(x) or x == '' else int(x) for x in train["ORIGIN_STAND"]])


train['TAXI_ID'] = pd.Series(pd.factorize(train['TAXI_ID'])[0]) + 1


train['DAY_TYPE'] = pd.Series([ord(x[0]) - ord('A')
                               for x in train['DAY_TYPE']])


# The array of long/lat coordinates per trip (row) is read in as a string.
# The function `ast.literal_eval(x)` evaluates the string into the
# expression it represents (safely). This happens below

polyline = pd.Series([ast.literal_eval(x) for x in train['POLYLINE']])


# Split into latitude/longitude

train['LATITUDE'] = pd.Series(
    [np.array([point[1] for point in poly], dtype=np.float32) for poly in polyline])


train['LONGITUDE'] = pd.Series(
    [np.array([point[0] for point in poly], dtype=np.float32) for poly in polyline])


utils.save_array(data_path + 'train/train.bc', train.as_matrix())


utils.save_array(data_path + 'train/meta_train.bc', meta.as_matrix())


# ## Further Feature Engineering

# After converting 'csv_to_hdf5.py' functionality to pandas, I saved that
# array and then simply constructed the rest of the features as specified
# in the paper using pandas. I didn't bother seeing how the author did it
# as it was extremely obtuse and involved the fuel module.

train = pd.DataFrame(utils.load_array(data_path + 'train/train.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                                                                              'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE'])


train.head()


# The paper discusses how many categorical variables there are per
# category. The following all check out

train['ORIGIN_CALL'].max()


train['ORIGIN_STAND'].max()


train['TAXI_ID'].max()


# Self-explanatory

train['DAY_OF_WEEK'] = pd.Series(
    [datetime.datetime.fromtimestamp(t).weekday() for t in train['TIMESTAMP']])


# Quarter hour of the day, i.e. 1 of the `4*24 = 96` quarter hours of the day

train['QUARTER_HOUR'] = pd.Series([int((datetime.datetime.fromtimestamp(t).hour * 60 + datetime.datetime.fromtimestamp(t).minute) / 15)
                                   for t in train['TIMESTAMP']])


# Self-explanatory

train['WEEK_OF_YEAR'] = pd.Series([datetime.datetime.fromtimestamp(
    t).isocalendar()[1] for t in train['TIMESTAMP']])


# Target coords are the last in the sequence (final position). If there
# are no positions, or only 1, then mark as invalid w/ nan in order to
# drop later

train['TARGET'] = pd.Series([[l[1][0][-1], l[1][1][-1]] if len(l[1][0]) >
                             1 else numpy.nan for l in train[['LONGITUDE', 'LATITUDE']].iterrows()])


# This function creates the continuous inputs, which are the concatened k first and k last coords in a sequence, as discussed in the paper.
#
# If there aren't at least 2* k coords excluding the target, then the k first and k last overlap. In this case the sequence (excluding target) is padded at the end with the last coord in the sequence. The paper mentioned they padded front and back but didn't specify in what manner.
#
# Also marks any invalid w/ na's

def start_stop_inputs(k):
    result = []
    for l in train[['LONGITUDE', 'LATITUDE']].iterrows():
        if len(l[1][0]) < 2 or len(l[1][1]) < 2:
            result.append(numpy.nan)
        elif len(l[1][0][:-1]) >= 2 * k:
            result.append(numpy.concatenate(
                [l[1][0][0:k], l[1][0][-(k + 1):-1], l[1][1][0:k], l[1][1][-(k + 1):-1]]).flatten())
        else:
            l1 = numpy.lib.pad(
                l[1][0][:-1], (0, 20 - len(l[1][0][:-1])), mode='edge')
            l2 = numpy.lib.pad(
                l[1][1][:-1], (0, 20 - len(l[1][1][:-1])), mode='edge')
            result.append(numpy.concatenate(
                [l1[0:k], l1[-k:], l2[0:k], l2[-k:]]).flatten())
    return pd.Series(result)


train['COORD_FEATURES'] = start_stop_inputs(5)


train.shape


train.dropna().shape


# Drop na's

train = train.dropna()


utils.save_array(data_path + 'train/train_features.bc', train.as_matrix())


# ## End to end feature transformation

train = pd.read_csv(data_path + 'train/train.csv', header=0)


test = pd.read_csv(data_path + 'test/test.csv', header=0)


def start_stop_inputs(k, data, test):
    result = []
    for l in data[['LONGITUDE', 'LATITUDE']].iterrows():
        if not test:
            if len(l[1][0]) < 2 or len(l[1][1]) < 2:
                result.append(np.nan)
            elif len(l[1][0][:-1]) >= 2 * k:
                result.append(np.concatenate(
                    [l[1][0][0:k], l[1][0][-(k + 1):-1], l[1][1][0:k], l[1][1][-(k + 1):-1]]).flatten())
            else:
                l1 = np.lib.pad(
                    l[1][0][:-1], (0, 4 * k - len(l[1][0][:-1])), mode='edge')
                l2 = np.lib.pad(
                    l[1][1][:-1], (0, 4 * k - len(l[1][1][:-1])), mode='edge')
                result.append(np.concatenate(
                    [l1[0:k], l1[-k:], l2[0:k], l2[-k:]]).flatten())
        else:
            if len(l[1][0]) < 1 or len(l[1][1]) < 1:
                result.append(np.nan)
            elif len(l[1][0]) >= 2 * k:
                result.append(np.concatenate(
                    [l[1][0][0:k], l[1][0][-k:], l[1][1][0:k], l[1][1][-k:]]).flatten())
            else:
                l1 = np.lib.pad(
                    l[1][0], (0, 4 * k - len(l[1][0])), mode='edge')
                l2 = np.lib.pad(
                    l[1][1], (0, 4 * k - len(l[1][1])), mode='edge')
                result.append(np.concatenate(
                    [l1[0:k], l1[-k:], l2[0:k], l2[-k:]]).flatten())
    return pd.Series(result)


# Pre-calculated below on train set

lat_mean = 41.15731
lat_std = 0.074120656
long_mean = -8.6161413
long_std = 0.057200309


def feature_ext(data, test=False):

    data['ORIGIN_CALL'] = pd.Series(pd.factorize(data['ORIGIN_CALL'])[0]) + 1

    data['ORIGIN_STAND'] = pd.Series(
        [0 if pd.isnull(x) or x == '' else int(x) for x in data["ORIGIN_STAND"]])

    data['TAXI_ID'] = pd.Series(pd.factorize(data['TAXI_ID'])[0]) + 1

    data['DAY_TYPE'] = pd.Series([ord(x[0]) - ord('A')
                                  for x in data['DAY_TYPE']])

    polyline = pd.Series([ast.literal_eval(x) for x in data['POLYLINE']])

    data['LATITUDE'] = pd.Series(
        [np.array([point[1] for point in poly], dtype=np.float32) for poly in polyline])

    data['LONGITUDE'] = pd.Series(
        [np.array([point[0] for point in poly], dtype=np.float32) for poly in polyline])

    if not test:

        data['TARGET'] = pd.Series([[l[1][0][-1], l[1][1][-1]] if len(
            l[1][0]) > 1 else np.nan for l in data[['LONGITUDE', 'LATITUDE']].iterrows()])

    data['LATITUDE'] = pd.Series(
        [(t - lat_mean) / lat_std for t in data['LATITUDE']])

    data['LONGITUDE'] = pd.Series(
        [(t - long_mean) / long_std for t in data['LONGITUDE']])

    data['COORD_FEATURES'] = start_stop_inputs(5, data, test)

    data['DAY_OF_WEEK'] = pd.Series(
        [datetime.datetime.fromtimestamp(t).weekday() for t in data['TIMESTAMP']])

    data['QUARTER_HOUR'] = pd.Series([int((datetime.datetime.fromtimestamp(t).hour * 60 + datetime.datetime.fromtimestamp(t).minute) / 15)
                                      for t in data['TIMESTAMP']])

    data['WEEK_OF_YEAR'] = pd.Series([datetime.datetime.fromtimestamp(
        t).isocalendar()[1] for t in data['TIMESTAMP']])

    data = data.dropna()

    return data


train = feature_ext(train)


test = feature_ext(test, test=True)


test.head()


utils.save_array(data_path + 'train/train_features.bc', train.as_matrix())


utils.save_array(data_path + 'test/test_features.bc', test.as_matrix())


train.head()


# ## MEANSHIFT

# Meanshift clustering as performed in the paper

train = pd.DataFrame(utils.load_array(data_path + 'train/train_features.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                                                                                       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'DAY_OF_WEEK',
                                                                                       'QUARTER_HOUR', "WEEK_OF_YEAR", "TARGET", "COORD_FEATURES"])


# Clustering performed on the targets

y_targ = np.vstack(train["TARGET"].as_matrix())


from sklearn.cluster import MeanShift, estimate_bandwidth


# Can use the commented out code for a estimate of bandwidth, which causes clustering to converge much quicker.
#
# This is not mentioned in the paper but is included in the code. In order to get results similar to the paper's,
# they manually chose the uncommented bandwidth

#bw = estimate_bandwidth(y_targ, quantile=.1, n_samples=1000)
bw = 0.001


# This takes some time

ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(y_targ)


cluster_centers = ms.cluster_centers_


# This is very close to the number of clusters mentioned in the paper

cluster_centers.shape


utils.save_array(data_path + "cluster_centers_bw_001.bc", cluster_centers)


# ## Formatting Features for Bcolz iterator / garbage

train = pd.DataFrame(utils.load_array(data_path + 'train/train_features.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                                                                                       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                                                                                       'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


cluster_centers = utils.load_array(data_path + "cluster_centers_bw_001.bc")


long = np.array([c[0] for c in cluster_centers])
lat = np.array([c[1] for c in cluster_centers])


X_train, X_val = train_test_split(train, test_size=0.2, random_state=42)


def get_features(data):
    return [np.vstack(data['COORD_FEATURES'].as_matrix()), np.vstack(data['ORIGIN_CALL'].as_matrix()),
            np.vstack(
        data['TAXI_ID'].as_matrix()), np.vstack(
        data['ORIGIN_STAND'].as_matrix()),
        np.vstack(
        data['QUARTER_HOUR'].as_matrix()), np.vstack(
        data['DAY_OF_WEEK'].as_matrix()),
        np.vstack(data['WEEK_OF_YEAR'].as_matrix()), np.array(
                [long for i in range(0, data.shape[0])]),
        np.array([lat for i in range(0, data.shape[0])])]


def get_target(data):
    return np.vstack(data["TARGET"].as_matrix())


X_train_features = get_features(X_train)


X_train_target = get_target(X_train)


utils.save_array(
    data_path +
    'train/X_train_features.bc',
    get_features(X_train))


# ## MODEL

# Load training data and cluster centers

train = pd.DataFrame(utils.load_array(data_path + 'train/train_features.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                                                                                       'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                                                                                       'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


# Validation cuts

cuts = [
    1376503200,  # 2013-08-14 18:00
    1380616200,  # 2013-10-01 08:30
    1381167900,  # 2013-10-07 17:45
    1383364800,  # 2013-11-02 04:00
    1387722600  # 2013-12-22 14:30
]


print(datetime.datetime.fromtimestamp(1376503200))


train.shape


val_indices = []
index = 0
for index, row in train.iterrows():
    time = row['TIMESTAMP']
    latitude = row['LATITUDE']
    for ts in cuts:
        if time <= ts and time + 15 * (len(latitude) - 1) >= ts:
            val_indices.append(index)
            break
    index += 1


X_valid = train.iloc[val_indices]


valid.head()


for d in valid['TIMESTAMP']:
    print(datetime.datetime.fromtimestamp(d))


X_train = train.drop(train.index[[val_indices]])


cluster_centers = utils.load_array(
    data_path + "/data/cluster_centers_bw_001.bc")


long = np.array([c[0] for c in cluster_centers])
lat = np.array([c[1] for c in cluster_centers])


utils.save_array(data_path + 'train/X_train.bc', X_train.as_matrix())


utils.save_array(data_path + 'valid/X_val.bc', X_valid.as_matrix())


X_train = pd.DataFrame(utils.load_array(data_path + 'train/X_train.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                                                                                  'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                                                                                  'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


X_val = pd.DataFrame(utils.load_array(data_path + 'valid/X_val.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                                                                              'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE', 'TARGET',
                                                                              'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


# The equirectangular loss function mentioned in the paper.
#
# Note: Very important that y[0] is longitude and y[1] is latitude.
#
# Omitted the radius of the earth constant "R" as it does not affect
# minimization and units were not given in the paper.

def equirectangular_loss(y_true, y_pred):
    deg2rad = 3.141592653589793 / 180
    long_1 = y_true[:, 0] * deg2rad
    long_2 = y_pred[:, 0] * deg2rad
    lat_1 = y_true[:, 1] * deg2rad
    lat_2 = y_pred[:, 1] * deg2rad
    return 6371 * K.sqrt(K.square((long_1 - long_2) * K.cos((lat_1 + lat_2) / 2.))
                         + K.square(lat_1 - lat_2))


def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1,
                          W_regularizer=l2(reg))(inp)


# The following returns a fully-connected model as mentioned in the paper. Takes as input k as defined before, and the cluster centers.
#
# Inputs: Embeddings for each category, concatenated w/ the 4*k continous variable representing the first/last k coords as mentioned above.
#
# Embeddings have no regularization, as it was not mentioned in paper, though are easily equipped to include.
#
# Paper mentions global normalization. Didn't specify exactly how they did that, whether thay did it sequentially or whatnot. I just included a batchnorm layer for the continuous inputs.
#
# After concatenation, 1 hidden layer of 500 neurons as called for in paper.
#
# Finally, output layer has as many outputs as there are cluster centers, w/ a softmax activation. Call this output P.
#
# The prediction is the weighted sum of each cluster center c_i w/ corresponding predicted prob P_i.
#
# To facilitate this, dotted output w/ cluster latitudes and longitudes separately. (this happens at variable y), then concatenated
#     into single tensor.
#
# NOTE!!: You will see that I have the cluster center coords as inputs.
# Ideally, This function should store the cluster longs/lats as a constant
# to be used in the model, but I could not figure out. As a consequence, I
# pass them in as a repeated input.

def taxi_mlp(k, cluster_centers):
    shp = cluster_centers.shape[0]
    nums = Input(shape=(4 * k,))

    center_longs = Input(shape=(shp,))
    center_lats = Input(shape=(shp,))

    emb_names = [
        'client_ID',
        'taxi_ID',
        "stand_ID",
        "quarter_hour",
        "day_of_week",
        "week_of_year"]
    emb_ins = [57106, 448, 64, 96, 7, 52]
    emb_outs = [10 for i in range(0, 6)]
    regs = [0 for i in range(0, 6)]

    embs = [
        embedding_input(
            e[0],
            e[1] + 1,
            e[2],
            e[3]) for e in zip(
            emb_names,
            emb_ins,
            emb_outs,
            regs)]

    x = merge([nums] + [Flatten()(e[1]) for e in embs], mode='concat')

    x = Dense(500, activation='relu')(x)

    x = Dense(shp, activation='softmax')(x)

    y = merge([merge([x, center_longs], mode='dot'), merge(
        [x, center_lats], mode='dot')], mode='concat')

    return Model(input=[nums] + [e[0] for e in embs] +
                 [center_longs, center_lats], output=y)


# As mentioned, construction of repeated cluster longs/lats for input

# Iterator for in memory `train` pandas dataframe. I did this as opposed
# to bcolz iterator due to the pre-processing

def data_iter(data, batch_size, cluster_centers):
    long = [c[0] for c in cluster_centers]
    lat = [c[1] for c in cluster_centers]
    i = 0
    N = data.shape[0]
    while True:
        yield ([np.vstack(data['COORD_FEATURES'][i:i + batch_size].as_matrix()), np.vstack(data['ORIGIN_CALL'][i:i + batch_size].as_matrix()),
                np.vstack(data['TAXI_ID'][i:i + batch_size].as_matrix()
                          ), np.vstack(data['ORIGIN_STAND'][i:i + batch_size].as_matrix()),
                np.vstack(data['QUARTER_HOUR'][i:i + batch_size].as_matrix()
                          ), np.vstack(data['DAY_OF_WEEK'][i:i + batch_size].as_matrix()),
                np.vstack(data['WEEK_OF_YEAR'][i:i + batch_size].as_matrix()
                          ), np.array([long for i in range(0, batch_size)]),
                np.array([lat for i in range(0, batch_size)])], np.vstack(data["TARGET"][i:i + batch_size].as_matrix()))
        i += batch_size


x = Lambda(thing)([x, long, lat])


# Of course, k in the model needs to match k from feature construction. We
# again use 5 as they did in the paper

model = taxi_mlp(5, cluster_centers)


# Paper used SGD opt w/ following paramerters

model.compile(
    optimizer=SGD(
        0.01,
        momentum=0.9),
    loss=equirectangular_loss,
    metrics=['mse'])


X_train_feat = get_features(X_train)


X_train_target = get_target(X_train)


X_val_feat = get_features(X_valid)


X_val_target = get_target(X_valid)


tqdm = TQDMNotebookCallback()


checkpoint = ModelCheckpoint(
    filepath=data_path +
    'models/tmp/weights.{epoch:03d}.{val_loss:.8f}.hdf5',
    save_best_only=True)


batch_size = 256


# ### original

model.fit(
    X_train_feat,
    X_train_target,
    nb_epoch=1,
    batch_size=batch_size,
    validation_data=(
        X_val_feat,
        X_val_target),
    callbacks=[
        tqdm,
        checkpoint],
    verbose=0)


model.fit(
    X_train_feat,
    X_train_target,
    nb_epoch=30,
    batch_size=batch_size,
    validation_data=(
        X_val_feat,
        X_val_target),
    callbacks=[
        tqdm,
        checkpoint],
    verbose=0)


model = load_model(
    data_path +
    'models/weights.0.0799.hdf5',
    custom_objects={
        'equirectangular_loss': equirectangular_loss})


model.fit(
    X_train_feat,
    X_train_target,
    nb_epoch=100,
    batch_size=batch_size,
    validation_data=(
        X_val_feat,
        X_val_target),
    callbacks=[
        tqdm,
        checkpoint],
    verbose=0)


model.save(data_path + 'models/current_model.hdf5')


# ### new valid

model.fit(
    X_train_feat,
    X_train_target,
    nb_epoch=1,
    batch_size=batch_size,
    validation_data=(
        X_val_feat,
        X_val_target),
    callbacks=[
        tqdm,
        checkpoint],
    verbose=0)


model.fit(
    X_train_feat,
    X_train_target,
    nb_epoch=400,
    batch_size=batch_size,
    validation_data=(
        X_val_feat,
        X_val_target),
    callbacks=[
        tqdm,
        checkpoint],
    verbose=0)


model.save(data_path + '/models/current_model.hdf5')


len(X_val_feat[0])


# It works, but it seems to converge unrealistically quick and the loss
# values are not the same. The paper does not mention what it's using as
# "error" in it's results. I assume the same equirectangular? Not very
# clear. The difference in values could be due to the missing Earth-radius
# factor

# ## Kaggle Entry

best_model = load_model(
    data_path +
    'models/weights.308.0.03373993.hdf5',
    custom_objects={
        'equirectangular_loss': equirectangular_loss})


best_model.evaluate(X_val_feat, X_val_target)


test = pd.DataFrame(utils.load_array(data_path + 'test/test_features.bc'), columns=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                                                                                    'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE', 'LATITUDE', 'LONGITUDE',
                                                                                    'COORD_FEATURES', "DAY_OF_WEEK", "QUARTER_HOUR", "WEEK_OF_YEAR"])


test['ORIGIN_CALL'] = pd.read_csv(
    data_path +
    'real_origin_call.csv',
    header=None)


test['TAXI_ID'] = pd.read_csv(data_path + 'real_taxi_id.csv', header=None)


X_test = get_features(test)


b = np.sort(X_test[1], axis=None)


test_preds = np.round(best_model.predict(X_test), decimals=6)


d = {0: test['TRIP_ID'], 1: test_preds[:, 1], 2: test_preds[:, 0]}
kaggle_out = pd.DataFrame(data=d)


kaggle_out.to_csv(
    data_path +
    'submission.csv',
    header=[
        'TRIP_ID',
        'LATITUDE',
        'LONGITUDE'],
    index=False)


def hdist(a, b):
    deg2rad = 3.141592653589793 / 180

    lat1 = a[:, 1] * deg2rad
    lon1 = a[:, 0] * deg2rad
    lat2 = b[:, 1] * deg2rad
    lon2 = b[:, 0] * deg2rad

    dlat = abs(lat1 - lat2)
    dlon = abs(lon1 - lon2)

    al = np.sin(dlat / 2)**2 + np.cos(lat1) * \
        np.cos(lat2) * (np.sin(dlon / 2)**2)
    d = np.arctan2(np.sqrt(al), np.sqrt(1 - al))

    hd = 2 * 6371 * d

    return hd


val_preds = best_model.predict(X_val_feat)


trn_preds = model.predict(X_train_feat)


er = hdist(val_preds, X_val_target)


er.mean()


K.equal()


# To-do: simple to extend to validation data

# ## Uh oh... training data not representative of test

cuts = [
    1376503200,  # 2013-08-14 18:00
    1380616200,  # 2013-10-01 08:30
    1381167900,  # 2013-10-07 17:45
    1383364800,  # 2013-11-02 04:00
    1387722600  # 2013-12-22 14:30
]


np.any([train['TIMESTAMP'].map(lambda x: x in cuts)])


train['TIMESTAMP']


np.any(train['TIMESTAMP'] == 1381167900)


times = train['TIMESTAMP'].as_matrix()


X_train.columns


times


count = 0
for index, row in X_val.iterrows():
    for ts in cuts:
        time = row['TIMESTAMP']
        latitude = row['LATITUDE']
        if time <= ts and time + 15 * (len(latitude) - 1) >= ts:
            count += 1


one = count


count + one


import h5py


h = h5py.File(data_path + 'original/data.hdf5', 'r')


evrData = h['/Configure:0000/Run:0000/CalibCycle:0000/EvrData::DataV3/NoDetector.0:Evr.0/data']


c = np.load(data_path + 'original/arrival-clusters.pkl')


# ### hd5f files

from fuel.utils import find_in_data_path
from fuel.datasets import H5PYDataset


original_path = '/data/bckenstler/data/taxi/original/'


train_set = H5PYDataset(
    original_path +
    'data.hdf5',
    which_sets=(
        'train',
    ),
    load_in_memory=True)


valid_set = H5PYDataset(
    original_path +
    'valid.hdf5',
    which_sets=(
        'cuts/test_times_0',
    ),
    load_in_memory=True)


print(train_set.num_examples)


print(valid_set.num_examples)


data = train_set.data_sources


data[0]


valid_data = valid_set.data_sources


valid_data[4][0]


stamps = valid_data[-3]


stamps[0]


for i in range(0, 304):
    print(np.any([t == int(stamps[i]) for t in X_val['TIMESTAMP']]))


type(X_train['TIMESTAMP'][0])


type(stamps[0])


check = [s in stamps for s in X_val['TIMESTAMP']]


for s in X_val['TIMESTAMP']:
    print(datetime.datetime.fromtimestamp(s))


for s in stamps:
    print(datetime.datetime.fromtimestamp(s))


ids = valid_data[-1]


type(ids[0])


ids


X_val
