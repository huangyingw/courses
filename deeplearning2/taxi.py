
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

import pickle


# Data Path

data_path = "/data/datasets/taxi/"


# #  CSV 2 DATA

# Need to check process and see if resulting data is the same

# ## Original data

# This is the author's feature extraction process. I have modified it so
# that it returns the tuple instead of saving as hdf5 file

import ast
import csv
import os
import sys

import h5py
import numpy
from fuel.converters.base import fill_hdf5_file

import data


taxi_id_dict = {}
origin_call_dict = {0: 0}


def get_unique_taxi_id(val):
    if val in taxi_id_dict:
        return taxi_id_dict[val]
    else:
        taxi_id_dict[val] = len(taxi_id_dict)
        return len(taxi_id_dict) - 1


def get_unique_origin_call(val):
    if val in origin_call_dict:
        return origin_call_dict[val]
    else:
        origin_call_dict[val] = len(origin_call_dict)
        return len(origin_call_dict) - 1


def read_stands(input_directory, h5file):
    stands_name = numpy.empty(shape=(data.stands_size,), dtype=('a', 24))
    stands_latitude = numpy.empty(
        shape=(data.stands_size,), dtype=numpy.float32)
    stands_longitude = numpy.empty(
        shape=(data.stands_size,), dtype=numpy.float32)
    stands_name[0] = 'None'
    stands_latitude[0] = stands_longitude[0] = 0
    with open(os.path.join(input_directory, 'metaData_taxistandsID_name_GPSlocation.csv'), 'r') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for line in reader:
            id = int(line[0])
            stands_name[id] = line[1].encode('utf-8')
            stands_latitude[id] = float(line[2])
            stands_longitude[id] = float(line[3])
    return (('stands', 'stands_name', stands_name),
            ('stands', 'stands_latitude', stands_latitude),
            ('stands', 'stands_longitude', stands_longitude))


def read_taxis(input_directory, h5file, dataset):
    size = getattr(data, '%s_size' % dataset)
    trip_id = numpy.empty(shape=(size,), dtype='S19')
    call_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    origin_call = numpy.empty(shape=(size,), dtype=numpy.int32)
    origin_stand = numpy.empty(shape=(size,), dtype=numpy.int8)
    taxi_id = numpy.empty(shape=(size,), dtype=numpy.int16)
    timestamp = numpy.empty(shape=(size,), dtype=numpy.int32)
    day_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    missing_data = numpy.empty(shape=(size,), dtype=numpy.bool)
    latitude = numpy.empty(shape=(size,), dtype=data.Polyline)
    longitude = numpy.empty(shape=(size,), dtype=data.Polyline)
    with open(os.path.join(input_directory, '%s.csv' % dataset), 'r') as f:
        reader = csv.reader(f)
        next(reader)  # header
        id = 0
        for line in reader:
            trip_id[id] = line[0].encode('utf-8')
            call_type[id] = ord(line[1][0]) - ord('A')
            origin_call[id] = 0 if line[2] == 'NA' or line[2] == '' else get_unique_origin_call(
                int(line[2]))
            origin_stand[id] = 0 if line[3] == 'NA' or line[3] == '' else int(
                line[3])
            taxi_id[id] = get_unique_taxi_id(int(line[4]))
            timestamp[id] = int(line[5])
            day_type[id] = ord(line[6][0]) - ord('A')
            missing_data[id] = line[7][0] == 'T'
            polyline = ast.literal_eval(line[8])
            latitude[id] = numpy.array(
                [point[1] for point in polyline], dtype=numpy.float32)
            longitude[id] = numpy.array(
                [point[0] for point in polyline], dtype=numpy.float32)
            id += 1
    splits = ()
    for name in ['trip_id', 'call_type', 'origin_call', 'origin_stand', 'taxi_id',
                 'timestamp', 'day_type', 'missing_data', 'latitude', 'longitude']:
        splits += ((dataset, name, locals()[name]),)
    return splits


def unique(h5file):
    unique_taxi_id = numpy.empty(shape=(data.taxi_id_size,), dtype=numpy.int32)
    assert len(taxi_id_dict) == data.taxi_id_size
    for k, v in taxi_id_dict.items():
        unique_taxi_id[v] = k

    unique_origin_call = numpy.empty(
        shape=(data.origin_call_size,), dtype=numpy.int32)
    assert len(origin_call_dict) == data.origin_call_size
    for k, v in origin_call_dict.items():
        unique_origin_call[v] = k

    return (('unique_taxi_id', 'unique_taxi_id', unique_taxi_id),
            ('unique_origin_call', 'unique_origin_call', unique_origin_call))


def get_data(input_directory):
    split = ()
    split += read_stands(input_directory, None)
    split += read_taxis(input_directory, None, 'train')
    split += read_taxis(input_directory, None, 'test')
    split += unique(None)
    return split


# ### Check

# manually go through data collection

taxi_id_dict = {}
origin_call_dict = {0: 0}


split = ()


split += read_stands(data_path + 'data', None)


split += read_taxis(data_path + 'data', None, 'train')


split += read_taxis(data_path + 'data', None, 'test')


split += unique(None)


# Data structure: Tuple of tuples. Each sub-tuple: ('dataset', 'column' 'data')

# Contains stands: metadata.

# Save tulpe

with open(data_path + '/data/data_tuple.pickle', 'wb') as f:
    pickle.dump(split, f)


# Load tuple

with open(data_path + '/data/data_tuple.pickle', 'r') as f:
    split = pickle.load(f)


# ### Validation Split

# Time cuts

cuts = [
    1376503200,  # 2013-08-14 18:00
    1380616200,  # 2013-10-01 08:30
    1381167900,  # 2013-10-07 17:45
    1383364800,  # 2013-11-02 04:00
    1387722600  # 2013-12-22 14:30
]


split[12][2][0]


split


split[7][1]


def make_valid(split):

    valid = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )

    for i in range(len(split[5][2])):
        trip_id = split[3][2][i]
        call_type = split[4][2][i]
        origin_call = split[5][2][i]
        origin_stand = split[6][2][i]
        taxi_id = split[7][2][i]
        time = split[8][2][i]
        day_type = split[9][2][i]
        missing_data = split[10][2][i]
        latitude = split[11][2][i]
        longitude = split[12][2][i]

        if len(latitude) == 0:
            continue

        for ts in cuts:
            if time <= ts and time + 15 * (len(latitude) - 1) >= ts:
                # keep it
                valid[0].append(trip_id)
                valid[1].append(call_type)
                valid[2].append(origin_call)
                valid[3].append(origin_stand)
                valid[4].append(taxi_id)
                valid[5].append(time)
                valid[6].append(day_type)
                valid[7].append(missing_data)
                n = (ts - time) / 15 + 1
                valid[8].append(latitude[:n])
                valid[9].append(longitude[:n])
                valid[10].append(latitude[-1])
                valid[11].append(longitude[-1])
                valid[12].append(15 * (len(latitude) - 1))
                break
    return (
        ('valid', 'trip_id', np.array(valid[0])),
        ('valid', 'call_type', np.array(valid[1])),
        ('valid', 'origin_call', np.array(valid[2])),
        ('valid', 'origin_stand', np.array(valid[3])),
        ('valid', 'taxi_id', np.array(valid[4])),
        ('valid', 'timestamp', np.array(valid[5])),
        ('valid', 'day_type', np.array(valid[6])),
        ('valid', 'missing_data', np.array(valid[7])),
        ('valid', 'latitude', np.array(valid[8])),
        ('valid', 'longitude', np.array(valid[9])),
        ('valid', 'destination_latitude', np.array(valid[10])),
        ('valid', 'destination_longitude', np.array(valid[11])),
        ('valid', 'travel_time', np.array(valid[12]))
    )


valid_split = make_valid(split)


valid_split


# ## Pandas Data

# ### Meta-data

stands = pd.read_csv(
    data_path +
    '/data/metaData_taxistandsID_name_GPSlocation.csv',
    header=0)


stands.head()


# Compare columns

stands['Descricao']


len(stands['Descricao'].as_matrix())


split[2][2]


len(split[0][2])


# Author's data has a "None" row w/ zeros. Will add to series

d = {'col1': 1, 'col2': 2}


stands = pd.DataFrame([['None', 0., 0.]], columns=[
                      'Descricao', 'Latitude', 'Longitude']).append(stands)


np.allclose(stands['Latitude'], split[1][2])


np.allclose(stands['Longitude'], split[2][2])


# Longs/Lats check out

# ### Train Data

data = pd.read_csv(data_path + 'data/train.csv', header=0)


data.columns


# Check data same size

data.shape


len(split[4][2])


split[4][1]


# #### CAll type

def call_type_f(x): return ord(x) - ord('A')


np.allclose(data['CALL_TYPE'].apply(call_type_f), split[4][2])


data['CALL_TYPE'] = data['CALL_TYPE'].apply(call_type_f)


# #### Origin Call

# Turn origin call into categorical variable

# Can do using factorize: we want the nulls to be zero. Factorize sets
# them as -1, add 1 to set as needed

np.allclose(pd.Series(pd.factorize(data['ORIGIN_CALL'])[0]) + 1, split[5][2])


# And it is the same

data['ORIGIN_CALL'] = pd.Series(pd.factorize(data['ORIGIN_CALL'])[0]) + 1


num_origin_call = len(data['ORIGIN_CALL'].unique())


s = set()


s.i


# #### Origin Stand

def origin_stand_f(x): return 0 if pd.isnull(x) or x == '' else int(x)


split[6]


np.allclose(data["ORIGIN_STAND"].apply(origin_stand_f), split[6][2])


data["ORIGIN_STAND"] = data["ORIGIN_STAND"].apply(origin_stand_f)


num_origin_stand = len(data['ORIGIN_STAND'].unique())


num_origin_stand


# #### Taxi ID

split[7][2]


pd.factorize(data['TAXI_ID'])[0]


np.allclose(pd.Series(pd.factorize(data['TAXI_ID'])[0]), split[7][2])


data['TAXI_ID'] = pd.Series(pd.factorize(data['TAXI_ID'])[0])


num_taxi_id = data['TAXI_ID'].unique()


# #### Day Type

split[9]


def day_type_f(x): return ord(x[0]) - ord('A')


np.allclose(data['DAY_TYPE'].apply(day_type_f), split[9][2])


data['DAY_TYPE'] = data['DAY_TYPE'].apply(day_type_f)


# #### Long/lat

def polyline_f(x): return ast.literal_eval(x)


polyline = data['POLYLINE'].apply(polyline_f)


polyline.to_pickle(data_path + 'data/polylines.pkl')


polyline = pd.read_pickle(data_path + 'data/polylines.pkl')


len(polyline)


polyline


lats = pd.Series([np.array([point[1] for point in poly],
                           dtype=np.float32) for poly in polyline])


split[11][2]


np.alltrue([np.allclose(lats[i], split[11][2][i]) for i in range(len(lats))])


# Latitudes check out

longs = pd.Series([np.array([point[0] for point in poly],
                            dtype=np.float32) for poly in polyline])


split[12]


np.alltrue([np.allclose(longs[i], split[12][2][i]) for i in range(len(longs))])


# Longitudes check out

data['LATITUDE'] = lats


data['LONGITUDE'] = longs


data


# SAVE DICTS

np.save(data_path + 'data/origin_call_dict.npy', origin_call_dict)


np.save(data_path + 'data/taxi_id_dict.npy', taxi_id_dict)


# ### Test DATA

test_data = pd.read_csv(data_path + 'data/test.csv', header=0)


test_data.columns


# Check data same size

test_data.shape


len(split[13][2])


# #### CAll type

def call_type_f(x): return ord(x) - ord('A')


np.allclose(test_data['CALL_TYPE'].apply(call_type_f), split[14][2])


test_data['CALL_TYPE'] = test_data['CALL_TYPE'].apply(call_type_f)


# #### Origin Call

# Turn origin call into categorical variable

# Can do using factorize: we want the nulls to be zero. Factorize sets
# them as -1, add 1 to set as needed

np.unique(split[15][2])


# Hold up! We need to use our previous mapping

import numpy as np


taxi_id_dict = np.load(data_path + 'data/taxi_id_dict.npy').item()


len(origin_call_dict)


def test_origin_call_f(x): return 0 if (
    np.isnan(x) or x == '' or x >= num_origin_call) else origin_call_dict[x]


np.allclose(test_data['ORIGIN_CALL'].apply(test_origin_call_f), split[15][2])


test_data['ORIGIN_CALL'] = test_data['ORIGIN_CALL'].apply(test_origin_call_f)


# #### Origin Stand

def origin_stand_f(x): return 0 if pd.isnull(x) or x == '' else int(x)


split[16]


np.allclose(test_data["ORIGIN_STAND"].apply(origin_stand_f), split[16][2])


test_data["ORIGIN_STAND"] = test_data["ORIGIN_STAND"].apply(origin_stand_f)


# #### Taxi ID

split[17][2]


def test_taxi_id_f(x): return taxi_id_dict[x]


np.allclose(test_data['TAXI_ID'].apply(test_taxi_id_f), split[17][2])


test_data['TAXI_ID'] = test_data['TAXI_ID'].apply(test_taxi_id_f)


test_data['ORIGIN_CALL'].unique()


test_data['TAXI_ID'].as_matrix()


# #### Day Type

split[19]


def day_type_f(x): return ord(x[0]) - ord('A')


np.allclose(test_data['DAY_TYPE'].apply(day_type_f), split[19][2])


test_data['DAY_TYPE'] = test_data['DAY_TYPE'].apply(day_type_f)


# #### Long/lat

def polyline_f(x): return ast.literal_eval(x)


test_polyline = test_data['POLYLINE'].apply(polyline_f)


test_polyline.to_pickle(data_path + 'data/test_polylines.pkl')


polyline


lats = pd.Series([np.array([point[1] for point in poly],
                           dtype=np.float32) for poly in test_polyline])


split[11][2]


np.alltrue([np.allclose(lats[i], split[21][2][i]) for i in range(len(lats))])


# Latitudes check out

longs = pd.Series([np.array([point[0] for point in poly],
                            dtype=np.float32) for poly in test_polyline])


split[12]


np.alltrue([np.allclose(longs[i], split[22][2][i]) for i in range(len(longs))])


# Longitudes check out

test_data['LATITUDE'] = lats


test_data['LONGITUDE'] = longs


# ### Make Validation Set

data.head()


def make_valid_pandas(data):
    valid = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )
    for row in data.itertuples():
        trip_id = row[1]
        call_type = row[2]
        origin_call = row[3]
        origin_stand = row[4]
        taxi_id = row[5]
        time = row[6]
        day_type = row[7]
        missing_data = row[8]
        latitude = row[10]
        longitude = row[11]
        if len(latitude) == 0:
            continue

        for ts in cuts:
            if time <= ts and time + 15 * (len(latitude) - 1) >= ts:
                # keep it
                valid[0].append(trip_id)
                valid[1].append(call_type)
                valid[2].append(origin_call)
                valid[3].append(origin_stand)
                valid[4].append(taxi_id)
                valid[5].append(time)
                valid[6].append(day_type)
                valid[7].append(missing_data)
                n = (ts - time) / 15 + 1
                valid[8].append(latitude[:n])
                valid[9].append(longitude[:n])
                valid[10].append(latitude[-1])
                valid[11].append(longitude[-1])
                valid[12].append(15 * (len(latitude) - 1))
                break
    return pd.DataFrame({
        'TRIP_ID': valid[0],
        'CALL_TYPE': valid[1],
        'ORIGIN_CALL': valid[2],
        'ORIGIN_STAND': valid[3],
        'TAXI_ID': valid[4],
        'TIMESTAMP': valid[5],
        'DAY_TYPE': valid[6],
        'MISSING_DATA': valid[7],
        'LATITUDE': valid[8],
        'LONGITUDE': valid[9],
        'DESTINATION_LATITUDE': valid[10],
        'DESTINATION_LONGITUDE': valid[11],
        'TRAVEL_TIME': valid[12]
    }
    )


valid_data = make_valid_pandas(data)


np.allclose(valid_data['CALL_TYPE'], valid_split[1][2])


np.allclose(valid_data['ORIGIN_CALL'], valid_split[2][2])


np.allclose(valid_data['ORIGIN_STAND'], valid_split[3][2])


np.allclose(valid_data['TAXI_ID'], valid_split[4][2])


np.allclose(valid_data['TIMESTAMP'], valid_split[5][2])


np.allclose(valid_data['DAY_TYPE'], valid_split[6][2])


np.allclose(valid_data['MISSING_DATA'], valid_split[7][2])


np.alltrue([np.allclose(valid_data['LATITUDE'][i], valid_split[8][2][i])
            for i in range(0, len(valid_data['LATITUDE']))])


np.alltrue([np.allclose(valid_data['LONGITUDE'][i], valid_split[9][2][i])
            for i in range(0, len(valid_data['LATITUDE']))])


np.allclose(valid_data['DESTINATION_LATITUDE'], valid_split[10][2])


np.allclose(valid_data['DESTINATION_LONGITUDE'], valid_split[11][2])


np.allclose(valid_data['TRAVEL_TIME'], valid_split[12][2])


# Values check out. Yay

# # Clustering

# ## Original Data

# ### Mean Shift Clusters

from sklearn.cluster import MeanShift, estimate_bandwidth


split[11][1]


dests = []
for i in range(0, len(split[5][2])):
    if len(split[11][2][i]) == 0:
        continue
    dests.append([split[11][2][i][-1], split[12][2][i][-1]])
pts = numpy.array(dests)


pts.shape


bw = 0.001


ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(pts)
cluster_centers = ms.cluster_centers_


cluster_centers.shape


# ## Pandas

dests = []
for row in data[['LATITUDE', 'LONGITUDE']].itertuples():
    if len(row[1]) == 0:
        continue
    dests.append([row[1][-1], row[2][-1]])
pts2 = numpy.array(dests)


np.allclose(pts, pts2)


# The points are the same. Hooray!

np.save(data_path + 'data/cluster_centers.npy', cluster_centers)


# # Feature Extraction

# ## Original Data

(1, 2) + (3,)


split[8][1]


valid_split[5][2]


def get_date_data(split):
    tmp = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )
    for ts in split[8][2]:
        date = datetime.datetime.utcfromtimestamp(ts)
        yearweek = date.isocalendar()[1] - 1
        tmp[0].append(numpy.int8(51 if yearweek == 52 else yearweek))
        tmp[1].append(numpy.int8(date.weekday()))
        tmp[2].append(numpy.int8(date.hour * 4 + date.minute / 15))
    for ts in split[18][2]:
        date = datetime.datetime.utcfromtimestamp(ts)
        yearweek = date.isocalendar()[1] - 1
        tmp[3].append(numpy.int8(51 if yearweek == 52 else yearweek))
        tmp[4].append(numpy.int8(date.weekday()))
        tmp[5].append(numpy.int8(date.hour * 4 + date.minute / 15))
    for ts in valid_split[5][2]:
        date = datetime.datetime.utcfromtimestamp(ts)
        yearweek = date.isocalendar()[1] - 1
        tmp[6].append(numpy.int8(51 if yearweek == 52 else yearweek))
        tmp[7].append(numpy.int8(date.weekday()))
        tmp[8].append(numpy.int8(date.hour * 4 + date.minute / 15))
    return (
        ('train', 'yearweek', np.array(tmp[0])),
        ('train', 'weekday', np.array(tmp[1])),
        ('train', 'quarterhour', np.array(tmp[2])),
        ('test', 'yearweek', np.array(tmp[3])),
        ('test', 'weekday', np.array(tmp[4])),
        ('test', 'quarterhour', np.array(tmp[5])),
        ('valid', 'yearweek', np.array(tmp[6])),
        ('valid', 'weekday', np.array(tmp[7])),
        ('valid', 'quarterhour', np.array(tmp[8]))
    )


dates = get_date_data(split)


dates[3][2].shape


dates[2][2].shape[0]


trn_size = dates[2][2].shape[0]


test_size = dates[4][2].shape[0]


val_size = dates[6][2].shape[0]


_size


train_gps_mean = [np.concatenate([split[11][2][i] for i in range(trn_size)]).mean(),
                  np.concatenate([split[12][2][i] for i in range(trn_size)]).mean()]


train_gps_std = [np.concatenate([split[11][2][i] for i in range(trn_size)]).std(),
                 np.concatenate([split[12][2][i] for i in range(trn_size)]).std()]


def at_least_k(k, v, pad_at_begin, is_longitude):
    if len(v) == 0:
        v = numpy.array([train_gps_mean[1 if is_longitude else 0]])
    if len(v) < k:
        if pad_at_begin:
            v = numpy.concatenate((numpy.full((k - len(v),), v[0]), v))
        else:
            v = numpy.concatenate((v, numpy.full((k - len(v),), v[-1])))
    return v


valid_split[9]


import theano


def get_first_last_k(split, k):
    first_k = (
        [],
        [],
        [],
        [],
        [],
        []
    )
    last_k = (
        [],
        [],
        [],
        [],
        [],
        []
    )
    for i in range(trn_size):
        first_k[0].append(
            np.array(
                at_least_k(
                    k,
                    split[11][2][i],
                    False,
                    False)[
                    :k]))
        first_k[1].append(
            np.array(
                at_least_k(
                    k,
                    split[12][2][i],
                    False,
                    True)[
                    :k]))
        last_k[0].append(
            np.array(at_least_k(k, split[11][2][i], True, False)[-k:]))
        last_k[1].append(
            np.array(at_least_k(k, split[12][2][i], True, True)[-k:]))
    for i in range(test_size):
        first_k[2].append(
            np.array(
                at_least_k(
                    k,
                    split[21][2][i],
                    False,
                    False)[
                    :k]))
        first_k[3].append(
            np.array(
                at_least_k(
                    k,
                    split[22][2][i],
                    False,
                    True)[
                    :k]))
        last_k[2].append(
            np.array(at_least_k(k, split[21][2][i], True, False)[-k:]))
        last_k[3].append(
            np.array(at_least_k(k, split[22][2][i], True, True)[-k:]))
    for i in range(val_size):
        first_k[4].append(
            np.array(
                at_least_k(
                    k,
                    valid_split[8][2][i],
                    False,
                    False)[
                    :k]))
        first_k[5].append(
            np.array(
                at_least_k(
                    k,
                    valid_split[9][2][i],
                    False,
                    True)[
                    :k]))
        last_k[4].append(
            np.array(at_least_k(k, valid_split[8][2][i], True, False)[-k:]))
        last_k[5].append(
            np.array(at_least_k(k, valid_split[9][2][i], True, True)[-k:]))
    return (
        ('train', 'first_latitude', np.array(first_k[0])),
        ('train', 'first_longitude', np.array(first_k[1])),
        ('train', 'last_latitude', np.array(last_k[0])),
        ('train', 'last_longitude', np.array(last_k[1])),
        ('test', 'first_latitude', np.array(first_k[2])),
        ('test', 'first_longitude', np.array(first_k[3])),
        ('test', 'last_latitude', np.array(last_k[2])),
        ('test', 'last_longitude', np.array(last_k[3])),
        ('valid', 'first_latitude', np.array(first_k[4])),
        ('valid', 'first_longitude', np.array(first_k[5])),
        ('valid', 'last_latitude', np.array(last_k[4])),
        ('valid', 'last_longitude', np.array(last_k[5])))


import warnings
warnings.filterwarnings("ignore")


coords = get_first_last_k(split, 5)


coords[0][2].shape


coords[4][2].shape


coords[4]


valid_split[8][2][0][-1]


valid_split[10][2][0]


np.sum([len(l) for l in data['LATITUDE']])


# # Move forward with Pandas

# ### Prepare Train

data.to_pickle(data_path + 'data/train_data.pkl')


data['DAY_OF_WEEK'] = data['TIMESTAMP'].apply(
    lambda t: datetime.datetime.fromtimestamp(t).weekday())


data['QUARTER_HOUR'] = data['TIMESTAMP'].apply(
    lambda t: int(
        (datetime.datetime.fromtimestamp(t).hour *
         60 +
         datetime.datetime.fromtimestamp(t).minute) /
        15))


data['WEEK_OF_YEAR'] = data['TIMESTAMP'].apply(
    lambda t: datetime.datetime.fromtimestamp(t).isocalendar()[1])


data['DESTINATION_LATITUDE'] = data['LATITUDE'].apply(
    lambda l: l[-1] if len(l) > 0 else np.nan)


data['DESTINATION_LONGITUDE'] = data['LONGITUDE'].apply(
    lambda l: l[-1] if len(l) > 0 else np.nan)


data = data.dropna()


train_gps_mean


data['LATITUDE'] = data['LATITUDE'].apply(
    lambda l: (l - train_gps_mean[0]) / train_gps_std[0])


data['LONGITUDE'] = data['LONGITUDE'].apply(
    lambda l: (l - train_gps_mean[1]) / train_gps_std[1])


data.columns


data['CALL_TYPE'].unique()


def at_least_k(k, v, pad_at_begin, is_longitude):
    if len(v) == 0:
        v = numpy.array([train_gps_mean[1 if is_longitude else 0]])
    if len(v) < k:
        if pad_at_begin:
            v = numpy.concatenate((numpy.full((k - len(v),), v[0]), v))
        else:
            v = numpy.concatenate((v, numpy.full((k - len(v),), v[-1])))
    return v


origin_call = []
origin_stand = []
taxi_id = []
day_of_week = []
quarter_hour = []
week_of_year = []
day_type = []
first_latitude = []
first_longitude = []
last_latitude = []
last_longitude = []
destination_latitude = []
destination_longitude = []


k = 5


def

origin_call = []
origin_stand = []
taxi_id = []
day_of_week = []
quarter_hour = []
week_of_year = []
day_type = []
first_latitude = []
first_longitude = []
last_latitude = []
last_longitude = []
destination_latitude = []
destination_longitude = []

for i in data.index:
    latitude = data['LATITUDE'][i][:-1]
    longitude = data['LONGITUDE'][i][:-1]
    l = len(latitude)
    if l == 0:
        continue
    if l < 100:
        for j in range(l):
            first_latitude.append(
                np.array(at_least_k(k, latitude[:j + 1], False, False)[:k]))
            first_longitude.append(
                np.array(at_least_k(k, longitude[:j + 1], False, True)[:k]))
            last_latitude.append(
                np.array(at_least_k(k, latitude[:j + 1], False, False)[-k:]))
            last_longitude.append(
                np.array(at_least_k(k, longitude[:j + 1], False, True)[-k:]))
            origin_call.append(data['ORIGIN_CALL'][i])
            origin_stand.append(data['ORIGIN_STAND'][i])
            taxi_id.append(data['TAXI_ID'][i])
            day_of_week.append(data['DAY_OF_WEEK'][i])
            quarter_hour.append(data['QUARTER_HOUR'][i])
            week_of_year.append(data['WEEK_OF_YEAR'][i])
            day_type.append(data['DAY_TYPE'][i])
            destination_latitude.append(data['DESTINATION_LATITUDE'][i])
            destination_longitude.append(data['DESTINATION_LONGITUDE'][i])
    else:
        indices = np.random.choice(range(l), 100, replace=False)
        for j in indices:
            first_latitude.append(
                np.array(at_least_k(k, latitude[:j + 1], False, False)[:k]))
            first_longitude.append(
                np.array(at_least_k(k, longitude[:j + 1], False, True)[:k]))
            last_latitude.append(
                np.array(at_least_k(k, latitude[:j + 1], False, False)[-k:]))
            last_longitude.append(
                np.array(at_least_k(k, longitude[:j + 1], False, True)[-k:]))
            origin_call.append(data['ORIGIN_CALL'][i])
            origin_stand.append(data['ORIGIN_STAND'][i])
            taxi_id.append(data['TAXI_ID'][i])
            day_of_week.append(data['DAY_OF_WEEK'][i])
            quarter_hour.append(data['QUARTER_HOUR'][i])
            week_of_year.append(data['WEEK_OF_YEAR'][i])
            day_type.append(data['DAY_TYPE'][i])
            destination_latitude.append(data['DESTINATION_LATITUDE'][i])
            destination_longitude.append(data['DESTINATION_LONGITUDE'][i])


print('I"m finished')


5


len(origin_stand)


data['LATITUDE'][0]


first_longitude[:22]


last_longitude[:22]


first_latitude[:22]


last_latitude[:22]


valid_data.to_pickle(data_path + 'data/valid_data.pkl')


test_data.to_pickle(data_path + 'data/test_data.pkl')


# # MODEL

n_origin_call = len(data['ORIGIN_CALL'].unique())
n_taxi_id = len(data['TAXI_ID'].unique())
n_origin_stand = len(data['ORIGIN_STAND'].unique())
n_quarter_hour = len(data['QUARTER_HOUR'].unique())
n_day_of_week = len(data['DAY_OF_WEEK'].unique())
n_week_of_year = len(data['WEEK_OF_YEAR'])
n_day_type = 3


def latitude_sum(x): return np.dot(x, cluster_centers[0])


def longitude_sum(x): return np.dot(x, cluster_centers[1])


def taxi_mlp(k, shp=cluster_centers.shape[0]):

    first_lat_in = Input(shape(k,))
    last_lat_in = Input(shape(k,))
    first_long_in = Input(shape(k,))
    last_long_in = Input(shape(k,))

    center_lats = Input(shape=(shp,))
    center_longs = Input(shape=(shp,))

    emb_names = [
        'origin_call',
        'taxi_ID',
        "origin_stand",
        "quarter_hour",
        "day_of_week",
        "week_of_year",
        "day_type"]
    emb_ins = [
        n_origin_call + 1,
        n_taxi_id + 1,
        n_origin_stand + 1,
        n_quarter_hour + 1,
        n_day_of_week + 1,
        n_week_of_year + 1,
        n_day_type + 1]
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

    x = merge([first_lat_in, last_lat_in, first_long_in, last_long_in] +
              [Flatten()(e[1]) for e in embs], mode='concat')

    x = Dense(500, activation='relu')(x)

    x = Dense(shp, activation='softmax')(x)

    # CHECK ON CLUSTERS!!!!

    y_latitude = Lambda(latitude_sum, (1,))(x)
    y_longitude = Lambda(longitude_sum, (1,))(x)

    return Model(input=[first_lat_in, last_lat_in, first_long_in, last_long_in] + [e[0] for e in embs] + [center_longs, center_lats],
                 output=[y_latitude, y_longitude])
