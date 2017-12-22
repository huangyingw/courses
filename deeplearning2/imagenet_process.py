
# coding: utf-8

# # Imagenet Processing in parallel

import importlib
import utils2
importlib.reload(utils2)
from utils2 import *


from bcolz_array_iterator import BcolzArrayIterator
from tqdm import tqdm


limit_mem()


# This is where our full dataset lives. It's slow spinning discs, but lots of room!
#
# **NB:** We can easily switch to and from using a sample. We'll use a sample for *everything*, except the final complete processing (which we'll use fast/expensive compute for, and time on the sample so we know how long it will take).

path = '/data/jhoward/imagenet/full/'
# path = '/data/jhoward/imagenet/sample/'


# This is on a RAID 1 SSD for fast access, so good for resized images and
# feature arrays

dpath = '/data/jhoward/fast/imagenet/full/'
# dpath = '/data/jhoward/fast/imagenet/sample/'
# %mkdir {dpath}


# Note that either way, AWS isn't going to be a great place for doing this kind of analysis - putting a model into production will cost at minimum $600/month for a P2 instance. For that price you can **buy** a GTX 1080 card, which has **double** the performance of the AWS P2 card! And you can set up your slow full data RAID 5 array and your fast preprocessed data RAID 1 array just as you like it. Since you'll want your own servers for production, you may as well use them for training, and benefit from the greater speed, lower cost, and greater control of storage resources.
#
# You can put your server inside a colo facility for very little money, paying just for the network and power. (Cloud providers aren't even allowed to provide GTX 1080's!)
#
# There's little need for distributed computing systems for the vast
# majority of training and production needs in deep learning.

# ## Get word vectors

# First we need to grab some word vectors, to use as our dependent
# variable for the image model (so that the image vectors and word vectors
# will be in the same space). After loading the word vectors, we'll make
# sure that the names of the wordnet/imagenet are in the word list.

# * Be careful not to just follow paper's approach - e.g. here word2vec better than custom wikipedia vectors. word2vec has multi-word tokens like 'golden retriever'
# * Take evaluations shown in papers with a grain of salt, and do your own tests on important bits. E.g. DeVISE (because it's an older paper) used an old and inaccurate image model, and poor word vectors, so recent papers that compare to it aren't so relevent

from gensim.models import word2vec
w2v_path = '/data/jhoward/datasets/nlp/GoogleNews-vectors-negative300'


model = word2vec.KeyedVectors.load_word2vec_format(
    w2v_path + '.bin', binary=True)
model.save_word2vec_format(w2v_path + '.txt', binary=False)


lines = open(w2v_path + '.txt').readlines()


def parse_w2v(l):
    i = l.index(' ')
    return l[:i], np.fromstring(l[i + 1:-2], 'float32', sep=' ')


w2v_list = list(map(parse_w2v, lines[1:]))


pickle.dump(w2v_list, open(path + '../w2vl.pkl', 'wb'))


w2v_list = pickle.load(open(path + '../w2vl.pkl', 'rb'))


# We save the processed file so we can access it quickly in the future.
# It's a good idea to save any intermediate results that take a while to
# recreate, so you can use them both in production and prototyping.

w2v_dict = dict(w2v_list)
words, vectors = zip(*w2v_list)


# Always test your inputs! If you're not sure what to look for, try to
# come up with some kind of reasonableness test.

np.corrcoef(w2v_dict['jeremy'], w2v_dict['Jeremy'])


np.corrcoef(w2v_dict['banana'], w2v_dict['Jeremy'])


lc_w2v = {w.lower(): w2v_dict[w] for w in reversed(words)}


# We're going to map word vectors for each of:
# * The 1000 categories in the Imagenet competition
# * The 82,000 nouns in Wordnet

fpath = get_file('imagenet_class_index.json',
                 'http://www.platform.ai/models/imagenet_class_index.json',
                 cache_subdir='models')
class_dict = json.load(open(fpath))
nclass = len(class_dict)
nclass


classids_1k = dict(class_dict.values())
classid_lines = open(path + '../classids.txt', 'r').readlines()
classids = dict(l.strip().split(' ') for l in classid_lines)
len(classids)


syn_wv = [(k, lc_w2v[v.lower()]) for k, v in classids.items()
          if v.lower() in lc_w2v]
syn_wv_1k = [(k, lc_w2v[v.lower()]) for k, v in classids_1k.items()
             if v.lower() in lc_w2v]
syn2wv = dict(syn_wv)
len(syn2wv)


nomatch = [v[0] for v in class_dict.values() if v[0] not in syn2wv]


# nm_path=path+'train_nm/'
# os.mkdir(nm_path)
# for nm in nomatch: os.rename(path+'train/'+nm, nm_path+nm)


ndim = len(list(syn2wv.values())[0])
ndim


# ## Resize images

# Now that we've got our word vectors, we need a model that can create image vectors. It's nearly always best to start with a pre-train image model, and these require a specific size input. We'll be using resnet, which requires 224x224 sized images. Reading jpegs and resizing them can be slow, so we'll store the result of this.
#
# First we create the filename list for the imagenet archive:

fnames = list(glob.iglob(path + 'train/*/*.JPEG'))
pickle.dump(fnames, open(path + 'fnames.pkl', 'wb'))


# Even scanning a large collection of files is slow, so we save the filenames:

fnames = pickle.load(open(path + 'fnames.pkl', 'rb'))


fnames = np.random.permutation(fnames)


pickle.dump(fnames, open(path + 'fnames_r.pkl', 'wb'))


fnames = pickle.load(open(path + 'fnames_r.pkl', 'rb'))


new_s = 224  # height and width to resize to
n = len(fnames)
n


bc_path = f'{dpath}/trn_resized_{new_s}_r.bc'


bc_path = f'{path}/results/trn_resized_{new_s}_r.bc'


# Using pillow to resize the image (recommendation: install pillow-simd for 600% speedup). To install, force remove the conda installed version, then:
#
#     CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

def _resize(img):
    shortest = min(img.width, img.height)
    resized = np.round(np.multiply(new_s / shortest, img.size)).astype(int)
    return img.resize(resized, Image.BILINEAR)


def resize_img(i):
    img = Image.open(fnames[i])
    s = np.array(img).shape
    if len(s) != 3 or s[2] != 3:
        return
    return _resize(img)


def resize_img_bw(i):
    return _resize(Image.open(fnames[i]).convert('L'))


# Pre-allocate memory in threadlocal storage

tl = threading.local()


tl.place = np.zeros((new_s, new_s, 3), 'uint8')
#tl.place = np.zeros((new_s,new_s), 'uint8')


# Bcolz is amazingly fast, easy to use, and provides a largely numpy-compatible interface. It creates file-backed arrays and are transparently cached in memory.
#
# Create (or open) compressed array for our resized images

arr = bcolz.carray(np.empty((0, new_s, new_s, 3), 'float32'),
                   chunklen=16, mode='w', rootdir=bc_path)


# Function that appends resized image with black border added to longer axis

def get_slice(p, n): return slice((p - n + 1) // 2, p - (p - n) // 2)


def app_img(r):
    tl.place[:] = (np.array(r)[get_slice(r.size[1], new_s), get_slice(r.size[0], new_s)]
                   if r else 0.)
    arr.append(tl.place)


# Serial version
for i in range(2000):
    app_img(resize_img(i))
arr.flush()


# Parallel version
step = 6400
for i in tqdm(range(0, n, step)):
    with ThreadPoolExecutor(max_workers=16) as execr:
        res = execr.map(resize_img, range(i, min(i + step, n)))
        for r in res:
            app_img(r)
    arr.flush()


# Times to process 2000 images that aren't in filesystem cache
# (tpe==ThreadPoolExecutor, ppe==ProcessPoolExecutor; number shows #jobs)

times = [('tpe 16', 3.22), ('tpe 12', 3.65), ('ppe 12', 3.97), ('ppe 8 ', 4.47),
         ('ppe 6 ', 4.89), ('ppe 3 ', 8.03), ('serial', 25.3)]

column_chart(*zip(*times))


arr = bcolz.open(bc_path)


plt.imshow(arr[-2].astype('uint8'))


# We do our prototyping in a notebook, and then use *'Download
# as->Notebook'* to get a python script we can run under tmux. Notebooks
# are great for running small experiments, since it's easy to make lots of
# changes and inspect the results in a wide variety of ways.

# ## Create model

# Now we're ready to create our first model. Step one: create our target
# labels, which is simply a case of grabbing the synset id from the
# filename, and looking up the word vector for each.

def get_synset(f): return f[f.rfind('/') + 1:f.find('_')]


labels = list(map(get_synset, fnames))
labels[:5]


vecs = np.stack([syn2wv[l] for l in labels])
vecs.shape


# We'll be using resnet as our model for these experiments.

rn_mean = np.array([123.68, 116.779, 103.939],
                   dtype=np.float32).reshape((1, 1, 3))
inp = Input((224, 224, 3))
preproc = Lambda(lambda x: (x - rn_mean)[:, :, :, ::-1])(inp)
model = ResNet50(include_top=False, input_tensor=preproc)


# In order to make each step faster, we'll save a couple of intermediate
# activations that we'll be using shortly. First, the last layer before
# the final convolutional bottleneck:

mid_start = model.get_layer('res5b_branch2a')
mid_out = model.layers[model.layers.index(mid_start) - 1]
shp = mid_out.output_shape
shp


# We put an average pooling layer on top to make it a more managable size.

rn_top = Model(model.input, mid_out.output)
rn_top_avg = Sequential([rn_top, AveragePooling2D((7, 7))])


shp = rn_top_avg.output_shape
shp


# We create this intermediate array a batch at a time, so we don't have to
# keep it in memory.

features_mid = bcolz.open(path + 'results/features_mid_1c_r.bc')


features_mid = bcolz.carray(np.empty((0,) + shp[1:]), rootdir=path + 'results/features_mid_1c_r.bc',
                            chunklen=16, mode='w')


def gen_features_mid(dirn):
    gen = (features_mid[i:min(i + 128, n)]
           for i in range(0, len(features_mid), 128))
    for i, batch in tqdm(enumerate(gen)):
        features_mid2.append(rn_top_avg.predict(batch[:, :, ::dirn]))
        if (i % 100 == 99):
            features_mid2.flush()
    features_mid2.flush()


gen_features_mid(1)


gen_features_mid(-1)


features_mid.shape


# Our final layers match the original resnet, although we add on extra
# resnet block at the top as well.

rn_bot_inp = Input(shp[1:])
x = rn_bot_inp
# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
# x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
x = Flatten()(x)
rn_bot = Model(rn_bot_inp, x)
rn_bot.output_shape


for i in range(len(rn_bot.layers) - 2):
    rn_bot.layers[-i - 2].set_weights(model.layers[-i - 2].get_weights())


# We save this layer's results too, although it's smaller so should fit in RAM.

get_ipython().magic(u'time features_last = rn_bot.predict(features_mid, batch_size=128)')


features_last = bcolz.carray(features_last, rootdir=path + 'results/features_last_r.bc',
                             chunklen=64, mode='w')


features_last = bcolz.open(path + 'results/features_last_r.bc')[:]


# We add a linear model on top to predict our word vectors.

lm_inp = Input(shape=(2048,))
lm = Model(lm_inp, Dense(ndim)(lm_inp))


# cosine distance is a good choice for anything involving nearest
# neighbors (which we'll use later).

def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


lm.compile('adam', 'cosine_proximity')


lm.evaluate(features_last, vecs, verbose=0)


lm.fit(features_last, v, verbose=2, nb_epoch=3)


# Be sure to save intermediate weights, to avoid recalculating them

lm.save_weights(path + 'results/lm_cos.h5')


lm.load_weights(path + 'results/lm_cos.h5')


# ## Nearest Neighbors

# Let's use nearest neighbors to look at a couple of examples, to see how
# well it's working. The first NN will be just looking at the word vectors
# of the 1,000 imagenet competition categories.

syns, wvs = list(zip(*syn_wv_1k))
wvs = np.array(wvs)


nn = NearestNeighbors(3, metric='cosine', algorithm='brute').fit(wvs)


nn = LSHForest(20, n_neighbors=3).fit(wvs)


get_ipython().magic(u'time pred_wv = lm.predict(features_last[:10000])')


get_ipython().magic(u'time dist, idxs = nn.kneighbors(pred_wv)')


[[classids[syns[id]] for id in ids] for ids in idxs[190:200]]


plt.imshow(arr[190].astype('uint8'))


# A much harder task is to look up every wordnet synset id.

all_syns, all_wvs = list(zip(*syn_wv))
all_wvs = np.array(all_wvs)


all_nn = LSHForest(20, n_neighbors=3).fit(all_wvs)


get_ipython().magic(u'time dist, idxs = all_nn.kneighbors(pred_wv[:200])')


[[classids[all_syns[id]] for id in ids] for ids in idxs[190:200]]


# ## Fine tune

# To improve things, let's fine tune more layers.

lm_inp2 = Input(shape=(2048,))
lm2 = Model(lm_inp2, Dense(ndim)(lm_inp2))


for l1, l2 in zip(lm.layers, lm2.layers):
    l2.set_weights(l1.get_weights())


rn_bot_seq = Sequential([rn_bot, lm2])
rn_bot_seq.compile('adam', 'cosine_proximity')
rn_bot_seq.output_shape


bc_it = BcolzArrayIterator(features_mid, v, shuffle=True, batch_size=128)


K.set_value(rn_bot_seq.optimizer.lr, 1e-3)


rn_bot_seq.fit_generator(bc_it, bc_it.N, verbose=2, nb_epoch=4)


K.set_value(rn_bot_seq.optimizer.lr, 1e-4)


rn_bot_seq.fit_generator(bc_it, bc_it.N, verbose=2, nb_epoch=8)


K.set_value(rn_bot_seq.optimizer.lr, 1e-5)


rn_bot_seq.fit_generator(bc_it, bc_it.N, verbose=2, nb_epoch=5)


rn_bot_seq.evaluate(features_mid, v, verbose=2)


rn_bot_seq.save_weights(path + 'results/rn_bot_seq_cos.h5')


rn_bot_seq.load_weights(path + 'results/rn_bot_seq_cos.h5')


# ## KNN again

get_ipython().magic(u'time pred_wv = rn_bot_seq.predict(features_mid)')


rng = slice(190, 200)


dist, idxs = nn.kneighbors(pred_wv[rng])


[[classids[syns[id]] for id in ids] for ids in idxs]


dist, idxs = all_nn.kneighbors(pred_wv[rng])


[[classids[all_syns[id]] for id in ids] for ids in idxs]


plt.imshow(arr[rng][1].astype('uint8'))


# ## Text -> Image

# Something very nice about this kind of model is we can go in the other
# direction as well - find images similar to a word or phrase!

img_nn = NearestNeighbors(3, metric='cosine', algorithm='brute').fit(pred_wv)


img_nn2 = LSHForest(20, n_neighbors=3).fit(pred_wv)


word = 'boat'
vec = w2v_dict[word]
# dist, idxs = img_nn.kneighbors(vec.reshape(1,-1))
dist, idxs = img_nn2.kneighbors(vec.reshape(1, -1))


ims = [Image.open(fnames[fn % n]) for fn in idxs[0]]
display(*ims)


vec = (w2v_dict['engine'] + w2v_dict['boat']) / 2
dist, idxs = img_nn.kneighbors(vec.reshape(1, -1))


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    omega = np.arccos(
        np.dot(
            low /
            np.linalg.norm(low),
            high /
            np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + \
        np.sin(val * omega) / so * high


vec = slerp(0.5, w2v_dict['paddle'], w2v_dict['boat'])
dist, idxs = img_nn.kneighbors(vec.reshape(1, -1))


# ## Image -> image

# Since that worked so well, let's try to find images with similar content
# to another image...

ft_model = Sequential([rn_top_avg, rn_bot_seq])


new_file = '/data/jhoward/imagenet/full/valid/n01498041/ILSVRC2012_val_00005642.JPEG'


new_file = '/data/jhoward/imagenet/full/valid/n01440764/ILSVRC2012_val_00007197.JPEG'


new_im = Image.open(new_file).resize((224, 224), Image.BILINEAR)
new_im


vec = ft_model.predict(np.expand_dims(new_im, 0))


dist, idxs = img_nn2.kneighbors(vec)


ims = [Image.open(fnames[fn % n]) for fn in idxs[0]]
display(*ims)
