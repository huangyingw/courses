
# coding: utf-8

get_ipython().magic(u'matplotlib inline')
import utils
reload(utils)
from utils import *
from __future__ import division, print_function


path = 'data/glove/'
res_path = path + 'results/'


# ## Preprocessing

# This section shows how we processed the original glove text files.
# However, there's no need for you to do this, since we provide the
# [pre-processed glove data](www.platform.ai/models/glove).

def get_glove(name):
    with open(path + 'glove.' + name + '.txt', 'r') as f:
        lines = [line.split() for line in f]
    words = [d[0] for d in lines]
    vecs = np.stack(np.array(d[1:], dtype=np.float32) for d in lines)
    wordidx = {o: i for i, o in enumerate(words)}
    save_array(res_path + name + '.dat', vecs)
    pickle.dump(words, open(res_path + name + '_words.pkl', 'wb'))
    pickle.dump(wordidx, open(res_path + name + '_idx.pkl', 'wb'))


get_glove('6B.50d')
get_glove('6B.100d')
get_glove('6B.200d')
get_glove('6B.300d')


# ## Looking at the vectors

# After you've downloaded the [pre-processed glove data](www.platform.ai/models/glove), you should use `tar -zxf` to untar them, and put them in the path that {res_path} points to. (If you don't have a great internet connection, feel free to only download the 50d version, since that's what we'll be using in class).
#
# Then the following function will return the word vectors as a matrix,
# the word list, and the mapping from word to index.

def load_glove(loc):
    return (load_array(loc + '.dat'),
            pickle.load(open(loc + '_words.pkl', 'rb')),
            pickle.load(open(loc + '_idx.pkl', 'rb')))


vecs, words, wordidx = load_glove(res_path + '6B.50d')
vecs.shape


# Here's the first 25 "words" in glove.

' '.join(words[:25])


# This is how you can look up a word vector.

def w2v(w): return vecs[wordidx[w]]


w2v('of')


# Just for fun, let's take a look at a 2d projection of the first 350
# words, using [T-SNE](http://distill.pub/2016/misread-tsne/).

reload(sys)
sys.setdefaultencoding('utf8')


tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(vecs[:500])

start = 0
end = 350
dat = Y[start:end]
plt.figure(figsize=(15, 15))
plt.scatter(dat[:, 0], dat[:, 1])
for label, x, y in zip(words[start:end], dat[:, 0], dat[:, 1]):
    plt.text(x, y, label, color=np.random.rand(3) * 0.7,
             fontsize=14)
plt.show()
