
# coding: utf-8

from theano.sandbox import cuda
cuda.use('gpu2')


get_ipython().magic(u'matplotlib inline')
import utils
reload(utils)
from utils import *
from __future__ import division, print_function


from keras.layers import TimeDistributed, Activation
from numpy.random import choice


# ## Setup

# We haven't really looked into the detail of how this works yet - so this
# is provided for self-study for those who are interested. We'll look at
# it closely next week.

path = get_file(
    'nietzsche.txt',
    origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))


get_ipython().system(u'tail {path} -n25')


#path = 'data/wiki/'
#text = open(path+'small.txt').read().lower()
#print('corpus length:', len(text))

#text = text[0:1000000]


chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
print('total chars:', vocab_size)


chars.insert(0, "\0")


''.join(chars[1:-6])


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


idx = [char_indices[c] for c in text]


idx[:10]


''.join(indices_char[i] for i in idx[:70])


# ## Preprocess and create model

maxlen = 40
sentences = []
next_chars = []
for i in range(0, len(idx) - maxlen + 1):
    sentences.append(idx[i: i + maxlen])
    next_chars.append(idx[i + 1: i + maxlen + 1])
print('nb sequences:', len(sentences))


sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])


sentences.shape, next_chars.shape


n_fac = 24


model = Sequential([
    Embedding(vocab_size, n_fac, input_length=maxlen),
    LSTM(512, input_dim=n_fac, return_sequences=True, dropout_U=0.2, dropout_W=0.2,
         consume_less='gpu'),
    Dropout(0.2),
    LSTM(512, return_sequences=True, dropout_U=0.2, dropout_W=0.2,
         consume_less='gpu'),
    Dropout(0.2),
    TimeDistributed(Dense(vocab_size)),
    Activation('softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())


# ## Train

def print_example():
    seed_string = "ethics is a basic foundation of all that"
    for i in range(320):
        x = np.array([char_indices[c]
                      for c in seed_string[-40:]])[np.newaxis, :]
        preds = model.predict(x, verbose=0)[0][-1]
        preds = preds / np.sum(preds)
        next_char = choice(chars, p=preds)
        seed_string = seed_string + next_char
    print(seed_string)


model.fit(sentences, np.expand_dims(next_chars, -1), batch_size=64, nb_epoch=1)


print_example()


model.fit(sentences, np.expand_dims(next_chars, -1), batch_size=64, nb_epoch=1)


print_example()


model.optimizer.lr = 0.001


model.fit(sentences, np.expand_dims(next_chars, -1), batch_size=64, nb_epoch=1)


print_example()


model.optimizer.lr = 0.0001


model.fit(sentences, np.expand_dims(next_chars, -1), batch_size=64, nb_epoch=1)


print_example()


model.save_weights('data/char_rnn.h5')


model.optimizer.lr = 0.00001


model.fit(sentences, np.expand_dims(next_chars, -1), batch_size=64, nb_epoch=1)


print_example()


model.fit(sentences, np.expand_dims(next_chars, -1), batch_size=64, nb_epoch=1)


print_example()


print_example()


model.save_weights('data/char_rnn.h5')
