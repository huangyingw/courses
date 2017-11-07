
# coding: utf-8

# # Babi End to End MemNN

get_ipython().magic(u'matplotlib inline')
import importlib
import utils2
importlib.reload(utils2)
from utils2 import *


np.set_printoptions(4)
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))


# A memory network is a network that can retain information; it can be trained on a structured story and will learn how to answer questions about said story.
#
# This notebook contains an implementation of an end-to-end memory network
# trained on the Babi tasks dataset.

# ## Create datasets

# Code from this section is mainly taken from the babi-memnn example in the keras repo.
#
# * [Popular Science](http://www.popsci.com/facebook-ai)
# * [Slate](http://www.slate.com/blogs/future_tense/2016/06/28/facebook_s_ai_researchers_are_making_bots_smarter_by_giving_them_memory.html)

# The Babi dataset is a collection of tasks (or stories) that detail
# events in a particular format. At the end of each task is a question
# with a labelled answer.

# This section shows how to construct the dataset from the raw data.

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


# This parser formats the story into a time-order labelled sequence of
# sentences, followed by the question and the labelled answer.

def parse_stories(lines):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        if int(nid) == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            substory = [[str(i) + ":"] + x for i, x in enumerate(story) if x]
            data.append((substory, q, a))
            story.append('')
        else:
            story.append(tokenize(line))
    return data


# Next we download and parse the data set.

path = get_file('babi-tasks-v1-2.tar.gz',
                origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)


challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    'two_supporting_facts_1k': 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
# challenge_type = 'two_supporting_facts_10k'
challenge = challenges[challenge_type]


def get_stories(f):
    data = parse_stories(f.readlines())
    return [(story, q, answer) for story, q, answer in data]


train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))


# Here we calculate upper bounds for things like words in sentence,
# sentences in a story, etc. for the corpus, which will be useful later.

stories = train_stories + test_stories


story_maxlen = max((len(s) for x, _, _ in stories for s in x))
story_maxsents = max((len(x) for x, _, _ in stories))
query_maxlen = max(len(x) for _, x, _ in stories)


def do_flatten(el):
    return isinstance(el, collections.Iterable) and not isinstance(
        el, (str, bytes))


def flatten(l):
    for el in l:
        if do_flatten(el):
            yield from flatten(el)
        else:
            yield el


# Create vocabulary of corpus and find size, including a padding element.

vocab = sorted(set(flatten(stories)))
vocab.insert(0, '<PAD>')
vocab_size = len(vocab)


story_maxsents, vocab_size, story_maxlen, query_maxlen, len(
    train_stories), len(test_stories)


# Now the dataset is in the correct format.
#
# Each task in the dataset contains a list of tokenized sentences ordered in time, followed by a question about the story with a given answer.
#
# In the example below, we go can backward through the sentences to find the answer to the question "Where is Daniel?" as sentence 12, the last sentence to mention Daniel.
#
# This task structure is called a "one supporting fact" structure, which
# means that we only need to find one sentence in the story to answer our
# question.

test_stories[534]


# Create an index mapping for the vocabulary.

word_idx = dict((c, i) for i, c in enumerate(vocab))


# Next we vectorize our dataset by mapping words to their indices. We
# enforce consistent dimension by padding vectors up to the upper bounds
# we calculated earlier with our pad element.

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [[word_idx[w] for w in s] for s in story]
        xq = [word_idx[w] for w in query]
        y = [word_idx[answer]]
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return ([pad_sequences(x, maxlen=story_maxlen) for x in X],
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx, story_maxlen, query_maxlen)


def stack_inputs(inputs):
    for i, it in enumerate(inputs):
        inputs[i] = np.concatenate([it,
                                    np.zeros((story_maxsents - it.shape[0], story_maxlen), 'int')])
    return np.stack(inputs)


inputs_train = stack_inputs(inputs_train)
inputs_test = stack_inputs(inputs_test)


inputs_train.shape, inputs_test.shape


# Our inputs for keras.

inps = [inputs_train, queries_train]
val_inps = [inputs_test, queries_test]


# ## Model

# The approach to solving this task relies not only on word embeddings, but sentence embeddings.
#
# The authors of the Babi paper constructed sentence embeddings by simply
# adding up the word embeddings; this might seem naive, but given the
# relatively small length of these sentences we can expect the sum to
# capture relevant information.

emb_dim = 20
parms = {'verbose': 2, 'callbacks': [TQDMNotebookCallback(leave_inner=False)]}


# We use <tt>TimeDistributed</tt> here to apply the embedding to every
# element of the sequence, then the <tt>Lambda</tt> layer adds them up

def emb_sent_bow(inp):
    emb = TimeDistributed(Embedding(vocab_size, emb_dim))(inp)
    return Lambda(lambda x: K.sum(x, 2))(emb)


# The embedding works as desired; the raw input has 10 sentences of 8
# words, and the output has 10 sentence embeddings of length 20.

inp_story = Input((story_maxsents, story_maxlen))
emb_story = emb_sent_bow(inp_story)
inp_story.shape, emb_story.shape


# We do the same for the queries, omitting the <tt>TimeDistributed</tt>
# since there is only one query. We use <tt>Reshape</tt> to match the rank
# of the input.

inp_q = Input((query_maxlen,))
emb_q = Embedding(vocab_size, emb_dim)(inp_q)
emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)
emb_q = Reshape((1, emb_dim))(emb_q)
inp_q.shape, emb_q.shape


# The actual memory network is incredibly simple.
#
# * For each story, we take the dot product of every sentence embedding with that story's query embedding. This gives us a list of numbers proportional to how similar each sentence is with the query.
# * We pass this vector of dot products through a softmax function to return a list of scalars that sum to one and tell us how similar the query is to each sentence.

x = merge([emb_story, emb_q], mode='dot', dot_axes=2)
x = Reshape((story_maxsents,))(x)
x = Activation('softmax')(x)
match = Reshape((story_maxsents, 1))(x)
match.shape


# * Next, we construct a second, separate, embedding function for the sentences
# * We then take the weighted average of these embeddings, using the softmax outputs as weights
# * Finally, we pass this weighted average though a dense layer and classify it w/ a softmax into one of the words in the vocabulary

emb_c = emb_sent_bow(inp_story)
x = merge([match, emb_c], mode='dot', dot_axes=1)
response = Reshape((emb_dim,))(x)
res = Dense(vocab_size, activation='softmax')(response)


answer = Model([inp_story, inp_q], res)


answer.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


# And it works extremely well

K.set_value(answer.optimizer.lr, 1e-2)
hist = answer.fit(inps, answers_train, **parms, nb_epoch=4, batch_size=32,
                  validation_data=(val_inps, answers_test))


# ## Test

# We can look inside our model to see how it's weighting the sentence
# embeddings.

f = Model([inp_story, inp_q], match)


qnum = 6


l_st = len(train_stories[qnum][0]) + 1
train_stories[qnum]


# Sure enough, for the question "Where is Sandra?", the largest weight is the last sentence with the name Sandra, sentence 1 with 0.98.
#
# The second highest is of course the first sentence, which also mentions
# Sandra. But the model has learned that the last occurring sentence is
# what is important; this is why we added the counter at the beginning of
# each sentence.

np.squeeze(f.predict([inputs_train[qnum:qnum + 1],
                      queries_train[qnum:qnum + 1]]))[:l_st]


answers_train[qnum:qnum + 10, 0]


np.argmax(answer.predict(
    [inputs_train[qnum:qnum + 10], queries_train[qnum:qnum + 10]]), 1)


answer.predict([inputs_train[qnum:qnum + 1], queries_train[qnum:qnum + 1]])


vocab[19]


# ## Multi hop

# Next, let's look at an example of a two-supporting fact story.

test_stories[534]


# We can see that the question "Where is the milk?" requires to supporting
# facts to answer, "Daniel traveled to the hallway" and "Daniel left the
# milk there".

inputs_train.shape, inputs_test.shape


# The approach is basically the same; we add more embedding dimensions to
# account for the increased task complexity.

parms = {'verbose': 2, 'callbacks': [TQDMNotebookCallback(leave_inner=False)]}
emb_dim = 30


def emb_sent_bow(inp):
    emb_op = TimeDistributed(Embedding(vocab_size, emb_dim))
    emb = emb_op(inp)
    emb = Lambda(lambda x: K.sum(x, 2))(emb)
#     return Elemwise(0, False)(emb), emb_op
    return emb, emb_op


inp_story = Input((story_maxsents, story_maxlen))
inp_q = Input((query_maxlen,))


emb_story, emb_story_op = emb_sent_bow(inp_story)


emb_q = emb_story_op.layer(inp_q)
emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)


h = Dense(emb_dim)


# The main difference is that we are going to do the same process twice.
# Here we've defined a "hop" as the operation that returns the weighted
# average of the input sentence embeddings.

def one_hop(u, A):
    C, _ = emb_sent_bow(inp_story)
    x = Reshape((1, emb_dim))(u)
    x = merge([A, x], mode='dot', dot_axes=2)
    x = Reshape((story_maxsents,))(x)
    x = Activation('softmax')(x)
    match = Reshape((story_maxsents, 1))(x)

    x = merge([match, C], mode='dot', dot_axes=1)
    x = Reshape((emb_dim,))(x)
    x = h(x)
    x = merge([x, emb_q], 'sum')
    return x, C


# We do one hop, and repeat the process using the resulting weighted sentence average as the new weights.
#
# This works because the first hop allows us to find the first fact relevant to the query, and then we can use that fact to find the next fact that answers the question. In our example, our model would first find the last sentence to mention "milk", and then use the information in that fact to know that it next has to find the last occurrence of "Daniel".
#
# This is facilitated by generating a new embedding function for the input
# story each time we hop. This means that the first embedding is learning
# things that help us find the first fact from the query, and the second
# is helping us find the second fact from the first.

# This approach can be extended to n-supporting factor problems by doing n
# hops.

response, emb_story = one_hop(emb_q, emb_story)
response, emb_story = one_hop(response, emb_story)
# response, emb_story = one_hop(response, emb_story)


res = Dense(vocab_size, activation='softmax')(response)


answer = Model([inp_story, inp_q], res)
answer.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


# Fitting this model can be tricky.

K.set_value(answer.optimizer.lr, 5e-3)
hist = answer.fit(inps, answers_train, **parms, nb_epoch=8, batch_size=32,
                  validation_data=(val_inps, answers_test))


np.array(hist.history['val_acc'])


# ## Custom bias layer

class Elemwise(Layer):
    def __init__(self, axis, is_mult, init='glorot_uniform', **kwargs):
        self.init = initializations.get(init)
        self.axis = axis
        self.is_mult = is_mult
        super(Elemwise, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dims = input_shape[1:]
        dims = [1] * len(input_dims)
        dims[self.axis] = input_dims[self.axis]
        self.b = self.add_weight(dims, self.init, '{}_bo'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        return x * self.b if self.is_mult else x + self.b

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'init': self.init.__name__, 'axis': self.axis}
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
