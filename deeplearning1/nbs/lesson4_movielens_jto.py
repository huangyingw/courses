
# coding: utf-8

# ## Fastai deep learning course lesson4 学习笔记（下）
# 分享者：胡智豪
# email: justinhochn@gmail.com

# ## 简述
# 本篇学习笔记是接着lesson4的下篇，上篇介绍了StateFarm的任务，下篇来介绍推荐系统的协同过滤算法，开始接触NLP方面的内容。

# ## 原理介绍
# 建议同学们先下载jeremy的excel表，来看看这个电影的协同过滤算法是如何进行推荐的。
#
# 我对图一和图二的表格进行了填色，这样便于解释各个方块的含义：
# 1. 图一：此表格为用户对他们所看过的电影的**真实评分**。
# 2. 图二蓝色区域：此区域是每个用户对于每部电影的**评分预测**
# 3. 图二绿色区域：左右两块绿色区域分别代表的是用户的特征以及电影的特征，对于每一个用户和每一部电影，这里各用5个数字进行表示。
# 4. 图二黄色区域：左右两块黄色区域分别代表**用户特征的偏置项**以及**电影特征的偏置项**。用户偏置项的意思是，预防有些用户是电影的狂热粉丝，有些用户不怎么看电影，这两个极端导致的评分相差太大。电影偏置项的意思是，预防有些电影是只是明星效应高实际不怎么好看，有些电影很好看但是演员不出名比较冷门，这两种极端情况导致的评分相差太大。
#
# **协同过滤算法的计算流程**
# 1. **用户特征**与**电影特征**进行**矩阵相乘**，并加上用户和电影特征各自的**偏置项（bias）**，获得用户对这部电影的**预测评分**。用图上的解释为：用户和电影的绿色区域相乘，再加上黄色区域的数字。
# 2. **预测评分**与**真实评分**相减，得出评分数值的误差。
# 3. 进行**梯度下降**，不断**更新用户特征及电影特征的数值**，最终使得评分误差最小。

# ## 代码解释
# 本文只对课程内核心代码进行解释，完整的代码可以点击这里下载。

# In[1]:


from theano.sandbox import cuda


# In[2]:


get_ipython().magic(u'matplotlib inline')
import utils
reload(utils)
from utils import *
from __future__ import division, print_funtion


# In[2]:


import pandas as pd
import numpy as np
import os


# In[3]:


path = 'F:/ml-latest-small/'
model_path = path + 'model/'
if not os.path.exists(model_path):
    os.mkdir(model_path)


# In[4]:


batch_size = 64


# ## 设置数据集

# In[5]:


ratings = pd.read_csv(path + 'ratings.csv')
ratings.head()


# In[6]:


len(ratings)


# In[7]:


movie_names = pd.read_csv(
    path + 'movies.csv').set_index('movieId')['title'].to_dict()


# In[8]:


pd.read_csv(path + 'movies.csv').set_index('movieId')['title']


# In[9]:


movie_names


# In[10]:


users = ratings.userId.unique()
movies = ratings.movieId.unique()


# In[11]:


userid2idx = {o: i for i, o in enumerate(users)}
movieid2idx = {o: i for i, o in enumerate(movies)}


# 对ratings的userid和movieid以升序排序，以变成连续的整数，用于后面的embedding层。

# In[12]:


ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])
ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])


# In[13]:


ratings.head()


# In[14]:


user_min, user_max, movie_min, movie_max = (ratings.userId.min(),
                                            ratings.userId.max(), ratings.movieId.min(), ratings.movieId.max())
user_min, user_max, movie_min, movie_max


# In[15]:


n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()
n_users, n_movies


# 设置潜在因子数量

# In[16]:


n_factors = 50


# In[17]:


np.random.seed = 42


# 随机分类出训练集和验证集

# In[18]:


msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]


# In[19]:


len(trn), len(val)


# ## 点乘 Dot Product

# In[78]:


from keras.layers import Input, Dense, merge, Flatten, Activation, Dropout
from keras.models import Model
from keras.layers import Embedding
from keras import regularizers
from keras import optimizers


# In[21]:


user_in = Input(shape=(1,), dtype='int64', name='user_in')
u = Embedding(n_users, n_factors, input_length=1,
              W_regularizer=regularizers.l2(1e-4))(user_in)
movie_in = Input(shape=(1,), dtype='int64', name='movie_in')
m = Embedding(n_movies, n_factors, input_length=1,
              W_regularizer=regularizers.l2(1e-4))(user_in)


# In[22]:


x = merge([u, m], mode='dot')
x = Flatten()(x)
model = Model([user_in, movie_in], x)
model.compile(optimizers.Adam(0.001), loss='mse')


# In[70]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=1,
          validation_data=([val.userId, val.movieId], val.rating))


# In[71]:


model.optimizer.lr = 0.01


# In[72]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=3,
          validation_data=([val.userId, val.movieId], val.rating))


# ## Bias 偏差

# In[26]:


def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1,
                          W_regularizer=regularizers.l2(reg))(inp)


# In[27]:


user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)


# In[28]:


def create_bias(inp, n_in):
    x = Embedding(n_in, 1, input_length=1)(inp)
    return Flatten()(x)


# In[29]:


ub = create_bias(user_in, n_users)
mb = create_bias(movie_in, n_movies)


# In[30]:


x = merge([u, m], mode='dot')
x = Flatten()(x)
x = merge([x, ub], mode='sum')
x = merge([x, mb], mode='sum')
model = Model([user_in, movie_in], x)
model.compile(optimizers.Adam(0.001), loss='mse')


# In[31]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=1,
          validation_data=([val.userId, val.movieId], val.rating))


# In[33]:


model.optimizer.lr = 0.01


# In[34]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=6,
          validation_data=([val.userId, val.movieId], val.rating))


# In[35]:


model.optimizer.lr = 0.001


# In[36]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=6,
          validation_data=([val.userId, val.movieId], val.rating))


# In[37]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=10,
          validation_data=([val.userId, val.movieId], val.rating))


# In[38]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=10,
          validation_data=([val.userId, val.movieId], val.rating))


# In[42]:


model.optimizer.lr = 0.001


# In[43]:


model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=5,
          validation_data=([val.userId, val.movieId], val.rating))


# ## 分析结果

# In[81]:


g = ratings.groupby('movieId')['rating'].count()
topMovies = g.sort_values(ascending=False)[:2000]
topMovies = np.array(topMovies.index)


# In[83]:


get_movie_bias = Model(movie_in, mb)
movie_bias = get_movie_bias.predict(topMovies)
movie_ratings = [(b[0], movie_names[movies[i]])
                 for i, b in zip(topMovies, movie_bias)]


# In[51]:


import operator


# In[53]:


sorted(movie_ratings, key=operator.itemgetter(0))[:15]


# In[54]:


sorted(movie_ratings, key=operator.itemgetter(0), reverse=True)[:15]


# 预测1号观众会为2号电影打多少分

# In[72]:


pred = model.predict([np.array([1]), np.array([2])])


# In[73]:


pred


# ## 神经网络
# 上面加了bias，费了好大劲都跑不到jeremy的0.8，下面利用单个隐藏层的神经网络，分分钟就state-of-the-art了...

# In[75]:


user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)


# In[79]:


x = merge([u, m], mode='concat')
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(70, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)
nn = Model([user_in, movie_in], x)
nn.compile(optimizers.Adam(0.001), loss='mse')


# In[80]:


nn.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=8,
       validation_data=([val.userId, val.movieId], val.rating))


# In[85]:


pred = nn.predict([np.array([1]), np.array([2])])
pred


# In[87]:


nn.save_weights(model_path + 'nn.h5')
