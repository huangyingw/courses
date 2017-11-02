
# coding: utf-8

# ## Fastai deep learning course lesson4 学习笔记（上）
# 分享者：胡智豪
# email: justinhochn@gmail.com

# ## 简述
# 本节课jeremy首先讲了一些数学概念，先是利用excel表来演示卷积层的工作，这是非常酷的演示方式，通过在excel表上可以清楚地看到每一层参数的变化情况以及图像的变化情况，同学们可以下载jeremy的excel表来玩玩。然后jeremy讲解了在神经网络更新梯度的过程中，几种最优化方法的对比，包括动量momentum、RMSprop、Adam、Eve等等，建议同学们参考wiki.fast.ai第四课的note以及第四课提供的延展阅读网页，会让你对这几种优化器有更深入的认识。
#
# 然后jeremy介绍了本节课的两个主要任务，第一个任务是进行StateFarm比赛的模型设计，先是使用最简单的网络来进行测试，然后再使用vgg网络来进行调试，最终得出了比较好的结果。第二个任务是介绍协同过滤算法，设计推荐系统，为观影者自动推荐感兴趣的电影。
#
# **本节课的内容相对较多，本节课学习笔记将分成上下两篇。上篇为介绍StateFarm的模型处理方法，下篇为介绍协同过滤算法。同时jeremy开始教授真正科学的模型搭建方法，建议同学们不要漏掉前面的statefarm的测试流程。**
#

# ## 任务流程

# ### 一.设置好数据集
# 1. 首先，原始数据集直接使用kaggle下载StateFarm的数据集，解压的目录如下，请创建必须的目录以确保程序的正确运行。
# ```
# statefarm/
#         vgg16.py
#         utils.py
#         vgg16bn.py
#         statefarm_jto.ipynb
#
#         data/
#             driver_imgs_list.csv
#
#             train/
#                 c0/
#                 c1/
#                 ...
#                 c9/
#
#             valid/
#                 c0/
#                 c1/
#                 ...
#                 c9/
#
#             test/
#
#             models/
#
#             sample/
#                 train/
#                 valid/
#                 test/
#
#             results/
# ```
#
# 2. 原始的数据集是没有valid验证集和sample集的。对于valid集，jeremy建议在训练集中随机挑选3到5位司机的图片，直接“剪切”到valid集中，注意是剪切，不是复制，训练集和验证集的图片不能有重复。设置验证集的代码可以参考论坛上一位同学的notebook，这边可以点击这里下载。SF_Dir_Organization.ipynb
#
# 至于sample集，可以参考jeremy的statefarm_sample.ipynb这个笔记本。
#
# **在本节课，valid集的选择很重要，jeremy选择了3位司机的图片，我在这里选择了5位，如果从训练集取出的司机图片多，验证集的准确率会更稳定，但同时训练集的图片就少了，需要自己权衡。但一般3到5位都问题不大。**

# ### 二.从头设计一个最简单的网络
# 在本节课之前，我们都是一直使用vgg模型来微调和训练，在猫狗大战上我们貌似做得还不错。但到了这一节课，我们面对的不是自然界的花花草草那样有一个明确的分类了，我们这次面对的是10种分心驾驶行为，但从图片上来说，无非就是一个人坐在汽车内，用普通的网络我们可能只能识别出这里有一个人或者这是一辆车，没办法识别出司机在做什么动作。因此为了锻炼我们的建模能力，jeremy开始教我们从头设计网络，最简单的就莫过于一个只有2个参数的线性回归模型了。我们最先开始也使用一个简单的Dense层，做一个简单的拟合。
# ```
# model = Sequential([
#         BatchNormalization(axis=1, input_shape=(3,224,224)),
#         Flatten(),
#         Dense(10, activation='softmax')
#         ])
# ```
# 然后我们开始不停地往这个网络里面加进新东西，例如加入L2正则化、加入单个隐藏层、加入一个卷积层，等等。逐渐使得我们的模型提高准确率，当然这个网络的效果不好，如此简单的模型很难去完成这种任务，但这就是我们算法工程师必备的技能，我们要认真对待。

# ### 三.引用VGG16模型
# Vgg16、googlenet这些imagenet冠军模型的卷积层在识别物体的能力上已经相当成熟，而且这些模型已经调好的参数也可以直接为我们所用。我们可以直接利用这些卷积层来识别物体的特征，然后加上由我们自己搭建的特征处理层，调整各参数，针对这个比赛设计一个新模型。
#

# ### 四.Data Augmentation , Pseudo-labeling and knowledge distillation
# 引入这几个方法到我们的模型中，本质上是扩大训练集的规模，让我们的模型能更好地泛化到不同的情况当中。在本篇学习笔记中主要使用Data Augmentation，因为这一步在本节课里面非常常用而且会有一些陷阱。另外两种方法可以直接参考jeremy的notebook。

# ## 概念解释
# **1.优化器Optimizer的作用**
#
# 神经网络通过优化器来进行权重的梯度更新，传统的梯度更新的方式是人工设定一个学习率，然后让优化器以固定的学习率更新各个权重。这样做的缺点是，一些权重原本可能很快就会到达最佳值，但由于设置了一个固定的学习率，使得这些权重更新过慢，效率低。为解决这个问题，科学家们发明了一些设置动态学习率更新权重的方法，对不同的权重设置不同的学习率，使各权重以自己合适的速度去更新，加快收敛。
#
# 动态学习率权重更新方法有：Adagard、Adam、RMSprop等等，其背后的数学原理在fastai上lesson4的notes里面有详细的解释，同时jeremy也引用了一个其他网站的文章，有过机器学习基础的同学理解起来不难，但要真不理解，可以遵循这位作者的经验：**在数学原理上，Adam、RMSprop和Adadelta的效果差不多，但对比起来，Adam的表现要比RMSprop好，因此Adam对于一般的神经网络来说是最好的选择。**
#
#
# **2.同时想使用图像增广，但同时也想改变Dropout参数，应该怎样做？**
#
# 如果想改变Dropout层参数，那么预计算前面的卷积层是最好的选择，因为模型的计算时间都用在了卷积层。但预计算卷积层就不能直接使用图像增广了，因为使用图像增广会让模型每一次都看到不一样的数据集，这样卷积层输出的feature就会不一样。折中的方法是，利用图像增广先把数据集进行扩展，扩大到3、4倍左右，然后再预计算这个数据集的卷积层输出，这样也相当于稍微使用了图像增广，这是一个妥协的方法。值得一提的是，Jeremy在notebook里面把数据集扩大到了6倍，在我们的P2实例上是做不了的，因为P2实例只有60G内存，一旦图像增广用在了训练集，将会占用50G左右的内存，后面没法进行拼接，导致内存错误。有意思的是，Jeremy在论坛上也告诉我们不要用这种方法来直接扩大数据集，他会教一些特别的技巧，可是我翻了论坛没发现他公布了这个技巧。所以目前就只能按这个方法扩大到4倍（已包括了原训练集），这样就不会提示内存错误。

# ## 代码解释
# 以下是课程notebook核心代码解释，可以直接点击这里，下载本篇ipynb文件。

# In[1]:


from theano.sandbox import cuda
cuda.use('gpu0')


# In[2]:


get_ipython().magic(u'matplotlib inline')
from __future__ import print_function, division
#path = "data/state/"
#path = "/home/ubuntu/statefarm/data/state/sample/"
path = "/home/ubuntu/statefarm/data/state/"
import utils
reload(utils)
from utils import *
from IPython.display import FileLink


# In[3]:


batch_size = 64


# In[6]:


batches = get_batches(path + 'train', batch_size=batch_size)
val_batches = get_batches(
    path + 'valid',
    batch_size=batch_size * 2,
    shuffle=False)


# In[7]:


(val_classes, trn_classes, val_labels, trn_labels,
 val_filenames, filenames, test_filenames) = get_classes(path)


# ## Basic Model - linear model
# 现在先从最简单的线性模型开始，慢慢加入新东西。

# In[9]:


model = Sequential([
    BatchNormalization(axis=1, input_shape=(3, 224, 224)),
    Flatten(),
    Dense(10, activation='softmax')
])


# In[15]:


model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=2,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)


# In[17]:


model.summary()


# try lower learning rate

# In[18]:


model = Sequential([
    BatchNormalization(axis=1, input_shape=(3, 224, 224)),
    Flatten(),
    Dense(10, activation='softmax')
])
model.compile(
    Adam(
        lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=2,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)


# In[19]:


model.optimizer.lr = 0.0001
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=2,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)


# ### Add L2 regularization
# 现在已经严重过拟合，我们加入L2正则化来控制过拟合。

# In[20]:


model = Sequential([
    BatchNormalization(axis=1, input_shape=(3, 224, 224)),
    Flatten(),
    Dense(10, activation='softmax', W_regularizer=l2(0.01))
])
model.compile(
    Adam(
        lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=2,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)


# In[21]:


model.optimizer.lr = 0.0001
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=4,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)


# In[22]:


model.optimizer.lr = 0.001
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=2,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)


# ## Add single hidden layer
# 加入单个隐藏层

# In[23]:


model = Sequential([
    BatchNormalization(axis=1, input_shape=(3, 224, 224)),
    Flatten(),
    Dense(100, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax', W_regularizer=l2(0.01))
])
model.compile(
    Adam(
        lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=2,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)

model.optimizer.lr = 0.01
model.fit_generator(
    batches,
    batches.nb_sample,
    nb_epoch=2,
    validation_data=val_batches,
    nb_val_samples=val_batches.nb_sample)


# ## Add single conv layer
# 加入单个卷积层，包括Conv2D、MaxPooling2D、BN、Conv2D、MaxPooling2D、BN

# In[26]:


def conv1(batches):
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax', W_regularizer=l2(0.01))
    ])

    model.compile(
        Adam(
            lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit_generator(
        batches,
        batches.nb_sample,
        nb_epoch=2,
        validation_data=val_batches,
        nb_val_samples=val_batches.nb_sample)

    model.optimizer.lr = 0.001
    model.fit_generator(
        batches,
        batches.nb_sample,
        nb_epoch=4,
        validation_data=val_batches,
        nb_val_samples=val_batches.nb_sample)
    return model


# In[27]:


model = conv1(batches)


# In[28]:


def conv2(batches):
    model = Sequential([
        BatchNormalization(axis=1, input_shape=(3, 224, 224)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((2, 2)),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax', W_regularizer=l2(0.01))
    ])

    model.compile(
        Adam(
            lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit_generator(
        batches,
        batches.nb_sample,
        nb_epoch=2,
        validation_data=val_batches,
        nb_val_samples=val_batches.nb_sample)

    model.optimizer.lr = 0.001
    model.fit_generator(
        batches,
        batches.nb_sample,
        nb_epoch=4,
        validation_data=val_batches,
        nb_val_samples=val_batches.nb_sample)
    return model


# In[29]:


model = conv2(batches)


# In[30]:


model.summary()


# ## Data Augmentation
# 图像增广，尝试多种增广方式，每次一种，选出每种方式最好的调节参数。

# ### 第一种：宽度偏移，向左或右偏移图像

# In[31]:


gen_t = image.ImageDataGenerator(width_shift_range=0.1)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


# In[32]:


model = conv2(batches)


# In[33]:


model = conv1(batches)


# 从以上两次调用conv1和conv2的情况可以看到，前三轮conv2的准确率比conv1的要高，因为conv2的maxpooling层下采样的大小为2×2， 比conv1的3×3保留了更多的信息，因此准确率会稍高。但到后面几轮发现conv2的准确率开始降低，但conv1的准确率基本维持在0.22左右。

# In[34]:


gen_t = image.ImageDataGenerator(width_shift_range=0.3)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


# In[35]:


model = conv2(batches)


# In[36]:


model = conv1(batches)


# 在以上两次测试可以看到，conv1的准确率对比conv2的准确率来说要相对稳定，我们后面也继续选取conv1作为model，进行后续的测试。这一次的偏移范围选择了0.3，可以看出准确率没有第一次偏移范围为0.1的好，同学们可以多试几次不同的偏移范围以得出最佳数值。从现在开始，我们就以jeremy给出的测试数值作为最佳数值，因篇幅有限，就不再多选参数来测试了。

# ### 第二种：高度偏移，上下移动图像 - Height shift: move the image up and down

# In[37]:


gen_t = image.ImageDataGenerator(height_shift_range=0.05)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


# In[38]:


model = conv1(batches)


# ### 第三种：随机剪切角（最大弧度） - Random shear angles (max in radians)

# In[39]:


gen_t = image.ImageDataGenerator(shear_range=0.1)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


# In[40]:


model = conv1(batches)


# ### 第四种：旋转度：最大度数 -Rotation: max in degrees

# In[41]:


gen_t = image.ImageDataGenerator(rotation_range=15)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


# In[42]:


model = conv1(batches)


# ### 综合使用前面四种处理方法

# In[43]:


gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                                 shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
batches = get_batches(path + 'train', gen_t, batch_size=batch_size)


# In[44]:


model = conv1(batches)


# 改变学习率，再跑多几轮

# In[45]:


model.optimizer.lr = 0.0001
model.fit_generator(batches, batches.nb_sample, nb_epoch=5, validation_data=val_batches,
                    nb_val_samples=val_batches.nb_sample)


# In[47]:


model.fit_generator(batches, batches.nb_sample, nb_epoch=10, validation_data=val_batches,
                    nb_val_samples=val_batches.nb_sample)


# ## 引入VGG16模型

# In[4]:


vgg = Vgg16()
model = vgg.model
last_conv_idx = [i for i, l in enumerate(
    model.layers) if isinstance(l, Convolution2D)][-1]
conv_layers = model.layers[:last_conv_idx + 1]


# In[5]:


conv_model = Sequential(conv_layers)


# In[6]:


(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)


# In[7]:


batches = get_batches(path + 'train', batch_size=batch_size, shuffle=False)
val_batches = get_batches(
    path + 'valid',
    batch_size=batch_size * 2,
    shuffle=False)
test_batches = get_batches(path + 'test', batch_size=batch_size)


# In[10]:


trn_data = get_data(path + 'train')
val_data = get_data(path + 'valid')


# In[12]:


save_array(path + 'results/val.dat', val_data)
save_array(path + 'results/trn.dat', trn_data)


# In[ ]:


val = load_array(path + 'results/val.dat')
trn = load_array(path + 'results/trn.dat')


# In[8]:


conv_feat = conv_model.predict_generator(batches, batches.nb_sample)
conv_val_feat = conv_model.predict_generator(
    val_batches, val_batches.nb_sample)


# In[8]:


conv_test_feat = conv_model.predict_generator(
    test_batches, test_batches.nb_sample)


# In[9]:


save_array(path + 'results/conv_val_feat.dat', conv_val_feat)
save_array(path + 'results/conv_feat.dat', conv_feat)


# In[9]:


save_array(path + 'results/conv_test_feat.dat', conv_test_feat)


# In[13]:


conv_feat = load_array(path + 'results/conv_feat.dat')
conv_val_feat = load_array(path + 'results/conv_val_feat.dat')
conv_val_feat.shape


# 设置bn_model，以修改dropout

# In[14]:


def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p / 2),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(p / 2),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]


# In[33]:


p = 0.5


# In[34]:


bn_model = Sequential(get_bn_layers(p))
bn_model.compile(
    Adam(
        lr=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# 下面这一步是已经运行了72轮才有这个结果，同学们可以多试几轮。

# In[43]:


bn_model.fit(
    conv_feat,
    trn_labels,
    batch_size=batch_size,
    nb_epoch=8,
    validation_data=(
        conv_val_feat,
        val_labels))


# 注意，这两次尝试都已经严重过拟合，可以设置高一点的P，同时设置新学习率看看效果，留给同学们去尝试。

# In[46]:


bn_model.optimizer.lr = 0.0001
bn_model.fit(
    conv_feat,
    trn_labels,
    batch_size=batch_size,
    nb_epoch=4,
    validation_data=(
        conv_val_feat,
        val_labels))


# In[45]:


bn_model.save_weights(path + 'models/conv_72e_p05_2dense100_lr6.h5')


# ### 引入Data Augmentation，预计算卷积层
# 由于使用图像增广技术，在fit model的时候必须使用整个数据集，因此并不能预计算卷积层。但有一个解决方法是，先用图像增广技术把整个训练集扩大到6倍，同时把trn_labels扩大到6倍（因为即使把图像增广用在了训练集，训练集的性质依然没变，猫依然是猫，狗依然是狗，只是图片稍微变了形），那么就相当于在原数据集上使用了图像增广。这样就可以预计算卷积层，后面就不需要等待长时间的训练了。

# In[9]:


gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
                                 shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
da_batches = get_batches(
    path + 'train',
    gen_t,
    batch_size=batch_size,
    shuffle=False)


# In[10]:


da_conv_feat = conv_model.predict_generator(
    da_batches, da_batches.nb_sample * 3)


# In[11]:


save_array(path + 'results/da_conv_feat3.dat', da_conv_feat)


# In[9]:


da_conv_feat = load_array(path + 'results/da_conv_feat2.dat')


# In[14]:


da_conv_feat = np.concatenate([da_conv_feat, conv_feat])


# In[16]:


da_trn_labels = np.concatenate([trn_labels] * 4)


# In[10]:


def get_bn_da_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]


# In[11]:


p = 0.8


# In[12]:


bn_model = Sequential(get_bn_da_layers(p))
bn_model.compile(
    Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# In[53]:


bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=1,
             validation_data=(conv_val_feat, val_labels))


# In[54]:


bn_model.optimizer.lr = 0.01


# In[56]:


bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=2,
             validation_data=(conv_val_feat, val_labels))


# In[57]:


bn_model.save_weights(path + 'models/da_p8_e2_4e.h5')


# In[13]:


bn_model.load_weights(path + 'models/da_p8_e2_4e.h5')


# 不用再训练了，再训练多4轮的话就会产生过拟合，以上这个结果是刚好。

# In[14]:


def do_clip(arr, mx): return np.clip(arr, (1 - mx) / 9, mx)


# In[ ]:


keras.metrics.categorical_crossentropy(
    val_labels, do_clip(val_preds, 0.93)).eval()


# In[ ]:


conv_test_feat = load_array(path + 'results/conv_test_feat.dat')


# In[16]:


preds = bn_model.predict(conv_test_feat, batch_size=batch_size * 2)


# In[17]:


subm = do_clip(preds, 0.93)


# In[18]:


subm_name = path + 'results/subm.gz'


# In[19]:


classes = sorted(batches.class_indices, key=batches.class_indices.get)


# In[20]:


# 注意，原文第二行代码原本为a[4:]，但这对于我们的目录来说应该是a[5:]。
submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'img', [a[5:] for a in test_filenames])
submission.head()


# In[21]:


submission.to_csv(subm_name, index=False, compression='gzip')


# In[22]:


FileLink(subm_name)
