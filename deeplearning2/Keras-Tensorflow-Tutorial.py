
# coding: utf-8

# In[1]:


import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)


# In[2]:


# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))


# In[3]:


from keras.layers import Dense

# Keras layers can be called on TensorFlow tensors:
# fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(img)
x = Dense(128, activation='relu')(x)
# output layer with 10 units and a softmax activation
preds = Dense(10, activation='softmax')(x)


# In[4]:


labels = tf.placeholder(tf.float32, shape=(None, 10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))


# In[6]:


from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})


# In[ ]:


mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
