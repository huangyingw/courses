
# coding: utf-8

import importlib
import utils2
importlib.reload(utils2)
from utils2 import *


cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.Session(config=cfg)


a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.multiply(a, b)

print (sess.run(y, feed_dict={a: 3, b: 3}))


# ## Simple regression

num_points = 1000
x_data = np.random.normal(0.0, 0.55, (num_points,))
y_data = x_data * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (num_points,))


plt.scatter(x_data, y_data)


loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()


sess = tf.InteractiveSession(config=cfg)


init.run()
for step in range(8):
    train.run()
    print (step, W.eval(), b.eval())


plt.scatter(x_data, y_data)
plt.plot(x_data, W.eval() * x_data + b.eval())


# ## Variable scope

tf.reset_default_graph()


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [2, 3])
v.name


v.get_shape()


with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [2, 3])


v1 == v


v2 = tf.get_variable("v2", [1])


v2.name


# ## K-means clustering

def plot_data(data, centroids):
    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i, centroid in enumerate(centroids):
        samples = data[i * n_samples:(i + 1) * n_samples]
        plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
        plt.plot(
            centroid[0],
            centroid[1],
            markersize=15,
            marker="x",
            color='k',
            mew=10)
        plt.plot(
            centroid[0],
            centroid[1],
            markersize=10,
            marker="x",
            color='m',
            mew=5)
    plt.show()


n_clusters = 10
n_samples = 250
centroids = np.random.uniform(-35, 35, (n_clusters, 2))
slices = [np.random.multivariate_normal(centroids[i], np.diag([5., 5.]), n_samples)
          for i in range(n_clusters)]
data = np.concatenate(slices).astype(np.float32)


plot_data(data, centroids)


# Numpy Version
def find_initial_centroids_numpy(data, k):
    r_index = np.random.randint(data.shape[0])
    r = data[r_index, :][np.newaxis]
    initial_centroids = []
    for i in range(k):
        diff = data - np.expand_dims(r, 1)
        dist = np.linalg.norm(diff, axis=2)  # 100x2  5x2 --> 100x5x2 --> 100x5
        farthest_index = np.argmax(np.min(dist, axis=0))
        farthest_point = data[farthest_index]
        initial_centroids.append(farthest_point)
        r = np.array(initial_centroids)
    return r


def find_initial_centroids(data, k):
    r_index = tf.random_uniform([1], 0, tf.shape(data)[0], dtype=tf.int32)
    r = tf.expand_dims(data[tf.squeeze(r_index)], dim=1)
    initial_centroids = []
    for i in range(k):
        diff = tf.squared_difference(
            tf.expand_dims(
                data, 0), tf.expand_dims(
                r, 1))
        dist = tf.reduce_sum(diff, axis=2)
        farthest_index = tf.argmax(tf.reduce_min(dist, axis=0), 0)
        farthest_point = data[tf.to_int32(farthest_index)]
        initial_centroids.append(farthest_point)
        r = tf.pack(initial_centroids)
    return r


samples = tf.placeholder(tf.float32, (None, None))


initial_centroids = find_initial_centroids(
    samples, n_clusters).eval({samples: data})


def choose_random_centroids(samples, n_clusters):
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    centroid_indices = random_indices[:n_clusters]
    return tf.gather(samples, centroid_indices)


initial_centroids = find_initial_centroids_numpy(data, n_clusters)


plot_data(data, initial_centroids)


def assign_to_nearest(samples, centroids):
    dim_dists = tf.squared_difference(
        tf.expand_dims(
            samples, 0), tf.expand_dims(
            centroids, 1))
    return tf.argmin(tf.reduce_sum(dim_dists, 2), 0)


def update_centroids(samples, nearest_indices, n_clusters):
    partitions = tf.dynamic_partition(
        samples, tf.to_int32(nearest_indices), n_clusters)
    return tf.concat(0, [tf.expand_dims(tf.reduce_mean(partition, 0), 0)
                         for partition in partitions])


initial_centroids = choose_random_centroids(
    samples, n_clusters).eval({samples: data})


curr_centroids = tf.Variable(initial_centroids)


nearest_indices = assign_to_nearest(samples, curr_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)


tf.global_variables_initializer().run()


c = initial_centroids
for i in range(10):
    # TODO animate
    c2 = curr_centroids.assign(updated_centroids).eval({samples: data})
    if np.allclose(c, c2):
        break
    c = c2


plot_data(data, curr_centroids.eval())


# ## Tf->Keras LR (not working)

class LinRegr(Layer):
    def __init__(self, **kwargs):
        super(LinRegr, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dims = input_shape[1:]
        self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        self.b = tf.Variable(tf.zeros([1]))
        self.built = True

    def call(self, x, mask=None):
        return self.W * x + self.b

    def get_output_shape_for(self, input_shape):
        return input_shape


inp = Input((1,))
res = LinRegr()(inp)


model = Model(inp, res)


model.compile('adam', 'mse')


model.fit(x_data, y_data, verbose=2)
