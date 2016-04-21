# import dataset
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf


def xavier_init(n_in, n_out, uniform=True):

  if uniform:
    init_range = tf.sqrt(6.0 / (n_in + n_out))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    std_dev = tf.sqrt(3.0 / (n_in + n_out))
    return tf.random_uniform_initializer(stddev = std_dev)

# init configs
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1
init = 'random'
dropout = True

# load dataset

X = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
Y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# initialize weights

if init == 'xavier':
  w1 = tf.get_variable("w1", shape=[784, 256], initializer=xavier_init(784, 256))
  w2 = tf.get_variable("w2", shape=[256, 256], initializer=xavier_init(256, 256))
  w3 = tf.get_variable("w3", shape=[256, 10], initializer=xavier_init(256,10))
else:
  w1 = tf.Variable(tf.random_normal([784, 256]))
  w2 = tf.Variable(tf.random_normal([256, 256]))
  w3 = tf.Variable(tf.random_normal([256, 10]))


# initialize bias
b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([10]))

# define hypothesis
L1 = tf.nn.relu(tf.add(tf.matmul(X, w1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, w2), b2))

hypothesis = tf.add(tf.matmul(L2, w3), b3)

# define cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()


# lanch training
with tf.Session() as sess:
  sess.run(init)

  for epoch in xrange(training_epochs):
    avg_cost = 0.0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in xrange(total_batch):
      x_batch, y_batch = mnist.train.next_batch(batch_size)
      sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch})
      avg_cost += sess.run(cost, feed_dict={X: x_batch, Y: y_batch})/total_batch

    if epoch % display_step == 0:
      print "epoch: :%04d" % (epoch + 1), "cost: %.9f" % avg_cost
  print "Finished optimization"

  c_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

  accuracy = tf.reduce_mean(tf.cast(c_prediction, "float"))
  print "Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})






