# import dataset
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np

import tensorflow as tf


# init configs
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1

# load dataset
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])
dropout_rate = tf.placeholder("float")
dropout_rate_hidden = tf.placeholder("float")

# initialize weights

w = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
c1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
r1 = tf.nn.relu(c1)
l1 = tf.nn.max_pool(r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, dropout_rate)

w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
c2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME')
r2 = tf.nn.relu(c2)
l2 = tf.nn.max_pool(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, dropout_rate)

w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
c3 = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME')
r3 = tf.nn.relu(c3)
l3 = tf.nn.max_pool(r3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w4 = tf.Variable(tf.random_normal([128*4*4, 625], stddev=0.01))
w_o = tf.Variable(tf.random_normal([625, 10], stddev=0.01))

l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, dropout_rate)

l4 = tf.nn.relu(tf.matmul(l3, w4))
l4 = tf.nn.dropout(l4, dropout_rate_hidden)

prediction = tf.matmul(l4, w_o)

# define cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()


# lanch training
with tf.Session() as sess:
  sess.run(init)

  for epoch in xrange(training_epochs):
    print 'epoch ', epoch
    avg_cost = 0.0
    total_batch = int(mnist.train.num_examples/batch_size)

    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
      sess.run(optimizer, feed_dict={X: trX[start:end],
                                     Y: trY[start:end],
                                     dropout_rate: 0.8,
                                     dropout_rate_hidden: 0.5})
      batch_cost = sess.run(cost, feed_dict={X: trX[start:end],
                                             Y: trY[start:end],
                                             dropout_rate: 1.0,
                                             dropout_rate_hidden: 0.5})
      avg_cost += batch_cost/total_batch

    if epoch % display_step == 0:
      print "epoch: :%04d" % (epoch + 1), "cost: %.9f" % avg_cost

    test_indices = np.arange(len(teX)) # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:256]
    c_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(c_prediction, "float"))
    print "Accuracy:", accuracy.eval({X: teX[test_indices],
                                      Y: teY[test_indices],
                                      dropout_rate: 1,
                                      dropout_rate_hidden: 1})
  print "Finished optimization"








