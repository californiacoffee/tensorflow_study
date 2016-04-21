'''
inspired by
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
    https://github.com/hunkim/TensorFlow-Tutorials/blob/master/7_lstm.py
    and its note
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/reccurent_network.ipynb
'''

import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

'''
for using rnn, we consider each image row as a sequence of pixels. So, we have
28 sequences as the image size is (28 x 28) in mnist dataset. Each sequence has
28 steps.
'''

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10


lstm_size = 28
time_step_size = 28

W = tf.Variable(tf.random_normal([lstm_size, 10], stddev=0.01))
B = tf.Variable(tf.random_normal([10], stddev=0.01))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
init_state = tf.placeholder("float", [None, 2*lstm_size])


def model(X, W, B, init_state, lstm_size):
  # X, input shape: (batch_size, input_vec_size, time_step_size)
  XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size

  # XT shape: (input_vec_size, batch_szie, time_step_size)
  XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size)

  # XR shape: (input vec_size, batch_size)
  X_split = tf.split(0, time_step_size, XR) # split them to time_step_size (28 arrays)
  # Each array shape: (batch_size, input_vec_size)

  # Make lstm with lstm_size (each input vector size)
  lstm = rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0)

  # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
  outputs, states = rnn.rnn(lstm, X_split, initial_state=init_state)

  # Get the last output
  return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

# Linear activation
# Get the last output
prediction, state_size  = model(X, W, B, init_state, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

# launch training
with tf.Session() as sess:
  sess.run(init)

  for epoch in xrange(training_iters):
    avg_cost = 0.0
    total_batch = int(mnist.train.num_examples/batch_size)

    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
      sess.run(optimizer, feed_dict={X: trX[start:end],
                                     Y: trY[start:end],
                                     init_state: np.zeros((batch_size, state_size))})
      batch_cost = sess.run(cost, feed_dict={X: trX[start:end],
                                            Y: trY[start:end],
                                            init_state: np.zeros((batch_size, state_size))})
      avg_cost += batch_cost/total_batch

    if epoch % display_step == 0:
      print "epoch: %04d" % (epoch + 1), "cost: %.9f" %avg_cost

      test_indices = np.arange(len(teX)) #0 ~ n_test
      np.random.shuffle(test_indices)
      test_indices = test_indices[0:batch_size] #get first batch_size exampl
      n_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

      accuracy = tf.reduce_mean(tf.cast(n_correct, "float"))
      print "Accuracy: ", sess.run(accuracy, feed_dict={X: teX[test_indices],
                                                      Y: teY[test_indices],
                                                      init_state: np.zeros((batch_size, state_size))})

  print "Finished optimization"
