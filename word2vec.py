'''
inspired by
  https://github.com/hunkim/TensorFlow-Tutorials/blob/master/8_word2vec.py
  https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html
'''

import collections
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 100
embedding_size = 2
num_sampled = 15 # number of negative examples to sample for nce loss

# Sample sentences
sentences = ["the quick brown fox jumped over the lazy dog",
             "I love cats and dogs",
             "we all love cats and dogs",
             "cats and dogs are great",
             "sung likes cats",
             "she loves dogs",
             "cats can be very independent",
             "cats are great companions when they want to be",
             "cats are playful",
             "cats are natural hunters",
             "It's raining cats and dogs",
             "dogs and cats love sung"]

# sentences to words and count
words = " ".join(sentences).split()
count = collections.Counter(words).most_common()
print ("Word count", count[:5])


# Build dictionaries
rdic = [i[0] for i in count] #reverse dic, idx -> word
dic = {w: i for i, w in enumerate(rdic)} #dic, word -> id
voc_size = len(dic)

data = [dic[word] for word in words]
print('Sample data', data[:10], [rdic[t] for t in data[:10]])

# build training data, using neighborhood words (e.g. -1, +1 words in case of window size is 1)
# ([the, brown], quick), ([quick, fox], brown), ....

window_size = 1
word_pairs = []

window_size = min(window_size, int(len(data)-1) / 2)
for i in range(window_size, len(data)-window_size): #from 1 to end-1
  word_pairs.append([data[i-window_size:i] + data[i+1:i+window_size+1], data[i]])

# build skip-gram pairs
skip_gram_pairs = []
for word_pair in word_pairs:
  print word_pair
  for w in word_pair[0]:
    skip_gram_pairs.append([word_pair[1], w])

print('skip-gram pairs', skip_gram_pairs[:5])

X = tf.placeholder(tf.int32, shape=[batch_size])
#nn.nce_loss requires 2d [batch_size,1]
Y = tf.placeholder(tf.int32, shape=[batch_size, 1])

#embed is prediction
with tf.device('/cpu:0'):
  embeddings = tf.Variable(
    tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, X)


# NCE loss
nce_W = tf.Variable(
  tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_B = tf.Variable(tf.zeros([voc_size]))

# Compute loss
loss = tf.reduce_mean(
  tf.nn.nce_loss(nce_W, nce_B, embed, Y, num_sampled, voc_size))

# optimizer adam
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
  tf.initialize_all_variables().run()

  for epoch in xrange(10000):
    total_batch = int(len(skip_gram_pairs)/batch_size)
    avg_cost = 0

    start_r = range(0, len(skip_gram_pairs), batch_size)
    end_r = range(batch_size, len(skip_gram_pairs), batch_size)
    if not end_r:
      end_r = [len(skip_gram_pairs)]

    for start, end in zip(start_r, end_r):
      y_data = []
      x_data = []
      for i in range(start, end):
          x_data.append(skip_gram_pairs[i][0])
          y_data.append([skip_gram_pairs[i][1]])

      sess.run(optimizer, feed_dict={X: x_data,
                                     Y: y_data})

      batch_cost = sess.run(loss, feed_dict={X: x_data,
                                             Y: y_data})
      avg_cost += (batch_cost / total_batch)

    print 'epoch: ', epoch, 'loss: ', avg_cost
  trained_embeddings = embeddings.eval()

if trained_embeddings.shape[1] == 2:

  labels = rdic[:10]
  print labels
  for i, label in enumerate(labels):
    print i
    print label
    x, y = trained_embeddings[i, :]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')


  plt.show()



