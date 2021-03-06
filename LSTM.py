import numpy as np
import tensorflow as tf
from itertools import izip
import csv
import os,codecs
import itertools
from string import punctuation
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import Counter
from sklearn.utils import shuffle
from nltk import stem
from nltk.tokenize import word_tokenize
LOG_DIR = os.path.join(os.path.dirname(__file__),"log")
if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

""" 
Data Reading & Preprocessing
"""
#read csv without encoding problem
def csv_unireader(f, encoding="utf-8"):
    for row in csv.reader(codecs.iterencode(codecs.iterdecode(f, encoding), "utf-8")):
        yield [e.decode("utf-8") for e in row]

def read_csv(file,num_line):
  row_vec = []
  labels = []
  ids =[]
  with open(file) as f:
    reader = csv_unireader(f)
    i = 0
    for row in reader:
      if i != 0: #skip the first row
        ids.append(row[0])
        labels.append(row[2])
        row_vec.append(row[1])
      if i >= num_line:
        break
      i += 1
  return ids,labels,np.array(row_vec)

def read_csv_test(file,num_line):
  row_vec = []
  labels = []
  ids =[]
  with open(file) as f:
    reader = csv_unireader(f)
    i = 0
    for row in reader:
      if i != 0: #skip the first row
        ids.append(row[0])
        row_vec.append(row[1])
      if i >= num_line:
        break
      i += 1
  return ids,np.array(row_vec)
#get ids, labels, sentences from dataset
ids,labels, sentences = read_csv('train.csv',19579)
ids_test,sentences_test = read_csv_test('test.csv',8392)



"""
Preprocessing for train data
"""
stemmer = stem.PorterStemmer()
prepared = []
tmp = []
for i in range(len(sentences)):
    # tokenization and anti capitalization
    tokens = word_tokenize(sentences[i].lower())

    # remove stop words and words less than length 1 e.g., "." ":" ";"
    tokens_stoprm = filter(lambda w: len(w) > 1, tokens)
    tmp = [stemmer.stem(token) for token in tokens_stoprm]
    prepared.append(" ".join(tmp))
    tmp = []

words = []
# print "here",prepared

# split words in prepared
prepared_tmp = [prepared[i].split() for i in range(len(prepared))]
# print prepared_tmp
words = map(str, list(itertools.chain.from_iterable(prepared_tmp)))
# print words

counts = Counter(words)
# print counts

vocab = sorted(counts, key=counts.get, reverse=True)
# print vocab

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
reviews_ints = []
# print vocab_to_int

for each in prepared:
    reviews_ints.append([vocab_to_int[str(word.lower())] for word in each.split()])

# print reviews_ints

# print labels
# print labels
for i in range(len(labels)):
    if labels[i] == u'EAP':
        labels[i] = [1, 0, 0]
    elif labels[i] == u'HPL':
        labels[i] = [0, 1, 0]
    if labels[i] == u'MWS':
        labels[i] = [0, 0, 1]
labels = np.array(labels)

# labels = np.array([1 if each == u'EAP' elif  each == u'EAP' else 0 for each in labels])
# print labels
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length sentence: {}".format(review_lens[0]))
print("Maximum sentence length: {}".format(max(review_lens)))

non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])

seq_len = 500
features = np.zeros((len(reviews_ints), seq_len), dtype=int)
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]


"""
Preprocessing for test data
"""

prepared_test = []
tmp_test = []
for i in range(len(sentences_test)):
    # tokenization and anti capitalization
    tokens = word_tokenize(sentences_test[i].lower())

    # remove stop words and words less than length 1 e.g., "." ":" ";"
    tokens_stoprm = filter(lambda w: len(w) > 1, tokens)
    tmp_test = [stemmer.stem(token) for token in tokens_stoprm]
    prepared_test.append(" ".join(tmp_test))
    tmp_test = []

words = []
# print "here",prepared

# split words in prepared
prepared_tmp_test = [prepared_test[i].split() for i in range(len(prepared_test))]
# print prepared_tmp
words = map(str, list(itertools.chain.from_iterable(prepared_tmp_test)))
# print words

counts = Counter(words)
# print counts

vocab_test = sorted(counts, key=counts.get, reverse=True)

vocab_to_int_test = {word: ii for ii, word in enumerate(vocab_test, 1)}
reviews_ints_test = []


for each in prepared_test:
    reviews_ints_test.append([vocab_to_int_test[str(word.lower())] for word in each.split()])
review_lens_test = Counter([len(x) for x in reviews_ints_test])
print("Zero-length sentence: {}".format(review_lens_test[0]))
print("Maximum sentence length: {}".format(max(review_lens_test)))

non_zero_idx = [ii for ii, review in enumerate(reviews_ints_test) if len(review) != 0]
reviews_ints_test = [reviews_ints_test[ii] for ii in non_zero_idx]

seq_len_test = 500
features_test = np.zeros((len(reviews_ints_test), seq_len_test), dtype=int)
for i, row in enumerate(reviews_ints_test):
    features_test[i, -len(row):] = np.array(row)[:seq_len_test]


""" 
Training & Validation & Test data Division
"""

split_frac = 300
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

sub_x = features_test #submission data
sub_size = len(sub_x)
print("Feature Shapes:")
print("Train set:{}".format(train_x.shape),
      "Validation set: {}".format(val_x.shape),
      "Test set:{}".format(test_x.shape))

print("label set: {}".format(train_y.shape),
      "Validation label set: {}".format(val_y.shape),
      "Test label set: {}".format(test_y.shape))

print train_x[0].shape


graph = tf.Graph()

def inference(inputs_,batch_size, n_hidden,keep_prob,lstm_layers=2):

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)


    with tf.variable_scope("LSTM_layers"):
        tf.get_variable_scope().reuse_variables()
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(lstm_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)

        # Size of the embedding vectors (number of units in the embedding layer)
        embed_size =5555500
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

    predictions = tf.nn.softmax(tf.contrib.layers.fully_connected(outputs[:, -1], 3))
    return predictions

def loss(labels_,predictions):
    loss = tf.losses.mean_squared_error(labels_, predictions)
    return loss

def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step

def accuracy(predictions,labels_):


    predictions = tf.cast(tf.argmax(predictions, 1, name=None), tf.int32)
    predictions = tf.reshape(predictions,[tf.size(predictions),1])
    labels_= tf.cast(tf.argmax(labels_,1, name=None), tf.int32)
    labels_= tf.reshape(labels_,[tf.size(labels_),1])

    correct_pred = tf.equal(predictions,labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


""" 
Model Configuation
"""

n_out = 1
lstm_size = 256
batch_size_int = 500


n_words = len(vocab_to_int) + 1
n_batch = len(train_x) // batch_size_int
N_validation = len(test_y)
N_train = len(train_x)

inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
batch_size = tf.placeholder(tf.int32, shape=[])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

y = inference(inputs_, n_hidden=lstm_size,keep_prob=keep_prob, batch_size=batch_size)
loss = loss(labels_,y)
train_step = training(loss)
accuracy = accuracy(y,labels_)

""" 
Model Learning
"""

epochs =25 

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
pre_val_loss = 0

tf.summary.FileWriter(LOG_DIR,sess.graph)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("./log/test_short/", sess.graph_def)
#print test_y.reshape(len(test_y), 3)

for epoch in range(epochs):
    train_x, train_y = shuffle(train_x,train_y)

    for i in range(n_batch):
        start = i * batch_size_int
        end = start + batch_size_int
        #print train_y[start:end].reshape(len(train_y[start:end]),3)
        #print train_x[start:end]
        inf = sess.run(train_step,feed_dict={
            inputs_: train_x[start:end],
            batch_size: batch_size_int,
            keep_prob: 0.5,
            labels_: train_y[start:end].reshape(len(train_y[start:end]),3)
        })


    # after one epoch,compute the loss
    val_loss = loss.eval(session=sess, feed_dict={
        inputs_: test_x,
        keep_prob: 1.0,
        batch_size: N_validation,
        labels_: test_y.reshape(len(test_y),3)
    })
    print('epoch:', epoch, ' validation loss:', val_loss-pre_val_loss)
    pre_val_loss = val_loss

    # after training,compute the accuracy
    accuracy_tmp = sess.run(accuracy, feed_dict={
        inputs_: train_x,
        keep_prob: 1.0,
        batch_size: N_train,
        labels_: train_y.reshape(len(train_y), 3)
    })
    print "train_accuracy", accuracy_tmp

# after training,compute the accuracy
accuracy = sess.run(accuracy, feed_dict={
        inputs_: test_x,
        keep_prob: 1.0,
        batch_size: N_validation,
        labels_: test_y.reshape(len(test_y), 3)
})
print "accuracy", accuracy
# get inference
print sub_x
prob = sess.run(y, feed_dict={
        inputs_: sub_x,
        keep_prob: 1.0,
        batch_size:sub_size
})


prob = [map(str,row) for row in prob]

for i in range(len(prob)):
    prob[i].insert(0,str(ids_test[i]))

with open("tf_submission_lstm_e"+str(epochs)+".csv", 'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(prob)



